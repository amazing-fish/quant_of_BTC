"""量化回测主脚本。

该模块实现了一套针对加密货币趋势策略的快速回测流程，功能包含：

- 通过 Binance 公共 REST API 抓取历史 K 线数据。
- 计算移动平均、RSI、ATR、ADX 等核心指标。
- 执行带仓位控制、冷却机制与多目标止盈的交易模拟。
- 输出权益曲线、成交明细与图形化报告。

命令行入口位于 ``main`` 函数，可运行 ``python quant.py backtest`` 执行回测。
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---- 可选绘图 ----
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:  # pragma: no cover - 仅在缺少 matplotlib 时触发
    HAS_PLOT = False

BINANCE_BASE = "https://api.binance.com"
INTERVALS_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}

SESSION = requests.Session()


def ensure_outputs() -> None:
    """确保输出目录存在。"""

    os.makedirs("outputs", exist_ok=True)


def now_utc() -> datetime:
    """返回当前 UTC 时间。"""

    return datetime.now(tz=timezone.utc)


def to_ms(dt: datetime) -> int:
    """将时间戳转换为毫秒整数。"""

    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    """毫秒时间戳转为 ``datetime``。"""

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _klines_to_df(data: List[List[Any]]) -> pd.DataFrame:
    """将 Binance 返回的原始 K 线列表转换为 DataFrame。"""

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qv",
        "ntrades",
        "tb_base",
        "tb_quote",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[
        ["open_time", "open_dt", "close_dt", "open", "high", "low", "close", "volume", "ntrades"]
    ]


def fetch_klines(
    symbol: str,
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int = 1000,
) -> pd.DataFrame:
    """抓取指定区间的 K 线数据。"""

    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(1000, limit)}
    frames: List[pd.DataFrame] = []

    try:
        if start is None and end is None:
            response = SESSION.get(url, params=params, timeout=12)
            response.raise_for_status()
            frames.append(_klines_to_df(response.json()))
        else:
            if start is None and end is not None:
                minutes = INTERVALS_MINUTES[interval] * (limit - 1)
                start = end - timedelta(minutes=minutes)
            if start is None:
                start = now_utc() - timedelta(days=90)
            start_ms, end_ms = to_ms(start), to_ms(end or now_utc())
            while True:
                request_params = dict(params, startTime=start_ms, endTime=end_ms)
                response = SESSION.get(url, params=request_params, timeout=12)
                response.raise_for_status()
                data = response.json()
                if not data:
                    break
                frames.append(_klines_to_df(data))
                last_open = int(data[-1][0])
                next_open = last_open + INTERVALS_MINUTES[interval] * 60_000
                if next_open >= end_ms:
                    break
                start_ms = next_open
                time.sleep(0.2)
    except Exception as exc:  # pragma: no cover - 网络异常时触发
        print("❌ 数据获取失败:", exc)
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    print(f"✅ 获取K线: {len(result)}")
    return result


def load_klines_from_file(path: str) -> pd.DataFrame:
    """从本地文件加载 K 线数据。"""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix == ".json":
        frame = pd.read_json(file_path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    else:
        raise ValueError("仅支持 CSV/JSON/Parquet 文件")

    if "open_time" not in frame.columns:
        raise ValueError("文件缺少 open_time 列，无法识别 K 线数据")

    frame = frame.copy()
    if "open_dt" not in frame.columns:
        frame["open_dt"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    else:
        frame["open_dt"] = pd.to_datetime(frame["open_dt"], utc=True)

    if "close_dt" not in frame.columns and "close_time" in frame.columns:
        frame["close_dt"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    elif "close_dt" in frame.columns:
        frame["close_dt"] = pd.to_datetime(frame["close_dt"], utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        if column not in frame.columns:
            raise ValueError(f"文件缺少 {column} 列，无法识别 K 线数据")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values("open_time").reset_index(drop=True)


# ---- 指标 ----
def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""

    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均。"""

    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算相对强弱指标 (RSI)。"""

    diff = series.diff()
    upward = diff.clip(lower=0).fillna(0)
    downward = (-diff.clip(upper=0)).fillna(0)
    ru = upward.ewm(alpha=1 / period, adjust=False).mean()
    rd = downward.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波动范围 (ATR)。"""

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均趋向指数 (ADX)。"""

    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_weighted = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_weighted + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_weighted + 1e-12))
    dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di + 1e-12))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---- 配置 ----
@dataclass
class StrategyConfig:
    """策略参数。"""

    fast: int = 30
    slow: int = 90
    rsi_long_min: float = 55.0
    adx_min: float = 18.0
    cross_buffer_atr: float = 0.28
    atr_sl_mult: float = 2.5
    atr_tp1_mult: float = 2.0
    atr_tp2_mult: float = 6.0
    tp1_pct: float = 0.25
    trail_mult: float = 3.0
    risk_per_trade: float = 0.005
    max_bars_in_trade: int = 240
    day_stop_pct: float = 0.015
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    ema_regime: int = 200
    slope_lookback: int = 5
    atrp_floor_const: float = 0.0025
    atrp_dyn_days: int = 60
    atrp_quantile: float = 0.35
    cooldown_bars: int = 3
    dd_scale_on: bool = True
    dd_lvl1: float = 0.10
    dd_lvl2: float = 0.20
    dd_scale1: float = 0.5
    dd_scale2: float = 0.25
    loss_cooldown_bars: int = 8


@dataclass
class BrokerConfig:
    """资金与交易成本设置。"""

    init_cash: float = 10_000.0
    fee_rate: float = 0.001
    slippage_bp: float = 1.0
    accounting: str = "spot"  # spot | futures


@dataclass
class Trade:
    """交易记录。"""

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str
    entry_price: float
    exit_price: float
    qty: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    pnl: float
    fee: float
    reason: str
    entry_fee: float = 0.0
    exit_fee: float = 0.0


@dataclass
class BacktestResult:
    """回测结果集合。"""

    equity_curve: pd.DataFrame
    trades: List[Trade]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为 ``dict``，便于序列化或调试。"""

        return {
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "metrics": self.metrics,
        }


# ---- 回测 ----
class Backtester:
    """执行策略回测。"""

    def __init__(self, df: pd.DataFrame, strategy: StrategyConfig, broker: BrokerConfig, interval: str):
        self.df = df.copy()
        self.s = strategy
        self.b = broker
        self.interval = interval
        self.mins = INTERVALS_MINUTES[interval]
        self._prepare_indicators()

    # --- 预处理 ---
    def _prepare_indicators(self) -> None:
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        self.df["sma_fast"] = sma(close, self.s.fast)
        self.df["sma_slow"] = sma(close, self.s.slow)
        self.df["ema_regime"] = ema(close, self.s.ema_regime)
        self.df["ema_slope"] = self.df["ema_regime"] - self.df["ema_regime"].shift(self.s.slope_lookback)
        self.df["rsi"] = rsi(close, self.s.rsi_period)
        self.df["atr"] = atr(high, low, close, self.s.atr_period)
        self.df["adx"] = adx(high, low, close, self.s.adx_period)
        self.df["atrp"] = self.df["atr"] / (close + 1e-12)

        window_bars = max(30, int((self.s.atrp_dyn_days * 1440) / self.mins))
        dynamic_floor = (
            self.df["atrp"].rolling(window_bars, min_periods=max(10, window_bars // 3)).quantile(self.s.atrp_quantile)
        ).fillna(self.s.atrp_floor_const)
        self.df["atrp_floor_eff"] = np.maximum(self.s.atrp_floor_const, dynamic_floor.values)

        prev_fast = self.df["sma_fast"].shift(1)
        prev_slow = self.df["sma_slow"].shift(1)
        cross_up = (prev_fast <= prev_slow) & (self.df["sma_fast"] > self.df["sma_slow"])
        buffer_ok = close > (self.df["sma_slow"] + self.s.cross_buffer_atr * self.df["atr"])
        regime_ok = (close > self.df["ema_regime"]) & (self.df["ema_slope"] > 0)
        adx_slope = self.df["adx"] - self.df["adx"].shift(self.s.slope_lookback)
        power_ok = (self.df["adx"] >= self.s.adx_min) & (adx_slope > 0)
        vol_ok = self.df["atrp"] >= self.df["atrp_floor_eff"]
        rsi_ok = self.df["rsi"] >= self.s.rsi_long_min
        self.df["long_signal"] = cross_up & buffer_ok & regime_ok & power_ok & vol_ok & rsi_ok

    # --- 工具函数 ---
    def _mark_equity(self, position: Optional[str], qty: float, entry_px: float, price: float) -> float:
        """按当前市价估算权益。"""

        cash = self.cash
        if position == "long":
            if self.mode == "spot":
                return cash + qty * price
            return cash + qty * (price - entry_px)
        return cash

    def _reset_state(self) -> None:
        """重置持仓相关变量。"""

        self.position: Optional[str] = None
        self.qty: float = 0.0
        self.entry_px: float = 0.0
        self.entry_ts: Optional[pd.Timestamp] = None
        self.stop: float = math.nan
        self.tp1: float = math.nan
        self.tp2: float = math.nan
        self.tp1_done: bool = False
        self.bars_in_trade: int = 0
        self.entry_fee_total: float = 0.0
        self.entry_fee_remain: float = 0.0

    def _record_trade(
        self,
        exit_ts: pd.Timestamp,
        exit_px: float,
        qty: float,
        reason: str,
        entry_fee_alloc: float,
    ) -> None:
        """记录成交并更新资金。"""

        fee_rate = self.b.fee_rate
        if self.mode == "spot":
            proceeds = qty * exit_px
            exit_fee = proceeds * fee_rate
            self.cash += proceeds - exit_fee
            gross = (exit_px - self.entry_px) * qty
            pnl = gross - entry_fee_alloc - exit_fee
        else:
            exit_fee = abs(qty) * exit_px * fee_rate
            gross = (exit_px - self.entry_px) * qty
            self.cash += gross - exit_fee
            pnl = gross - entry_fee_alloc - exit_fee

        self.trades.append(
            Trade(
                self.entry_ts or exit_ts,
                exit_ts,
                self.position or "long",
                self.entry_px,
                exit_px,
                qty,
                self.stop,
                self.tp1,
                self.tp2,
                pnl,
                entry_fee_alloc + exit_fee,
                reason,
                entry_fee_alloc,
                exit_fee,
            )
        )
        self.entry_fee_remain = max(0.0, self.entry_fee_remain - entry_fee_alloc)
        self.loss_streak = self.loss_streak + 1 if pnl <= 0 else 0

    def run(self) -> BacktestResult:
        """执行回测并返回结果。"""

        self.cash = self.b.init_cash
        self.mode = self.b.accounting.lower()
        if self.mode not in {"spot", "futures"}:
            raise ValueError(f"暂不支持的账户类型: {self.b.accounting}")
        self.trades: List[Trade] = []
        self.loss_streak = 0
        self._reset_state()

        equity_records: List[Dict[str, Any]] = []
        cooled_until = -1
        fee_rate = self.b.fee_rate
        slippage = self.b.slippage_bp / 10_000.0
        current_day: Optional[datetime.date] = None
        day_anchor: Optional[float] = None
        day_stop = False
        peak = self.cash

        df = self.df
        for i in range(2, len(df) - 1):
            row = df.iloc[i]
            nxt = df.iloc[i + 1]
            ts = row["open_dt"]
            if (current_day is None) or (ts.date() != current_day):
                current_day = ts.date()
                day_anchor = self._mark_equity(self.position, self.qty, self.entry_px, row["close"])
                day_stop = False

            buy_px = nxt["open"] * (1 + slippage)
            sell_px = nxt["open"] * (1 - slippage)

            if self.position and self.tp1_done:
                current_atr = row["atr"]
                if np.isfinite(current_atr):
                    trail = row["close"] - self.s.trail_mult * current_atr
                    self.stop = max(self.stop, trail)
                    breakeven = self.entry_px + self.entry_px * fee_rate * 2.0 + 0.3 * current_atr
                    self.stop = max(self.stop, breakeven)

            if self.position:
                low = row["low"]
                high = row["high"]
                if low <= self.stop:
                    exit_px = max(self.stop * (1 - slippage), 1e-9)
                    self._record_trade(ts, exit_px, self.qty, "stop", self.entry_fee_remain)
                    self._reset_state()
                else:
                    if (not self.tp1_done) and high >= self.tp1 and self.qty > 0:
                        quantity = self.qty * self.s.tp1_pct
                        fill = self.tp1 * (1 - slippage)
                        alloc = self.entry_fee_total * (quantity / max(1e-12, self.qty))
                        self._record_trade(ts, fill, quantity, "tp1", alloc)
                        self.qty -= quantity
                        self.tp1_done = True
                    if self.position and high >= self.tp2 and self.qty > 0:
                        fill = self.tp2 * (1 - slippage)
                        self._record_trade(ts, fill, self.qty, "tp2", self.entry_fee_remain)
                        self._reset_state()
                    if self.position:
                        self.bars_in_trade += 1
                        if self.s.max_bars_in_trade and self.bars_in_trade >= self.s.max_bars_in_trade:
                            self._record_trade(nxt["open_dt"], sell_px, self.qty, "timeout", self.entry_fee_remain)
                            self._reset_state()

            equity_now = self._mark_equity(self.position, self.qty, self.entry_px, row["close"])
            peak = max(peak, equity_now)
            if day_anchor and equity_now <= day_anchor * (1 - self.s.day_stop_pct):
                day_stop = True

            if (not self.position) and (i > cooled_until) and (not day_stop) and row["long_signal"]:
                atr_value = row["atr"]
                if np.isfinite(atr_value) and atr_value > 0:
                    entry_px = buy_px
                    stop = entry_px - self.s.atr_sl_mult * atr_value
                    tp1 = entry_px + self.s.atr_tp1_mult * atr_value
                    tp2 = entry_px + self.s.atr_tp2_mult * atr_value
                    risk_per_unit = entry_px - stop
                    if risk_per_unit > 0:
                        scale = 1.0
                        if self.s.dd_scale_on:
                            drawdown = 1.0 - (equity_now / (peak + 1e-12))
                            if drawdown >= self.s.dd_lvl2:
                                scale = self.s.dd_scale2
                            elif drawdown >= self.s.dd_lvl1:
                                scale = self.s.dd_scale1
                        risk_cash = self.cash * self.s.risk_per_trade * scale
                        qty = max(0.0, math.floor((risk_cash / risk_per_unit) * 1e6) / 1e6)
                        if self.mode == "spot" and entry_px > 0:
                            capital_cap = max(0.0, (self.cash - 1e-6) / entry_px)
                            qty = min(qty, math.floor(capital_cap * 1e6) / 1e6)
                        if qty > 0:
                            entry_fee_total = qty * entry_px * fee_rate
                            if self.mode == "spot":
                                self.cash -= qty * entry_px + entry_fee_total
                            else:
                                self.cash -= entry_fee_total

                            self.position = "long"
                            self.qty = qty
                            self.entry_px = entry_px
                            self.entry_ts = nxt["open_dt"]
                            self.stop = stop
                            self.tp1 = tp1
                            self.tp2 = tp2
                            self.tp1_done = False
                            self.bars_in_trade = 0
                            self.entry_fee_total = entry_fee_total
                            self.entry_fee_remain = entry_fee_total

                            base_cool = self.s.cooldown_bars
                            add_cool = (
                                self.s.loss_cooldown_bars
                                if len(self.trades) >= 2
                                and self.trades[-1].pnl <= 0
                                and self.trades[-2].pnl <= 0
                                else 0
                            )
                            cooled_until = i + base_cool + add_cool

            equity_records.append({"time": row["close_dt"], "equity": equity_now})

        if self.position and self.qty > 0:
            last_price = self.df["close"].iloc[-1] * (1 - slippage)
            last_ts = self.df["open_dt"].iloc[-1]
            self._record_trade(last_ts, last_price, self.qty, "eod_close", self.entry_fee_remain)
            self._reset_state()

        equity_df = pd.DataFrame(equity_records).set_index("time") if equity_records else pd.DataFrame()
        metrics = self._metrics(equity_df["equity"], self.trades) if not equity_df.empty else {}
        pnl_sum = float(sum(trade.pnl for trade in self.trades))
        if not equity_df.empty and abs(equity_df["equity"].iloc[-1] - (self.b.init_cash + pnl_sum)) > 1e-2:
            print("⚠️ 自洽校验存在微小偏差")

        return BacktestResult(equity_df, self.trades, metrics)

    def _metrics(self, equity: pd.Series, trades: List[Trade]) -> Dict[str, Any]:
        """计算绩效指标。"""

        if equity.empty:
            return {}
        daily = equity.resample("1D").last().ffill()
        rets = daily.pct_change().dropna()
        days = max(1, (daily.index[-1] - daily.index[0]).days + 1)
        cagr = (daily.iloc[-1] / daily.iloc[0]) ** (365.0 / days) - 1.0
        sharpe = (rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(365.0)
        roll = daily.cummax()
        max_dd = ((daily / roll) - 1).min()

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / max(1, len(trades))
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = -float(np.mean([t.pnl for t in losses])) if losses else 0.0
        payoff = (avg_win / (avg_loss + 1e-12)) if avg_loss > 0 else float("nan")
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        profit_factor = (gross_profit / (gross_loss + 1e-12)) if gross_loss > 0 else float("nan")
        return {
            "start": str(equity.index[0]),
            "end": str(equity.index[-1]),
            "days": int(days),
            "bars": int(len(equity)),
            "initial_cash": float(daily.iloc[0]),
            "final_equity": float(daily.iloc[-1]),
            "total_return": float(daily.iloc[-1] / daily.iloc[0] - 1.0),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": float(win_rate),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff": float(payoff),
            "profit_factor": float(profit_factor),
        }


def plot_results(eq: pd.DataFrame, trades: List[Trade], symbol: str, interval: str) -> None:
    """绘制回测结果图表。"""

    if not HAS_PLOT or eq.empty:
        return
    ensure_outputs()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{symbol} {interval} 回测结果", fontsize=16, fontweight="bold")
    ax1 = axes[0]
    ax1.plot(eq.index, eq["equity"], lw=2, label="权益")
    ax1.axhline(eq["equity"].iloc[0], ls="--", alpha=0.5, label="初始")
    ax1.set_ylabel("权益")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    roll = eq["equity"].cummax()
    drawdown = (eq["equity"] / roll - 1.0) * 100
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.25)
    ax2.plot(drawdown.index, drawdown, lw=1.5)
    ax2.set_ylabel("回撤(%)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    cumulative = 0.0
    xs: List[pd.Timestamp] = []
    ys: List[float] = []
    for trade in trades:
        if trade.exit_time:
            cumulative += trade.pnl
            xs.append(trade.exit_time)
            ys.append(cumulative)
    if xs:
        ax3.bar(xs, ys, width=0.8, alpha=0.6)
    ax3.axhline(0, lw=0.8)
    ax3.set_ylabel("累计盈亏")
    ax3.grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join("outputs", f"backtest_result_{symbol}_{interval}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def print_report(metrics: Dict[str, Any], trades: List[Trade]) -> None:
    """在终端打印回测摘要。"""

    if not metrics:
        print("❌ 无结果")
        return
    print("=" * 66, "\n📊 回测报告")
    print(
        f"期间: {metrics['start']}→{metrics['end']}  天数:{metrics['days']}  K线:{metrics['bars']}"
    )
    print(
        f"资金: 初始${metrics['initial_cash']:.2f}  期末${metrics['final_equity']:.2f}  "
        f"总收益{metrics['total_return'] * 100:+.2f}%  年化{metrics['cagr'] * 100:+.2f}%  夏普{metrics['sharpe']:.3f}"
    )
    print(f"风险: 最大回撤 {metrics['max_drawdown'] * 100:.2f}%")
    print(
        f"成交: 段落{metrics['trades']}  胜{metrics['wins']}  负{metrics['losses']}  "
        f"胜率{metrics['win_rate'] * 100:.2f}%  盈亏比{metrics['payoff']:.3f}  PF{metrics['profit_factor']:.3f}"
    )
    if trades:
        print("\n最近10条：")
        print(f"{'入场时间':<20}{'方向':<6}{'入':>9}{'出':>9}{'数量':>11}{'盈亏':>13}{'原因':>8}")
        for trade in trades[-10:]:
            entry_time = trade.entry_time.strftime("%Y-%m-%d %H:%M") if trade.entry_time else "N/A"
            print(
                f"{entry_time:<20}{trade.side:<6}{trade.entry_price:>9.2f}{trade.exit_price:>9.2f}"
                f"{trade.qty:>11.6f}{trade.pnl:>13.2f}{trade.reason:>8}"
            )
    print("=" * 66)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """解析命令行时间参数，统一转换为 UTC。"""

    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_data(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    """根据命令行参数加载或下载 K 线数据。"""

    start = _parse_datetime(getattr(args, "start", None))
    end = _parse_datetime(getattr(args, "end", None))

    if getattr(args, "input_file", None):
        try:
            df = load_klines_from_file(args.input_file)
        except (FileNotFoundError, ValueError) as exc:
            print(f"❌ 无法加载本地数据: {exc}")
            return None

        if start is not None:
            df = df[df["open_dt"] >= start]
        if end is not None:
            df = df[df["open_dt"] <= end]
        if df.empty:
            print("❌ 本地数据在指定时间范围内为空")
            return None
        return df

    lookback_days = getattr(args, "lookback_days", None)
    if lookback_days and not start:
        start = now_utc() - timedelta(days=lookback_days)
    if end is None:
        end = now_utc()
    limit = getattr(args, "limit", 1000)
    df = fetch_klines(args.symbol, args.interval, start, end, limit=limit)
    if df.empty:
        print("❌ 无K线")
        return None
    return df


def _build_strategy_from_args(args: argparse.Namespace) -> StrategyConfig:
    """根据命令行参数构建策略配置。"""

    defaults = StrategyConfig()
    overrides: Dict[str, Any] = {}
    for field in dataclasses.fields(StrategyConfig):
        if hasattr(args, field.name):
            overrides[field.name] = getattr(args, field.name)
    return dataclasses.replace(defaults, **overrides)


def _build_broker_from_args(args: argparse.Namespace) -> BrokerConfig:
    """根据命令行参数构建资金配置。"""

    defaults = BrokerConfig()
    overrides: Dict[str, Any] = {}
    for field in dataclasses.fields(BrokerConfig):
        if hasattr(args, field.name):
            overrides[field.name] = getattr(args, field.name)
    return dataclasses.replace(defaults, **overrides)


def _parse_range_spec(spec: Optional[str], caster) -> List[Any]:
    """解析 ``start:end:step`` 或逗号分隔的范围描述。"""

    if not spec:
        return []
    text = spec.strip()
    if not text:
        return []

    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError("范围格式需形如 start:end:step")
        start_raw, end_raw, step_raw = parts
        if caster is int:
            start = int(start_raw)
            end = int(end_raw)
            step = int(step_raw)
            if step <= 0:
                raise ValueError("步长必须为正数")
            if start > end:
                raise ValueError("起始值需不大于结束值")
            values: List[int] = []
            current = start
            while current <= end:
                values.append(current)
                current += step
            return values
        start = float(start_raw)
        end = float(end_raw)
        step = float(step_raw)
        if step <= 0:
            raise ValueError("步长必须为正数")
        if start > end:
            raise ValueError("起始值需不大于结束值")
        values_float: List[float] = []
        current = start
        while current <= end + 1e-9:
            values_float.append(round(current, 10))
            current += step
        return [caster(str(value)) for value in values_float]

    if "," in text:
        return [caster(part.strip()) for part in text.split(",") if part.strip()]
    return [caster(text)]


def run_backtest(args: argparse.Namespace) -> None:
    """根据命令行参数执行回测。"""

    df = _load_data(args)
    if df is None:
        return

    strategy = _build_strategy_from_args(args)
    broker = _build_broker_from_args(args)

    backtester = Backtester(df, strategy, broker, args.interval)
    result = backtester.run()

    ensure_outputs()
    result.equity_curve.to_csv("outputs/equity_curve.csv")
    trades_df = pd.DataFrame([dataclasses.asdict(trade) for trade in result.trades])
    if not trades_df.empty:
        trades_df.to_csv("outputs/trades.csv", index=False)
    if HAS_PLOT:
        plot_results(result.equity_curve, result.trades, args.symbol, args.interval)
    print_report(result.metrics, result.trades)


def run_optimize(args: argparse.Namespace) -> None:
    """执行参数网格搜索优化。"""

    df = _load_data(args)
    if df is None:
        return

    base_strategy = _build_strategy_from_args(args)
    broker = _build_broker_from_args(args)

    try:
        fast_values = _parse_range_spec(getattr(args, "fast_range", None), int)
        slow_values = _parse_range_spec(getattr(args, "slow_range", None), int)
        rsi_values = _parse_range_spec(getattr(args, "rsi_range", None), float)
        adx_values = _parse_range_spec(getattr(args, "adx_range", None), float)
    except ValueError as exc:
        print(f"❌ 范围参数解析失败: {exc}")
        return

    if not fast_values:
        fast_values = [base_strategy.fast]
    if not slow_values:
        slow_values = [base_strategy.slow]
    if not rsi_values:
        rsi_values = [base_strategy.rsi_long_min]
    if not adx_values:
        adx_values = [base_strategy.adx_min]

    combos: List[Tuple[int, int, float, float]] = []
    for fast, slow, rsi, adx in itertools.product(fast_values, slow_values, rsi_values, adx_values):
        if fast >= slow:
            continue
        combos.append((fast, slow, rsi, adx))

    if not combos:
        print("❌ 未生成有效的参数组合，请检查范围设置")
        return

    print(f"🚀 启动优化，共 {len(combos)} 组组合")
    records: List[Dict[str, Any]] = []
    for idx, (fast, slow, rsi, adx) in enumerate(combos, start=1):
        strategy = dataclasses.replace(
            base_strategy,
            fast=fast,
            slow=slow,
            rsi_long_min=rsi,
            adx_min=adx,
        )
        backtester = Backtester(df, strategy, broker, args.interval)
        result = backtester.run()
        metrics = result.metrics
        if not metrics:
            continue
        record = {
            "fast": fast,
            "slow": slow,
            "rsi_long_min": rsi,
            "adx_min": adx,
        }
        record.update(metrics)
        records.append(record)
        if args.verbose:
            ret_pct = metrics.get("total_return", float("nan")) * 100
            sharpe = metrics.get("sharpe", float("nan"))
            print(
                f"[{idx:>4}/{len(combos)}] fast={fast} slow={slow} rsi={rsi:.2f} adx={adx:.2f}"
                f" -> 收益{ret_pct:+.2f}% 夏普{sharpe:.3f}"
            )

    if not records:
        print("❌ 无有效回测结果")
        return

    sort_by = args.sort_by

    def sort_value(item: Dict[str, Any]) -> float:
        value = item.get(sort_by, float("-inf"))
        if isinstance(value, float) and math.isnan(value):
            return float("-inf")
        return float(value)

    records.sort(key=sort_value, reverse=True)

    top_n = min(args.top, len(records))
    print(f"🏁 完成优化，展示前 {top_n} 组（按 {sort_by} 降序）")
    header = (
        f"{'排名':<4}{'fast':>6}{'slow':>6}{'RSI':>8}{'ADX':>8}"
        f"{'收益%':>10}{'夏普':>9}{'回撤%':>10}{'笔数':>8}"
    )
    print(header)
    for rank, row in enumerate(records[:top_n], start=1):
        total_return = row.get("total_return", float("nan")) * 100
        max_dd = row.get("max_drawdown", float("nan")) * 100
        sharpe = row.get("sharpe", float("nan"))
        trades = row.get("trades", 0)
        print(
            f"{rank:<4}{row['fast']:>6}{row['slow']:>6}{row['rsi_long_min']:>8.2f}{row['adx_min']:>8.2f}"
            f"{total_return:>10.2f}{sharpe:>9.3f}{max_dd:>10.2f}{trades:>8}"
        )

    if args.output:
        ensure_outputs()
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(records)
        df_out.to_csv(output_path, index=False)
        print(f"💾 已保存 {len(records)} 条结果至 {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser("BTC量化回测(精简验证版)")
    sub = parser.add_subparsers(dest="cmd")
    strategy_defaults = StrategyConfig()
    broker_defaults = BrokerConfig()
    backtest = sub.add_parser("backtest", help="运行回测")
    backtest.add_argument("--symbol", default="BTCUSDT")
    backtest.add_argument("--interval", default="1h", choices=list(INTERVALS_MINUTES.keys()))
    backtest.add_argument("--lookback_days", type=int, default=365)
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--input_file", help="使用已下载的本地 K 线文件")
    backtest.add_argument("--limit", type=int, default=1000, help="单次 API 拉取的最大 K 线数量")
    backtest.add_argument("--fast", type=int, default=30)
    backtest.add_argument("--slow", type=int, default=90)
    backtest.add_argument("--rsi_long_min", type=float, default=55.0)
    backtest.add_argument("--adx_min", type=float, default=18.0)
    backtest.add_argument("--cross_buffer_atr", type=float, default=0.28)
    backtest.add_argument("--atr_sl_mult", type=float, default=2.5)
    backtest.add_argument("--atr_tp1_mult", type=float, default=2.0)
    backtest.add_argument("--atr_tp2_mult", type=float, default=6.0)
    backtest.add_argument("--tp1_pct", type=float, default=0.25)
    backtest.add_argument("--trail_mult", type=float, default=3.0)
    backtest.add_argument("--risk_per_trade", type=float, default=0.005)
    backtest.add_argument("--max_bars_in_trade", type=int, default=240)
    backtest.add_argument("--day_stop_pct", type=float, default=0.015)
    backtest.add_argument("--init_cash", type=float, default=10_000.0)
    backtest.add_argument("--fee_rate", type=float, default=0.001)
    backtest.add_argument("--slippage_bp", type=float, default=1.0)
    backtest.add_argument("--accounting", choices=["spot", "futures"], default="spot")

    fetch = sub.add_parser("fetch", help="下载 K 线数据并保存")
    fetch.add_argument("--symbol", default="BTCUSDT")
    fetch.add_argument("--interval", default="1h", choices=list(INTERVALS_MINUTES.keys()))
    fetch.add_argument("--lookback_days", type=int, default=365)
    fetch.add_argument("--start")
    fetch.add_argument("--end")
    fetch.add_argument("--limit", type=int, default=1000, help="单次 API 拉取的最大 K 线数量")
    fetch.add_argument("--output", default="outputs/klines.csv", help="输出文件路径")

    optimize = sub.add_parser("optimize", help="网格搜索策略参数")
    optimize.add_argument("--symbol", default="BTCUSDT")
    optimize.add_argument("--interval", default="1h", choices=list(INTERVALS_MINUTES.keys()))
    optimize.add_argument("--lookback_days", type=int, default=365)
    optimize.add_argument("--start")
    optimize.add_argument("--end")
    optimize.add_argument("--input_file", help="使用已下载的本地 K 线文件")
    optimize.add_argument("--limit", type=int, default=1000, help="单次 API 拉取的最大 K 线数量")
    optimize.add_argument("--fast", type=int, default=strategy_defaults.fast)
    optimize.add_argument("--slow", type=int, default=strategy_defaults.slow)
    optimize.add_argument("--rsi_long_min", type=float, default=strategy_defaults.rsi_long_min)
    optimize.add_argument("--adx_min", type=float, default=strategy_defaults.adx_min)
    optimize.add_argument("--cross_buffer_atr", type=float, default=strategy_defaults.cross_buffer_atr)
    optimize.add_argument("--atr_sl_mult", type=float, default=strategy_defaults.atr_sl_mult)
    optimize.add_argument("--atr_tp1_mult", type=float, default=strategy_defaults.atr_tp1_mult)
    optimize.add_argument("--atr_tp2_mult", type=float, default=strategy_defaults.atr_tp2_mult)
    optimize.add_argument("--tp1_pct", type=float, default=strategy_defaults.tp1_pct)
    optimize.add_argument("--trail_mult", type=float, default=strategy_defaults.trail_mult)
    optimize.add_argument("--risk_per_trade", type=float, default=strategy_defaults.risk_per_trade)
    optimize.add_argument("--max_bars_in_trade", type=int, default=strategy_defaults.max_bars_in_trade)
    optimize.add_argument("--day_stop_pct", type=float, default=strategy_defaults.day_stop_pct)
    optimize.add_argument("--init_cash", type=float, default=broker_defaults.init_cash)
    optimize.add_argument("--fee_rate", type=float, default=broker_defaults.fee_rate)
    optimize.add_argument("--slippage_bp", type=float, default=broker_defaults.slippage_bp)
    optimize.add_argument("--accounting", choices=["spot", "futures"], default=broker_defaults.accounting)
    optimize.add_argument("--fast-range", help="快均线范围，格式 start:end:step 或以逗号分隔")
    optimize.add_argument("--slow-range", help="慢均线范围，格式 start:end:step 或以逗号分隔")
    optimize.add_argument("--rsi-range", help="RSI 门槛范围，格式 start:end:step 或以逗号分隔")
    optimize.add_argument("--adx-range", help="ADX 门槛范围，格式 start:end:step 或以逗号分隔")
    optimize.add_argument("--sort-by", choices=["total_return", "sharpe"], default="total_return")
    optimize.add_argument("--top", type=int, default=10, help="打印排名前 N 名结果")
    optimize.add_argument("--output", help="保存完整结果的 CSV 路径，例如 outputs/optimization.csv")
    optimize.add_argument("--verbose", action="store_true", help="逐组合输出中间结果")
    return parser


def run_fetch(args: argparse.Namespace) -> None:
    """下载 Binance K 线并保存到本地文件。"""

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else None
    if args.lookback_days and not start:
        start = now_utc() - timedelta(days=args.lookback_days)
    if end is None:
        end = now_utc()

    df = fetch_klines(args.symbol, args.interval, start, end, limit=args.limit)
    if df.empty:
        print("❌ 未获取到任何数据")
        return

    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records")
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        print("❌ 输出格式仅支持 CSV/JSON/Parquet")
        return

    print(f"✅ 已保存 {len(df)} 条K线至 {output_path}")


def main() -> None:
    """脚本入口。"""

    ensure_outputs()
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "backtest":
        run_backtest(args)
    elif args.cmd == "fetch":
        run_fetch(args)
    elif args.cmd == "optimize":
        run_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
