# quant_of_BTC

`quant_of_BTC` 是一个面向加密货币趋势策略的轻量级回测脚本，集成数据获取、指标计算、交易模拟与报表生成。1.2 版本在继承原有功能的基础上，对日志系统、默认参数与文档结构进行了全面升级，便于快速验证策略想法并复现回测结果。

## 功能概览

- 通过 Binance 公共 REST API 或本地文件加载多周期 K 线数据。
- 计算移动平均、RSI、ATR、ADX 等核心指标，并实现带冷却、止盈止损与资金管理的多目标策略。
- 输出权益曲线、交易明细、图形化报告，并提供 `optimize` 子命令执行参数网格搜索。
- 新增基于 `logging` 的结构化日志，可全局控制日志级别并清晰区分提示、告警与错误信息。

## 环境要求

- Python 3.9+
- 依赖库：`requests`、`numpy`、`pandas`、`matplotlib`

安装依赖：

```bash
pip install requests numpy pandas matplotlib
```

## 快速开始

1. 直接运行默认参数（默认以 Binance 接口获取近一年 1 小时 K 线）：

   ```bash
   python quant.py --log-level INFO backtest --symbol BTCUSDT --interval 1h --lookback_days 365
   ```

2. 使用仓库附带的数据离线回测：

   ```bash
   python quant.py --log-level INFO backtest --interval 1h --input_file btcusdt_1h.csv
   ```

运行后将在 `outputs/` 目录生成：

- `equity_curve.csv`：权益曲线
- `trades.csv`：成交明细（若有交易）
- `backtest_result_<symbol>_<interval>.png`：图形化回测报告

## 版本 1.2 更新摘要

- **日志重构**：引入 `--log-level` 全局参数（支持 `CRITICAL/ERROR/WARNING/INFO/DEBUG`），输出统一采用 `logging` 模块，错误与告警信息清晰可辨，同时新增关键结果的结构化说明。
- **默认参数更新**：基于最新数据网格搜索，默认策略改为 `fast=34`、`slow=100`、`rsi_long_min=58`、`adx_min=18`，其余仓位管理参数保持稳定。
- **性能表现刷新**：使用 `btcusdt_1h.csv` 回测一年数据，可获得约 **+1.15%** 的总收益、**-1.85%** 的最大回撤、**1.511** 的利润因子与 **0.635** 的夏普比率，显著优于旧版本的基准表现。【9e7769†L1-L28】

## 参数与命令速览

- `backtest`：执行回测，常用参数包括：
  - `--fast` / `--slow`：均线窗口。
  - `--rsi_long_min`、`--adx_min`：动量筛选阈值。
  - `--atr_sl_mult`、`--atr_tp1_mult`、`--atr_tp2_mult`：ATR 止损/止盈倍数。
  - `--risk_per_trade`：单笔风险资金比例。
  - `--accounting`：账户模式，支持 `spot` 或 `futures`。
  - `--input_file`：使用本地 K 线文件（CSV/JSON/Parquet）。
- `fetch`：下载 Binance K 线并保存到文件。
- `optimize`：网格搜索策略参数，支持 `--fast-range`/`--slow-range`/`--rsi-range`/`--adx-range` 指定搜索区间，`--sort-by` 选择排序指标，`--top` 指定输出前 N 组。

所有子命令均支持全局 `--log-level` 用于控制输出密度，默认 INFO。

## 参数优化示例

使用仓库自带的 1 小时数据，对快慢均线与动量阈值执行小范围网格搜索：

```bash
python quant.py --log-level INFO optimize \
  --interval 1h --input_file btcusdt_1h.csv \
  --fast-range 30,34,38 --slow-range 90,100,110 \
  --rsi-range 58,60 --adx-range 18,20 \
  --sort-by sharpe --top 10
```

搜索 36 组组合后，表现最优的配置为 `fast=34`、`slow=100`、`rsi_long_min=58`、`adx_min=18`，其收益率约 **1.15%**、夏普 **0.635**，回撤约 **-1.85%**。【1b8ad0†L1-L11】

若需更大范围搜索，可增大范围参数或启用 `--verbose` 查看逐组合结果，注意组合数量上升会显著增加运行时间。

## 常见问题

- Binance 公共接口存在频率限制，脚本已自带轻量节流，仍建议合理控制拉取区间或使用本地数据。
- `matplotlib` 在缺少中文字体时会提示“Glyph missing”告警，不影响结果；如需完整中文显示，可安装合适字体并在脚本中设置。
- 回测结论仅供研究参考，不构成投资建议。

## 目录结构

```text
quant.py      # 核心回测脚本
btcusdt_1h.csv# 示例数据集
outputs/      # 运行后生成的结果文件夹
```

欢迎根据需求继续扩展策略逻辑，并在更新时同步调整文档与协作规范。

