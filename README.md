# quant_of_BTC

一个用于快速评估 BTC/USDT 等币种趋势策略的精简回测框架，提供数据抓取、指标计算、仓位管理与绩效报告等功能。脚本基于 Binance K 线数据，可通过命令行直接运行回测并导出关键结果。

## 功能概览

- 使用 Binance 公共 REST API 自动拉取历史 K 线数据。
- 内置多种常见指标：移动平均、RSI、ATR、ADX 等。
- 提供可配置的资金管理、止损止盈、回撤控制与交易冷却机制。
- 自动生成权益曲线、成交明细 CSV，可选绘制图形报告。

## 环境要求

- Python 3.9+
- 依赖库：`requests`、`numpy`、`pandas`、`matplotlib`（可选）

可通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

如仓库内暂未提供 `requirements.txt`，可直接安装：

```bash
pip install requests numpy pandas matplotlib
```

## 快速开始

```bash
python quant.py backtest \
  --symbol BTCUSDT \
  --interval 1h \
  --lookback_days 365
```

运行后将在 `outputs/` 目录生成权益曲线、成交明细以及（若已安装 matplotlib）图形化回测报告。

若所在网络无法直接访问 Binance，可先使用 `fetch` 子命令或外部工具下载 K 线数据，再通过 `--input_file` 选项离线回测：

```bash
# 1.（可选）尝试直接拉取并保存 CSV
python quant.py fetch \
  --symbol BTCUSDT \
  --interval 1h \
  --lookback_days 365 \
  --output outputs/btcusdt_1h.csv

# 2. 使用本地文件进行回测
python quant.py backtest \
  --interval 1h \
  --input_file outputs/btcusdt_1h.csv
```

## 常用参数

- `--fast` / `--slow`：快慢均线窗口。
- `--rsi_long_min`、`--adx_min`：动量阈值。
- `--atr_sl_mult`、`--atr_tp1_mult`、`--atr_tp2_mult`：ATR 止损/止盈倍数。
- `--risk_per_trade`：单笔风险占用的资金比例。
- `--accounting`：账户模式，支持 `spot` 或 `futures`。
- `--input_file`：指定已下载的 K 线文件（CSV/JSON/Parquet），用于离线回测。
- `fetch` 子命令支持 `--output` 指定保存路径，后缀自动识别 CSV/JSON/Parquet。

使用 `-h` 查看全部参数说明：

```bash
python quant.py backtest -h
```

## 目录结构

```
quant.py      # 核心回测脚本
outputs/      # 运行后生成的结果文件（首次执行自动创建）
```

## 注意事项

- Binance 公共接口存在频率限制，脚本内部已添加轻量节流，仍建议适当控制拉取范围。
- 若因地区限制导致 `fetch` 命令报错，可使用代理或在其他环境下载后通过 `--input_file` 导入。
- 回测结果仅供研究参考，不构成投资建议。
- 修改策略逻辑或增加依赖时，请同步更新本文档与 `AGENTS.md`。
