# quant_of_BTC

一个用于快速评估 BTC/USDT 等币种趋势策略的精简回测框架，提供数据抓取、指标计算、仓位管理与绩效报告等功能。脚本基于 Binance K 线数据，可通过命令行直接运行回测并导出关键结果。

## 功能概览

- 使用 Binance 公共 REST API 自动拉取历史 K 线数据。
- 内置多种常见指标：移动平均、RSI、ATR、ADX 等。
- 提供可配置的资金管理、止损止盈、回撤控制与交易冷却机制。
- 自动生成权益曲线、成交明细 CSV，可选绘制图形报告。

## 环境要求

- Python 3.9+
- 依赖库：`requests`、`numpy`、`pandas`、`matplotlib`

通过以下命令安装依赖：

```bash
pip install requests numpy pandas matplotlib
```

## 快速开始

```bash
python quant.py backtest --symbol BTCUSDT --interval 1h --lookback_days 365
```

运行后将在 `outputs/` 目录生成权益曲线、成交明细以及图形化回测报告。

## 日志输出

- 默认输出等级为 `INFO`，可通过 `--log-level` 控制详细程度，例如：

  ```bash
  python quant.py --log-level DEBUG backtest --interval 1h --input_file btcusdt_1h.csv
  ```

- 日志包含回测摘要、优化进度以及文件保存信息，便于快速审计流程。

## 版本 1.2 更新摘要

- 回测状态改用数据类封装，减少重复赋值，提升可维护性。
- 日志改为标准 `logging` 管理，支持等级切换与结构化输出。
- 结合网格搜索，将默认参数调整为 `fast=24`、`slow=100`、`rsi_long_min=63`、`adx_min=20`，强调稳健的顺势过滤。

使用仓库附带的 `btcusdt_1h.csv` 数据执行：

```bash
python quant.py backtest --interval 1h --input_file btcusdt_1h.csv
```

可得到约 **+1.52%** 的总收益、**-0.64%** 的最大回撤与 **0.998** 的夏普值，利润因子约 **3.61**，较旧版本显著改善收益与风险匹配度。

若所在网络无法直接访问 Binance，可先使用 `fetch` 子命令或外部工具下载 K 线数据，再通过 `--input_file` 选项离线回测：

```bash
# 1.尝试直接拉取并保存 CSV
python quant.py fetch --symbol BTCUSDT --interval 1h --lookback_days 365 --output outputs/btcusdt_1h.csv

# 2. 使用本地文件进行回测
python quant.py backtest --interval 1h --input_file outputs/btcusdt_1h.csv
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

## 参数优化

`optimize` 子命令可在同一份数据上执行网格搜索，组合不同的快慢均线、RSI 与 ADX 阈值。范围参数支持 `start:end:step` 或逗号分隔列表，例如 `--fast-range 22:28:2 --rsi-range 63,65,67`。完整示例：

```bash
python quant.py optimize \
  --interval 1h --input_file btcusdt_1h.csv \
  --fast-range 22:28:2 --slow-range 95:110:5 \
  --rsi-range 63,65,67 --adx-range 20:24:2 \
  --sort-by sharpe --top 5 --output outputs/optimization.csv
```

上述设置共评估 81 组组合，按夏普排序的前五名如下：

| 排名 | fast | slow | RSI  | ADX  | 收益% | 夏普 | 回撤% | 笔数 |
|------|-----:|-----:|-----:|-----:|------:|-----:|------:|-----:|
| 1    |   24 |  100 | 63.0 | 20.0 |  1.52 | 0.998 | -0.64 |   13 |
| 2    |   24 |  105 | 63.0 | 24.0 |  1.39 | 0.924 | -0.64 |   11 |
| 3    |   24 |  105 | 63.0 | 22.0 |  1.39 | 0.924 | -0.64 |   11 |
| 4    |   24 |  100 | 63.0 | 24.0 |  1.39 | 0.924 | -0.64 |   11 |
| 5    |   24 |  100 | 63.0 | 22.0 |  1.39 | 0.924 | -0.64 |   11 |

完整记录会写入 `outputs/optimization.csv`，若搜索空间更大（如包含更多指标），运行耗时会急剧增长，建议逐步扩大范围并留意组合数量。

## 目录结构

```
quant.py      # 核心回测脚本
outputs/      # 运行后生成的结果文件（首次执行自动创建）
```

## 注意事项

- Binance 公共接口存在频率限制，脚本内部已添加轻量节流，仍建议适当控制拉取范围。
- 若 `fetch` 命令报错，可在其他环境下载后通过 `--input_file` 导入。
- 回测结果仅供研究参考，不构成投资建议。
- 修改策略逻辑或增加依赖时，请同步更新本文档与 `AGENTS.md`。
