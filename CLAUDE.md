# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Competition Overview

**零氪杯·电动汽车充电站充电负荷预测** (Zero-ke Cup · EV Charging Station Load Forecast)

Build an AI model to predict charging load curves for the next 24 hours based on real-world EV charging data. The goal is to help station operators with day-ahead energy scheduling and enable effective grid management of aggregated charging loads.

### Prediction Task (A榜)

- **Target Period**: 2024/11/1 0:00 to 2024/12/31 23:45 (61 days, 5856 data points)
- **Time Granularity**: 15-minute intervals (96 intervals per day)
- **Submission Limit**: 5 submissions per day
- **Scoring Formula**: `Score = 1 / (1 + RMSE)` (higher is better)

## Data Files

| File | Description |
|------|-------------|
| `A榜-充电站充电负荷训练数据.csv` | Training data (2024/1/1 - 2024/10/31), columns: NAME, SENID, TIME, V (MW), AVGV, MAXV, MAXT, MINV, MINT, S, AVGS, MAXS, MINS, SPAN |
| `submit_example.csv` | Submission template: TIME, V columns, 5856 rows |
| `附件1-V2G站向电网售电及从电网购电电价.csv` | V2G station buy/sell prices (hourly, 元/kWh) |
| `附件2-光伏典型出力.xlsx` | Solar PV output: 96 rows (15-min intervals), columns: 时段, 光伏(kW) |
| `附件3-EV用户充放电电价.csv` | EV user charge/discharge prices (time-of-use, 元/kWh) |
| `附件4-线路基本参数.xlsx` | 9-bus network line parameters (R, X, B in per-unit) |

## Key Domain Concepts

- **V2G (Vehicle-to-Grid)**: EVs discharge power back to grid during peak hours
- **Time-of-Use Pricing**: Prices vary by time period (5 time blocks: 0-6h, 6-10h, 10-14h, 14-18h, 18-24h)
- **15-minute Intervals**: 96 intervals per day
- **Per-Unit System**: Line parameters use per-unit normalization
- **Target Variable**: V (充电功率/Charging Power in MW)

## Development Environment

```bash
pip install pandas numpy openpyxl matplotlib scikit-learn scipy
```

## Data Loading

CSV files use GBK encoding:
```python
train_df = pd.read_csv('A榜-充电站充电负荷训练数据.csv', encoding='gbk')
submit_df = pd.read_csv('submit_example.csv', encoding='gbk')
pv_df = pd.read_excel('附件2-光伏典型出力.xlsx')
line_df = pd.read_excel('附件4-线路基本参数.xlsx')
```

## Submission Format

CSV with two columns:
- `TIME`: Format "2024/11/1 0:00", "2024/11/1 0:15", etc.
- `V`: Predicted charging power in MW

Total: 5856 rows (61 days × 96 intervals)
