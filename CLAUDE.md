# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science competition project for **EV Charging Station Load Prediction** (电动汽车充电站充电负荷预测). The goal is to predict the 24-hour charging load curve for electric vehicle charging stations using historical charging data.

## Competition Task

Build an AI-based model to predict future 24-hour charging load based on historical charging records from EV charging stations. The predictions support day-ahead energy scheduling and grid management.

## Scoring Metric

Score = `1 / (1 + RMSE)`

Where RMSE is the Root Mean Square Error between predicted and actual charging power.

## Data Files

All raw data files are located in `data/raw/`:

| File | Description |
|------|-------------|
| `A榜-充电站充电负荷训练数据.csv` | Training data (2024/1/1 - 2024/10/31), 15-min intervals |
| `附件1-V2G站向电网售电及从电网购电电价.csv` | V2G station buy/sell electricity prices by time period |
| `附件2-光伏典型出力.xlsx` | Typical photovoltaic output (96 time slots, 15-min intervals) |
| `附件3 -EV用户充放电电价.csv` | EV user charge/discharge prices (hourly) |
| `附件4-线路基本参数.xlsx` | Line parameters (resistance, reactance, susceptance) |
| `submit_example.csv` | Submission format example |

## Training Data Schema

Actual columns in training data (second row is header, first row is Chinese names):

| Column | Description |
|--------|-------------|
| `NAME` | Station name (电动汽车充电站) |
| `SENID` | Station ID (1001-1012) |
| `TIME` | Timestamp (e.g., "2024/1/1 0:00") |
| `V` | Charging power (MW) - **Target variable** |
| `AVGV` | Daily average charging power (MW) |
| `MAXV` | Daily maximum charging power (MW) |
| `MAXT` | Time of daily maximum |
| `MINV` | Daily minimum charging power (MW) |
| `MINT` | Time of daily minimum |
| `S` | Charging duration (h) |
| `AVGS` | Average S value (h) |
| `MAXS` | Maximum S value (h) |
| `MINS` | Minimum S value (h) |
| `SPAN` | Duration time span (h) |

## Submission Format

CSV with two columns:
- `TIME`: Timestamp (e.g., "2024/11/1 0:00")
- `V`: Predicted charging power (MW)

Prediction period: 2024/11/1 0:00 to 2024/12/31 23:45 (15-minute intervals).

## Running Experiments

The main experiments are in `experiments/calendar_feature_ablation/`. Run all experiments and generate comparison summary:

```bash
python3 experiments/calendar_feature_ablation/run_all.py
```

This runs 3 LightGBM variants with different feature combinations and outputs:
- `experiments/calendar_feature_ablation/outputs/<experiment_name>/submission.csv` - Submission file
- `experiments/calendar_feature_ablation/outputs/<experiment_name>/report.json` - Detailed metrics
- `experiments/calendar_feature_ablation/summary.md` - Cross-experiment comparison

## Validation Strategy

5-fold time-series cross-validation defined in `experiments/calendar_feature_ablation/common.py`:

| Fold | Train Period | Valid Period |
|------|--------------|--------------|
| 1 | Jan-Apr | May-Jun |
| 2 | Feb-May | Jun-Jul |
| 3 | Mar-Jun | Jul-Aug |
| 4 | Apr-Jul | Aug-Sep |
| 5 | May-Aug | Sep-Oct |

## Current Best Model

Best performing model: `lgbm_weekday_slot_is_non_workday` (avg_score = 0.729951)

Features used:
- `weekday`: Day of week (0-6)
- `slot`: 15-minute time slot (0-95)
- `is_non_workday`: Binary flag for Chinese 2024 holidays/weekends (accounts for 调休)

## Dependencies

Standard data science stack:
- pandas
- numpy
- scikit-learn
- LightGBM

## Python Environment

```python
# Read training data (note: first row is Chinese names, second row is actual header)
import pandas as pd
df = pd.read_csv('data/raw/A榜-充电站充电负荷训练数据.csv', encoding='gbk').iloc[1:]
```

Note: Data files use GBK encoding.
