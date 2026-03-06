# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science competition project for **EV Charging Station Load Prediction** (电动汽车充电站充电负荷预测). The goal is to predict the 24-hour charging load curve for electric vehicle charging stations using historical charging data.

## Project Structure

```
configs/         - Configuration files
data/
  raw/           - Raw competition data files (CSV, XLSX, PNG)
  processed/     - Processed/feature-engineered data
models/          - Saved model files
notebooks/       - Jupyter notebooks for analysis
src/
  features/      - Feature engineering code
  models/        - Model implementations
  utils/         - Utility functions
  validation/    - Validation scripts
submissions/     - Generated submission files
```

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

- `TIME`: Timestamp (e.g., "2024/1/1 0:00")
- `V`: Charging power (MW) - **Target variable**
- `AVGV`: Daily average charging power (MW)
- `MAXV`: Daily maximum charging power (MW)
- `MAXT`: Time of daily maximum
- `MINV`: Daily minimum charging power (MW)
- `MINT`: Time of daily minimum
- `S`: Charging duration (h)
- Additional features: AVGS, MAXS, MINS, SPAN

## Submission Format

CSV with two columns:
- `TIME`: Timestamp (e.g., "2024/11/1 0:00")
- `V`: Predicted charging power (MW)

Prediction period: 2024/11/1 0:00 to 2024/12/31 23:45 (15-minute intervals).

## Dependencies

Standard data science stack:
- pandas
- numpy
- scikit-learn
- LightGBM (optional, for gradient boosting)

## Python Environment

```python
# Read training data
import pandas as pd
df = pd.read_csv('data/raw/A榜-充电站充电负荷训练数据.csv', encoding='gbk')
```

Note: Data files use GBK encoding.
