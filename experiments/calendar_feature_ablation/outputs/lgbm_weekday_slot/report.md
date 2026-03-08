# LGBM weekday + slot

- experiment = `lgbm_weekday_slot`
- features = `weekday`, `slot`
- note = Use only day-of-week and 15-minute slot as the online-safe baseline.

## Validation folds

- `1234-56`: score = 0.763956, rmse = 0.308976
- `2345-67`: score = 0.741836, rmse = 0.348006
- `3456-78`: score = 0.637426, rmse = 0.568809
- `4567-89`: score = 0.744831, rmse = 0.342586
- `5678-910`: score = 0.759940, rmse = 0.315893

## Summary

- avg_score = 0.729598
- min_score = 0.637426
- max_score = 0.763956
- std_score = 0.046859

## Output files

- submission = /Users/z/基于多源数据融合的电动汽车充电协同优化挑战/experiments/calendar_feature_ablation/outputs/lgbm_weekday_slot/submission.csv
- report_json = /Users/z/基于多源数据融合的电动汽车充电协同优化挑战/experiments/calendar_feature_ablation/outputs/lgbm_weekday_slot/report.json