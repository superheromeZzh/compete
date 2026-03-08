# LGBM slot + is_non_workday

- experiment = `lgbm_slot_is_non_workday`
- features = `slot`, `is_non_workday`
- note = Keep only time-of-day plus the 2024 China non-workday flag.

## Validation folds

- `1234-56`: score = 0.565956, rmse = 0.766923
- `2345-67`: score = 0.618887, rmse = 0.615804
- `3456-78`: score = 0.598768, rmse = 0.670096
- `4567-89`: score = 0.566880, rmse = 0.764043
- `5678-910`: score = 0.481058, rmse = 1.078753

## Summary

- avg_score = 0.566310
- min_score = 0.481058
- max_score = 0.618887
- std_score = 0.047088

## Output files

- submission = /Users/z/基于多源数据融合的电动汽车充电协同优化挑战/experiments/calendar_feature_ablation/outputs/lgbm_slot_is_non_workday/submission.csv
- report_json = /Users/z/基于多源数据融合的电动汽车充电协同优化挑战/experiments/calendar_feature_ablation/outputs/lgbm_slot_is_non_workday/report.json