# LGBM weekday + slot + is_non_workday

- experiment = `lgbm_weekday_slot_is_non_workday`
- features = `weekday`, `slot`, `is_non_workday`
- note = Add a 2024 China non-workday flag on top of weekday and slot.

## Validation folds

- `1234-56`: score = 0.762980, rmse = 0.310651
- `2345-67`: score = 0.734167, rmse = 0.362088
- `3456-78`: score = 0.637318, rmse = 0.569075
- `4567-89`: score = 0.748609, rmse = 0.335810
- `5678-910`: score = 0.766681, rmse = 0.304324

## Summary

- avg_score = 0.729951
- min_score = 0.637318
- max_score = 0.766681
- std_score = 0.047721

## Output files

- submission = /Users/z/基于多源数据融合的电动汽车充电协同优化挑战/experiments/calendar_feature_ablation/outputs/lgbm_weekday_slot_is_non_workday/submission.csv
- report_json = /Users/z/基于多源数据融合的电动汽车充电协同优化挑战/experiments/calendar_feature_ablation/outputs/lgbm_weekday_slot_is_non_workday/report.json