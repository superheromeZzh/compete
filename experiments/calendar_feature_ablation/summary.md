# Calendar Feature Ablation Summary

## Ranking

1. `lgbm_weekday_slot_is_non_workday` | features = `weekday`, `slot`, `is_non_workday` | avg_score = 0.729951 | delta_vs_baseline = +0.000353
2. `lgbm_weekday_slot` | features = `weekday`, `slot` | avg_score = 0.729598 | delta_vs_baseline = +0.000000
3. `lgbm_slot_is_non_workday` | features = `slot`, `is_non_workday` | avg_score = 0.566310 | delta_vs_baseline = -0.163288

## Conclusion

- Best offline variant is `lgbm_weekday_slot_is_non_workday` with avg_score = 0.729951.
- Adding `is_non_workday` on top of `weekday + slot` changes avg_score from 0.729598 to 0.729951 (+0.000353).
- Replacing `weekday` with `is_non_workday` changes avg_score from 0.729598 to 0.566310 (-0.163288).