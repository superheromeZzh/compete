# Calendar Feature Ablation

This folder keeps the three calendar-feature variants in one place:

- `lgbm_weekday_slot.py`
- `lgbm_weekday_slot_is_non_workday.py`
- `lgbm_slot_is_non_workday.py`

Run all variants and refresh the comparison summary with:

```bash
python3 experiments/calendar_feature_ablation/run_all.py
```

Generated outputs are written to `outputs/`, and the latest cross-variant conclusion is written to:

- `summary.md`
- `summary.json`
