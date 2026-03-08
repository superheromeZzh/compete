from common import EXPERIMENTS_BY_NAME, run_experiment


if __name__ == "__main__":
    run_experiment(EXPERIMENTS_BY_NAME["lgbm_weekday_slot_is_non_workday"])
