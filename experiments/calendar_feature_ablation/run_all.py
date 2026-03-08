from __future__ import annotations

import json

from common import EXPERIMENTS, build_submission_feature_table, load_feature_table, run_experiment, write_summary


def main() -> None:
    df = load_feature_table()
    submit_features = build_submission_feature_table()

    summaries = [
        run_experiment(experiment, df=df, submit_features=submit_features)
        for experiment in EXPERIMENTS
    ]
    summary = write_summary(summaries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
