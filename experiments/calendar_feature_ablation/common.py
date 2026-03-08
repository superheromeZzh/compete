from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
SUMMARY_MD_PATH = EXPERIMENT_ROOT / "summary.md"
SUMMARY_JSON_PATH = EXPERIMENT_ROOT / "summary.json"
TRAIN_PATH = ROOT / "data/raw/A榜-充电站充电负荷训练数据.csv"
SUBMIT_EXAMPLE_PATH = ROOT / "data/raw/submit_example.csv"

FOLDS = [
    ("1234-56", "2024-01-01", "2024-04-30 23:45", "2024-05-01", "2024-06-30 23:45"),
    ("2345-67", "2024-02-01", "2024-05-31 23:45", "2024-06-01", "2024-07-31 23:45"),
    ("3456-78", "2024-03-01", "2024-06-30 23:45", "2024-07-01", "2024-08-31 23:45"),
    ("4567-89", "2024-04-01", "2024-07-31 23:45", "2024-08-01", "2024-09-30 23:45"),
    ("5678-910", "2024-05-01", "2024-08-31 23:45", "2024-09-01", "2024-10-31 23:45"),
]

EXPERIMENTS = [
    {
        "name": "lgbm_weekday_slot",
        "title": "LGBM weekday + slot",
        "feature_columns": ["weekday", "slot"],
        "description": "Use only day-of-week and 15-minute slot as the online-safe baseline.",
    },
    {
        "name": "lgbm_weekday_slot_is_non_workday",
        "title": "LGBM weekday + slot + is_non_workday",
        "feature_columns": ["weekday", "slot", "is_non_workday"],
        "description": "Add a 2024 China non-workday flag on top of weekday and slot.",
    },
    {
        "name": "lgbm_slot_is_non_workday",
        "title": "LGBM slot + is_non_workday",
        "feature_columns": ["slot", "is_non_workday"],
        "description": "Keep only time-of-day plus the 2024 China non-workday flag.",
    },
]
EXPERIMENTS_BY_NAME = {experiment["name"]: experiment for experiment in EXPERIMENTS}

# 2024 年中国法定节假日调休安排。这里只保留会改变“工作日/休息日”判断的特殊日期：
# 常规周末默认视为休息日，调休上班周末单独列在 SPECIAL_WORKDAYS_2024 中。
SPECIAL_HOLIDAYS_2024 = pd.to_datetime(
    [
        "2024-01-01",
        "2024-02-12",
        "2024-02-13",
        "2024-02-14",
        "2024-02-15",
        "2024-02-16",
        "2024-04-04",
        "2024-04-05",
        "2024-05-01",
        "2024-05-02",
        "2024-05-03",
        "2024-06-10",
        "2024-09-16",
        "2024-09-17",
        "2024-10-01",
        "2024-10-02",
        "2024-10-03",
        "2024-10-04",
        "2024-10-07",
    ]
)
SPECIAL_WORKDAYS_2024 = pd.to_datetime(
    [
        "2024-02-04",
        "2024-02-18",
        "2024-04-07",
        "2024-04-28",
        "2024-05-11",
        "2024-09-14",
        "2024-09-29",
        "2024-10-12",
    ]
)


def log(message: str) -> None:
    print(f"[INFO] {message}")


def competition_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "score": 1.0 / (1.0 + rmse)}


def build_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )


def build_calendar_features(timestamps: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps)
    dates = ts.dt.normalize()

    is_special_holiday = dates.isin(SPECIAL_HOLIDAYS_2024)
    is_special_workday = dates.isin(SPECIAL_WORKDAYS_2024)
    is_weekend = ts.dt.weekday >= 5

    return pd.DataFrame(
        {
            "weekday": ts.dt.weekday.astype(int),
            "slot": ((ts.dt.hour * 60 + ts.dt.minute) // 15).astype(int),
            "is_non_workday": ((is_weekend & ~is_special_workday) | is_special_holiday).astype(int),
        },
        index=ts.index,
    )


def load_feature_table() -> pd.DataFrame:
    log("Loading training data")
    raw = pd.read_csv(TRAIN_PATH, encoding="gbk").iloc[1:].copy()
    raw["timestamp"] = pd.to_datetime(raw["日期"])
    raw["target"] = pd.to_numeric(raw["充电功率/MW"], errors="raise")

    df = (
        pd.DataFrame(
            {
                "timestamp": raw["timestamp"],
                "target": raw["target"],
            }
        )
        .join(build_calendar_features(raw["timestamp"]))
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    log(
        "History range: "
        f"{df['timestamp'].min()} -> {df['timestamp'].max()}, rows={len(df)}"
    )
    return df


def build_submission_feature_table() -> pd.DataFrame:
    submit = pd.read_csv(SUBMIT_EXAMPLE_PATH, encoding="gbk")
    submit["TIME_TEXT"] = submit["TIME"]
    submit["TIME"] = pd.to_datetime(submit["TIME"])

    return (
        pd.DataFrame(
            {
                "TIME_TEXT": submit["TIME_TEXT"],
                "timestamp": submit["TIME"],
            }
        )
        .join(build_calendar_features(submit["TIME"]))
    )


def evaluate_fold(
    df: pd.DataFrame,
    feature_columns: list[str],
    fold: tuple[str, str, str, str, str],
) -> dict[str, float | str | int]:
    fold_name, train_start, train_end, valid_start, valid_end = fold

    train_df = df[(df["timestamp"] >= pd.Timestamp(train_start)) & (df["timestamp"] <= pd.Timestamp(train_end))].copy()
    valid_df = df[(df["timestamp"] >= pd.Timestamp(valid_start)) & (df["timestamp"] <= pd.Timestamp(valid_end))].copy()

    log(
        f"Fold {fold_name} | train_rows={len(train_df)}, valid_rows={len(valid_df)} "
        f"| train={train_start}~{train_end}, valid={valid_start}~{valid_end}"
    )

    model = build_model()
    model.fit(train_df[feature_columns], train_df["target"])
    pred = model.predict(valid_df[feature_columns])
    pred = np.clip(pred, df["target"].min(), df["target"].max())

    metrics = competition_score(valid_df["target"], pred)
    metrics.update(
        {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
        }
    )
    log(f"Fold {fold_name} result | rmse={metrics['rmse']:.6f}, score={metrics['score']:.6f}")
    return metrics


def feature_importance(model: LGBMRegressor, feature_columns: list[str]) -> list[dict[str, int | str]]:
    pairs = sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": name, "importance": int(value)} for name, value in pairs]


def build_experiment_report(summary: dict[str, object]) -> str:
    feature_text = ", ".join(f"`{column}`" for column in summary["feature_columns"])
    lines = [
        f"# {summary['title']}",
        "",
        f"- experiment = `{summary['name']}`",
        f"- features = {feature_text}",
        f"- note = {summary['description']}",
        "",
        "## Validation folds",
        "",
    ]

    for fold_name, metrics in summary["folds"].items():
        lines.append(
            f"- `{fold_name}`: score = {metrics['score']:.6f}, rmse = {metrics['rmse']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- avg_score = {summary['score_summary']['avg_score']:.6f}",
            f"- min_score = {summary['score_summary']['min_score']:.6f}",
            f"- max_score = {summary['score_summary']['max_score']:.6f}",
            f"- std_score = {summary['score_summary']['std_score']:.6f}",
            "",
            "## Output files",
            "",
            f"- submission = {summary['submission_path']}",
            f"- report_json = {summary['report_json_path']}",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    experiment: dict[str, object],
    df: pd.DataFrame | None = None,
    submit_features: pd.DataFrame | None = None,
) -> dict[str, object]:
    feature_columns = list(experiment["feature_columns"])
    experiment_name = str(experiment["name"])
    experiment_dir = OUTPUT_ROOT / experiment_name
    submission_path = experiment_dir / "submission.csv"
    report_json_path = experiment_dir / "report.json"
    report_md_path = experiment_dir / "report.md"

    experiment_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_feature_table()
    if submit_features is None:
        submit_features = build_submission_feature_table()

    log(f"[{experiment_name}] Using online-safe features: {feature_columns}")

    folds = {}
    scores = []
    for fold in FOLDS:
        metrics = evaluate_fold(df, feature_columns, fold)
        folds[fold[0]] = metrics
        scores.append(metrics["score"])

    score_summary = {
        "avg_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "std_score": float(np.std(scores)),
    }
    log(
        f"[{experiment_name}] Summary | "
        f"avg={score_summary['avg_score']:.6f}, min={score_summary['min_score']:.6f}, "
        f"max={score_summary['max_score']:.6f}, std={score_summary['std_score']:.6f}"
    )

    final_model = build_model()
    final_model.fit(df[feature_columns], df["target"])
    pred = final_model.predict(submit_features[feature_columns])
    pred = np.clip(pred, df["target"].min(), df["target"].max())

    submission = pd.DataFrame(
        {
            "TIME": submit_features["TIME_TEXT"],
            "V": np.round(pred, 6),
        }
    )
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    summary = {
        "name": experiment_name,
        "title": experiment["title"],
        "description": experiment["description"],
        "online_safe": True,
        "feature_columns": feature_columns,
        "folds": folds,
        "score_summary": score_summary,
        "top_features": feature_importance(final_model, feature_columns),
        "submission_path": str(submission_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
    }

    report_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(build_experiment_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def write_summary(summaries: list[dict[str, object]]) -> dict[str, object]:
    baseline = next(summary for summary in summaries if summary["name"] == "lgbm_weekday_slot")
    weekday_slot_non_workday = next(
        summary for summary in summaries if summary["name"] == "lgbm_weekday_slot_is_non_workday"
    )
    slot_non_workday = next(
        summary for summary in summaries if summary["name"] == "lgbm_slot_is_non_workday"
    )
    ranking = sorted(summaries, key=lambda summary: summary["score_summary"]["avg_score"], reverse=True)

    summary_payload = {
        "baseline_experiment": baseline["name"],
        "best_experiment": ranking[0]["name"],
        "ranking": [
            {
                "name": summary["name"],
                "title": summary["title"],
                "feature_columns": summary["feature_columns"],
                "avg_score": summary["score_summary"]["avg_score"],
                "delta_vs_baseline": summary["score_summary"]["avg_score"] - baseline["score_summary"]["avg_score"],
                "submission_path": summary["submission_path"],
                "report_json_path": summary["report_json_path"],
                "report_md_path": summary["report_md_path"],
            }
            for summary in ranking
        ],
        "conclusion": {
            "best_experiment": ranking[0]["name"],
            "best_avg_score": ranking[0]["score_summary"]["avg_score"],
            "weekday_slot_to_weekday_slot_is_non_workday": (
                weekday_slot_non_workday["score_summary"]["avg_score"] - baseline["score_summary"]["avg_score"]
            ),
            "weekday_slot_to_slot_is_non_workday": (
                slot_non_workday["score_summary"]["avg_score"] - baseline["score_summary"]["avg_score"]
            ),
        },
    }

    lines = [
        "# Calendar Feature Ablation Summary",
        "",
        "## Ranking",
        "",
    ]

    for index, summary in enumerate(ranking, start=1):
        delta = summary["score_summary"]["avg_score"] - baseline["score_summary"]["avg_score"]
        feature_text = ", ".join(f"`{column}`" for column in summary["feature_columns"])
        lines.append(
            f"{index}. `{summary['name']}` | features = {feature_text} | "
            f"avg_score = {summary['score_summary']['avg_score']:.6f} | delta_vs_baseline = {delta:+.6f}"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"- Best offline variant is `{ranking[0]['name']}` with avg_score = "
                f"{ranking[0]['score_summary']['avg_score']:.6f}."
            ),
            (
                f"- Adding `is_non_workday` on top of `weekday + slot` changes avg_score from "
                f"{baseline['score_summary']['avg_score']:.6f} to "
                f"{weekday_slot_non_workday['score_summary']['avg_score']:.6f} "
                f"({weekday_slot_non_workday['score_summary']['avg_score'] - baseline['score_summary']['avg_score']:+.6f})."
            ),
            (
                f"- Replacing `weekday` with `is_non_workday` changes avg_score from "
                f"{baseline['score_summary']['avg_score']:.6f} to "
                f"{slot_non_workday['score_summary']['avg_score']:.6f} "
                f"({slot_non_workday['score_summary']['avg_score'] - baseline['score_summary']['avg_score']:+.6f})."
            ),
        ]
    )

    SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary_payload
