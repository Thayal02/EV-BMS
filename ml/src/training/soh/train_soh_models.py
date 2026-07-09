"""End-to-end SOH model training: compares 7 regression algorithms with
group-aware cross-validated hyperparameter tuning, then evaluates each
tuned model across many independent held-out battery splits (not just one)
before promoting the best one to the model registry.

Usage:
    python -m src.training.soh.train_soh_models \
        --dataset data/processed/nasa_cycle_features.parquet \
        --registry-dir models/registry \
        --reports-dir reports \
        --n-trials 25 \
        --n-repeats 15

Why a group split (not a random row split): consecutive discharge cycles
from the same battery are highly autocorrelated, so a random split would
let the model see near-duplicate rows from a battery at both train and test
time and report inflated generalization metrics. Every split in this
pipeline - the hyperparameter tuning's inner cross-validation and every
held-out evaluation split - is grouped on `battery_id`.

Why *repeated* holdout, not a single 80/20 split: with only ~34 batteries,
a single grouped split is a high-variance estimate of generalization - one
unlucky draw can concentrate the hardest batteries into the test side. This
was observed directly in this project's first run: a single split happened
to put several deliberately multi-condition batteries (see
ml/reports/eda_summary.md) together in one test fold, and every model's
single-split test MAE came out 1.3-2.6x worse than its cross-validation MAE.
Model selection here is therefore based on the mean test MAE across many
independent random grouped splits (see robust_evaluation.py), not any one
split - see ml/reports/soh_training_report.md for the full comparison.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.training.soh.dataset import FEATURE_COLUMNS, TARGET_COLUMN, load_soh_dataset
from src.training.soh.evaluate import compute_regression_metrics
from src.training.soh.model_specs import get_model_specs
from src.training.soh.robust_evaluation import (
    repeated_group_holdout_evaluation,
    summarize_repeated_evaluation,
)
from src.training.soh.tuning import tune_model


def run_training(
    dataset_path: str | Path,
    n_trials: int = 25,
    n_cv_splits: int = 5,
    n_repeats: int = 15,
    test_size: float = 0.20,
    tuning_random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    """Returns (tuning_leaderboard, robust_leaderboard, best_model_record,
    final_fitted_pipeline)."""
    dataset = load_soh_dataset(dataset_path)

    # Phase 1: tune each model's hyperparameters once, on one grouped split.
    # The resulting single-split test metrics are kept only as a diagnostic
    # (see module docstring for why they must not drive model selection).
    tuning_splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=tuning_random_state
    )
    train_idx, test_idx = next(
        tuning_splitter.split(dataset.X, dataset.y, groups=dataset.groups)
    )
    X_train, y_train, groups_train = (
        dataset.X.iloc[train_idx].reset_index(drop=True),
        dataset.y.iloc[train_idx].reset_index(drop=True),
        dataset.groups.iloc[train_idx].reset_index(drop=True),
    )
    X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]

    tuning_results = []
    specs_with_params: list[tuple] = []

    for spec in get_model_specs():
        tuning_result = tune_model(
            spec, X_train, y_train, groups_train, n_trials=n_trials, n_splits=n_cv_splits
        )
        specs_with_params.append((spec, tuning_result.best_params))

        pipeline = spec.build_pipeline(tuning_result.best_params)
        pipeline.fit(X_train, y_train)
        single_split_metrics = compute_regression_metrics(y_test.to_numpy(), pipeline.predict(X_test))

        tuning_results.append(
            {
                "model": spec.name,
                "cv_mae_mean": tuning_result.cv_mae_mean,
                "cv_mae_std": tuning_result.cv_mae_std,
                "single_split_test_mae": single_split_metrics.mae,
                "single_split_test_r2": single_split_metrics.r2,
                "best_params": json.dumps(tuning_result.best_params),
            }
        )

    tuning_leaderboard = pd.DataFrame(tuning_results).sort_values("cv_mae_mean").reset_index(drop=True)

    # Phase 2: with hyperparameters now fixed, re-evaluate every model across
    # many independent grouped splits and select based on the distribution,
    # not the one split used for tuning.
    per_repeat = repeated_group_holdout_evaluation(
        dataset, specs_with_params, n_repeats=n_repeats, test_size=test_size
    )
    robust_leaderboard = summarize_repeated_evaluation(per_repeat)

    best_model_name = robust_leaderboard.iloc[0]["model"]
    best_spec, best_params = next(
        (s, p) for s, p in specs_with_params if s.name == best_model_name
    )

    # Model selection is now complete - refit the winner on ALL available
    # data (every battery) for the artifact that actually gets deployed.
    # This is standard practice once evaluation is done: more data only
    # helps a fixed, already-chosen model/hyperparameter combination, and
    # the repeated-holdout metrics above remain the honest generalization
    # estimate for this algorithm/hyperparameter combination regardless of
    # what the artifact was subsequently refit on.
    final_pipeline = best_spec.build_pipeline(best_params)
    final_pipeline.fit(dataset.X, dataset.y)

    best_row = robust_leaderboard.iloc[0]
    best_model_record = {
        "model": best_model_name,
        "best_params": best_params,
        "n_repeats": n_repeats,
        "test_mae_mean": float(best_row["test_mae_mean"]),
        "test_mae_std": float(best_row["test_mae_std"]),
        "test_rmse_mean": float(best_row["test_rmse_mean"]),
        "test_rmse_std": float(best_row["test_rmse_std"]),
        "test_mape_mean": float(best_row["test_mape_mean"]),
        "test_r2_mean": float(best_row["test_r2_mean"]),
        "test_r2_std": float(best_row["test_r2_std"]),
        "n_total_rows": len(dataset.X),
        "n_total_batteries": dataset.n_batteries,
        "n_rows_dropped_outliers": dataset.n_rows_dropped_outliers,
        "n_rows_dropped_missing_features": dataset.n_rows_dropped_missing_features,
    }

    return tuning_leaderboard, robust_leaderboard, best_model_record, final_pipeline


def _write_manifest(record: dict, registry_dir: Path, version: str) -> None:
    manifest = {
        "task": "soh",
        "version": version,
        "algorithm": record["model"],
        "hyperparameters": record["best_params"],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": {
            "repeated_holdout": {
                "n_repeats": record["n_repeats"],
                "mae_mean": record["test_mae_mean"],
                "mae_std": record["test_mae_std"],
                "rmse_mean": record["test_rmse_mean"],
                "rmse_std": record["test_rmse_std"],
                "mape_mean": record["test_mape_mean"],
                "r2_mean": record["test_r2_mean"],
                "r2_std": record["test_r2_std"],
            }
        },
        "dataset": {
            "n_total_rows": record["n_total_rows"],
            "n_total_batteries": record["n_total_batteries"],
            "n_rows_dropped_outliers": record["n_rows_dropped_outliers"],
            "n_rows_dropped_missing_features": record["n_rows_dropped_missing_features"],
        },
        "notes": (
            "metrics.repeated_holdout is the mean/std of held-out test "
            "performance across n_repeats independent random 80/20 "
            "grouped (by battery_id) train/test splits - not a single "
            "split - since a single split over only ~34 batteries is a "
            "high-variance generalization estimate (see "
            "src/training/soh/robust_evaluation.py and "
            "ml/reports/soh_training_report.md). The deployed model.joblib "
            "artifact is refit on ALL batteries with these hyperparameters "
            "once model selection was complete."
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (registry_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_training_report(
    tuning_leaderboard: pd.DataFrame,
    robust_leaderboard: pd.DataFrame,
    best_record: dict,
    reports_dir: Path,
) -> None:
    lines = [
        "# SOH Model Training Report",
        "",
        f"- Total rows: {best_record['n_total_rows']} across {best_record['n_total_batteries']} batteries",
        f"- Rows dropped (flagged capacity outliers): {best_record['n_rows_dropped_outliers']}",
        f"- Selected model: **{best_record['model']}**",
        f"- Repeated holdout ({best_record['n_repeats']} independent grouped 80/20 splits): "
        f"MAE {best_record['test_mae_mean']:.3f} ± {best_record['test_mae_std']:.3f} SOH%, "
        f"R² {best_record['test_r2_mean']:.3f} ± {best_record['test_r2_std']:.3f}",
        "",
        "## Why repeated holdout, not a single train/test split",
        "",
        "A single 80/20 split over only ~34 batteries is high-variance: one "
        "unlucky draw can concentrate the hardest batteries into the test "
        "side. This is not hypothetical - it happened on the first run of "
        "this pipeline. The table below is that single split's diagnostic "
        "(used only to tune hyperparameters), and every model's single-split "
        "test MAE is substantially worse than its cross-validation MAE, "
        "because that split's test battery set happened to include several "
        "of the deliberately multi-condition batteries documented in "
        "`ml/reports/eda_summary.md` (B0038, B0042, B0044) plus known-"
        "anomalous batteries (B0047, B0049, B0050). Model *selection* below "
        "is based on the repeated-holdout leaderboard instead, which "
        "averages over many different random battery splits and is not "
        "sensitive to which specific batteries any one split happens to "
        "hold out.",
        "",
        "### Single-split tuning diagnostic (do not use for model selection)",
        "",
        tuning_leaderboard.to_markdown(index=False),
        "",
        "### Repeated holdout leaderboard (used for model selection)",
        "",
        robust_leaderboard.to_markdown(index=False),
        "",
    ]
    (reports_dir / "soh_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/nasa_cycle_features.parquet")
    parser.add_argument("--registry-dir", default="models/registry")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-cv-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=15)
    args = parser.parse_args()

    tuning_leaderboard, robust_leaderboard, best_record, final_pipeline = run_training(
        args.dataset,
        n_trials=args.n_trials,
        n_cv_splits=args.n_cv_splits,
        n_repeats=args.n_repeats,
    )

    version = datetime.now(UTC).strftime("v%Y%m%dT%H%M%SZ")
    registry_dir = Path(args.registry_dir) / "soh" / version
    registry_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, registry_dir / "model.joblib")
    _write_manifest(best_record, registry_dir, version)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tuning_leaderboard.to_csv(reports_dir / "soh_tuning_diagnostic.csv", index=False)
    robust_leaderboard.to_csv(reports_dir / "soh_model_comparison.csv", index=False)
    _write_training_report(tuning_leaderboard, robust_leaderboard, best_record, reports_dir)

    print("Single-split tuning diagnostic (NOT used for model selection):")
    print(tuning_leaderboard.to_string(index=False))
    print("\nRepeated holdout leaderboard (used for model selection):")
    print(robust_leaderboard.to_string(index=False))
    print(f"\nBest model: {best_record['model']}")
    print(
        f"Repeated holdout test MAE: {best_record['test_mae_mean']:.3f} "
        f"+/- {best_record['test_mae_std']:.3f} SOH%  "
        f"R2: {best_record['test_r2_mean']:.3f}"
    )
    print(f"Saved to {registry_dir}/")


if __name__ == "__main__":
    main()
