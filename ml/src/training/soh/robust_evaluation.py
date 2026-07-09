"""Repeated grouped-holdout evaluation.

A single 80/20 GroupShuffleSplit over only ~34 batteries is a high-variance
estimate of generalization performance: with 7 batteries in the test side,
one unlucky (or just unrepresentative) draw can concentrate the hardest
batteries in either split and swing every model's reported test metric.
This was observed directly in this project - a first single-split run put
several of the deliberately multi-condition batteries (B0038, B0042, B0044 -
see ml/reports/eda_summary.md) and known-anomalous batteries (B0047, B0049,
B0050) together in one test fold, and every model's held-out MAE came out
1.3-2.6x worse than its cross-validation MAE as a result.

Rather than re-running the expensive hyperparameter search per repeat, this
module holds each model's already-tuned hyperparameters fixed and repeats
only the cheap part - refit on a fresh grouped train/test split, evaluate -
across many random splits, then reports the mean and standard deviation of
each metric. That distribution, not any single split, is the honest
generalization estimate and what model selection should be based on.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.training.soh.dataset import SohDataset
from src.training.soh.evaluate import compute_regression_metrics
from src.training.soh.model_specs import ModelSpec


def repeated_group_holdout_evaluation(
    dataset: SohDataset,
    specs_with_params: list[tuple[ModelSpec, dict]],
    n_repeats: int = 15,
    test_size: float = 0.20,
    base_seed: int = 1000,
) -> pd.DataFrame:
    """Refit each (model, fixed hyperparameters) pair on `n_repeats`
    independent grouped train/test splits and return one row of metrics per
    (repeat, model).
    """
    records = []
    for repeat in range(n_repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=base_seed + repeat)
        train_idx, test_idx = next(splitter.split(dataset.X, dataset.y, groups=dataset.groups))

        X_train, y_train = dataset.X.iloc[train_idx], dataset.y.iloc[train_idx]
        X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]

        for spec, params in specs_with_params:
            pipeline = spec.build_pipeline(params)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            metrics = compute_regression_metrics(y_test.to_numpy(), predictions)

            records.append(
                {
                    "repeat": repeat,
                    "model": spec.name,
                    "test_mae": metrics.mae,
                    "test_rmse": metrics.rmse,
                    "test_mape": metrics.mape,
                    "test_r2": metrics.r2,
                }
            )

    return pd.DataFrame(records)


def summarize_repeated_evaluation(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repeat metrics into mean/std per model, sorted by the
    metric that should drive model selection (mean test MAE, ascending)."""
    summary = per_repeat.groupby("model").agg(
        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),
        test_mape_mean=("test_mape", "mean"),
        test_r2_mean=("test_r2", "mean"),
        test_r2_std=("test_r2", "std"),
    )
    return summary.sort_values("test_mae_mean").reset_index()
