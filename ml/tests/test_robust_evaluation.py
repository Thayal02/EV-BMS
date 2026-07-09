from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.soh.dataset import SohDataset
from src.training.soh.model_specs import get_model_specs
from src.training.soh.robust_evaluation import (
    repeated_group_holdout_evaluation,
    summarize_repeated_evaluation,
)


@pytest.fixture
def tiny_dataset() -> SohDataset:
    rng = np.random.default_rng(0)
    n_batteries = 10
    rows_per_battery = 20
    battery_ids = []
    feature_rows = []
    targets = []
    for b in range(n_batteries):
        for i in range(rows_per_battery):
            battery_ids.append(f"B{b:04d}")
            feature_rows.append(rng.normal(size=3))
            targets.append(100.0 - i * 0.5 + rng.normal(scale=0.1))

    X = pd.DataFrame(feature_rows, columns=["f1", "f2", "f3"])
    y = pd.Series(targets, name="soh_percent")
    groups = pd.Series(battery_ids, name="battery_id")

    return SohDataset(
        X=X, y=y, groups=groups, n_rows_dropped_outliers=0, n_rows_dropped_missing_features=0
    )


def test_repeated_group_holdout_evaluation_shape(tiny_dataset: SohDataset) -> None:
    specs = get_model_specs()[:2]
    # Each model's default (empty-params) build is enough for this plumbing
    # test - correctness of the search spaces themselves is covered
    # elsewhere; this test only exercises the repeat/aggregate mechanics.
    specs_with_params = [(spec, {}) for spec in specs]

    result = repeated_group_holdout_evaluation(
        tiny_dataset, specs_with_params, n_repeats=3, test_size=0.3
    )

    assert len(result) == 3 * len(specs)
    assert set(result["model"].unique()) == {spec.name for spec in specs}
    assert set(result["repeat"].unique()) == {0, 1, 2}


def test_summarize_repeated_evaluation_sorts_by_mean_mae(tiny_dataset: SohDataset) -> None:
    per_repeat = pd.DataFrame(
        {
            "repeat": [0, 1, 0, 1],
            "model": ["a", "a", "b", "b"],
            "test_mae": [2.0, 4.0, 1.0, 1.0],
            "test_rmse": [2.0, 4.0, 1.0, 1.0],
            "test_mape": [2.0, 4.0, 1.0, 1.0],
            "test_r2": [0.9, 0.8, 0.95, 0.95],
        }
    )

    summary = summarize_repeated_evaluation(per_repeat)

    assert list(summary["model"]) == ["b", "a"]
    assert summary.loc[0, "test_mae_mean"] == pytest.approx(1.0)
    assert summary.loc[1, "test_mae_mean"] == pytest.approx(3.0)
