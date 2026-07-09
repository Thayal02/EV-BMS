from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.soh.dataset import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    load_soh_dataset,
)


def test_feature_columns_exclude_leakage_and_identifiers() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert GROUP_COLUMN not in FEATURE_COLUMNS
    for leaky_column in LEAKAGE_COLUMNS:
        assert leaky_column not in FEATURE_COLUMNS


def _write_fixture_parquet(path: Path, n_per_battery: int = 5) -> None:
    rows = []
    for battery_id in ["B0005", "B0006"]:
        for i in range(n_per_battery):
            row = {col: float(i) for col in FEATURE_COLUMNS}
            row.update(
                {
                    "battery_id": battery_id,
                    TARGET_COLUMN: 90.0 - i,
                    "is_capacity_outlier": False,
                }
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_load_soh_dataset_basic_shape(tmp_path: Path) -> None:
    path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(path)

    dataset = load_soh_dataset(path)

    assert list(dataset.X.columns) == FEATURE_COLUMNS
    assert len(dataset.X) == 10
    assert dataset.n_batteries == 2
    assert dataset.n_rows_dropped_outliers == 0
    assert dataset.n_rows_dropped_missing_features == 0


def test_load_soh_dataset_drops_flagged_outliers(tmp_path: Path) -> None:
    path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(path)
    df = pd.read_parquet(path)
    df.loc[0, "is_capacity_outlier"] = True
    df.to_parquet(path, index=False)

    dataset = load_soh_dataset(path)

    assert dataset.n_rows_dropped_outliers == 1
    assert len(dataset.X) == 9


def test_load_soh_dataset_drops_rows_with_missing_features(tmp_path: Path) -> None:
    path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(path)
    df = pd.read_parquet(path)
    df.loc[0, FEATURE_COLUMNS[0]] = np.nan
    df.to_parquet(path, index=False)

    dataset = load_soh_dataset(path)

    assert dataset.n_rows_dropped_missing_features == 1
    assert len(dataset.X) == 9
    assert not dataset.X[FEATURE_COLUMNS[0]].isna().any()
