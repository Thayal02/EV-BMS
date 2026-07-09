from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import DatasetValidationError, validate_and_clean, validate_schema


def _base_df(n: int = 10, battery_id: str = "B0005") -> pd.DataFrame:
    capacity = np.linspace(1.9, 1.5, n)
    return pd.DataFrame(
        {
            "battery_id": [battery_id] * n,
            "cycle_index": list(range(n)),
            "discharge_sequence": list(range(n)),
            "timestamp": pd.date_range("2008-01-01", periods=n, freq="D"),
            "ambient_temperature_c": [24.0] * n,
            "capacity_ah": capacity,
            "discharge_duration_s": [2000.0] * n,
            "voltage_mean": [3.8] * n,
            "current_mean": [-2.0] * n,
            "temperature_mean": [25.0] * n,
            "nearest_impedance_re_ohm": [0.045] * n,
            "nearest_impedance_rct_ohm": [0.07] * n,
            "nearest_charge_duration_s": [500.0] * n,
            "nearest_impedance_lag_s": [300.0] * n,
        }
    )


def test_validate_schema_raises_on_missing_columns() -> None:
    df = _base_df().drop(columns=["capacity_ah"])
    with pytest.raises(DatasetValidationError):
        validate_schema(df)


def test_validate_and_clean_removes_exact_duplicates() -> None:
    df = pd.concat([_base_df(5), _base_df(5).iloc[[0]]], ignore_index=True)

    cleaned, report = validate_and_clean(df)

    assert report.n_duplicates_removed == 1
    assert len(cleaned) == 5


def test_validate_and_clean_drops_missing_label_rows() -> None:
    df = _base_df(5)
    df.loc[2, "capacity_ah"] = np.nan

    cleaned, report = validate_and_clean(df)

    assert report.n_missing_label_dropped == 1
    assert len(cleaned) == 4
    assert not cleaned["capacity_ah"].isna().any()


def test_validate_and_clean_flags_capacity_outlier() -> None:
    df = _base_df(10)
    # A single implausible sensor glitch in the middle of a smooth fade curve.
    df.loc[5, "capacity_ah"] = 0.0

    cleaned, report = validate_and_clean(df)

    assert report.n_outliers_flagged >= 1
    flagged_row = cleaned[cleaned["cycle_index"] == 5].iloc[0]
    assert flagged_row["is_capacity_outlier"]
    # Outliers are flagged, not silently dropped.
    assert len(cleaned) == 10


def test_validate_and_clean_imputes_missing_feature_context() -> None:
    df = _base_df(5)
    df.loc[2, "nearest_impedance_re_ohm"] = np.nan

    cleaned, report = validate_and_clean(df)

    assert report.n_feature_values_imputed >= 1
    assert not cleaned["nearest_impedance_re_ohm"].isna().any()


def test_validate_and_clean_handles_multiple_batteries_independently() -> None:
    df = pd.concat([_base_df(5, "B0005"), _base_df(5, "B0006")], ignore_index=True)

    cleaned, report = validate_and_clean(df)

    assert set(cleaned["battery_id"].unique()) == {"B0005", "B0006"}
    assert len(cleaned) == 10


def test_validate_and_clean_does_not_flag_deliberate_multi_current_regime() -> None:
    """Batteries like B0038-B0044 deliberately alternate discharge current
    levels (and so, capacity) as part of the test design - those blocks must
    not be misflagged as sensor faults just because they differ sharply from
    the other current level's capacity."""
    n_per_block = 10
    df = pd.concat(
        [_base_df(n_per_block), _base_df(n_per_block)], ignore_index=True
    ).reset_index(drop=True)
    df["cycle_index"] = range(len(df))
    df["discharge_sequence"] = range(len(df))
    # First block discharged at ~2A with capacity ~1.7-1.9 Ah (from _base_df);
    # second block discharged at ~4A, which genuinely yields much lower
    # measured capacity due to the higher rate - not a fault.
    df.loc[n_per_block:, "current_mean"] = -4.0
    df.loc[n_per_block:, "capacity_ah"] = df.loc[n_per_block:, "capacity_ah"] * 0.4

    cleaned, report = validate_and_clean(df)

    assert report.n_outliers_flagged == 0


def test_validate_and_clean_still_flags_glitch_within_a_multi_regime_battery() -> None:
    n_per_block = 10
    df = pd.concat(
        [_base_df(n_per_block), _base_df(n_per_block)], ignore_index=True
    ).reset_index(drop=True)
    df["cycle_index"] = range(len(df))
    df["discharge_sequence"] = range(len(df))
    df.loc[n_per_block:, "current_mean"] = -4.0
    df.loc[n_per_block:, "capacity_ah"] = df.loc[n_per_block:, "capacity_ah"] * 0.4
    # A genuine sensor glitch within the second (4A) regime.
    df.loc[n_per_block + 3, "capacity_ah"] = 0.0

    cleaned, report = validate_and_clean(df)

    assert report.n_outliers_flagged == 1
    flagged = cleaned[cleaned["cycle_index"] == n_per_block + 3].iloc[0]
    assert flagged["is_capacity_outlier"]
