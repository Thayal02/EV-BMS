from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.battery_features import (
    add_calendar_age_feature,
    add_degradation_dynamics,
    add_operational_context_features,
    add_rul_labels,
    add_soh_and_fade,
    engineer_features,
)


def _fading_battery_df(battery_id: str, capacities: list[float]) -> pd.DataFrame:
    n = len(capacities)
    return pd.DataFrame(
        {
            "battery_id": [battery_id] * n,
            "discharge_sequence": list(range(n)),
            "capacity_ah": capacities,
            "current_mean": [-2.0] * n,
            "nearest_charge_duration_s": [500.0 + 10 * i for i in range(n)],
            "timestamp": pd.date_range("2008-01-01", periods=n, freq="3D"),
        }
    )


def test_add_soh_and_fade_with_scalar_rated_capacity() -> None:
    df = _fading_battery_df("B0005", [2.0, 1.8, 1.4])

    result = add_soh_and_fade(df, rated_capacity_ah=2.0)

    assert list(result["soh_percent"]) == pytest.approx([100.0, 90.0, 70.0])
    assert list(result["capacity_fade_ah"]) == pytest.approx([0.0, 0.2, 0.6])


def test_add_soh_and_fade_with_per_battery_rated_capacity() -> None:
    df = pd.concat(
        [_fading_battery_df("B0005", [2.0, 1.0]), _fading_battery_df("B0045", [1.0, 0.5])],
        ignore_index=True,
    )

    result = add_soh_and_fade(df, rated_capacity_ah={"B0005": 2.0, "B0045": 1.0})

    b5 = result[result["battery_id"] == "B0005"]
    b45 = result[result["battery_id"] == "B0045"]
    assert list(b5["soh_percent"]) == [100.0, 50.0]
    assert list(b45["soh_percent"]) == [100.0, 50.0]


def test_add_rul_labels_reaches_eol() -> None:
    # Rated capacity 2.0, EOL at 70% -> 1.4 Ah. Reaches EOL at sequence 3.
    df = add_soh_and_fade(
        _fading_battery_df("B0005", [2.0, 1.8, 1.6, 1.4, 1.3]), rated_capacity_ah=2.0
    )

    result = add_rul_labels(df, eol_capacity_fraction=0.70)

    assert list(result["rul_cycles"]) == [3.0, 2.0, 1.0, 0.0, 0.0]
    assert not result["rul_is_censored"].any()


def test_add_rul_labels_censored_when_eol_never_reached() -> None:
    # Never drops to 70% of rated capacity within the recorded cycles.
    df = add_soh_and_fade(
        _fading_battery_df("B0005", [2.0, 1.95, 1.9]), rated_capacity_ah=2.0
    )

    result = add_rul_labels(df, eol_capacity_fraction=0.70)

    assert result["rul_cycles"].isna().all()
    assert result["rul_is_censored"].all()


def test_add_degradation_dynamics_causal_rolling_stats() -> None:
    df = _fading_battery_df("B0005", [2.0, 1.8, 1.6])

    result = add_degradation_dynamics(df, window=2)

    assert np.isnan(result.loc[0, "capacity_delta_prev"])
    assert result.loc[1, "capacity_delta_prev"] == pytest.approx(1.8 - 2.0)
    # Rolling mean at row 1 uses only rows 0..1 (causal, no future leakage).
    assert result.loc[1, "capacity_rolling_mean"] == pytest.approx((2.0 + 1.8) / 2)


def test_add_calendar_age_feature_elapsed_days_since_first_cycle() -> None:
    df = _fading_battery_df("B0005", [2.0, 1.8, 1.6])

    result = add_calendar_age_feature(df)

    assert list(result["days_since_first_cycle"]) == pytest.approx([0.0, 3.0, 6.0])


def test_add_calendar_age_feature_independent_per_battery() -> None:
    df = pd.concat(
        [_fading_battery_df("B0005", [2.0, 1.8]), _fading_battery_df("B0045", [1.0, 0.9])],
        ignore_index=True,
    )

    result = add_calendar_age_feature(df)

    assert list(result[result["battery_id"] == "B0045"]["days_since_first_cycle"]) == pytest.approx(
        [0.0, 3.0]
    )


def test_add_operational_context_features_c_rate_and_charge_ratio() -> None:
    df = _fading_battery_df("B0005", [2.0, 1.8])

    result = add_operational_context_features(df, rated_capacity_ah=2.0)

    assert result.loc[0, "c_rate_mean"] == pytest.approx(1.0)
    assert result.loc[0, "charge_duration_ratio"] == pytest.approx(1.0)
    assert result.loc[1, "charge_duration_ratio"] == pytest.approx(510.0 / 500.0)


def test_engineer_features_end_to_end_runs_in_order() -> None:
    df = _fading_battery_df("B0005", [2.0, 1.8, 1.6, 1.4])

    result = engineer_features(df, rated_capacity_ah=2.0, eol_capacity_fraction=0.70)

    expected_columns = {
        "soh_percent",
        "capacity_fade_ah",
        "rul_cycles",
        "rul_is_censored",
        "capacity_delta_prev",
        "capacity_rolling_mean",
        "capacity_rolling_std",
        "days_since_first_cycle",
        "c_rate_mean",
        "charge_duration_ratio",
    }
    assert expected_columns.issubset(result.columns)
    assert result.loc[3, "rul_cycles"] == 0.0
