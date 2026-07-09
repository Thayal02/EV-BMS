"""Dataset assembly for SOH (State of Health) regression training.

The single most important property enforced here is that nothing derived
from the target leaks back in as a feature. `soh_percent` is a direct linear
rescaling of `capacity_ah`, so `capacity_ah`, `capacity_fade_ah`,
`rated_capacity_ah`, and every rolling/diff feature computed over
`capacity_ah` (`capacity_delta_prev`, `capacity_rolling_mean`,
`capacity_rolling_std`) are excluded - a model trained with those included
would trivially "solve" SOH by inverting arithmetic rather than learning
from the actual sensor signals (voltage/current/temperature/impedance) a
deployed system would have to rely on.

`battery_id` is kept only as a *group* key for splitting, never as a
feature: encoding battery identity would let a model memorize per-battery
averages instead of generalizing from operating conditions, which is
exactly what evaluating on held-out batteries is meant to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "soh_percent"
GROUP_COLUMN = "battery_id"

# Every one of these is either the target itself, a deterministic
# transform of it, or a different task's label - see module docstring.
LEAKAGE_COLUMNS = [
    "capacity_ah",
    "capacity_fade_ah",
    "rated_capacity_ah",
    "capacity_delta_prev",
    "capacity_rolling_mean",
    "capacity_rolling_std",
    "rul_cycles",
    "rul_is_censored",
]

FEATURE_COLUMNS = [
    "discharge_sequence",
    "ambient_temperature_c",
    "discharge_duration_s",
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "current_std",
    "current_min",
    "current_max",
    "temperature_mean",
    "temperature_std",
    "temperature_min",
    "temperature_max",
    "nearest_charge_duration_s",
    "nearest_impedance_re_ohm",
    "nearest_impedance_rct_ohm",
    "nearest_impedance_lag_s",
    "days_since_first_cycle",
    "c_rate_mean",
    "charge_duration_ratio",
]


@dataclass
class SohDataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    n_rows_dropped_outliers: int
    n_rows_dropped_missing_features: int

    @property
    def n_batteries(self) -> int:
        return self.groups.nunique()


def load_soh_dataset(dataset_path: str | Path) -> SohDataset:
    """Load the processed NASA cycle-features table and assemble the SOH
    training frame: feature matrix, target, and group key.

    Rows are excluded (and counted) in two cases:
      - `is_capacity_outlier` is True: the capacity reading behind the
        target itself was flagged as a likely sensor/logging fault (see
        src/data/validation.py) - the ground truth is untrustworthy, so
        these rows are excluded from both training and evaluation rather
        than either taught to the model or used to judge it.
      - Any selected feature is missing - the feature columns are already
        imputed upstream where physically defensible (see
        src/data/validation.py), so a residual gap here means a genuine,
        un-recoverable hole (e.g. a battery with zero recorded impedance
        cycles) rather than something safe to guess.
    """
    df = pd.read_parquet(dataset_path)

    n_before_outliers = len(df)
    df = df.loc[~df["is_capacity_outlier"]].copy()
    n_rows_dropped_outliers = n_before_outliers - len(df)

    n_before_missing = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    n_rows_dropped_missing_features = n_before_missing - len(df)

    return SohDataset(
        X=df[FEATURE_COLUMNS].reset_index(drop=True),
        y=df[TARGET_COLUMN].reset_index(drop=True),
        groups=df[GROUP_COLUMN].reset_index(drop=True),
        n_rows_dropped_outliers=n_rows_dropped_outliers,
        n_rows_dropped_missing_features=n_rows_dropped_missing_features,
    )
