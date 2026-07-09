"""Feature engineering for the NASA discharge-cycle summary table.

Operates purely on the tidy per-discharge-cycle DataFrame produced by
`src.data.cycle_summary.build_discharge_summary` and cleaned by
`src.data.validation.validate_and_clean`. Every feature here is causal (uses
only the current and past cycles of its own battery) so nothing here leaks
future information into a row - that property matters more than usual
because SOH/RUL models are evaluated on held-out *batteries*, and a feature
that peeked ahead would make offline metrics look better than deployment
performance ever could.

Normalization/scaling is deliberately NOT done here - it happens inside each
model's training pipeline (fit on the training split only) so it never
leaks test-set statistics. Feature *selection* is likewise a training-time
concern (driven by model-based importance) and not performed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RolloingWindow = 5


def add_soh_and_fade(df: pd.DataFrame, rated_capacity_ah: float | dict[str, float]) -> pd.DataFrame:
    """Add `rated_capacity_ah`, `soh_percent`, and `capacity_fade_ah` columns.

    SOH is defined the standard way for this dataset: measured discharge
    capacity as a percentage of the cell's rated (nameplate) capacity.
    """
    df = df.copy()
    if isinstance(rated_capacity_ah, dict):
        df["rated_capacity_ah"] = df["battery_id"].map(rated_capacity_ah)
    else:
        df["rated_capacity_ah"] = rated_capacity_ah

    df["soh_percent"] = (df["capacity_ah"] / df["rated_capacity_ah"]) * 100.0
    df["capacity_fade_ah"] = df["rated_capacity_ah"] - df["capacity_ah"]
    return df


def add_rul_labels(df: pd.DataFrame, eol_capacity_fraction: float) -> pd.DataFrame:
    """Add `rul_cycles` and `rul_is_censored` columns.

    End-of-life (EOL) is reached at the first discharge cycle whose SOH
    drops to or below `eol_capacity_fraction * 100`. RUL for every cycle
    before that point is the number of discharge cycles remaining until
    EOL. If a battery's recorded cycles never reach EOL, RUL is
    right-censored: we do not extrapolate a fabricated end-of-life cycle,
    we report `rul_cycles = NaN` and `rul_is_censored = True` for that
    battery's rows so training/evaluation code can decide how to treat
    censored samples (e.g. exclude them, or use a survival-analysis loss)
    rather than silently training against a guessed label.
    """
    df = df.sort_values(["battery_id", "discharge_sequence"]).copy()
    df["rul_cycles"] = np.nan
    df["rul_is_censored"] = True

    for _battery_id, group in df.groupby("battery_id"):
        eol_mask = group["soh_percent"] <= (eol_capacity_fraction * 100.0)
        if not eol_mask.any():
            continue
        eol_sequence = group.loc[eol_mask, "discharge_sequence"].iloc[0]
        rul = (eol_sequence - group["discharge_sequence"]).clip(lower=0)
        df.loc[group.index, "rul_cycles"] = rul.values
        df.loc[group.index, "rul_is_censored"] = False

    return df.reset_index(drop=True)


def add_degradation_dynamics(df: pd.DataFrame, window: int = RolloingWindow) -> pd.DataFrame:
    """Add trailing (causal) rolling statistics and cycle-to-cycle deltas of capacity."""
    df = df.sort_values(["battery_id", "discharge_sequence"]).copy()
    grouped = df.groupby("battery_id")["capacity_ah"]

    df["capacity_delta_prev"] = grouped.diff()
    df["capacity_rolling_mean"] = grouped.transform(
        lambda s: s.rolling(window=window, min_periods=1).mean()
    )
    df["capacity_rolling_std"] = grouped.transform(
        lambda s: s.rolling(window=window, min_periods=1).std()
    ).fillna(0.0)
    return df.reset_index(drop=True)


def add_calendar_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add `days_since_first_cycle` - elapsed wall-clock time since each
    battery's first recorded discharge cycle.

    This captures calendar/rest-time aging (e.g. time spent idle between
    test sessions) as distinct from `discharge_sequence`, which only counts
    cycles and is blind to how much real time elapsed between them. It's
    derived purely from timestamps, not from capacity, so it carries no risk
    of leaking the SOH/RUL targets.
    """
    df = df.sort_values(["battery_id", "discharge_sequence"]).copy()
    first_timestamp = df.groupby("battery_id")["timestamp"].transform("first")
    df["days_since_first_cycle"] = (df["timestamp"] - first_timestamp).dt.total_seconds() / 86400.0
    return df.reset_index(drop=True)


def add_operational_context_features(
    df: pd.DataFrame, rated_capacity_ah: float | dict[str, float]
) -> pd.DataFrame:
    """Add C-rate and normalized charge-duration features.

    `charge_duration_ratio` compares each battery's charge time to its own
    first recorded cycle, so it reflects *relative* growth in charge time
    (a known symptom of rising internal resistance) rather than an absolute
    duration that would vary with the current-limiting profile.
    """
    df = df.sort_values(["battery_id", "discharge_sequence"]).copy()

    if isinstance(rated_capacity_ah, dict):
        rated = df["battery_id"].map(rated_capacity_ah)
    else:
        rated = rated_capacity_ah
    df["c_rate_mean"] = df["current_mean"].abs() / rated

    def _normalize_charge_duration(group: pd.DataFrame) -> pd.Series:
        baseline = group["nearest_charge_duration_s"].iloc[0]
        if not baseline or np.isnan(baseline) or baseline == 0:
            return pd.Series(np.nan, index=group.index)
        return group["nearest_charge_duration_s"] / baseline

    charge_duration_ratio = pd.Series(np.nan, index=df.index)
    for _, group_index in df.groupby("battery_id").groups.items():
        charge_duration_ratio.loc[group_index] = _normalize_charge_duration(df.loc[group_index])
    df["charge_duration_ratio"] = charge_duration_ratio
    return df.reset_index(drop=True)


def engineer_features(
    df: pd.DataFrame,
    rated_capacity_ah: float | dict[str, float],
    eol_capacity_fraction: float,
) -> pd.DataFrame:
    """Run the full feature engineering sequence in the correct order.

    Order matters: SOH must exist before RUL labels (RUL's EOL threshold is
    defined in terms of SOH), and rows must be sorted by
    (battery_id, discharge_sequence) throughout for the rolling/diff
    operations to be meaningful.
    """
    df = add_soh_and_fade(df, rated_capacity_ah)
    df = add_rul_labels(df, eol_capacity_fraction)
    df = add_degradation_dynamics(df)
    df = add_calendar_age_feature(df)
    df = add_operational_context_features(df, rated_capacity_ah)
    return df
