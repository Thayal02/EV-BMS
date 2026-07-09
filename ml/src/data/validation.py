"""Dataset validation and cleaning for NASA discharge-cycle summaries.

Three concerns, kept deliberately separate from feature engineering:

1. Schema validation - the summary table has the columns training code expects.
2. Missing value handling - `capacity_ah` is the regression label; a missing
   label cannot be imputed without fabricating ground truth, so those rows
   are dropped (and counted). Feature columns (impedance/charge context) are
   forward/backward-filled per battery, since impedance is sampled less
   often than every discharge cycle and is a slowly-varying quantity.
3. Outlier detection - the NASA READMEs themselves flag "several discharge
   runs where the capacity was very low" from unexplained sensor/logging
   faults, which our own scan of the raw files confirmed (e.g. literal 0.0
   Ahr readings inside otherwise-smooth degradation curves). These are
   flagged via a rolling-median/MAD test rather than silently dropped, so
   callers can decide whether to exclude them.

   Critically, a subset of batteries (confirmed from their batch READMEs -
   e.g. B0038-B0044) were deliberately cycled through *multiple* discharge
   current levels and ambient temperatures as part of the test design, not
   as a single fixed profile. Measured capacity (Ah) is rate-dependent, so
   a battery discharged at 4A will show a much lower "capacity" than the
   same cell at the same age discharged at 1A - a naive rolling-median test
   over the raw sequence would misflag this deliberate, repeatable,
   condition-driven variation as sensor faults. Outlier detection therefore
   first partitions each battery's cycles into "established" discharge
   current regimes (>=5 cycles and >=10% of that battery's cycles at a
   given rounded current level) and tests deviations within each regime
   rather than across the whole mixed-condition sequence. Cycles that don't
   belong to any established regime (isolated, one-off low readings) are
   compared against the nearest established regime instead, which is
   exactly how the genuine single-cycle sensor faults (e.g. in B0033/B0045)
   get caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = [
    "battery_id",
    "cycle_index",
    "discharge_sequence",
    "timestamp",
    "ambient_temperature_c",
    "capacity_ah",
    "discharge_duration_s",
    "voltage_mean",
    "current_mean",
    "temperature_mean",
    "nearest_impedance_re_ohm",
    "nearest_impedance_rct_ohm",
]

_FORWARD_FILLABLE_COLUMNS = [
    "nearest_charge_duration_s",
    "nearest_impedance_re_ohm",
    "nearest_impedance_rct_ohm",
    "nearest_impedance_lag_s",
]


class DatasetValidationError(ValueError):
    """Raised when the input does not satisfy the required schema."""


@dataclass
class ValidationReport:
    n_rows_in: int
    n_duplicates_removed: int = 0
    n_missing_label_dropped: int = 0
    n_feature_values_imputed: int = 0
    n_outliers_flagged: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_rows_out(self) -> int:
        return self.n_rows_in - self.n_duplicates_removed - self.n_missing_label_dropped

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_duplicates_removed": self.n_duplicates_removed,
            "n_missing_label_dropped": self.n_missing_label_dropped,
            "n_feature_values_imputed": self.n_feature_values_imputed,
            "n_outliers_flagged": self.n_outliers_flagged,
            "notes": self.notes,
        }


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")


def _rolling_mad_outliers(
    capacity: pd.Series, window: int, mad_multiplier: float, mad_floor: float = 0.01
) -> pd.Series:
    """Flag values far from their local rolling median.

    MAD (median absolute deviation) is used instead of std because it isn't
    itself distorted by the outliers it's trying to detect. A zero MAD (e.g.
    a run of identical values) would make every nonzero deviation register
    as infinitely many MADs away, so a small absolute floor keeps flat-but-
    clean regions from all being flagged.
    """
    rolling_median = capacity.rolling(window=window, center=True, min_periods=1).median()
    deviation = (capacity - rolling_median).abs()
    mad = deviation.rolling(window=window, center=True, min_periods=1).median()
    threshold = mad_multiplier * mad.clip(lower=mad_floor)
    return deviation > threshold


def _established_current_regimes(
    current_regime: pd.Series, min_count: int = 5, min_fraction: float = 0.10
) -> set[float]:
    """Discharge current levels (rounded to the nearest amp) that recur often
    enough within a battery to represent a deliberate, sustained test
    condition rather than a one-off anomalous reading."""
    counts = current_regime.value_counts()
    threshold = max(min_count, min_fraction * len(current_regime))
    return set(counts[counts >= threshold].index)


def _flag_capacity_outliers(
    group: pd.DataFrame, window: int = 5, mad_multiplier: float = 6.0
) -> pd.Series:
    """Flag capacity readings that look like sensor/logging faults rather
    than genuine aging or a deliberate change in test condition, for one
    battery's discharge cycles (already sorted by discharge_sequence).

    See module docstring for why this is regime-aware: measured capacity is
    confounded by discharge current, and several batteries in this dataset
    deliberately vary it as part of the experiment design.
    """
    current_regime = group["current_mean"].round()
    established = _established_current_regimes(current_regime)

    flags = pd.Series(False, index=group.index)

    if len(established) <= 1:
        # Single (or no clearly dominant) discharge condition - the simple
        # whole-battery rolling test is sufficient and avoids slicing an
        # already-small series into even smaller, noisier pieces.
        flags[:] = _rolling_mad_outliers(group["capacity_ah"], window, mad_multiplier).values
        return flags

    for regime_value in established:
        regime_mask = current_regime == regime_value
        flags.loc[regime_mask] = _rolling_mad_outliers(
            group.loc[regime_mask, "capacity_ah"], window, mad_multiplier
        ).values

    # Cycles that don't belong to any established regime (isolated one-off
    # readings) are compared against whichever established regime's typical
    # current level they're closest to, since they're too rare to support
    # their own rolling baseline.
    unestablished_mask = ~current_regime.isin(established)
    if unestablished_mask.any():
        established_list = sorted(established)
        for idx in group.index[unestablished_mask]:
            nearest_regime = min(established_list, key=lambda r: abs(r - current_regime[idx]))
            reference_capacity = group.loc[current_regime == nearest_regime, "capacity_ah"]
            reference_median = reference_capacity.median()
            reference_mad = (reference_capacity - reference_median).abs().median()
            threshold = mad_multiplier * max(reference_mad, 0.01)
            flags[idx] = abs(group.loc[idx, "capacity_ah"] - reference_median) > threshold

    return flags


def validate_and_clean(
    df: pd.DataFrame,
    *,
    outlier_window: int = 5,
    outlier_mad_multiplier: float = 6.0,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate schema, drop duplicates/missing labels, impute feature gaps,
    and flag (without dropping) capacity outliers.

    Returns the cleaned DataFrame (with an added `is_capacity_outlier` column)
    and a report of every action taken, so cleaning is auditable rather than
    a silent black box.
    """
    validate_schema(df)
    report = ValidationReport(n_rows_in=len(df))

    working = df.sort_values(["battery_id", "discharge_sequence"]).reset_index(drop=True)

    before = len(working)
    working = working.drop_duplicates(subset=["battery_id", "cycle_index"], keep="first")
    report.n_duplicates_removed = before - len(working)

    missing_label_mask = working["capacity_ah"].isna()
    report.n_missing_label_dropped = int(missing_label_mask.sum())
    if report.n_missing_label_dropped:
        report.notes.append(
            f"Dropped {report.n_missing_label_dropped} row(s) with missing capacity_ah "
            "label (empty measurement in source .mat file - see ml/README.md)."
        )
    working = working.loc[~missing_label_mask].copy()

    fillable = [c for c in _FORWARD_FILLABLE_COLUMNS if c in working.columns]
    na_before = int(working[fillable].isna().sum().sum())
    for _, group_index in working.groupby("battery_id").groups.items():
        working.loc[group_index, fillable] = (
            working.loc[group_index, fillable].ffill().bfill()
        )
    na_after = int(working[fillable].isna().sum().sum())
    report.n_feature_values_imputed = na_before - na_after

    outlier_flags = pd.Series(False, index=working.index)
    for _, group_index in working.groupby("battery_id").groups.items():
        group = working.loc[group_index]
        outlier_flags.loc[group_index] = _flag_capacity_outliers(
            group, outlier_window, outlier_mad_multiplier
        )
    working["is_capacity_outlier"] = outlier_flags
    report.n_outliers_flagged = int(working["is_capacity_outlier"].sum())
    if report.n_outliers_flagged:
        report.notes.append(
            f"Flagged {report.n_outliers_flagged} capacity reading(s) as outliers "
            "(rolling-median/MAD test) - not dropped, see 'is_capacity_outlier' column."
        )

    return working.reset_index(drop=True), report
