"""Reduce a battery's raw charge/discharge/impedance cycles to one tidy
row per discharge cycle - the unit of analysis for SOH/RUL modeling.

Each discharge cycle in the NASA dataset carries the ground-truth capacity
measurement (`Capacity`) that SOH is derived from, plus terminal
voltage/current/temperature time series. Charge and impedance cycles don't
carry a capacity label, but the closest-in-time charge and impedance cycles
around a given discharge still carry predictive signal (e.g. internal
resistance drift, charge duration) - those are attached as extra columns
rather than kept as separate tables, since a per-discharge-cycle table is
what every downstream step (validation, feature engineering, model
training) operates on.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.data.nasa_loader import RawCycle


def _matlab_datevec_to_datetime(vec: np.ndarray) -> datetime:
    year, month, day, hour, minute, second = vec
    second_int = int(second)
    microsecond = int(round((float(second) - second_int) * 1_000_000))
    return datetime(int(year), int(month), int(day), int(hour), int(minute), second_int, microsecond)


def _safe_scalar(value: object) -> float:
    """Coerce a .mat measurement to a float, treating empty arrays as missing.

    Some discharge cycles in this dataset (see ml/README.md /
    docs/data_quality_notes) have `Capacity` stored as an empty MATLAB
    matrix rather than NaN when the measurement failed - `np.asarray(value)`
    on an empty array has size 0, which we must not silently coerce to 0.0.

    A handful of impedance cycles (observed in the low-temperature B0049/
    B0051 batches) report `Re`/`Rct` as complex numbers - a non-physical
    result from a nonlinear equivalent-circuit fit that failed to converge,
    not a real resistance measurement with an imaginary component. Those are
    treated as missing rather than silently reduced to their real part,
    which would fabricate a value from a known-bad fit.
    """
    arr = np.atleast_1d(np.asarray(value))
    if arr.size == 0:
        return float("nan")
    scalar = arr.reshape(-1)[0]
    if np.iscomplexobj(scalar):
        return float("nan")
    return float(scalar)


# A physically implausible bound for the internal-resistance components (Re,
# Rct) of these small 18650-format research cells: real fitted values in this
# dataset fall in roughly the 0.02-0.3 Ohm range even at end-of-life. A small
# number of impedance cycles (observed in B0050) have a *real-valued* (not
# complex) but wildly diverged nonlinear curve fit - e.g. Re on the order of
# 1e14 Ohm - which the complex-number check above doesn't catch since the
# result isn't complex, just non-physical.
_MAX_PLAUSIBLE_RESISTANCE_OHM = 10.0


def _safe_resistance_scalar(value: object) -> float:
    """Like `_safe_scalar`, but additionally rejects real-valued fit results
    that are non-physical for these cells' internal resistance (a diverged
    curve fit, not a genuine measurement) - see `_MAX_PLAUSIBLE_RESISTANCE_OHM`.
    """
    scalar = _safe_scalar(value)
    if np.isnan(scalar) or scalar < 0 or scalar > _MAX_PLAUSIBLE_RESISTANCE_OHM:
        return float("nan")
    return scalar


def _array_stats(value: object, prefix: str) -> dict[str, float]:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"),
                f"{prefix}_min": float("nan"), f"{prefix}_max": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _duration_seconds(time_array: object) -> float:
    arr = np.atleast_1d(np.asarray(time_array, dtype=float))
    if arr.size < 2:
        return float("nan")
    return float(arr[-1] - arr[0])


def _nearest_cycle(target_dt: datetime, candidates: list[RawCycle]) -> RawCycle | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: abs((_matlab_datevec_to_datetime(c.timestamp) - target_dt).total_seconds()),
    )


def build_discharge_summary(cycles: list[RawCycle]) -> pd.DataFrame:
    """Build a per-discharge-cycle summary DataFrame for one battery.

    Columns:
        battery_id, cycle_index, discharge_sequence, timestamp,
        ambient_temperature_c, capacity_ah,
        voltage_{mean,std,min,max}, current_{mean,std,min,max},
        temperature_{mean,std,min,max}, discharge_duration_s,
        nearest_charge_duration_s, nearest_impedance_re_ohm,
        nearest_impedance_rct_ohm, nearest_impedance_lag_s
    """
    discharge_cycles = [c for c in cycles if c.type == "discharge"]
    charge_cycles = [c for c in cycles if c.type == "charge"]
    impedance_cycles = [c for c in cycles if c.type == "impedance"]

    rows: list[dict[str, object]] = []
    for seq, cycle in enumerate(discharge_cycles):
        dt = _matlab_datevec_to_datetime(cycle.timestamp)
        row: dict[str, object] = {
            "battery_id": cycle.battery_id,
            "cycle_index": cycle.cycle_index,
            "discharge_sequence": seq,
            "timestamp": dt,
            "ambient_temperature_c": cycle.ambient_temperature_c,
            "capacity_ah": _safe_scalar(cycle.data.get("Capacity")),
            "discharge_duration_s": _duration_seconds(cycle.data.get("Time")),
        }
        row.update(_array_stats(cycle.data.get("Voltage_measured"), "voltage"))
        row.update(_array_stats(cycle.data.get("Current_measured"), "current"))
        row.update(_array_stats(cycle.data.get("Temperature_measured"), "temperature"))

        nearest_charge = _nearest_cycle(dt, charge_cycles)
        row["nearest_charge_duration_s"] = (
            _duration_seconds(nearest_charge.data.get("Time")) if nearest_charge else float("nan")
        )

        nearest_impedance = _nearest_cycle(dt, impedance_cycles)
        if nearest_impedance is not None:
            row["nearest_impedance_re_ohm"] = _safe_resistance_scalar(
                nearest_impedance.data.get("Re")
            )
            row["nearest_impedance_rct_ohm"] = _safe_resistance_scalar(
                nearest_impedance.data.get("Rct")
            )
            row["nearest_impedance_lag_s"] = abs(
                (_matlab_datevec_to_datetime(nearest_impedance.timestamp) - dt).total_seconds()
            )
        else:
            row["nearest_impedance_re_ohm"] = float("nan")
            row["nearest_impedance_rct_ohm"] = float("nan")
            row["nearest_impedance_lag_s"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)
