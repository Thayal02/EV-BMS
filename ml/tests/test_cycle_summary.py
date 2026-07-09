from __future__ import annotations

import numpy as np

from src.data.cycle_summary import build_discharge_summary
from src.data.nasa_loader import RawCycle


def _cycle(
    cycle_index: int,
    cycle_type: str,
    timestamp: list[float],
    data: dict,
    ambient_temperature_c: float = 24.0,
    battery_id: str = "B9999",
) -> RawCycle:
    return RawCycle(
        battery_id=battery_id,
        cycle_index=cycle_index,
        type=cycle_type,
        ambient_temperature_c=ambient_temperature_c,
        timestamp=np.array(timestamp, dtype=float),
        data=data,
    )


def test_build_discharge_summary_basic_fields() -> None:
    cycles = [
        _cycle(0, "charge", [2008, 4, 2, 10, 0, 0], {"Time": np.array([0.0, 500.0])}),
        _cycle(
            1,
            "discharge",
            [2008, 4, 2, 12, 0, 0],
            {
                "Voltage_measured": np.array([4.2, 3.8, 3.0]),
                "Current_measured": np.array([-2.0, -2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 25.0, 26.0]),
                "Time": np.array([0.0, 1000.0, 2000.0]),
                "Capacity": 1.8,
            },
        ),
        _cycle(
            2,
            "impedance",
            [2008, 4, 2, 12, 5, 0],
            {"Re": 0.045, "Rct": 0.07},
        ),
    ]

    summary = build_discharge_summary(cycles)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["battery_id"] == "B9999"
    assert row["cycle_index"] == 1
    assert row["discharge_sequence"] == 0
    assert row["capacity_ah"] == 1.8
    assert row["discharge_duration_s"] == 2000.0
    assert row["voltage_mean"] == np.mean([4.2, 3.8, 3.0])
    assert row["nearest_charge_duration_s"] == 500.0
    # Impedance cycle is 5 minutes after discharge - nearest and only candidate.
    assert row["nearest_impedance_re_ohm"] == 0.045
    assert row["nearest_impedance_rct_ohm"] == 0.07
    assert row["nearest_impedance_lag_s"] == 300.0


def test_build_discharge_summary_handles_empty_capacity_array() -> None:
    cycles = [
        _cycle(
            0,
            "discharge",
            [2008, 4, 2, 12, 0, 0],
            {
                "Voltage_measured": np.array([4.2, 3.0]),
                "Current_measured": np.array([-2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 24.0]),
                "Time": np.array([0.0, 1000.0]),
                "Capacity": np.array([]),
            },
        ),
    ]

    summary = build_discharge_summary(cycles)

    assert np.isnan(summary.iloc[0]["capacity_ah"])


def test_build_discharge_summary_treats_complex_impedance_as_missing() -> None:
    cycles = [
        _cycle(
            0,
            "discharge",
            [2008, 4, 2, 12, 0, 0],
            {
                "Voltage_measured": np.array([4.2, 3.0]),
                "Current_measured": np.array([-2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 24.0]),
                "Time": np.array([0.0, 1000.0]),
                "Capacity": 1.5,
            },
        ),
        _cycle(
            1,
            "impedance",
            [2008, 4, 2, 12, 1, 0],
            {"Re": complex(0.05, -0.03), "Rct": complex(0.05, 0.03)},
        ),
    ]

    summary = build_discharge_summary(cycles)

    assert np.isnan(summary.iloc[0]["nearest_impedance_re_ohm"])
    assert np.isnan(summary.iloc[0]["nearest_impedance_rct_ohm"])


def test_build_discharge_summary_treats_diverged_resistance_fit_as_missing() -> None:
    """A handful of impedance cycles (observed in B0050) have a real-valued
    but physically impossible Re/Rct (~1e14 Ohm) from a nonlinear curve fit
    that diverged rather than failing outright - these aren't caught by the
    complex-number check and must be rejected on physical plausibility."""
    cycles = [
        _cycle(
            0,
            "discharge",
            [2008, 4, 2, 12, 0, 0],
            {
                "Voltage_measured": np.array([4.2, 3.0]),
                "Current_measured": np.array([-2.0, -2.0]),
                "Temperature_measured": np.array([24.0, 24.0]),
                "Time": np.array([0.0, 1000.0]),
                "Capacity": 1.5,
            },
        ),
        _cycle(
            1,
            "impedance",
            [2008, 4, 2, 12, 1, 0],
            {"Re": -968924452345684.0, "Rct": 2055843366793053.5},
        ),
    ]

    summary = build_discharge_summary(cycles)

    assert np.isnan(summary.iloc[0]["nearest_impedance_re_ohm"])
    assert np.isnan(summary.iloc[0]["nearest_impedance_rct_ohm"])


def test_build_discharge_summary_no_discharge_cycles_returns_empty() -> None:
    cycles = [_cycle(0, "charge", [2008, 4, 2, 10, 0, 0], {"Time": np.array([0.0, 1.0])})]

    summary = build_discharge_summary(cycles)

    assert summary.empty
