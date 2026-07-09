"""Tests for src.data.nasa_loader.

These monkeypatch scipy.io.loadmat rather than shipping a real .mat binary
fixture: `simplify_cells=True` output is just nested Python dicts/lists/
scalars/ndarrays, so a hand-built dict in that shape exercises exactly the
same parsing logic (list-normalization, KeyError handling, RawCycle
construction) as a real file would, without a binary fixture to maintain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import nasa_loader


def _fake_raw_mat(cycles: list[dict], battery_id: str = "B9999") -> dict:
    return {
        "__header__": b"fake",
        "__version__": "1.0",
        "__globals__": [],
        battery_id: {"cycle": cycles},
    }


def _charge_cycle(ambient_temperature: float = 24.0) -> dict:
    return {
        "type": "charge",
        "ambient_temperature": ambient_temperature,
        "time": [2008, 4, 2, 13, 8, 17.921],
        "data": {
            "Voltage_measured": np.array([3.7, 3.8, 4.0]),
            "Current_measured": np.array([1.5, 1.5, 1.4]),
            "Temperature_measured": np.array([24.0, 24.2, 24.5]),
            "Current_charge": np.array([1.5, 1.5, 1.4]),
            "Voltage_charge": np.array([3.7, 3.8, 4.0]),
            "Time": np.array([0.0, 100.0, 200.0]),
        },
    }


def test_load_battery_file_normalizes_single_cycle_to_list(monkeypatch: pytest.MonkeyPatch) -> None:
    single_cycle = _charge_cycle()
    monkeypatch.setattr(
        nasa_loader.sio, "loadmat", lambda path, simplify_cells: _fake_raw_mat(single_cycle)
    )

    records = nasa_loader.load_battery_file("B9999.mat", battery_id="B9999")

    assert len(records) == 1
    assert records[0].type == "charge"
    assert records[0].battery_id == "B9999"
    assert records[0].cycle_index == 0


def test_load_battery_file_parses_multiple_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    cycles = [_charge_cycle(), _charge_cycle(ambient_temperature=25.0)]
    monkeypatch.setattr(
        nasa_loader.sio, "loadmat", lambda path, simplify_cells: _fake_raw_mat(cycles)
    )

    records = nasa_loader.load_battery_file("B9999.mat", battery_id="B9999")

    assert [r.cycle_index for r in records] == [0, 1]
    assert records[1].ambient_temperature_c == 25.0


def test_load_battery_file_raises_on_missing_battery_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nasa_loader.sio,
        "loadmat",
        lambda path, simplify_cells: _fake_raw_mat([_charge_cycle()], battery_id="B0005"),
    )

    with pytest.raises(KeyError):
        nasa_loader.load_battery_file("B9999.mat", battery_id="B9999")


def test_load_battery_file_infers_battery_id_from_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nasa_loader.sio,
        "loadmat",
        lambda path, simplify_cells: _fake_raw_mat([_charge_cycle()], battery_id="B0042"),
    )

    records = nasa_loader.load_battery_file("/some/path/B0042.mat")

    assert records[0].battery_id == "B0042"


def test_discover_battery_files(tmp_path: Path) -> None:
    (tmp_path / "B0005.mat").touch()
    (tmp_path / "B0006.mat").touch()
    (tmp_path / "README.txt").touch()

    files = nasa_loader.discover_battery_files(tmp_path)

    assert set(files.keys()) == {"B0005", "B0006"}
    assert files["B0005"] == tmp_path / "B0005.mat"
