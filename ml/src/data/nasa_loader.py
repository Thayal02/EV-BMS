"""Low-level parser for NASA Prognostics Center of Excellence battery .mat files.

Each .mat file contains a single top-level struct (keyed by the battery id,
e.g. "B0005") with one field, `cycle`, holding a struct array. Each element
of `cycle` is one charge, discharge, or impedance operation with:

    type: "charge" | "discharge" | "impedance"
    ambient_temperature: float (degrees C)
    time: MATLAB date vector [year, month, day, hour, minute, second]
    data: dict of measurement arrays/scalars, whose keys depend on `type`

This module only turns that raw structure into plain Python objects - no
cleaning, validation, or feature engineering happens here (see
`validation.py` and `../features/`). Keeping this boundary means a change in
raw file layout (e.g. a new NASA sub-experiment format) only ever touches
this one module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.io as sio

CycleType = Literal["charge", "discharge", "impedance"]


@dataclass
class RawCycle:
    """One charge/discharge/impedance operation, as recorded in the .mat file."""

    battery_id: str
    cycle_index: int
    type: CycleType
    ambient_temperature_c: float
    timestamp: np.ndarray
    """MATLAB date vector [year, month, day, hour, minute, second]."""
    data: dict[str, Any]
    """Raw measurement fields for this cycle type - keys vary by `type`."""


def load_battery_file(mat_path: str | Path, battery_id: str | None = None) -> list[RawCycle]:
    """Parse one NASA battery .mat file into a list of RawCycle records.

    Args:
        mat_path: path to a B00XX.mat file.
        battery_id: expected top-level struct key. If omitted, it's inferred
            from the filename stem (e.g. "B0005.mat" -> "B0005"), which
            matches every NASA battery file observed in this dataset.

    Raises:
        KeyError: if the inferred/given battery_id is not a top-level key in
            the .mat file - this indicates the file doesn't follow the
            expected NASA layout and should not be silently skipped.
    """
    mat_path = Path(mat_path)
    battery_id = battery_id or mat_path.stem

    raw = sio.loadmat(str(mat_path), simplify_cells=True)
    if battery_id not in raw:
        available = [k for k in raw if not k.startswith("__")]
        raise KeyError(
            f"Expected top-level key '{battery_id}' in {mat_path}, found {available}"
        )

    cycles = raw[battery_id]["cycle"]
    # A .mat file with exactly one cycle would be simplified to a single
    # dict rather than a length-1 list by simplify_cells - normalize so
    # callers always get a list.
    if isinstance(cycles, dict):
        cycles = [cycles]

    records: list[RawCycle] = []
    for idx, cycle in enumerate(cycles):
        records.append(
            RawCycle(
                battery_id=battery_id,
                cycle_index=idx,
                type=cycle["type"],
                ambient_temperature_c=float(cycle["ambient_temperature"]),
                timestamp=np.asarray(cycle["time"], dtype=float),
                data=cycle["data"],
            )
        )
    return records


def discover_battery_files(batch_root: str | Path) -> dict[str, Path]:
    """Map battery_id -> .mat file path for every battery file in a batch directory."""
    batch_root = Path(batch_root)
    return {p.stem: p for p in sorted(batch_root.glob("*.mat"))}
