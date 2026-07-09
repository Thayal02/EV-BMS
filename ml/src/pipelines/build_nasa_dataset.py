"""End-to-end pipeline: raw NASA .mat files -> cleaned, feature-engineered,
per-discharge-cycle dataset ready for SOH/RUL model training.

Usage:
    python -m src.pipelines.build_nasa_dataset \
        --config configs/nasa_battery.yaml \
        --output-dir data/processed

Writes:
    data/processed/nasa_cycle_features.parquet   (full dataset, all batteries)
    data/processed/nasa_cycle_features.csv       (same, human-readable)
    data/processed/validation_report.json        (per-battery cleaning report)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cycle_summary import build_discharge_summary  # noqa: E402
from src.data.nasa_loader import discover_battery_files, load_battery_file  # noqa: E402
from src.data.validation import validate_and_clean  # noqa: E402
from src.features.battery_features import engineer_features  # noqa: E402


def _resolve_battery_sources(config: dict) -> dict[str, tuple[Path, float]]:
    """Map battery_id -> (mat file path, rated_capacity_ah), applying the
    configured duplicate-resolution policy when a battery_id appears in
    more than one batch (e.g. B0025-28 exist in both the FY08Q4-adjacent
    'P1' batch and the full 25-44 batch)."""
    raw_root = Path(config["raw_data_root"])
    resolution = config.get("duplicate_battery_id_resolution", "prefer_last_batch")
    if resolution != "prefer_last_batch":
        raise NotImplementedError(f"Unsupported duplicate resolution policy: {resolution}")

    sources: dict[str, tuple[Path, float]] = {}
    for batch in config["batches"]:
        batch_dir = raw_root / batch["name"]
        files_by_id = discover_battery_files(batch_dir)
        rated_capacity = batch["rated_capacity_ah"]
        for battery_id in batch["battery_ids"]:
            if battery_id not in files_by_id:
                raise FileNotFoundError(f"{battery_id}.mat not found in {batch_dir}")
            # Later batches overwrite earlier ones - "prefer_last_batch".
            sources[battery_id] = (files_by_id[battery_id], rated_capacity)
    return sources


def build_dataset(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    config = yaml.safe_load(Path(config_path).read_text())
    sources = _resolve_battery_sources(config)
    eol_fraction = config["eol_capacity_fraction"]

    per_battery_frames: list[pd.DataFrame] = []
    validation_reports: dict[str, dict] = {}

    for battery_id, (mat_path, rated_capacity_ah) in sorted(sources.items()):
        cycles = load_battery_file(mat_path, battery_id=battery_id)
        summary = build_discharge_summary(cycles)
        if summary.empty:
            validation_reports[battery_id] = {"error": "no discharge cycles found"}
            continue

        cleaned, report = validate_and_clean(summary)
        featured = engineer_features(cleaned, rated_capacity_ah, eol_fraction)

        per_battery_frames.append(featured)
        validation_reports[battery_id] = report.as_dict()

    full_dataset = pd.concat(per_battery_frames, ignore_index=True)
    return full_dataset, validation_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nasa_battery.yaml")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, reports = build_dataset(args.config)

    dataset.to_parquet(output_dir / "nasa_cycle_features.parquet", index=False)
    dataset.to_csv(output_dir / "nasa_cycle_features.csv", index=False)
    (output_dir / "validation_report.json").write_text(json.dumps(reports, indent=2))

    print(f"Built dataset: {len(dataset)} rows, {dataset['battery_id'].nunique()} batteries")
    print(f"Written to {output_dir}/nasa_cycle_features.{{parquet,csv}}")
    total_outliers = sum(r.get("n_outliers_flagged", 0) for r in reports.values())
    total_dropped = sum(r.get("n_missing_label_dropped", 0) for r in reports.values())
    print(f"Total capacity outliers flagged: {total_outliers}")
    print(f"Total rows dropped for missing label: {total_dropped}")


if __name__ == "__main__":
    main()
