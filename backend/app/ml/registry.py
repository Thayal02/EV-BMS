"""Model registry loader.

Mirrors the model registry contract documented in ml/README.md: training
code in ml/ writes `model.joblib` + `manifest.json` per task/version under
`ml/models/registry/<task>/<version>/`. This module only ever *reads* those
files - it never imports or invokes training code, which is what makes
retraining possible without redeploying the API: a new version directory
appears, and the next call to `load_model` (or the next process restart,
for the cached path) picks it up.

Version directories are named `v<UTC timestamp>` (e.g. `v20260709T093758Z`),
which sorts lexicographically the same as chronologically, so "latest" is
just the last entry in a sorted directory listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import joblib

from app.core.config import get_settings


class ModelNotFoundError(RuntimeError):
    """Raised when no trained model version exists for a requested task."""


@dataclass
class LoadedModel:
    task: str
    version: str
    pipeline: Any
    manifest: dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return self.manifest["feature_columns"]

    @property
    def target_column(self) -> str:
        return self.manifest["target_column"]

    @property
    def algorithm(self) -> str:
        return self.manifest["algorithm"]


def _resolve_version_dir(task: str, version: str | None) -> Path:
    task_dir = get_settings().model_registry_dir / task
    if not task_dir.is_dir():
        raise ModelNotFoundError(f"No models registered for task '{task}' (expected {task_dir})")

    if version is not None:
        version_dir = task_dir / version
        if not version_dir.is_dir():
            raise ModelNotFoundError(
                f"Version '{version}' not found for task '{task}' (expected {version_dir})"
            )
        return version_dir

    version_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
    if not version_dirs:
        raise ModelNotFoundError(f"No model versions found for task '{task}' in {task_dir}")
    return version_dirs[-1]


@cache
def _load_artifacts(version_dir: str) -> tuple[Any, dict[str, Any]]:
    """Cached by exact resolved version directory - loading a joblib model
    is expensive and the artifact for a given version never changes, but
    resolving *which* version is "latest" is cheap and deliberately left
    uncached in `load_model` so a newly deployed version is picked up
    without needing this cache to be invalidated.
    """
    path = Path(version_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    pipeline = joblib.load(path / "model.joblib")
    return pipeline, manifest


def load_model(task: str, version: str | None = None) -> LoadedModel:
    """Load a task's model - the latest version by default, or a pinned one."""
    version_dir = _resolve_version_dir(task, version)
    pipeline, manifest = _load_artifacts(str(version_dir))
    return LoadedModel(task=task, version=manifest["version"], pipeline=pipeline, manifest=manifest)
