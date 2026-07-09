from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from app.core.config import get_settings
from app.ml.registry import ModelNotFoundError, load_model


class _DummyPipeline:
    def predict(self, X):
        return [42.0] * len(X)


def _write_version(task_dir: Path, version: str, algorithm: str = "dummy") -> None:
    version_dir = task_dir / version
    version_dir.mkdir(parents=True)
    joblib.dump(_DummyPipeline(), version_dir / "model.joblib")
    manifest = {
        "task": task_dir.name,
        "version": version,
        "algorithm": algorithm,
        "feature_columns": ["f1", "f2"],
        "target_column": "soh_percent",
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_load_model_resolves_latest_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_root = tmp_path / "registry"
    task_dir = registry_root / "soh"
    _write_version(task_dir, "v20260101T000000Z")
    _write_version(task_dir, "v20260201T000000Z")

    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry_root))

    model = load_model("soh")

    assert model.version == "v20260201T000000Z"
    assert model.feature_columns == ["f1", "f2"]
    assert model.target_column == "soh_percent"


def test_load_model_pins_specific_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_root = tmp_path / "registry"
    task_dir = registry_root / "soh"
    _write_version(task_dir, "v20260101T000000Z")
    _write_version(task_dir, "v20260201T000000Z")

    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry_root))

    model = load_model("soh", version="v20260101T000000Z")

    assert model.version == "v20260101T000000Z"


def test_load_model_raises_when_task_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "registry"))

    with pytest.raises(ModelNotFoundError):
        load_model("soh")


def test_load_model_raises_when_pinned_version_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "registry"
    _write_version(registry_root / "soh", "v20260101T000000Z")
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry_root))

    with pytest.raises(ModelNotFoundError):
        load_model("soh", version="v-does-not-exist")
