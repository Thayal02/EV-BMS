from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


class _DummyPipeline:
    def predict(self, X):
        return [88.5] * len(X)


def _write_soh_model(registry_root: Path, version: str = "v1") -> None:
    version_dir = registry_root / "soh" / version
    version_dir.mkdir(parents=True)
    joblib.dump(_DummyPipeline(), version_dir / "model.joblib")
    manifest = {
        "task": "soh",
        "version": version,
        "algorithm": "dummy",
        "feature_columns": ["voltage_mean", "current_mean"],
        "target_column": "soh_percent",
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@localhost:5432/db")
    get_settings.cache_clear()

    _write_soh_model(tmp_path / "registry")

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


def test_predict_soh_success(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predictions/soh",
        json={"features": {"voltage_mean": 3.8, "current_mean": -2.0}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_soh_percent"] == pytest.approx(88.5)
    assert body["model_version"] == "v1"
    assert body["model_algorithm"] == "dummy"


def test_predict_soh_missing_feature_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predictions/soh",
        json={"features": {"voltage_mean": 3.8}},
    )

    assert response.status_code == 422
    assert "current_mean" in response.json()["detail"]


def test_predict_soh_unexpected_feature_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predictions/soh",
        json={"features": {"voltage_mean": 3.8, "current_mean": -2.0, "bogus_feature": 1.0}},
    )

    assert response.status_code == 422
    assert "bogus_feature" in response.json()["detail"]


def test_predict_soh_no_model_returns_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "empty_registry"))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@localhost:5432/db")
    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/v1/predictions/soh",
            json={"features": {"voltage_mean": 3.8, "current_mean": -2.0}},
        )

    assert response.status_code == 503
    get_settings.cache_clear()
