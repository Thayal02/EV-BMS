from __future__ import annotations

import numpy as np
import pytest

from src.training.soh.evaluate import compute_regression_metrics


def test_compute_regression_metrics_perfect_prediction() -> None:
    y_true = np.array([90.0, 80.0, 70.0, 60.0])

    metrics = compute_regression_metrics(y_true, y_true.copy())

    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mape == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)


def test_compute_regression_metrics_known_values() -> None:
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 110.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics.mae == pytest.approx(10.0)
    assert metrics.rmse == pytest.approx(10.0)
    assert metrics.mape == pytest.approx(10.0)


def test_compute_regression_metrics_guards_against_zero_division() -> None:
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 90.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    # MAPE should only average over the nonzero-true-value row, not raise.
    assert metrics.mape == pytest.approx(10.0)
