"""Regression metrics for SOH model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "mape": self.mape, "r2": self.r2}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """MAE, RMSE, MAPE (%), and R^2 between true and predicted SOH percentages.

    MAPE guards against division by zero: `soh_percent` is bounded away from
    zero for every physically meaningful battery reading in this dataset
    (rows are already filtered to exclude flagged capacity outliers, which
    is where near-zero readings come from - see dataset.py), but the guard
    is kept here anyway since this function has no way to enforce that
    precondition on an arbitrary caller.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100.0
        )
    else:
        mape = float("nan")

    return RegressionMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2)
