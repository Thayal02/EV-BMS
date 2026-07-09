"""Group-aware hyperparameter tuning for a single SOH model spec.

Cross-validation uses `GroupKFold` keyed on `battery_id` rather than a plain
`KFold`: consecutive discharge cycles from the same battery are highly
autocorrelated (SOH barely moves cycle-to-cycle), so a plain random fold
would put near-duplicate rows in both the training and validation side of a
fold and report an optimistic score that doesn't reflect how the model
performs on a battery it has never seen - exactly the generalization the
final held-out test set (see train_soh_models.py) is also protecting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.training.soh.model_specs import ModelSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    best_params: dict
    cv_mae_mean: float
    cv_mae_std: float


def _cross_val_mae(
    spec: ModelSpec,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> np.ndarray:
    splitter = GroupKFold(n_splits=n_splits)
    fold_scores = []
    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        pipeline = spec.build_pipeline(params)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = pipeline.predict(X.iloc[val_idx])
        mae = float(np.mean(np.abs(y.iloc[val_idx].to_numpy() - predictions)))
        fold_scores.append(mae)
    return np.array(fold_scores)


def tune_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_trials: int = 25,
    n_splits: int = 5,
    seed: int = 42,
) -> TuningResult:
    """Run an Optuna study minimizing mean grouped-CV MAE for one model spec."""

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)
        scores = _cross_val_mae(spec, params, X, y, groups, n_splits)
        return float(scores.mean())

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_scores = _cross_val_mae(spec, study.best_params, X, y, groups, n_splits)
    return TuningResult(
        best_params=study.best_params,
        cv_mae_mean=float(best_scores.mean()),
        cv_mae_std=float(best_scores.std()),
    )
