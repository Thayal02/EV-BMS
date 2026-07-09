"""Candidate SOH regression models: pipeline construction + hyperparameter
search spaces for each.

Every model is wrapped in the same `SimpleImputer -> StandardScaler ->
estimator` shape so training/evaluation code can treat them uniformly. The
imputer is a safety net (the training dataset itself has no missing values
in the selected feature columns - see dataset.py) that matters once these
pipelines are applied to new, real-world inference data that may have gaps.
Scaling is a no-op for tree-based split decisions but is included
uniformly rather than conditionally, to keep every pipeline the same shape
and to avoid a second code path just for SVR.

Hyperparameter ranges are deliberately modest: the training split is ~27
batteries and ~2,200 discharge cycles, which is a small-data regime where
wide search spaces (e.g. thousands of trees, unbounded depth) mostly just
overfit noise rather than find genuine structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STATE = 42

# n_jobs=-1 (use all logical cores) causes severe thread-oversubscription
# overhead for XGBoost/LightGBM on this dataset's size (measured: a single
# fit on ~1,500 rows went from ~1s at n_jobs=4 to 15s+ at n_jobs=-1 for
# XGBoost, and LightGBM did not complete within 60s at all) - the
# synchronization cost of coordinating many threads swamps the actual
# per-thread work available on a small training set. A fixed, modest worker
# count avoids this without sacrificing meaningful parallelism at this data
# scale. RandomForest/ExtraTrees (sklearn's own joblib-based parallelism)
# and CatBoost (its own thread pool) were measured unaffected, so they keep
# using every core.
N_JOBS_BOUNDED = 4


class SuggestParams(Protocol):
    def __call__(self, trial: optuna.Trial) -> dict[str, Any]: ...


class BuildEstimator(Protocol):
    def __call__(self, params: dict[str, Any]) -> Any: ...


@dataclass
class ModelSpec:
    name: str
    build_estimator: BuildEstimator
    suggest_params: SuggestParams

    def build_pipeline(self, params: dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("estimator", self.build_estimator(params)),
            ]
        )


def _random_forest_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    def build(params: dict[str, Any]) -> RandomForestRegressor:
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)

    return ModelSpec("random_forest", build, suggest)


def _extra_trees_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    def build(params: dict[str, Any]) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)

    return ModelSpec("extra_trees", build, suggest)


def _gradient_boosting_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    def build(params: dict[str, Any]) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)

    return ModelSpec("gradient_boosting", build, suggest)


def _svr_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    def build(params: dict[str, Any]) -> SVR:
        # max_iter caps worst-case fit time: an unlucky (C, gamma) combination
        # on unscaled or poorly-conditioned data can otherwise iterate for a
        # very long time chasing a tight tolerance that never quite converges.
        return SVR(max_iter=50_000, **params)

    return ModelSpec("svr", build, suggest)


def _xgboost_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def build(params: dict[str, Any]) -> XGBRegressor:
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_BOUNDED,
            verbosity=0,
            tree_method="hist",
            **params,
        )

    return ModelSpec("xgboost", build, suggest)


def _lightgbm_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }

    def build(params: dict[str, Any]) -> LGBMRegressor:
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS_BOUNDED, verbose=-1, **params)

    return ModelSpec("lightgbm", build, suggest)


def _catboost_spec() -> ModelSpec:
    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }

    def build(params: dict[str, Any]) -> CatBoostRegressor:
        return CatBoostRegressor(
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False, **params
        )

    return ModelSpec("catboost", build, suggest)


def get_model_specs() -> list[ModelSpec]:
    return [
        _random_forest_spec(),
        _extra_trees_spec(),
        _gradient_boosting_spec(),
        _svr_spec(),
        _xgboost_spec(),
        _lightgbm_spec(),
        _catboost_spec(),
    ]
