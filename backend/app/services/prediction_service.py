"""Prediction business logic - independent of HTTP framing.

Validates a caller's feature dict against the *registered model's own*
expected schema (from its manifest) rather than any hardcoded feature list,
so this stays correct across retraining without a code change.
"""

from __future__ import annotations

import pandas as pd

from app.ml.registry import LoadedModel, load_model


class FeatureValidationError(ValueError):
    """Raised when the request's features don't match the model's expected schema."""


def predict_soh(features: dict[str, float], version: str | None = None) -> tuple[float, LoadedModel]:
    model = load_model("soh", version=version)
    expected = model.feature_columns

    missing = [c for c in expected if c not in features]
    unexpected = [c for c in features if c not in expected]
    if missing:
        raise FeatureValidationError(f"Missing required features: {missing}")
    if unexpected:
        raise FeatureValidationError(f"Unexpected features not used by this model: {unexpected}")

    row = pd.DataFrame([{column: features[column] for column in expected}])
    prediction = model.pipeline.predict(row)[0]
    return float(prediction), model
