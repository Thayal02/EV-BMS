"""Request/response contracts for prediction endpoints.

`features` is an open dict rather than fixed named fields deliberately: the
set of features a task's model expects lives in that model's manifest
(`feature_columns`), not in backend code, so a retrained model with a
different feature set never requires a schema change here - only a new
registry version.
"""

from pydantic import BaseModel, Field


class SohPredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=(
            "Feature name -> value. Must exactly match the registered SOH "
            "model's feature_columns (see its manifest.json)."
        ),
    )
    model_version: str | None = Field(
        default=None,
        description="Pin a specific registry version instead of using the latest.",
    )


class SohPredictionResponse(BaseModel):
    predicted_soh_percent: float
    model_task: str = "soh"
    model_version: str
    model_algorithm: str
