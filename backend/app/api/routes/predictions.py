"""Prediction endpoints - thin HTTP layer over app.services.prediction_service."""

from fastapi import APIRouter, HTTPException

from app.ml.registry import ModelNotFoundError
from app.schemas.prediction import SohPredictionRequest, SohPredictionResponse
from app.services.prediction_service import FeatureValidationError, predict_soh

router = APIRouter(tags=["predictions"])


@router.post("/predictions/soh", response_model=SohPredictionResponse)
def predict_soh_endpoint(request: SohPredictionRequest) -> SohPredictionResponse:
    try:
        predicted_value, model = predict_soh(request.features, version=request.model_version)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SohPredictionResponse(
        predicted_soh_percent=predicted_value,
        model_version=model.version,
        model_algorithm=model.algorithm,
    )
