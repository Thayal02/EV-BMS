"""Top-level API router - feature routers are included here as they're built.

e.g. api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
"""

from fastapi import APIRouter

from app.api.routes import health, predictions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(predictions.router)
