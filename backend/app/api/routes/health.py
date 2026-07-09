"""Liveness/readiness endpoints used by Docker healthchecks and monitoring."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def liveness() -> dict[str, str]:
    """Process-level liveness check - no external dependencies."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(db: Session = Depends(get_db)) -> dict[str, str]:
    """Readiness check - verifies the database connection is usable."""
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}
