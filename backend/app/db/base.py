"""Declarative base for all ORM models.

Concrete models (Battery, Dataset, Prediction, Report, ChatSession, ...) are
added alongside the features that need them, and must import this Base so
Alembic's autogenerate can discover them via app.db.base.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
