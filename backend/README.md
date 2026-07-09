# Backend

FastAPI serving layer: REST API, database access, and multi-agent
orchestration. Contains no training code - it only reads finished model
artifacts from `ml/models/registry/` (see [ml/README.md](../ml/README.md)).

## Layout

```
backend/
├── app/
│   ├── main.py          FastAPI app instance, middleware, router mounting
│   ├── core/            Settings (env-driven config) and logging setup
│   ├── db/              SQLAlchemy engine/session and declarative Base
│   ├── api/
│   │   ├── router.py    Aggregates all feature routers
│   │   └── routes/      One module per resource (health, datasets, predictions, ...)
│   ├── schemas/         Pydantic request/response models (mirrors shared/schemas/)
│   ├── services/        Business logic, independent of HTTP/DB framing
│   ├── agents/          Multi-agent pipeline (BaseAgent contract + concrete agents)
│   └── ml/              Model registry loader - reads ml/models/registry, no training
├── alembic/             Database migrations
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
cp ../.env.example ../.env
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs
Health check: http://localhost:8000/api/v1/health

## Tests

```bash
pytest
```

## Migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```
