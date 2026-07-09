# Agentic AI-Driven Intelligent Battery Management System

Final year engineering project: a decision-support system for Electric
Vehicle batteries that predicts **State of Health (SOH)**, **Remaining
Useful Life (RUL)**, and **Failure Risk** from uploaded historical battery
datasets (NASA Battery Dataset primary, CALCE optional), explains every
prediction with SHAP/LIME, and exposes the results through a multi-agent
architecture and a conversational AI battery expert.

This system works entirely from uploaded historical CSV datasets - there is
no live sensor, IoT, CAN bus, or real-time streaming integration.

## Repository layout

```
.
├── backend/    FastAPI serving layer, API, DB access, multi-agent orchestration
├── ml/         Offline training/experimentation workspace (NASA/CALCE pipelines, model registry)
├── frontend/   React + TypeScript + Tailwind dashboard
├── shared/     Canonical schemas (e.g. battery metadata) shared by backend and frontend
└── docker-compose.yml
```

See each workspace's own README for details: [backend/README.md](backend/README.md),
[ml/README.md](ml/README.md), [shared/README.md](shared/README.md).

## Architecture at a glance

- **Training/serving split**: `ml/` trains models and writes versioned
  artifacts to `ml/models/registry/<task>/<version>/`; `backend/` only reads
  from that registry at inference time. Training code never runs inside the
  API process, so retraining never requires redeploying the API.
- **Flexible battery support**: `shared/schemas/battery_metadata.schema.json`
  is the single source of truth for battery attributes. Pack capacity is an
  open numeric field (not an enum of fixed kWh sizes), so new EV pack sizes
  never require a schema or code change. Packs far outside the training
  distribution are still accepted, but predictions must carry a confidence
  score and explicit limitations.
- **Multi-agent pipeline** (`backend/app/agents/`): Data Agent → Prediction
  Agent → Diagnosis Agent → Maintenance Agent → LLM Expert Agent → Decision
  Agent, coordinated through a shared `AgentContext`.

## Getting started (local development)

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt
cp ../.env.example ../.env    # then edit .env
uvicorn app.main:app --reload
```

API docs at http://localhost:8000/docs.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard at http://localhost:5173.

### Database

```bash
docker compose up postgres -d
cd backend
alembic upgrade head
```

### Full stack via Docker Compose

```bash
docker compose up --build
```

## ML workspace

See [ml/README.md](ml/README.md) for the training workspace layout and the
model registry contract that connects it to the backend.

## Status

Project scaffold only - see the project charter for the full roadmap
(data pipeline, SOH/RUL/failure model training, explainability, agents,
dashboard features, AI battery expert chat).
