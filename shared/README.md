# Shared Contracts

Canonical, language-agnostic schemas that both `backend/` (Python/Pydantic)
and `frontend/` (TypeScript) must conform to. This is the mechanism that
keeps battery metadata (and later, prediction/report contracts) from
drifting between the two codebases or from being hardcoded independently.

## Battery metadata

`schemas/battery_metadata.schema.json` is the source of truth for what a
"battery" is in this system. Notably it does **not** enumerate supported
pack sizes (24/30/40/... kWh) - `nominal_capacity_kwh` is an open numeric
field, so new pack sizes never require a schema, backend, or frontend
change. Chemistry is a small enum with an `"Other"` escape hatch for the
same reason.

- Backend: `backend/app/schemas/battery.py` hand-mirrors this schema as a
  Pydantic model. Keep the two in sync manually until a codegen step
  (`datamodel-code-generator`) is introduced.
- Frontend: a matching TypeScript type will be added under
  `frontend/src/types/` when the upload feature is built.

## Adding a new shared contract

1. Add the JSON Schema file here.
2. Mirror it as a Pydantic model in `backend/app/schemas/`.
3. Mirror it as a TypeScript type in `frontend/src/types/`.
4. Reference this README so the mapping stays discoverable.
