"""Pydantic mirror of shared/schemas/battery_metadata.schema.json.

Keep in sync with that file manually - see shared/README.md.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Chemistry(StrEnum):
    NMC = "NMC"
    NCA = "NCA"
    LFP = "LFP"
    LMO = "LMO"
    LTO = "LTO"
    OTHER = "Other"


class FormFactor(StrEnum):
    CYLINDRICAL = "cylindrical"
    POUCH = "pouch"
    PRISMATIC = "prismatic"


class TemperatureRange(BaseModel):
    min: float
    max: float


class BatteryMetadata(BaseModel):
    battery_id: str = Field(..., min_length=1)
    manufacturer: str | None = None
    model_name: str | None = None
    nominal_capacity_kwh: float = Field(
        ...,
        gt=0,
        description=(
            "Rated pack energy capacity in kWh. Intentionally unconstrained - "
            "any pack size is accepted, not just the reference sizes used "
            "during model development."
        ),
    )
    nominal_voltage_v: float = Field(..., gt=0)
    chemistry: Chemistry
    pack_configuration: str | None = None
    cell_count: int | None = Field(default=None, ge=1)
    form_factor: FormFactor | None = None
    manufacture_year: int | None = Field(default=None, ge=1990)
    cycle_count_at_upload: int | None = Field(default=None, ge=0)
    operating_temperature_range_c: TemperatureRange | None = None
    notes: str | None = None
