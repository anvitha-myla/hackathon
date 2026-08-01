"""Batch package schemas — initial conditions and feed recipes."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator


class InitialCharge(BaseModel):
    """Initial liquid charge composition and volume."""

    model_config = ConfigDict(extra="forbid")

    volume_m3: PositiveFloat
    temperature_c: float
    pressure_bar: PositiveFloat
    concentrations_kmol_m3: dict[str, float] = Field(
        ...,
        description="Initial liquid concentrations [kmol/m³].",
    )
    catalyst_loading_kg_m3: PositiveFloat


class FeedPulse(BaseModel):
    """Semi-batch feed pulse / continuous feed segment."""

    model_config = ConfigDict(extra="forbid")

    start_time_s: float = Field(..., ge=0)
    end_time_s: float = Field(..., gt=0)
    species: str
    molar_flow_kmol_s: float = Field(..., ge=0)
    temperature_c: Optional[float] = None

    @model_validator(mode="after")
    def _check_window(self) -> FeedPulse:
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be > start_time_s")
        return self


class GasFeedSpec(BaseModel):
    """Gas-phase H2 / inert feed pressure target profile."""

    model_config = ConfigDict(extra="forbid")

    species: str = "H2"
    target_pressure_bar: PositiveFloat
    max_flow_slpm: Optional[PositiveFloat] = None


class BatchPackage(BaseModel):
    """Batch package: IC + feed recipe linking equipment and reaction packages."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    name: str
    reaction_id: str = Field(..., description="Master reaction package ID.")
    reactor_equipment_id: str
    tcu_equipment_id: str
    mfc_equipment_id: Optional[str] = None
    dosing_pump_equipment_id: Optional[str] = None
    initial_charge: InitialCharge
    liquid_feeds: list[FeedPulse] = Field(default_factory=list)
    gas_feed: GasFeedSpec
    batch_time_s: PositiveFloat = Field(..., description="Nominal batch horizon [s].")
    sample_interval_s: PositiveFloat = Field(default=10.0, description="Output sampling interval [s].")
    control_package_id: str = "CTRL-NX-H2-001"
    notes: Optional[str] = None
