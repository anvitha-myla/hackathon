"""Batch package schemas — recipe setpoints and initial run state."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator


class InitialConditions(BaseModel):
    """Initial liquid charge for a specific batch run."""

    model_config = ConfigDict(extra="forbid")

    initial_liquid_volume_m3: PositiveFloat
    initial_temperature_c: float
    initial_pressure_bar: PositiveFloat
    initial_concentrations_kmol_m3: dict[str, float]
    catalyst_loading_kg: PositiveFloat


class RecipeTargets(BaseModel):
    """Operating setpoints for the batch recipe."""

    model_config = ConfigDict(extra="forbid")

    target_operating_temp_c: float
    target_operating_pressure_bar: PositiveFloat
    agitator_speed_rpm: PositiveFloat
    coolant_inlet_temp_c: float


class BatchPackage(BaseModel):
    """Batch package: recipe + IC linking reaction and equipment packages."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    reaction_package_id: str
    equipment_package_id: str
    initial_conditions: InitialConditions
    recipe_targets: RecipeTargets
    notes: Optional[str] = None

    @property
    def catalyst_loading_kg_m3(self) -> float:
        return self.initial_conditions.catalyst_loading_kg / self.initial_conditions.initial_liquid_volume_m3


# ---------------------------------------------------------------------------
# Legacy batch format (pipeline demo) kept for backward compatibility
# ---------------------------------------------------------------------------


class InitialCharge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_m3: PositiveFloat
    temperature_c: float
    pressure_bar: PositiveFloat
    concentrations_kmol_m3: dict[str, float]
    catalyst_loading_kg_m3: PositiveFloat


class FeedPulse(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    species: str = "H2"
    target_pressure_bar: PositiveFloat
    max_flow_slpm: Optional[PositiveFloat] = None


class LegacyBatchPackage(BaseModel):
    """Earlier pipeline batch format (batch_nx_h2_001.json)."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    name: str
    reaction_id: str
    reactor_equipment_id: str
    tcu_equipment_id: str
    mfc_equipment_id: Optional[str] = None
    dosing_pump_equipment_id: Optional[str] = None
    initial_charge: InitialCharge
    liquid_feeds: list[FeedPulse] = Field(default_factory=list)
    gas_feed: GasFeedSpec
    batch_time_s: PositiveFloat
    sample_interval_s: PositiveFloat = 10.0
    control_package_id: str = "CTRL-NX-H2-001"
    notes: Optional[str] = None


def parse_batch_package(data: dict) -> Union[BatchPackage, LegacyBatchPackage]:
    if "reaction_package_id" in data and "initial_conditions" in data:
        return BatchPackage.model_validate(data)
    return LegacyBatchPackage.model_validate(data)
