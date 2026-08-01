"""Equipment package schemas for the ARIP Digital Twin (Modules 1–6).

Pydantic v2 models aligned to the equipment_packages JSON contracts:
working volumes, heat-transfer coefficient U / area A where applicable,
pressure ratings, and thermal / vacuum operating limits.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EquipmentMeta(BaseModel):
    """Common identity fields present on every equipment package."""

    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    module: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Module 1 — Reactor Systems
# ---------------------------------------------------------------------------


class PBRDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tube_length_m: float = Field(..., gt=0)
    inner_diameter_m: float = Field(..., gt=0)
    bed_volume_m3: float = Field(..., gt=0)


class CatalystBed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    porosity: float = Field(..., gt=0, lt=1)
    pellet_diameter_m: float = Field(..., gt=0)
    max_catalyst_mass_kg: float = Field(..., gt=0)


class PBRLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    max_operating_temp_c: float
    max_gas_flow_slpm: float = Field(..., gt=0)


class PackedBedReactor(EquipmentMeta):
    """EQ-PBR-100 — fixed-bed tubular catalytic reactor."""

    reactor_type: str
    construction_material: str
    dimensions: PBRDimensions
    catalyst_bed: CatalystBed
    limits: PBRLimits


class CSTRPFRDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cstr_volume_m3: float = Field(..., gt=0)
    pfr_volume_m3: float = Field(..., gt=0)
    pfr_tube_length_m: float = Field(..., gt=0)


class CSTRPFRLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    max_operating_temp_c: float
    max_residence_time_min: float = Field(..., gt=0)


class CSTRPFRSystem(EquipmentMeta):
    """EQ-CSTR-PFR-01 — continuous cascade CSTR + PFR suite."""

    reactor_type: str
    construction_material: str
    dimensions: CSTRPFRDimensions
    limits: CSTRPFRLimits


class SkidDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_volume_m3: float = Field(..., gt=0)
    working_volume_m3: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _check_volumes(self) -> SkidDimensions:
        if self.working_volume_m3 > self.total_volume_m3:
            raise ValueError("working_volume_m3 must be <= total_volume_m3")
        return self


class PressureTempLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    max_operating_temp_c: float


class ReactorSkid(EquipmentMeta):
    """EQ-SKID-MINI-01 — multipurpose miniplant skid."""

    reactor_type: str
    construction_material: str
    dimensions: SkidDimensions
    limits: PressureTempLimits


class STBRDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_volume_m3: float = Field(..., gt=0)
    working_volume_m3: float = Field(..., gt=0)
    inner_diameter_m: Optional[float] = Field(default=None, gt=0)
    straight_side_height_m: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_volumes(self) -> STBRDimensions:
        if self.working_volume_m3 > self.total_volume_m3:
            raise ValueError("working_volume_m3 must be <= total_volume_m3")
        return self


class STBRThermalParameters(BaseModel):
    """Jacket heat-transfer characterisation (U and A)."""

    model_config = ConfigDict(extra="forbid")

    overall_heat_transfer_coeff_U_W_m2K: float = Field(..., gt=0, description="U [W/(m²·K)]")
    jacket_heat_transfer_area_m2: float = Field(..., gt=0, description="Jacket area A [m²]")
    jacket_volume_m3: Optional[float] = Field(default=None, gt=0)


class STBRAgitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impeller_type: str
    max_agitator_speed_rpm: float = Field(..., gt=0)
    power_number: Optional[float] = Field(default=None, ge=0)
    baffled: bool = True


class STBRLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    max_operating_temp_c: float
    min_operating_temp_c: float


class STBRReactor(EquipmentMeta):
    """EQ-STBR-5000L — stirred-tank batch reactor with jacket U·A."""

    reactor_type: str
    construction_material: str
    dimensions: STBRDimensions
    thermal_parameters: STBRThermalParameters
    agitation: STBRAgitation
    limits: STBRLimits


class HPOXDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_volume_m3: float = Field(..., gt=0)
    working_volume_m3: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _check_volumes(self) -> HPOXDimensions:
        if self.working_volume_m3 > self.total_volume_m3:
            raise ValueError("working_volume_m3 must be <= total_volume_m3")
        return self


class HPOXLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    max_operating_temp_c: float
    max_o2_partial_pressure_bar: float = Field(..., ge=0)


class HPOXReactor(EquipmentMeta):
    """EQ-HPOX-050 — high-pressure catalytic oxidation autoclave."""

    reactor_type: str
    construction_material: str
    dimensions: HPOXDimensions
    limits: HPOXLimits


# ---------------------------------------------------------------------------
# Module 2 — Thermal Management & Process Control
# ---------------------------------------------------------------------------


class ThermalPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heating_capacity_kw: float = Field(..., ge=0)
    cooling_capacity_kw: float = Field(..., ge=0)
    temperature_range_c: list[float] = Field(..., min_length=2, max_length=2)
    max_coolant_flow_rate_m3_h: float = Field(..., gt=0)
    pump_pressure_bar: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> ThermalPerformance:
        t_min, t_max = self.temperature_range_c
        if t_max <= t_min:
            raise ValueError("temperature_range_c[1] must be > temperature_range_c[0]")
        return self


class ThermalControlUnit(EquipmentMeta):
    """EQ-TCU-SF-01 — single-fluid temperature control unit."""

    utility_medium: str
    thermal_performance: ThermalPerformance


class IOChannels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analog_inputs: int = Field(..., ge=0)
    analog_outputs: int = Field(..., ge=0)
    digital_inputs: int = Field(..., ge=0)
    digital_outputs: int = Field(..., ge=0)


class SCADADCSSystem(EquipmentMeta):
    """EQ-SCADA-DCS-01 — distributed control & SCADA system."""

    architecture: str
    sampling_interval_ms: int = Field(..., gt=0)
    io_channels: IOChannels
    safety_interlock_latency_ms: int = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Module 3 — Dosing & Flow Regulation
# ---------------------------------------------------------------------------


class FlowRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float = Field(..., ge=0)
    max: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _check_window(self) -> FlowRange:
        if self.max <= self.min:
            raise ValueError("flow range max must be greater than min")
        return self


class MassFlowController(EquipmentMeta):
    """EQ-MFC-GAS-01 — high-pressure gas mass-flow controller."""

    gas_compatibility: list[str] = Field(..., min_length=1)
    flow_range_slpm: FlowRange
    accuracy_percentage_full_scale: float = Field(..., ge=0)
    max_inlet_pressure_bar: float = Field(..., ge=0)


class DosingPump(EquipmentMeta):
    """EQ-PUMP-DOSING-01 — diaphragm high-pressure liquid metering pump."""

    head_material: str
    flow_range_l_h: FlowRange
    max_discharge_pressure_bar: float = Field(..., ge=0)
    dosing_precision_percentage: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Module 4 — Catalyst Separation & Solids Handling
# ---------------------------------------------------------------------------


class ANFDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter_area_m2: float = Field(..., gt=0)
    total_volume_l: float = Field(..., gt=0)
    cake_capacity_l: float = Field(..., gt=0)


class ANFAgitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stroke_height_mm: float = Field(..., gt=0)
    agitator_rpm_range: list[float] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def _check_rpm(self) -> ANFAgitation:
        lo, hi = self.agitator_rpm_range
        if hi <= lo:
            raise ValueError("agitator_rpm_range[1] must be > agitator_rpm_range[0]")
        return self


class ANFLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    vacuum_rating_mbar: float = Field(..., gt=0)
    max_temp_c: float


class AgitatedNutscheFilter(EquipmentMeta):
    """EQ-ANF-3L — agitated Nutsche filter dryer."""

    construction_material: str
    dimensions: ANFDimensions
    agitation: ANFAgitation
    limits: ANFLimits


class PNFDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_range_l: list[float] = Field(..., min_length=2, max_length=2)
    filter_mesh_micron: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _check_volume(self) -> PNFDimensions:
        lo, hi = self.volume_range_l
        if hi < lo:
            raise ValueError("volume_range_l[1] must be >= volume_range_l[0]")
        return self


class PNFLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_operating_pressure_bar: float = Field(..., ge=0)
    inert_gas_purge_pressure_bar: float = Field(..., ge=0)


class PressureNutscheFilter(EquipmentMeta):
    """EQ-PNF-1TO3L — pressurized liquid–solid Nutsche filter."""

    construction_material: str
    dimensions: PNFDimensions
    limits: PNFLimits


# ---------------------------------------------------------------------------
# Module 5 — Separation & Purification
# ---------------------------------------------------------------------------


class ATFEThermalParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_jacket_temp_c: float
    overall_heat_transfer_coeff_U: float = Field(
        ...,
        gt=0,
        description="Overall heat-transfer coefficient U [W/(m²·K)].",
    )


class ATFELimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operating_vacuum_mbar: float = Field(..., gt=0)
    max_feed_rate_kg_h: float = Field(..., gt=0)


class ThinFilmEvaporatorATFE(EquipmentMeta):
    """EQ-ATFE-01 — agitated thin-film evaporator."""

    evaporator_surface_area_m2: float = Field(..., gt=0, description="Evaporator area A [m²].")
    rotor_speed_rpm: float = Field(..., gt=0)
    thermal_parameters: ATFEThermalParameters
    limits: ATFELimits


class WFELimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_vacuum_mbar: float = Field(..., gt=0)
    max_operating_temp_c: float


class ThinFilmEvaporatorWFE(EquipmentMeta):
    """EQ-WFE-01 — short-path wiped-film evaporator."""

    evaporator_surface_area_m2: float = Field(..., gt=0, description="Evaporator area A [m²].")
    internal_condenser_area_m2: float = Field(..., gt=0, description="Internal condenser area A [m²].")
    limits: WFELimits


class DistillationLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boilup_rate_l_h: float = Field(..., gt=0)
    reflux_ratio_range: list[float] = Field(..., min_length=2, max_length=2)
    max_pot_temp_c: float

    @model_validator(mode="after")
    def _check_reflux(self) -> DistillationLimits:
        lo, hi = self.reflux_ratio_range
        if hi < lo:
            raise ValueError("reflux_ratio_range[1] must be >= reflux_ratio_range[0]")
        return self


class DistillationUnit(EquipmentMeta):
    """EQ-PILODIST-DIST-01 — automated fractional distillation column."""

    column_type: str
    theoretical_plates: int = Field(..., gt=0)
    limits: DistillationLimits


class VLELimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boiler_capacity_ml: float = Field(..., gt=0)
    max_pressure_bar: float = Field(..., ge=0)
    vacuum_mbar: float = Field(..., gt=0)
    max_temp_c: float


class VLEApparatus(EquipmentMeta):
    """EQ-PILODIST-VLE-01 — vapour–liquid equilibrium unit."""

    operating_mode: str
    limits: VLELimits


class RCVDLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vacuum_mbar: float = Field(..., gt=0)
    jacket_temp_c: float


class RotaryConeVacuumDryer(EquipmentMeta):
    """EQ-RCVD-50L — rotary cone vacuum dryer."""

    total_volume_l: float = Field(..., gt=0)
    working_volume_l: float = Field(..., gt=0)
    rotation_speed_rpm: float = Field(..., gt=0)
    limits: RCVDLimits

    @model_validator(mode="after")
    def _check_volumes(self) -> RotaryConeVacuumDryer:
        if self.working_volume_l > self.total_volume_l:
            raise ValueError("working_volume_l must be <= total_volume_l")
        return self


# ---------------------------------------------------------------------------
# Module 6 — Process Safety & Calorimetry
# ---------------------------------------------------------------------------


class RC1eResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heat_flow_detection_limit_w: float = Field(..., gt=0)
    temperature_accuracy_k: float = Field(..., gt=0)


class ReactionCalorimeter(EquipmentMeta):
    """EQ-METTLER-RC1E — reaction calorimeter."""

    operating_modes: list[str] = Field(..., min_length=1)
    vessel_volume_ml: float = Field(..., gt=0)
    measurement_resolution: RC1eResolution
    calorimetric_outputs: list[str] = Field(..., min_length=1)


class DSCLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_pressure_bar: float = Field(..., ge=0)
    atmosphere: list[str] = Field(..., min_length=1)


class DSCInstrument(EquipmentMeta):
    """EQ-DSC-2500 — high-pressure differential scanning calorimeter."""

    temperature_range_c: list[float] = Field(..., min_length=2, max_length=2)
    heating_rate_k_min: list[float] = Field(..., min_length=2, max_length=2)
    limits: DSCLimits
    primary_detection: str

    @model_validator(mode="after")
    def _check_ranges(self) -> DSCInstrument:
        t_min, t_max = self.temperature_range_c
        if t_max <= t_min:
            raise ValueError("temperature_range_c[1] must be > temperature_range_c[0]")
        r_min, r_max = self.heating_rate_k_min
        if r_max <= r_min:
            raise ValueError("heating_rate_k_min[1] must be > heating_rate_k_min[0]")
        return self


class ARCInstrument(EquipmentMeta):
    """EQ-ARC-254 — accelerating rate calorimeter."""

    operating_mode: str
    thermal_sensitivity_c_min: float = Field(..., gt=0)
    bomb_materials: list[str] = Field(..., min_length=1)
    calorimetric_outputs: list[str] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Registry / parser
# ---------------------------------------------------------------------------

SCHEMA_BY_EQUIPMENT_ID: dict[str, type[BaseModel]] = {
    "EQ-PBR-100": PackedBedReactor,
    "EQ-CSTR-PFR-01": CSTRPFRSystem,
    "EQ-SKID-MINI-01": ReactorSkid,
    "EQ-STBR-5000L": STBRReactor,
    "EQ-HPOX-050": HPOXReactor,
    "EQ-TCU-SF-01": ThermalControlUnit,
    "EQ-SCADA-DCS-01": SCADADCSSystem,
    "EQ-MFC-GAS-01": MassFlowController,
    "EQ-PUMP-DOSING-01": DosingPump,
    "EQ-ANF-3L": AgitatedNutscheFilter,
    "EQ-PNF-1TO3L": PressureNutscheFilter,
    "EQ-ATFE-01": ThinFilmEvaporatorATFE,
    "EQ-WFE-01": ThinFilmEvaporatorWFE,
    "EQ-PILODIST-DIST-01": DistillationUnit,
    "EQ-PILODIST-VLE-01": VLEApparatus,
    "EQ-RCVD-50L": RotaryConeVacuumDryer,
    "EQ-METTLER-RC1E": ReactionCalorimeter,
    "EQ-DSC-2500": DSCInstrument,
    "EQ-ARC-254": ARCInstrument,
}

# Backward-compatible aliases used by package exports
CalorimetryTool = ReactionCalorimeter
EquipmentType = str  # packages now identify via equipment_id + module


def parse_equipment_package(data: dict[str, Any]) -> BaseModel:
    """Validate raw JSON against the schema registered for ``equipment_id``."""
    eq_id = data.get("equipment_id")
    if not eq_id:
        raise ValueError("equipment package missing required field 'equipment_id'")
    model_cls = SCHEMA_BY_EQUIPMENT_ID.get(eq_id)
    if model_cls is None:
        # Unknown IDs: require at least EquipmentMeta fields
        return EquipmentMeta.model_validate(
            {k: data[k] for k in ("equipment_id", "name", "module") if k in data}
            | {"equipment_id": eq_id, "name": data.get("name", eq_id), "module": data.get("module", "unknown")}
        )
    return model_cls.model_validate(data)
