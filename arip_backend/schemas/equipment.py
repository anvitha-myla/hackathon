"""Equipment package schemas for the ARIP Digital Twin.

Pydantic v2 models covering Modules 1–6: reactors, thermal control,
dosing/flow, catalyst/solids handling, purification, and calorimetry.
Standard process-engineering parameters include working volume, overall
heat-transfer coefficient U, heat-transfer area A, MAWP, and thermal limits.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EquipmentType(str, Enum):
    """Supported equipment package kinds (Modules 1–6)."""

    # Module 1 — reactors
    STBR = "stbr"
    PBR = "pbr"
    CSTR_PFR = "cstr_pfr"
    REACTOR_SKID = "reactor_skid"
    HPOX = "hpox"
    # Module 2 — thermal / control
    THERMAL_CONTROL = "thermal_control"
    SCADA_DCS = "scada_dcs"
    # Module 3 — dosing / flow
    DOSING_PUMP = "dosing_pump"
    MFC = "mfc"
    # Module 4 — catalyst / solids
    ANF = "anf"
    PNF = "pnf"
    # Module 5 — purification
    ATFE = "atfe"
    WFE = "wfe"
    DISTILLATION = "distillation"
    VLE = "vle"
    RCVD = "rcvd"
    # Module 6 — calorimetry
    CALORIMETRY = "calorimetry"
    DSC = "dsc"
    ARC = "arc"


class ThermalLimits(BaseModel):
    """Allowable temperature window for safe equipment operation."""

    model_config = ConfigDict(extra="forbid")

    t_min_c: float = Field(..., description="Minimum allowable operating temperature [°C].")
    t_max_c: float = Field(..., description="Maximum allowable operating temperature [°C].")

    @model_validator(mode="after")
    def _check_window(self) -> ThermalLimits:
        if self.t_max_c <= self.t_min_c:
            raise ValueError("t_max_c must be greater than t_min_c")
        return self


class HeatTransferSpec(BaseModel):
    """Jacket / coil / exchanger heat-transfer characterisation."""

    model_config = ConfigDict(extra="forbid")

    U_W_m2K: float = Field(..., gt=0, description="Overall heat-transfer coefficient U [W/(m²·K)].")
    A_m2: float = Field(..., gt=0, description="Effective heat-transfer area A [m²].")
    side: Literal["jacket", "coil", "external_exchanger", "internal", "evaporator", "condenser"] = Field(
        default="jacket",
        description="Heat-transfer surface location.",
    )


class EquipmentBase(BaseModel):
    """Common metadata shared by process hardware packages."""

    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(..., min_length=1, description="Unique equipment package ID.")
    name: str = Field(..., min_length=1, description="Human-readable equipment name.")
    manufacturer: Optional[str] = Field(default=None, description="OEM / vendor.")
    model_number: Optional[str] = Field(default=None, description="Vendor model designation.")
    material_of_construction: Optional[str] = Field(
        default=None,
        description="Wetted-parts material (e.g. SS316L, Hastelloy C-276).",
    )
    max_operating_pressure_barg: float = Field(
        ...,
        ge=0,
        description="Maximum allowable working pressure (MAWP) [barg].",
    )
    thermal_limits: ThermalLimits = Field(..., description="Allowable equipment temperature window [°C].")
    description: Optional[str] = Field(default=None, description="Free-text notes.")
    tags: list[str] = Field(default_factory=list, description="Search / classification tags.")


# ---------------------------------------------------------------------------
# Module 1 — Reactors
# ---------------------------------------------------------------------------


class STBRReactor(EquipmentBase):
    """Stirred-tank batch reactor (STBR)."""

    equipment_type: Literal[EquipmentType.STBR] = EquipmentType.STBR
    working_volume_L: float = Field(..., gt=0, description="Nominal working (liquid) volume [L].")
    total_volume_L: Optional[float] = Field(default=None, gt=0, description="Total geometric vessel volume [L].")
    vessel_inner_diameter_m: Optional[float] = Field(default=None, gt=0)
    fill_height_m: Optional[float] = Field(default=None, gt=0)
    heat_transfer: HeatTransferSpec
    max_agitation_rpm: Optional[float] = Field(default=None, gt=0)
    impeller_type: Optional[str] = None
    power_number: Optional[float] = Field(default=None, ge=0)
    baffled: bool = True

    @model_validator(mode="after")
    def _default_total_volume(self) -> STBRReactor:
        if self.total_volume_L is None:
            object.__setattr__(self, "total_volume_L", self.working_volume_L)
        elif self.total_volume_L < self.working_volume_L:
            raise ValueError("total_volume_L must be >= working_volume_L")
        return self


class PackedBedReactor(EquipmentBase):
    """Packed-bed / fixed-bed catalytic reactor (PBR)."""

    equipment_type: Literal[EquipmentType.PBR] = EquipmentType.PBR
    bed_volume_L: float = Field(..., gt=0, description="Catalyst bed volume [L].")
    bed_diameter_m: float = Field(..., gt=0, description="Internal bed diameter [m].")
    bed_length_m: float = Field(..., gt=0, description="Packed-bed length [m].")
    void_fraction: float = Field(..., gt=0, lt=1, description="Bed void fraction ε [-].")
    max_catalyst_mass_kg: float = Field(..., gt=0, description="Maximum catalyst loading [kg].")
    heat_transfer: HeatTransferSpec = Field(..., description="Wall / tube-side U·A.")
    tube_count: Optional[int] = Field(default=None, ge=1, description="Multi-tubular count if applicable.")
    flow_orientation: Literal["upflow", "downflow", "horizontal"] = "downflow"


class CSTRPFRSystem(EquipmentBase):
    """Combined CSTR + PFR reactor train."""

    equipment_type: Literal[EquipmentType.CSTR_PFR] = EquipmentType.CSTR_PFR
    cstr_working_volume_L: float = Field(..., gt=0, description="CSTR working volume [L].")
    pfr_volume_L: float = Field(..., gt=0, description="PFR geometric volume [L].")
    pfr_length_m: float = Field(..., gt=0, description="PFR tube length [m].")
    pfr_inner_diameter_m: float = Field(..., gt=0, description="PFR inner diameter [m].")
    heat_transfer: HeatTransferSpec
    max_agitation_rpm: Optional[float] = Field(default=None, gt=0)
    max_throughput_L_h: float = Field(..., gt=0, description="Design liquid throughput [L/h].")


class ReactorSkid(EquipmentBase):
    """Integrated mini / pilot reactor skid."""

    equipment_type: Literal[EquipmentType.REACTOR_SKID] = EquipmentType.REACTOR_SKID
    working_volume_L: float = Field(..., gt=0, description="Primary reactor working volume [L].")
    heat_transfer: HeatTransferSpec
    has_dosing: bool = True
    has_gas_feed: bool = True
    has_condenser: bool = True
    max_agitation_rpm: Optional[float] = Field(default=None, gt=0)
    utilities: list[str] = Field(default_factory=list, description="Required utilities (N2, vacuum, etc.).")


class HPOXReactor(EquipmentBase):
    """High-pressure oxidation (HPOX) reactor."""

    equipment_type: Literal[EquipmentType.HPOX] = EquipmentType.HPOX
    working_volume_L: float = Field(..., gt=0, description="Working volume [L].")
    heat_transfer: HeatTransferSpec
    oxidant: str = Field(default="O2", description="Primary oxidant species.")
    max_oxygen_partial_pressure_barg: float = Field(..., ge=0)
    max_agitation_rpm: Optional[float] = Field(default=None, gt=0)
    lining: Optional[str] = Field(default=None, description="Corrosion lining / cladding.")


# ---------------------------------------------------------------------------
# Module 2 — Thermal control & automation
# ---------------------------------------------------------------------------


class ThermalControlUnit(EquipmentBase):
    """External thermal control unit (TCU) / circulator."""

    equipment_type: Literal[EquipmentType.THERMAL_CONTROL] = EquipmentType.THERMAL_CONTROL
    heating_capacity_kW: float = Field(..., ge=0)
    cooling_capacity_kW: float = Field(..., ge=0)
    heat_transfer_fluid: str = Field(..., min_length=1)
    max_flow_rate_L_min: float = Field(..., gt=0)
    setpoint_resolution_c: float = Field(default=0.1, gt=0)
    single_fluid: bool = Field(default=True, description="True for single-fluid (SF) TCU architecture.")
    heat_transfer: Optional[HeatTransferSpec] = None


class SCADADCSSystem(BaseModel):
    """SCADA / DCS supervisory control package (soft equipment)."""

    model_config = ConfigDict(extra="forbid")

    equipment_type: Literal[EquipmentType.SCADA_DCS] = EquipmentType.SCADA_DCS
    equipment_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    protocol: str = Field(..., min_length=1, description="Primary industrial protocol (OPC UA, Modbus TCP, …).")
    max_io_points: int = Field(..., gt=0, description="Licensed / configured I/O point capacity.")
    historian_enabled: bool = True
    alarm_classes: list[str] = Field(default_factory=lambda: ["advisory", "warning", "critical"])
    scan_rate_Hz: float = Field(default=1.0, gt=0, description="Base control scan rate [Hz].")
    redundant: bool = False
    # Soft packages still carry envelope metadata for registry uniformity
    max_operating_pressure_barg: float = Field(default=0.0, ge=0)
    thermal_limits: ThermalLimits = Field(
        default_factory=lambda: ThermalLimits(t_min_c=-40.0, t_max_c=60.0),
        description="Cabinet / electronics ambient limits [°C].",
    )


# ---------------------------------------------------------------------------
# Module 3 — Dosing & flow
# ---------------------------------------------------------------------------


class DosingPump(EquipmentBase):
    """Metering / dosing pump for liquid reagent addition."""

    equipment_type: Literal[EquipmentType.DOSING_PUMP] = EquipmentType.DOSING_PUMP
    min_flow_rate_mL_min: float = Field(..., ge=0)
    max_flow_rate_mL_min: float = Field(..., gt=0)
    stroke_volume_uL: Optional[float] = Field(default=None, gt=0)
    flow_accuracy_pct: float = Field(default=1.0, ge=0)
    wetted_materials: list[str] = Field(default_factory=list)
    pulse_free: bool = False

    @model_validator(mode="after")
    def _check_flow_window(self) -> DosingPump:
        if self.max_flow_rate_mL_min <= self.min_flow_rate_mL_min:
            raise ValueError("max_flow_rate_mL_min must be greater than min_flow_rate_mL_min")
        return self


class MassFlowController(EquipmentBase):
    """Gas mass-flow controller (MFC)."""

    equipment_type: Literal[EquipmentType.MFC] = EquipmentType.MFC
    gas_species: str = Field(..., min_length=1, description="Calibrated gas (e.g. H2, N2, air).")
    min_flow_sccm: float = Field(..., ge=0, description="Minimum controllable flow [sccm].")
    max_flow_sccm: float = Field(..., gt=0, description="Full-scale flow [sccm].")
    accuracy_pct_FS: float = Field(default=1.0, ge=0, description="Accuracy as % of full scale.")
    inlet_pressure_max_barg: float = Field(..., ge=0)
    control_signal: Literal["0-5V", "0-10V", "4-20mA", "EtherCAT", "Modbus"] = "0-5V"

    @model_validator(mode="after")
    def _check_flow(self) -> MassFlowController:
        if self.max_flow_sccm <= self.min_flow_sccm:
            raise ValueError("max_flow_sccm must be greater than min_flow_sccm")
        return self


# ---------------------------------------------------------------------------
# Module 4 — Catalyst / solids handling
# ---------------------------------------------------------------------------


class AgitatedNutscheFilter(EquipmentBase):
    """Agitated Nutsche filter (ANF) for solid–liquid separation."""

    equipment_type: Literal[EquipmentType.ANF] = EquipmentType.ANF
    working_volume_L: float = Field(..., gt=0)
    filter_area_m2: float = Field(..., gt=0, description="Filtration area [m²].")
    cake_volume_L: float = Field(..., gt=0)
    heat_transfer: Optional[HeatTransferSpec] = None
    max_agitation_rpm: Optional[float] = Field(default=None, gt=0)
    vacuum_capable: bool = True
    min_absolute_pressure_mbar: Optional[float] = Field(default=None, gt=0)


class PressureNutscheFilter(EquipmentBase):
    """Pressure Nutsche filter (PNF)."""

    equipment_type: Literal[EquipmentType.PNF] = EquipmentType.PNF
    working_volume_L_min: float = Field(..., gt=0, description="Minimum working volume [L].")
    working_volume_L_max: float = Field(..., gt=0, description="Maximum working volume [L].")
    filter_area_m2: float = Field(..., gt=0)
    heat_transfer: Optional[HeatTransferSpec] = None
    vacuum_capable: bool = True

    @model_validator(mode="after")
    def _check_volume(self) -> PressureNutscheFilter:
        if self.working_volume_L_max < self.working_volume_L_min:
            raise ValueError("working_volume_L_max must be >= working_volume_L_min")
        return self


# ---------------------------------------------------------------------------
# Module 5 — Purification
# ---------------------------------------------------------------------------


class ThinFilmEvaporator(EquipmentBase):
    """Agitated or wiped thin-film evaporator (ATFE / WFE)."""

    equipment_type: Literal[EquipmentType.ATFE, EquipmentType.WFE]
    evaporating_area_m2: float = Field(..., gt=0, description="Evaporating surface area A [m²].")
    heat_transfer: HeatTransferSpec
    feed_rate_max_kg_h: float = Field(..., gt=0)
    vacuum_min_mbar: float = Field(..., gt=0, description="Minimum achievable absolute pressure [mbar].")
    rotor_max_rpm: Optional[float] = Field(default=None, gt=0)
    condenser_area_m2: Optional[float] = Field(default=None, gt=0)


class DistillationUnit(EquipmentBase):
    """Batch / continuous distillation column (e.g. Pilodist)."""

    equipment_type: Literal[EquipmentType.DISTILLATION] = EquipmentType.DISTILLATION
    boiler_volume_L: float = Field(..., gt=0)
    column_diameter_m: float = Field(..., gt=0)
    packed_height_m: float = Field(..., gt=0)
    theoretical_stages: float = Field(..., gt=0)
    heat_transfer: HeatTransferSpec = Field(..., description="Reboiler U·A.")
    condenser_duty_kW: float = Field(..., ge=0)
    vacuum_min_mbar: Optional[float] = Field(default=None, gt=0)


class VLEApparatus(EquipmentBase):
    """Vapor–liquid equilibrium (VLE) measurement apparatus."""

    equipment_type: Literal[EquipmentType.VLE] = EquipmentType.VLE
    working_volume_L: float = Field(..., gt=0)
    heat_transfer: Optional[HeatTransferSpec] = None
    pressure_range_min_mbar: float = Field(..., gt=0)
    pressure_range_max_barg: float = Field(..., ge=0)
    composition_analysis: list[str] = Field(
        default_factory=list,
        description="On-line / off-line analysis methods (GC, densitometry, …).",
    )


class RotaryConeVacuumDryer(EquipmentBase):
    """Rotary cone vacuum dryer (RCVD)."""

    equipment_type: Literal[EquipmentType.RCVD] = EquipmentType.RCVD
    working_volume_L: float = Field(..., gt=0)
    heat_transfer: HeatTransferSpec
    vacuum_min_mbar: float = Field(..., gt=0)
    max_rotation_rpm: float = Field(..., gt=0)
    condenser_capable: bool = True


# ---------------------------------------------------------------------------
# Module 6 — Calorimetry
# ---------------------------------------------------------------------------


class CalorimetryTool(EquipmentBase):
    """Reaction calorimeter (e.g. Mettler RC1e)."""

    equipment_type: Literal[EquipmentType.CALORIMETRY] = EquipmentType.CALORIMETRY
    working_volume_L: float = Field(..., gt=0)
    heat_detection_limit_W: float = Field(..., gt=0)
    sensitivity_mW: float = Field(..., gt=0)
    baseline_stability_mW: float = Field(..., ge=0)
    sampling_rate_Hz: float = Field(default=1.0, gt=0)
    heat_transfer: Optional[HeatTransferSpec] = None
    supports_isothermal: bool = True
    supports_adiabatic: bool = False


class DSCInstrument(EquipmentBase):
    """Differential scanning calorimeter (DSC)."""

    equipment_type: Literal[EquipmentType.DSC] = EquipmentType.DSC
    temperature_ramp_max_K_min: float = Field(..., gt=0)
    heat_flow_range_mW: float = Field(..., gt=0)
    sensitivity_uW: float = Field(..., gt=0)
    sample_mass_max_mg: float = Field(..., gt=0)
    atmosphere: list[str] = Field(default_factory=lambda: ["N2", "air"])
    # DSC cells are small; working volume reported as crucible volume in µL via optional field
    crucible_volume_uL: Optional[float] = Field(default=None, gt=0)


class ARCInstrument(EquipmentBase):
    """Accelerating rate calorimeter (ARC)."""

    equipment_type: Literal[EquipmentType.ARC] = EquipmentType.ARC
    bomb_volume_mL: float = Field(..., gt=0, description="Sample bomb / bomb volume [mL].")
    heat_detection_threshold_C_min: float = Field(
        ...,
        gt=0,
        description="Self-heat-rate detection threshold [°C/min].",
    )
    phi_factor_min: float = Field(..., gt=0, description="Minimum thermal inertia (φ) factor [-].")
    max_operating_temperature_c: float = Field(..., description="Instrument T_max [°C].")
    supports_adiabatic: bool = True
    supports_isothermal: bool = False


EquipmentModel = Union[
    STBRReactor,
    PackedBedReactor,
    CSTRPFRSystem,
    ReactorSkid,
    HPOXReactor,
    ThermalControlUnit,
    SCADADCSSystem,
    DosingPump,
    MassFlowController,
    AgitatedNutscheFilter,
    PressureNutscheFilter,
    ThinFilmEvaporator,
    DistillationUnit,
    VLEApparatus,
    RotaryConeVacuumDryer,
    CalorimetryTool,
    DSCInstrument,
    ARCInstrument,
]

EquipmentPackage = Annotated[EquipmentModel, Field(discriminator="equipment_type")]

_TYPE_MAP: dict[Any, type[BaseModel]] = {
    EquipmentType.STBR.value: STBRReactor,
    EquipmentType.PBR.value: PackedBedReactor,
    EquipmentType.CSTR_PFR.value: CSTRPFRSystem,
    EquipmentType.REACTOR_SKID.value: ReactorSkid,
    EquipmentType.HPOX.value: HPOXReactor,
    EquipmentType.THERMAL_CONTROL.value: ThermalControlUnit,
    EquipmentType.SCADA_DCS.value: SCADADCSSystem,
    EquipmentType.DOSING_PUMP.value: DosingPump,
    EquipmentType.MFC.value: MassFlowController,
    EquipmentType.ANF.value: AgitatedNutscheFilter,
    EquipmentType.PNF.value: PressureNutscheFilter,
    EquipmentType.ATFE.value: ThinFilmEvaporator,
    EquipmentType.WFE.value: ThinFilmEvaporator,
    EquipmentType.DISTILLATION.value: DistillationUnit,
    EquipmentType.VLE.value: VLEApparatus,
    EquipmentType.RCVD.value: RotaryConeVacuumDryer,
    EquipmentType.CALORIMETRY.value: CalorimetryTool,
    EquipmentType.DSC.value: DSCInstrument,
    EquipmentType.ARC.value: ARCInstrument,
}


def parse_equipment_package(data: dict) -> EquipmentModel:
    """Validate a raw dict into the appropriate equipment schema via type discrimination."""
    eq_type = data.get("equipment_type")
    # Normalise enum members to their value
    if isinstance(eq_type, EquipmentType):
        eq_type = eq_type.value
    model_cls = _TYPE_MAP.get(eq_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown or missing equipment_type={eq_type!r}; "
            f"expected one of {sorted({e.value for e in EquipmentType})}"
        )
    return model_cls.model_validate(data)  # type: ignore[return-value]
