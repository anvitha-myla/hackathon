"""Equipment package schemas for the ARIP Digital Twin.

Defines Pydantic v2 models for stirred-tank batch reactors (STBR), thermal
control units, dosing pumps, and reaction calorimetry tools using standard
process-engineering parameters (working volume, overall heat-transfer
coefficient U, heat-transfer area A, pressure ratings, and thermal limits).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EquipmentType(str, Enum):
    """Supported equipment package kinds."""

    STBR = "stbr"
    THERMAL_CONTROL = "thermal_control"
    DOSING_PUMP = "dosing_pump"
    CALORIMETRY = "calorimetry"


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
    """Jacket / coil heat-transfer characterisation."""

    model_config = ConfigDict(extra="forbid")

    U_W_m2K: float = Field(
        ...,
        gt=0,
        description="Overall heat-transfer coefficient U [W/(m²·K)].",
    )
    A_m2: float = Field(
        ...,
        gt=0,
        description="Effective heat-transfer area A [m²].",
    )
    side: Literal["jacket", "coil", "external_exchanger", "internal"] = Field(
        default="jacket",
        description="Heat-transfer surface location.",
    )


class EquipmentBase(BaseModel):
    """Common metadata shared by all equipment packages."""

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
    thermal_limits: ThermalLimits = Field(
        ...,
        description="Allowable equipment temperature window [°C].",
    )
    description: Optional[str] = Field(default=None, description="Free-text notes.")


class STBRReactor(EquipmentBase):
    """Stirred-tank batch reactor (STBR) package.

    Captures geometry, agitation, and jacket heat-transfer parameters needed
    for digital-twin mass/energy balances.
    """

    equipment_type: Literal[EquipmentType.STBR] = EquipmentType.STBR

    working_volume_L: float = Field(
        ...,
        gt=0,
        description="Nominal working (liquid) volume [L].",
    )
    total_volume_L: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total geometric vessel volume [L]. Defaults to working volume if omitted.",
    )
    vessel_inner_diameter_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Internal vessel diameter [m].",
    )
    fill_height_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Liquid fill height at working volume [m].",
    )
    heat_transfer: HeatTransferSpec = Field(
        ...,
        description="Jacket/coil heat-transfer coefficient U and area A.",
    )
    max_agitation_rpm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum impeller speed [rpm].",
    )
    impeller_type: Optional[str] = Field(
        default=None,
        description="Impeller style (e.g. pitched-blade, Rushton, anchor).",
    )
    power_number: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dimensionless impeller power number Np [-].",
    )
    baffled: bool = Field(default=True, description="Whether the vessel is baffled.")

    @model_validator(mode="after")
    def _default_total_volume(self) -> STBRReactor:
        if self.total_volume_L is None:
            object.__setattr__(self, "total_volume_L", self.working_volume_L)
        elif self.total_volume_L < self.working_volume_L:
            raise ValueError("total_volume_L must be >= working_volume_L")
        return self


class ThermalControlUnit(EquipmentBase):
    """External thermal control unit (TCU) / circulator package."""

    equipment_type: Literal[EquipmentType.THERMAL_CONTROL] = EquipmentType.THERMAL_CONTROL

    heating_capacity_kW: float = Field(..., ge=0, description="Heating duty capacity [kW].")
    cooling_capacity_kW: float = Field(..., ge=0, description="Cooling duty capacity [kW].")
    heat_transfer_fluid: str = Field(
        ...,
        min_length=1,
        description="Circulating heat-transfer fluid (e.g. silicone oil, Syltherm, water/glycol).",
    )
    max_flow_rate_L_min: float = Field(
        ...,
        gt=0,
        description="Maximum circulating flow rate [L/min].",
    )
    setpoint_resolution_c: float = Field(
        default=0.1,
        gt=0,
        description="Smallest controllable temperature setpoint increment [°C].",
    )
    heat_transfer: Optional[HeatTransferSpec] = Field(
        default=None,
        description="Optional U·A characterisation of the TCU exchanger.",
    )


class DosingPump(EquipmentBase):
    """Metering / dosing pump package for reagent addition."""

    equipment_type: Literal[EquipmentType.DOSING_PUMP] = EquipmentType.DOSING_PUMP

    min_flow_rate_mL_min: float = Field(..., ge=0, description="Minimum controllable flow [mL/min].")
    max_flow_rate_mL_min: float = Field(..., gt=0, description="Maximum controllable flow [mL/min].")
    stroke_volume_uL: Optional[float] = Field(
        default=None,
        gt=0,
        description="Displaced volume per stroke for pulse pumps [µL].",
    )
    flow_accuracy_pct: float = Field(
        default=1.0,
        ge=0,
        description="Flow metering accuracy as percent of setpoint [%].",
    )
    wetted_materials: list[str] = Field(
        default_factory=list,
        description="Wetted-path materials of construction.",
    )
    pulse_free: bool = Field(
        default=False,
        description="True if the pump delivers continuous (pulse-free) flow.",
    )

    @model_validator(mode="after")
    def _check_flow_window(self) -> DosingPump:
        if self.max_flow_rate_mL_min <= self.min_flow_rate_mL_min:
            raise ValueError("max_flow_rate_mL_min must be greater than min_flow_rate_mL_min")
        return self


class CalorimetryTool(EquipmentBase):
    """Reaction calorimetry instrument package (RC1e / EasyMax HFCal style)."""

    equipment_type: Literal[EquipmentType.CALORIMETRY] = EquipmentType.CALORIMETRY

    working_volume_L: float = Field(
        ...,
        gt=0,
        description="Calorimeter reaction-vessel working volume [L].",
    )
    heat_detection_limit_W: float = Field(
        ...,
        gt=0,
        description="Minimum detectable heat-flow signal [W].",
    )
    sensitivity_mW: float = Field(
        ...,
        gt=0,
        description="Instrument heat-flow sensitivity [mW].",
    )
    baseline_stability_mW: float = Field(
        ...,
        ge=0,
        description="Long-term baseline drift / noise floor [mW].",
    )
    sampling_rate_Hz: float = Field(
        default=1.0,
        gt=0,
        description="Heat-flow data acquisition rate [Hz].",
    )
    heat_transfer: Optional[HeatTransferSpec] = Field(
        default=None,
        description="Optional vessel U·A used for heat-balance calibration.",
    )
    supports_isothermal: bool = Field(default=True, description="Supports isothermal mode.")
    supports_adiabatic: bool = Field(default=False, description="Supports adiabatic / ARC-like mode.")


EquipmentPackage = Annotated[
    Union[STBRReactor, ThermalControlUnit, DosingPump, CalorimetryTool],
    Field(discriminator="equipment_type"),
]


def parse_equipment_package(data: dict) -> STBRReactor | ThermalControlUnit | DosingPump | CalorimetryTool:
    """Validate a raw dict into the appropriate equipment schema via type discrimination."""
    eq_type = data.get("equipment_type")
    mapping = {
        EquipmentType.STBR.value: STBRReactor,
        EquipmentType.THERMAL_CONTROL.value: ThermalControlUnit,
        EquipmentType.DOSING_PUMP.value: DosingPump,
        EquipmentType.CALORIMETRY.value: CalorimetryTool,
        # Allow enum members if already parsed
        EquipmentType.STBR: STBRReactor,
        EquipmentType.THERMAL_CONTROL: ThermalControlUnit,
        EquipmentType.DOSING_PUMP: DosingPump,
        EquipmentType.CALORIMETRY: CalorimetryTool,
    }
    if eq_type not in mapping:
        raise ValueError(
            f"Unknown or missing equipment_type={eq_type!r}; "
            f"expected one of {[e.value for e in EquipmentType]}"
        )
    return mapping[eq_type].model_validate(data)
