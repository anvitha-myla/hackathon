"""Process control package schemas — TCU thermal and pressure loops."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class PIDGains(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Kp: float
    Ki: float = 0.0
    Kd: float = 0.0
    output_min: float
    output_max: float


class ThermalLoopConfig(BaseModel):
    """Jacket / TCU temperature control loop."""

    model_config = ConfigDict(extra="forbid")

    loop_id: str = "TCU_TEMP"
    mode: Literal["isothermal", "isoperibolic", "ramp"] = "isothermal"
    setpoint_c: float
    jacket_setpoint_c: Optional[float] = None
    pid: PIDGains
    ua_override_W_K: Optional[PositiveFloat] = Field(
        default=None,
        description="Optional UA [W/K] override; else derived from equipment U·A.",
    )
    max_jacket_temp_c: float = 200.0
    min_jacket_temp_c: float = -30.0


class PressureLoopConfig(BaseModel):
    """Reactor headspace / H2 pressure control loop."""

    model_config = ConfigDict(extra="forbid")

    loop_id: str = "H2_PRESSURE"
    setpoint_bar: PositiveFloat
    pid: PIDGains
    max_pressure_bar: PositiveFloat
    min_pressure_bar: float = Field(0.0, ge=0)


class InterlockConfig(BaseModel):
    """Hard interlocks tied to reaction safety limits / equipment envelopes."""

    model_config = ConfigDict(extra="forbid")

    max_reactor_temp_c: float
    max_rha_mole_fraction: float = Field(..., ge=0, le=1)
    max_pressure_bar: PositiveFloat
    trip_on_interlock: bool = True


class ControlPackage(BaseModel):
    """Process control logic package for a batch digital-twin run."""

    model_config = ConfigDict(extra="forbid")

    control_id: str
    name: str
    thermal_loop: ThermalLoopConfig
    pressure_loop: PressureLoopConfig
    interlocks: InterlockConfig
    description: Optional[str] = None
