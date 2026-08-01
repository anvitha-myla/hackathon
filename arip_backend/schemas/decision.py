"""Decision-engine output schemas (PSM two-tier architecture)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SafetyStatus(str, Enum):
    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class InterlockAction(BaseModel):
    """Hard actuator overrides forced by Tier-1 interlocks (non-overridable)."""

    model_config = ConfigDict(extra="forbid")

    cooling_jacket_flow_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Forced jacket coolant flow [%]; 100 = max cooling",
    )
    h2_feed_cutoff: bool = Field(
        False,
        description="True → signal H₂ feed trip / MFC closed",
    )
    h2_mfc_valve_closed: bool = Field(
        False,
        description="True → close H₂ mass-flow control valve",
    )
    max_dT_dt_c_per_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Restrict reactor temperature rise rate [°C/min]",
    )


class DecisionOutput(BaseModel):
    """Validated two-tier decision engine result."""

    model_config = ConfigDict(extra="forbid")

    safety_status: Literal["NOMINAL", "WARNING", "CRITICAL"] = Field(
        ...,
        description="Highest active severity across Tier-1 evaluations",
    )
    active_interlocks: list[str] = Field(
        default_factory=list,
        description="Triggered Tier-1 interlock IDs / alarm tags",
    )
    optimization_recommendations: list[str] = Field(
        default_factory=list,
        description="Tier-2 soft process optimization advice",
    )
    alarms: list[str] = Field(
        default_factory=list,
        description="Human-readable alarm / warning messages",
    )
    actuator_overrides: InterlockAction = Field(
        default_factory=InterlockAction,
        description="Hard overrides that the DCS/PLC must enforce",
    )
    tier1_blocked_optimizations: bool = Field(
        False,
        description="True when CRITICAL interlocks suppress soft advice",
    )
    evaluated_at_s: float = Field(0.0, description="Process time stamp [s]")
    metadata: dict[str, Any] = Field(default_factory=dict)
