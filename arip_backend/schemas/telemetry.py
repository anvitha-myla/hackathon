"""Industrial telemetry schemas — Aspen / LabPlot / SCADA aesthetic."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class MetricValue(BaseModel):
    """Scientific metric with explicit engineering units for dense UI tables."""

    model_config = ConfigDict(extra="forbid")

    tag: str = Field(..., description="Historian / SCADA tag name")
    value: float
    unit: str = Field(..., description="e.g. °C, bar, mol/L, W/m²K, RPM, kg/min")
    display_name: str
    precision: int = Field(2, ge=0, le=8)
    quality: Literal["GOOD", "UNCERTAIN", "BAD", "SUBSTITUTE"] = "GOOD"
    category: Literal[
        "thermal",
        "pressure",
        "composition",
        "kinetics",
        "mechanical",
        "energy",
        "quality",
        "status",
    ] = "status"


class TwinStepRequest(BaseModel):
    """Inputs for a single digital-twin time step."""

    model_config = ConfigDict(extra="forbid")

    dt_s: float = Field(1.0, gt=0.0, le=60.0, description="Integration step [s]")
    t_s: Optional[float] = Field(
        default=None,
        description="Absolute batch time [s]; if omitted, runtime advances from last step",
    )
    sensors: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional live sensors: T_reactor, P_headspace, MFC_H2_rate",
    )
    agitator_rpm: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    T_jacket_c: Optional[float] = Field(default=None)
    P_sp_bar: Optional[float] = Field(default=None, gt=0.0)
    h2_valve: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stage: Optional[int] = Field(default=None, ge=1, le=5)
    include_ai_advisory: bool = Field(
        True,
        description="Call LocalAIExplainer (falls back to template if Ollama offline)",
    )
    sensor_noise: bool = Field(
        False,
        description="If no sensors provided, synthesize noisy measurements from truth",
    )


class StreamControlMessage(BaseModel):
    """Client → server WebSocket control."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["start", "pause", "reset", "configure"] = "start"
    dt_s: float = Field(0.1, gt=0.0, le=5.0, description="Stream physics step [s]")
    hz: float = Field(10.0, gt=0.0, le=50.0, description="Telemetry publish rate")
    include_ai_advisory: bool = False
    agitator_rpm: Optional[float] = None


class UnifiedTwinFrame(BaseModel):
    """Single SCADA / Aspen-style telemetry frame after a twin step."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "arip.twin.v1"
    batch_id: str
    t_s: float
    t_min: float
    stage: str
    stage_index: int
    safety_status: Literal["NOMINAL", "WARNING", "CRITICAL"]
    scada_badge: Literal["RUN", "WARN", "TRIP", "HOLD", "IDLE"]

    # Dense metric tables (frontend renders Aspen-like grids)
    metrics: list[MetricValue]

    # Module outputs
    physics_state: dict[str, float]
    refined_state: dict[str, float]
    ekf: dict[str, Any]
    decision: dict[str, Any]
    residual: dict[str, Any]
    ai_advisory: Optional[dict[str, Any]] = None

    # Plot-friendly series point (LabPlot / trend)
    trend_point: dict[str, float] = Field(
        default_factory=dict,
        description="Compact {tag: value} for live scientific plots",
    )
    units_map: dict[str, str] = Field(
        default_factory=dict,
        description="tag → unit lookup for frontend axis labels",
    )
    pipeline_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage wall time for diagnostics",
    )
