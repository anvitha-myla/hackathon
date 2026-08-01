"""Hybrid residual-correction output schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScientificFeatures(BaseModel):
    """Inputs to the residual ML model."""

    model_config = ConfigDict(extra="forbid")

    conversion: float = Field(..., description="Nitro conversion fraction X ∈ [0, 1]")
    Q_rxn_W: float = Field(..., description="Instantaneous heat release rate [W]")
    mass_transfer_ratio: float = Field(
        ...,
        description="Dimensionless 3·r / (k_L a · C_H2*)",
    )
    T_reactor_c: float = Field(..., description="Reactor operating temperature [°C]")
    agitator_rpm: float = Field(..., description="Agitator speed [RPM]")


class ResidualCorrection(BaseModel):
    """ML residual vector δ_ML."""

    delta_T_exotherm_c: float = Field(0.0, description="ΔT_exotherm correction [°C]")
    delta_C_nitro: float = Field(0.0, description="ΔC_nitro correction [kmol/m³]")
    delta_yield: float = Field(0.0, description="ΔYield correction [fraction]")


class RefinedPrediction(BaseModel):
    """Hybrid output: y_real ≈ y_physics + δ_ML."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    physics_baseline: dict[str, float] = Field(
        ...,
        description="ODE physics baseline state / KPIs",
    )
    residual: ResidualCorrection = Field(
        ...,
        description="ML residual corrections δ_ML",
    )
    refined_state: dict[str, float] = Field(
        ...,
        description="Corrected prediction y_physics + δ_ML",
    )
    is_pure_physics: bool = Field(
        ...,
        description="True when no ML model was applied (cold-start / δ_ML = 0)",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Loaded joblib model stem, if any",
    )
    features_used: Optional[dict[str, float]] = Field(
        default=None,
        description="Scientific feature vector passed to predict_residual",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
