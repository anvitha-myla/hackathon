"""EKF sensor-fusion output schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

N_STATE = 6


class FusedStateEstimate(BaseModel):
    """Posterior EKF estimate after a predict+update cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fused_state: dict[str, float] = Field(
        ...,
        description="Posterior state estimate x̂_k keyed by EKF state names",
    )
    sensor_residuals: dict[str, float] = Field(
        ...,
        description="Innovation ν = z − h(x̂⁻) for each measurement channel",
    )
    covariance_matrix: list[list[float]] = Field(
        ...,
        description="Posterior error covariance P_k (6×6)",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="0–100% confidence from Tr(P_k)",
    )
    timestamp_s: float = Field(0.0, description="Filter time stamp [s]")
    kalman_gain: Optional[list[list[float]]] = Field(
        default=None,
        description="Optional Kalman gain K_k (6×3) for diagnostics",
    )

    @field_validator("covariance_matrix")
    @classmethod
    def _check_cov_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != N_STATE or any(len(row) != N_STATE for row in v):
            raise ValueError(f"covariance_matrix must be {N_STATE}×{N_STATE}")
        return v
