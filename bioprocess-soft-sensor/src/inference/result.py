"""Hybrid inference output for one timestamp.

Reference biomass is not a field of the live result. The evaluation layer may
attach it after prediction (see ``attach_reference_for_evaluation``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class HybridInferenceResult:
    """One causal hybrid step. Matches docs/07_HYBRID_INFERENCE_SPEC.md."""

    batch_id: str | None
    timestamp: float
    X_mechanistic: float
    delta_X_raw: float
    beta_trust: float
    delta_X_applied: float
    X_hybrid: float
    phase_indicators: dict[str, float]
    ood_distance: float
    latency: float
    solver_status: str
    physics_contribution: float = field(init=False)
    ml_correction: float = field(init=False)

    def __post_init__(self) -> None:
        self.physics_contribution = float(self.X_mechanistic)
        self.ml_correction = float(self.delta_X_applied)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def attach_reference_for_evaluation(
    result: HybridInferenceResult,
    X_reference: float,
) -> dict[str, Any]:
    """Evaluation-layer helper. Must never be called from the inference engine."""
    payload = result.as_dict()
    payload["X_reference"] = float(X_reference)
    return payload


def combine_hybrid(
    X_mechanistic: float,
    delta_X_pred: float,
    beta_trust: float = 1.0,
) -> tuple[float, float]:
    """X_hybrid = X_mechanistic + beta_trust * delta_X_pred.

    Returns (X_hybrid, delta_X_applied). ``beta_trust`` is clipped to [0, 1].
    """
    from src.inference.trust import clamp_beta

    beta = clamp_beta(float(beta_trust))
    delta_applied = beta * float(delta_X_pred)
    return float(X_mechanistic) + delta_applied, delta_applied


def result_from_parts(
    *,
    batch_id: str | None,
    timestamp: float,
    X_mechanistic: float,
    delta_X_raw: float,
    beta_trust: float,
    phase_indicators: Mapping[str, float],
    ood_distance: float,
    latency: float,
    solver_status: str,
) -> HybridInferenceResult:
    x_hybrid, delta_applied = combine_hybrid(X_mechanistic, delta_X_raw, beta_trust)
    return HybridInferenceResult(
        batch_id=batch_id,
        timestamp=float(timestamp),
        X_mechanistic=float(X_mechanistic),
        delta_X_raw=float(delta_X_raw),
        beta_trust=float(beta_trust),
        delta_X_applied=float(delta_applied),
        X_hybrid=float(x_hybrid),
        phase_indicators=dict(phase_indicators),
        ood_distance=float(ood_distance),
        latency=float(latency),
        solver_status=str(solver_status),
    )
