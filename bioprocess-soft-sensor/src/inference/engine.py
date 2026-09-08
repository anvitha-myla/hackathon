"""Hybrid inference helpers.

Prompt 8 sequential engine: ``src.inference.pipeline.SequentialHybridEngine``.
Prompt 9 one-timestamp combine lives here when ``ood`` is available:

    X_hybrid = X_mechanistic + beta_trust * delta_X_pred
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.inference.pipeline import HybridInferenceEngine, SequentialHybridEngine
from src.inference.result import HybridInferenceResult, attach_reference_for_evaluation
from src.inference.result import combine_hybrid as combine_hybrid_tuple
from src.inference.state import InferenceState
from src.inference.trust import MahalanobisTrustHook, PlaceholderTrustHook, TrustAssessment

try:
    from numpy.typing import ArrayLike

    from src.inference.ood import HybridCombine, MahalanobisOOD, OODScore, apply_hybrid
except ImportError:  # Prompt 9 module not present yet
    ArrayLike = Any  # type: ignore[misc,assignment]
    HybridCombine = None  # type: ignore[misc,assignment]
    MahalanobisOOD = None  # type: ignore[misc,assignment]
    OODScore = None  # type: ignore[misc,assignment]
    apply_hybrid = None  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class HybridStepResult:
    """One causal timestamp: physics + attenuated residual + OOD log fields."""

    batch_id: Any
    timestamp: Any
    X_mechanistic: float
    delta_X_raw: float
    beta_trust: float
    delta_X_applied: float
    X_hybrid: float
    d_m: float
    d_threshold: float
    ood_state: str
    fallback_state: str
    physics_contribution: float
    ml_correction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "X_mechanistic": self.X_mechanistic,
            "delta_X_raw": self.delta_X_raw,
            "beta_trust": self.beta_trust,
            "delta_X_applied": self.delta_X_applied,
            "X_hybrid": self.X_hybrid,
            "d_m": self.d_m,
            "d_threshold": self.d_threshold,
            "ood_state": self.ood_state,
            "fallback_state": self.fallback_state,
            "physics_contribution": self.physics_contribution,
            "ml_correction": self.ml_correction,
        }


def hybrid_step(
    *,
    x_mechanistic: float,
    delta_x_pred: float,
    features_t: ArrayLike,
    ood: Any,
    batch_id: Any = None,
    timestamp: Any = None,
) -> HybridStepResult:
    """Apply OOD beta_trust and the hybrid equation at a single timestamp."""
    if apply_hybrid is None:
        raise ImportError("src.inference.ood is required for hybrid_step")
    combine, score = ood.hybrid_at(
        x_mechanistic=x_mechanistic,
        delta_x_pred=delta_x_pred,
        phi=features_t,
    )
    return HybridStepResult(
        batch_id=batch_id,
        timestamp=timestamp,
        X_mechanistic=combine.X_mechanistic,
        delta_X_raw=combine.delta_X_raw,
        beta_trust=combine.beta_trust,
        delta_X_applied=combine.delta_X_applied,
        X_hybrid=combine.X_hybrid,
        d_m=score.d_m,
        d_threshold=score.d_threshold,
        ood_state=score.ood_state,
        fallback_state=score.fallback_state,
        physics_contribution=combine.physics_contribution,
        ml_correction=combine.ml_correction,
    )


def combine_hybrid(
    x_mechanistic: float,
    delta_x_pred: float,
    beta_trust: float = 1.0,
) -> tuple[float, float]:
    """X_hybrid = X_mechanistic + beta_trust * delta_X_pred.

    Returns ``(X_hybrid, delta_X_applied)``. Prompt 9 ``apply_hybrid`` returns
    a ``HybridCombine`` dataclass with the same equation.
    """
    return combine_hybrid_tuple(x_mechanistic, delta_x_pred, beta_trust)


__all__ = [
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "HybridStepResult",
    "InferenceState",
    "MahalanobisOOD",
    "MahalanobisTrustHook",
    "PlaceholderTrustHook",
    "SequentialHybridEngine",
    "TrustAssessment",
    "attach_reference_for_evaluation",
    "combine_hybrid",
    "hybrid_step",
]
