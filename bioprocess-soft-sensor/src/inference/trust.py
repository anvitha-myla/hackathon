"""OOD trust insertion for the sequential hybrid engine.

``SequentialHybridEngine.step(observation)`` calls ``TrustHook.assess``.
Prompt 9 injects ``MahalanobisTrustHook`` without changing that step API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.inference.ood import MahalanobisOOD, OODScore
from src.inference.state import InferenceState


@dataclass(frozen=True)
class TrustAssessment:
    """beta_trust ∈ [0, 1]. Prompt 9 fills ood_distance / states."""

    beta_trust: float = 1.0
    ood_distance: float = math.nan
    ood_state: str = "unscored"
    fallback_state: str = "none"
    d_threshold: float = math.nan


@runtime_checkable
class TrustHook(Protocol):
    def assess(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        delta_X_raw: float,
        state: InferenceState,
    ) -> TrustAssessment: ...


class PlaceholderTrustHook:
    """Always beta_trust=1.0. Use MahalanobisTrustHook for live OOD."""

    def assess(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        delta_X_raw: float,
        state: InferenceState,
    ) -> TrustAssessment:
        del features, X_mechanistic, delta_X_raw, state
        return TrustAssessment(beta_trust=1.0, ood_distance=math.nan)


class MahalanobisTrustHook:
    """Sequential per-timestamp Mahalanobis beta_trust (train-fit detector)."""

    def __init__(
        self,
        detector: MahalanobisOOD,
        feature_order: Sequence[str] | None = None,
    ) -> None:
        self.detector = detector
        self.feature_order = list(feature_order) if feature_order is not None else None

    def assess(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        delta_X_raw: float,
        state: InferenceState,
    ) -> TrustAssessment:
        del X_mechanistic, delta_X_raw, state
        order = self.feature_order or (
            list(self.detector.feature_names_) if self.detector.feature_names_ else None
        )
        if order is None:
            raise ValueError("MahalanobisTrustHook requires feature_order or a detector fit with feature_names")
        phi = self.detector.vector_from_features(features, order)
        score: OODScore = self.detector.score(phi)
        d_m = float(score.d_m) if score.valid and math.isfinite(score.d_m) else float("nan")
        if not math.isfinite(d_m):
            d_m = math.nan
        return TrustAssessment(
            beta_trust=float(score.beta_trust),
            ood_distance=d_m,
            ood_state=score.ood_state,
            fallback_state=score.fallback_state,
            d_threshold=float(score.d_threshold),
        )


def clamp_beta(beta: float) -> float:
    if not math.isfinite(beta):
        return 0.0
    if beta < 0.0:
        return 0.0
    if beta > 1.0:
        return 1.0
    return float(beta)
