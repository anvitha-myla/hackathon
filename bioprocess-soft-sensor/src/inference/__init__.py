"""Hybrid inference engine (Prompt 8)."""

from src.inference.pipeline import HybridInferenceEngine, SequentialHybridEngine
from src.inference.result import HybridInferenceResult, attach_reference_for_evaluation, combine_hybrid
from src.inference.state import InferenceState
from src.inference.trust import MahalanobisTrustHook, PlaceholderTrustHook, TrustAssessment

try:
    from src.inference.ood import MahalanobisOOD, apply_hybrid, beta_trust
except ImportError:
    MahalanobisOOD = None  # type: ignore[misc,assignment]
    apply_hybrid = None  # type: ignore[misc,assignment]
    beta_trust = None  # type: ignore[misc,assignment]

__version__ = "0.0.0"

__all__ = [
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "InferenceState",
    "MahalanobisOOD",
    "MahalanobisTrustHook",
    "PlaceholderTrustHook",
    "SequentialHybridEngine",
    "TrustAssessment",
    "apply_hybrid",
    "attach_reference_for_evaluation",
    "beta_trust",
    "combine_hybrid",
]
