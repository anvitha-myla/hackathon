"""Compatibility shim. Spec source of truth: ``src.inference.ood`` (Prompt 9)."""

from src.inference.ood import (
    FALLBACK_ATTENUATED,
    FALLBACK_NONE,
    FALLBACK_PHYSICS,
    HybridCombine,
    MahalanobisOOD,
    OODScore,
    apply_hybrid,
    beta_trust,
    beta_trust_from_distance,
    load_ood_config,
)

__all__ = [
    "FALLBACK_ATTENUATED",
    "FALLBACK_NONE",
    "FALLBACK_PHYSICS",
    "HybridCombine",
    "MahalanobisOOD",
    "OODScore",
    "apply_hybrid",
    "beta_trust",
    "beta_trust_from_distance",
    "load_ood_config",
]
