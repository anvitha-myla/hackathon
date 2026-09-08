"""Causal feature engineering.

Source of truth: ``src.features.feature_engine`` and ``src.features.registry``.
This module wraps those APIs so older import paths keep working.
"""

from src.features.feature_engine import FeatureEngine, engineer_batch
from src.features.registry import FEATURE_REGISTRY, unimplemented_due_to_missing_columns


def engineer(frame, engine: FeatureEngine | None = None):
    """Evaluate the causal feature registry on a batch (or concatenated batches)."""
    return engineer_batch(frame, engine=engine)


__all__ = [
    "FEATURE_REGISTRY",
    "FeatureEngine",
    "engineer",
    "engineer_batch",
    "unimplemented_due_to_missing_columns",
]
