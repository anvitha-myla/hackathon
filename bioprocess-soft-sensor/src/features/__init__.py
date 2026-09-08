"""Causal feature engineering (Prompt 4)."""

from src.features.engineering import engineer, engineer_batch
from src.features.feature_engine import FeatureEngine
from src.features.registry import (
    FEATURE_REGISTRY,
    unimplemented_due_to_missing_columns,
    unimplemented_features,
)

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureEngine",
    "engineer",
    "engineer_batch",
    "unimplemented_due_to_missing_columns",
    "unimplemented_features",
]
