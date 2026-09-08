"""Prompt 0 smoke tests: packages import; configs match locked architecture."""

from __future__ import annotations

import src
import src.data
import src.evaluation
import src.features
import src.inference
import src.mechanistic
import src.models
import src.monitoring
from src.config import load_default


def test_src_version() -> None:
    assert getattr(src, "__version__", None)


def test_locked_split_and_target() -> None:
    cfg = load_default()
    assert cfg["target"]["primary"] == "biomass"
    assert cfg["split"]["train_batches"] == 60
    assert cfg["split"]["validation_batches"] == 20
    assert cfg["split"]["test_batches"] == 20
    assert cfg["split"]["unit"] == "complete_batch"


def test_locked_mlp_and_ood() -> None:
    cfg = load_default()
    assert cfg["model"]["residual_mlp"]["hidden"] == [64, 32, 16]
    assert cfg["model"]["residual_mlp"]["activation"] == "relu"
    assert cfg["ood"]["method"] == "mahalanobis"
    assert cfg["mechanistic"]["use_full_indpensim_equations"] is False
    assert cfg["ui"]["tabs"] == [
        "data_cleaning",
        "feature_engineering",
        "live_inference",
        "computational_monitor",
    ]
