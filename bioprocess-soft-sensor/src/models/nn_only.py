"""Thin NN-only biomass predictor (Prompt 6 training was not executed).

Provides the same SmallMLP backbone as the residual network and a predict
function the evaluation runner can call if weights exist. Does not train,
does not download IndPenSim, and does not invent biomass trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import PROJECT_ROOT
from src.models.mlp import LOCKED_HIDDEN, SmallMLP

OUTPUT_NAME = "biomass"


class BiomassMLP(nn.Module):
    """Standalone MLP: features → biomass X. One network, not phase-specific."""

    n_networks = 1
    phase_specific = False
    output_name = OUTPUT_NAME

    def __init__(self, n_in: int, hidden: tuple[int, int, int] = LOCKED_HIDDEN) -> None:
        super().__init__()
        self.backbone = SmallMLP(n_in=n_in, n_out=1, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def default_artifact_dir() -> Path:
    return PROJECT_ROOT / "models" / "nn_only"


def artifacts_exist(artifact_dir: str | Path | None = None) -> bool:
    path = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir()
    weights = path / "nn_only_weights.pt"
    scaler = path / "scaler.joblib"
    return weights.is_file() and scaler.is_file()


def prompt6_status() -> dict[str, Any]:
    """Document whether a trained NN-only artifact is available."""
    path = default_artifact_dir()
    present = artifacts_exist(path)
    return {
        "prompt": 6,
        "implemented": "stub_interface_only",
        "trained_weights_present": present,
        "artifact_dir": str(path),
        "note": (
            "Prompt 6 NN-only training was never executed in this workspace. "
            "The evaluation runner will skip live IndPenSim scoring until "
            "trained NN-only (and residual/hybrid) artifacts exist."
        ),
    }


@torch.no_grad()
def predict_biomass(
    model: BiomassMLP,
    scaler: Any,
    features: np.ndarray,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cpu")
    model.eval()
    x = scaler.transform(np.asarray(features, dtype=np.float64))
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    pred = model(tensor).cpu().numpy()
    return np.asarray(pred, dtype=np.float64)


def load_training_metadata(artifact_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(artifact_dir) if artifact_dir is not None else default_artifact_dir()
    meta = path / "training_metadata.json"
    if not meta.is_file():
        return {}
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("NN-only training_metadata.json must be a mapping")
    return payload
