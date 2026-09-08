"""Shared small MLP backbone: Input → 64-ReLU → 32-ReLU → 16-ReLU → output.

Used by the residual network (Prompt 7) and as a stub for NN-only (Prompt 6).
Do not replace with LSTM, Transformer, Neural ODE, or XGBoost.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

LOCKED_HIDDEN: tuple[int, int, int] = (64, 32, 16)
LOCKED_ACTIVATION = "relu"


def _assert_locked_hidden(hidden: Sequence[int]) -> tuple[int, int, int]:
    hidden_t = tuple(int(h) for h in hidden)
    if hidden_t != LOCKED_HIDDEN:
        raise ValueError(
            f"Locked residual/NN MLP hidden sizes are {LOCKED_HIDDEN}, got {hidden_t}"
        )
    return hidden_t


class SmallMLP(nn.Module):
    """Single MLP. Not three phase-specific networks."""

    n_networks = 1
    phase_specific = False

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        hidden: Sequence[int] = LOCKED_HIDDEN,
    ) -> None:
        super().__init__()
        if n_in < 1:
            raise ValueError("n_in must be >= 1")
        if n_out < 1:
            raise ValueError("n_out must be >= 1")
        h1, h2, h3 = _assert_locked_hidden(hidden)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.net = nn.Sequential(
            nn.Linear(self.n_in, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, self.n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_linear_layers(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if type(m) is nn.Linear)
