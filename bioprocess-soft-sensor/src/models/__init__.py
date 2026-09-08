"""NN-only and residual MLP modules.

Prompt 7 residual training is implemented. NN-only shares ``SmallMLP``.
"""

from src.models.mlp import LOCKED_HIDDEN, SmallMLP
from src.models.residual_nn import ResidualMLP, residual_target, train

__version__ = "0.0.0"

__all__ = [
    "LOCKED_HIDDEN",
    "ResidualMLP",
    "SmallMLP",
    "residual_target",
    "train",
]
