"""Causal carry between sequential timestamps.

At time t the engine may use current measurements, previous measurements,
this state, and accumulated quantities — never t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceState:
    """Persistent sequential state. One engine instance tracks one live batch."""

    batch_id: str | None = None
    last_timestamp: float | None = None
    n_steps: int = 0
    X_mechanistic: float = 0.0
    last_cleaned: dict[str, Any] = field(default_factory=dict)
    last_features: dict[str, float] = field(default_factory=dict)
    last_phase_indicators: dict[str, float] = field(default_factory=dict)
    last_delta_X_raw: float = 0.0
    last_solver_status: str = "idle"
    accumulated: dict[str, float] = field(default_factory=dict)
    cleaner_state: Any = None
    feature_state: Any = None
    phase_state: Any = None

    def reset(self, batch_id: str | None = None) -> None:
        self.batch_id = batch_id
        self.last_timestamp = None
        self.n_steps = 0
        self.X_mechanistic = 0.0
        self.last_cleaned = {}
        self.last_features = {}
        self.last_phase_indicators = {}
        self.last_delta_X_raw = 0.0
        self.last_solver_status = "idle"
        self.accumulated = {}
        self.cleaner_state = None
        self.feature_state = None
        self.phase_state = None
