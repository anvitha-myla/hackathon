"""Sequential reduced-order mechanistic biomass model.

Primary live output is X_mechanistic. Does not consume reference biomass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.mechanistic.config import MechanisticConfig, load_mechanistic_config
from src.mechanistic.equations import (
    IDX_P,
    IDX_S,
    IDX_V,
    IDX_X,
    instantaneous_rates,
    pack_state,
    rhs,
    specific_growth_rate,
)
from src.mechanistic.solver import integrate_step


@dataclass
class MechanisticStepResult:
    """One causal update. Primary estimate is X_mechanistic."""

    X_mechanistic: float
    S: float
    V: float
    P: float | None
    t: float
    rates: dict[str, float]
    solver_status: str
    solver_latency: float
    solver_message: str
    method_used: str
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "X_mechanistic": self.X_mechanistic,
            "S": self.S,
            "V": self.V,
            "P": self.P,
            "t": self.t,
            "rates": dict(self.rates),
            "solver_status": self.solver_status,
            "solver_latency": self.solver_latency,
            "solver_message": self.solver_message,
            "method_used": self.method_used,
            "success": self.success,
        }


def assert_no_reference_biomass(
    payload: Mapping[str, Any],
    forbidden: frozenset[str],
) -> None:
    """Live prediction must never receive reference biomass."""
    hits = [k for k in payload if k in forbidden]
    if hits:
        raise ValueError(
            "Reference / live-forbidden biomass keys in mechanistic inputs: "
            + ", ".join(sorted(hits))
        )


class ReducedMechanisticModel:
    """Piecewise-constant input, sequential solve_ivp (BDF/Radau)."""

    def __init__(self, config: MechanisticConfig | None = None) -> None:
        self.config = config if config is not None else load_mechanistic_config()
        self._t = float(self.config.initial_conditions.t)
        ic = self.config.initial_conditions
        self._y = pack_state(ic.X, ic.S, ic.V, ic.P)
        self._last: MechanisticStepResult | None = None

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> ReducedMechanisticModel:
        return cls(load_mechanistic_config(path))

    def reset(
        self,
        *,
        X: float | None = None,
        S: float | None = None,
        V: float | None = None,
        P: float | None = None,
        t: float | None = None,
        reference_biomass: float | None = None,
    ) -> None:
        if reference_biomass is not None:
            raise ValueError(
                "Never initialize the live model from reference biomass "
                "(including future values)."
            )
        ic = self.config.initial_conditions
        self._y = pack_state(
            ic.X if X is None else float(X),
            ic.S if S is None else float(S),
            ic.V if V is None else float(V),
            ic.P if P is None else float(P),
        )
        self._t = ic.t if t is None else float(t)
        self._last = None

    @property
    def t(self) -> float:
        return self._t

    @property
    def state_vector(self) -> np.ndarray:
        return self._y.copy()

    @property
    def X_mechanistic(self) -> float:
        return float(self._y[IDX_X])

    @property
    def last_result(self) -> MechanisticStepResult | None:
        return self._last

    def _apply_stability(self, y: np.ndarray, status: str) -> tuple[np.ndarray, str]:
        stab = self.config.stability
        y = np.asarray(y, dtype=float).copy()
        flags: list[str] = []
        if status not in ("success", ""):
            flags.extend(part for part in status.split("+") if part and part != "success")
        if not np.all(np.isfinite(y)):
            if "nan" not in flags:
                flags.append("nan")
            return y, "+".join(flags) if flags else "nan"
        if np.any(y < -abs(stab.negative_tol)):
            if "negative_state" not in flags:
                flags.append("negative_state")
            if stab.clamp_negative:
                y = np.maximum(y, 0.0)
        caps = np.array(
            [stab.max_X, stab.max_S, stab.max_V, stab.max_P], dtype=float
        )
        if np.any(y > caps):
            if "divergence" not in flags:
                flags.append("divergence")
            y = np.minimum(y, caps)
        if not flags:
            return y, "success"
        return y, "+".join(flags)

    def step(
        self,
        dt: float,
        observables: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> MechanisticStepResult:
        """Advance one sample using only current/past observables.

        Required live inputs: F (feed), DO (dissolved oxygen).
        Optional: V (measured volume), S_f (feed concentration).
        """
        payload: dict[str, Any] = {}
        if observables is not None:
            payload.update(dict(observables))
        payload.update(kwargs)
        assert_no_reference_biomass(payload, self.config.forbidden_live_keys)

        if "F" not in payload or "DO" not in payload:
            raise TypeError("step requires F (feed rate) and DO (dissolved oxygen)")

        F = float(payload["F"])
        DO = float(payload["DO"])
        S_f = float(payload["S_f"]) if "S_f" in payload else self.config.parameters.S_f
        hold_volume = "V" in payload and payload["V"] is not None
        if hold_volume:
            self._y[IDX_V] = float(payload["V"])

        inputs = {
            "F": F,
            "DO": DO,
            "S_f": S_f,
            "hold_volume": 1.0 if hold_volume else 0.0,
            "enable_product": 1.0 if self.config.enable_product else 0.0,
        }

        def fun(t: float, y: np.ndarray) -> np.ndarray:
            return rhs(t, y, self.config.parameters, inputs)

        sol_cfg = self.config.solver
        stab = self.config.stability
        t0 = self._t
        outcome = integrate_step(
            fun,
            t0,
            self._y,
            float(dt),
            method=sol_cfg.method,
            fallback_method=sol_cfg.fallback_method,
            rtol=sol_cfg.rtol,
            atol=sol_cfg.atol,
            max_step=sol_cfg.max_step,
            timeout_s=sol_cfg.timeout_s,
            negative_tol=stab.negative_tol,
            maxima=(stab.max_X, stab.max_S, stab.max_V, stab.max_P),
        )
        y, status = self._apply_stability(outcome.y, outcome.status)
        success = outcome.success and status == "success"
        self._y = y
        dt_f = float(dt)
        self._t = t0 + dt_f if dt_f >= 0.0 else t0
        rates = instantaneous_rates(
            self._y,
            self.config.parameters,
            F=F,
            DO=DO,
            S_f=S_f,
            hold_volume=hold_volume,
            enable_product=self.config.enable_product,
        )
        rates["mu"] = specific_growth_rate(
            float(self._y[IDX_S]), DO, self.config.parameters
        )
        result = MechanisticStepResult(
            X_mechanistic=float(self._y[IDX_X]),
            S=float(self._y[IDX_S]),
            V=float(self._y[IDX_V]),
            P=float(self._y[IDX_P]) if self.config.enable_product else None,
            t=float(self._t),
            rates=rates,
            solver_status=status,
            solver_latency=float(outcome.latency_s),
            solver_message=outcome.message,
            method_used=outcome.method_used,
            success=success,
        )
        self._last = result
        return result

    def run_sequential(
        self,
        dt: float | Sequence[float],
        steps: Sequence[Mapping[str, Any]],
    ) -> list[MechanisticStepResult]:
        """Causal loop: step(t), step(t+1), ... without future trajectories."""
        out: list[MechanisticStepResult] = []
        for i, obs in enumerate(steps):
            dti = float(dt[i]) if not isinstance(dt, (int, float)) else float(dt)
            out.append(self.step(dti, obs))
        return out
