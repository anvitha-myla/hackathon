"""Reduced-order Monod / Luedeking-Piret ODEs.

Deliberately simpler than IndPenSim. DO is an exogenous measurement
(no kLa / OUR oxygen mass-balance from the full simulator).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.mechanistic.config import KineticParameters

# State index: X, S, V, P
IDX_X = 0
IDX_S = 1
IDX_V = 2
IDX_P = 3
N_STATES = 4


def specific_growth_rate(
    S: float,
    DO: float,
    params: KineticParameters,
) -> float:
    """mu(S, DO) = mu_max * S/(K_s+S) * DO/(K_DO+DO)."""
    s = max(float(S), 0.0)
    do = max(float(DO), 0.0)
    return float(
        params.mu_max * (s / (params.K_s + s)) * (do / (params.K_DO + do))
    )


def _volume(V: float, params: KineticParameters) -> float:
    return max(float(V), params.V_min)


def instantaneous_rates(
    y: np.ndarray,
    params: KineticParameters,
    *,
    F: float,
    DO: float,
    S_f: float | None = None,
    hold_volume: bool = False,
    enable_product: bool = True,
) -> dict[str, float]:
    """Mechanistic rates at the current state (not IndPenSim internals)."""
    X = float(y[IDX_X])
    S = float(y[IDX_S])
    V = _volume(float(y[IDX_V]), params)
    P = float(y[IDX_P])
    Sf = params.S_f if S_f is None else float(S_f)
    F = float(F)
    mu = specific_growth_rate(S, DO, params)
    dilution = F / V
    growth = mu * X
    decay = params.k_d * X
    dilution_X = dilution * X
    feed_contrib = (F * Sf) / V
    consumption = (mu / params.Y_xs) * X if params.Y_xs != 0.0 else 0.0
    maintenance = params.m_s * X
    dilution_S = dilution * S
    product_formation = (params.alpha * mu + params.beta) * X if enable_product else 0.0
    dilution_P = dilution * P if enable_product else 0.0
    dVdt = 0.0 if hold_volume else F
    dXdt = growth - decay - dilution_X
    dSdt = feed_contrib - consumption - maintenance - dilution_S
    dPdt = product_formation - dilution_P if enable_product else 0.0
    return {
        "mu": mu,
        "growth": growth,
        "decay": decay,
        "dilution_X": dilution_X,
        "feed_contribution": feed_contrib,
        "consumption": consumption,
        "maintenance": maintenance,
        "dilution_S": dilution_S,
        "product_formation": product_formation,
        "dilution_P": dilution_P,
        "dXdt": dXdt,
        "dSdt": dSdt,
        "dVdt": dVdt,
        "dPdt": dPdt,
        "F": F,
        "DO": float(DO),
        "S_f": Sf,
        "X": X,
        "S": S,
        "V": V,
        "P": P,
    }


def rhs(
    t: float,
    y: np.ndarray,
    params: KineticParameters,
    inputs: Mapping[str, float],
) -> np.ndarray:
    """dy/dt for y = [X, S, V, P]. ``t`` unused (piecewise-constant inputs)."""
    del t
    rates = instantaneous_rates(
        y,
        params,
        F=float(inputs["F"]),
        DO=float(inputs["DO"]),
        S_f=float(inputs["S_f"]) if "S_f" in inputs else None,
        hold_volume=bool(inputs.get("hold_volume", 0.0)),
        enable_product=bool(inputs.get("enable_product", 1.0)),
    )
    out = np.empty(N_STATES, dtype=float)
    out[IDX_X] = rates["dXdt"]
    out[IDX_S] = rates["dSdt"]
    out[IDX_V] = rates["dVdt"]
    out[IDX_P] = rates["dPdt"]
    return out


def pack_state(X: float, S: float, V: float, P: float = 0.0) -> np.ndarray:
    return np.array([X, S, V, P], dtype=float)
