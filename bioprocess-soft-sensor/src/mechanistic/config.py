"""Load reduced-order mechanistic parameters from configs/mechanistic.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import load_yaml

DEFAULT_CONFIG_NAME = "mechanistic.yaml"

FORBIDDEN_LIVE_KEYS_DEFAULT = frozenset(
    {
        "X_reference",
        "x_reference",
        "biomass_reference",
        "reference_biomass",
        "Penicillin_biomass",
        "biomass",
    }
)


@dataclass(frozen=True)
class SolverSettings:
    method: str = "BDF"
    fallback_method: str = "Radau"
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float | None = None
    timeout_s: float = 1.0


@dataclass(frozen=True)
class InitialConditions:
    """Documented start state. Must not be taken from future reference biomass."""

    X: float = 0.15
    S: float = 15.0
    P: float = 0.0
    V: float = 100.0
    t: float = 0.0


@dataclass(frozen=True)
class KineticParameters:
    mu_max: float = 0.12
    K_s: float = 0.5
    K_DO: float = 2.0
    k_d: float = 0.01
    Y_xs: float = 0.45
    m_s: float = 0.03
    S_f: float = 500.0
    alpha: float = 0.05
    beta: float = 0.008
    V_min: float = 1e-6


@dataclass(frozen=True)
class StabilitySettings:
    clamp_negative: bool = True
    max_X: float = 1e6
    max_S: float = 1e6
    max_P: float = 1e6
    max_V: float = 1e6
    negative_tol: float = 1e-12


@dataclass(frozen=True)
class MechanisticConfig:
    family: str = "reduced_order_monod"
    use_full_indpensim_equations: bool = False
    primary_estimate: str = "biomass"
    enable_product: bool = True
    initialize_from_reference_biomass: bool = False
    solver: SolverSettings = field(default_factory=SolverSettings)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    parameters: KineticParameters = field(default_factory=KineticParameters)
    stability: StabilitySettings = field(default_factory=StabilitySettings)
    forbidden_live_keys: frozenset[str] = FORBIDDEN_LIVE_KEYS_DEFAULT

    def __post_init__(self) -> None:
        if self.use_full_indpensim_equations:
            raise ValueError(
                "Full IndPenSim equations are forbidden (circularity). "
                "Keep use_full_indpensim_equations: false."
            )
        if self.initialize_from_reference_biomass:
            raise ValueError(
                "Live model must not initialize from reference biomass "
                "(including future values)."
            )
        if self.primary_estimate != "biomass":
            raise ValueError("Primary live estimate must be biomass X.")


def _mapping(raw: dict[str, Any] | None) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def parse_mechanistic_config(raw: dict[str, Any]) -> MechanisticConfig:
    states = _mapping(raw.get("states"))
    enable_product = bool(states.get("product", True))
    solver_raw = _mapping(raw.get("solver"))
    ic_raw = _mapping(raw.get("initial_conditions"))
    par_raw = _mapping(raw.get("parameters"))
    stab_raw = _mapping(raw.get("stability"))
    keys = raw.get("forbidden_live_keys")
    forbidden = (
        frozenset(str(k) for k in keys) if keys is not None else FORBIDDEN_LIVE_KEYS_DEFAULT
    )
    max_step = solver_raw.get("max_step", None)
    if max_step is not None:
        max_step = float(max_step)
    return MechanisticConfig(
        family=str(raw.get("family", "reduced_order_monod")),
        use_full_indpensim_equations=bool(raw.get("use_full_indpensim_equations", False)),
        primary_estimate=str(raw.get("primary_estimate", "biomass")),
        enable_product=enable_product,
        initialize_from_reference_biomass=bool(
            raw.get("initialize_from_reference_biomass", False)
        ),
        solver=SolverSettings(
            method=str(solver_raw.get("method", "BDF")),
            fallback_method=str(solver_raw.get("fallback_method", "Radau")),
            rtol=float(solver_raw.get("rtol", 1e-6)),
            atol=float(solver_raw.get("atol", 1e-8)),
            max_step=max_step,
            timeout_s=float(solver_raw.get("timeout_s", 1.0)),
        ),
        initial_conditions=InitialConditions(
            X=float(ic_raw.get("X", 0.15)),
            S=float(ic_raw.get("S", 15.0)),
            P=float(ic_raw.get("P", 0.0)),
            V=float(ic_raw.get("V", 100.0)),
            t=float(ic_raw.get("t", 0.0)),
        ),
        parameters=KineticParameters(
            mu_max=float(par_raw.get("mu_max", 0.12)),
            K_s=float(par_raw.get("K_s", 0.5)),
            K_DO=float(par_raw.get("K_DO", 2.0)),
            k_d=float(par_raw.get("k_d", 0.01)),
            Y_xs=float(par_raw.get("Y_xs", 0.45)),
            m_s=float(par_raw.get("m_s", 0.03)),
            S_f=float(par_raw.get("S_f", 500.0)),
            alpha=float(par_raw.get("alpha", 0.05)),
            beta=float(par_raw.get("beta", 0.008)),
            V_min=float(par_raw.get("V_min", 1e-6)),
        ),
        stability=StabilitySettings(
            clamp_negative=bool(stab_raw.get("clamp_negative", True)),
            max_X=float(stab_raw.get("max_X", 1e6)),
            max_S=float(stab_raw.get("max_S", 1e6)),
            max_P=float(stab_raw.get("max_P", 1e6)),
            max_V=float(stab_raw.get("max_V", 1e6)),
            negative_tol=float(stab_raw.get("negative_tol", 1e-12)),
        ),
        forbidden_live_keys=forbidden,
    )


def load_mechanistic_config(path: str | Path | None = None) -> MechanisticConfig:
    if path is None:
        raw = load_yaml(DEFAULT_CONFIG_NAME)
    else:
        raw = load_yaml(path)
    return parse_mechanistic_config(raw)
