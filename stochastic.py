"""
Stochastic SCADA artifact layer for EQ-STBR-5000L telemetry.

- Ornstein–Uhlenbeck (AR(1)) process noise on thermal / pressure channels
- Discrete PLC quantization (0.1 °C temperatures)
- Raw-material CoA purity sampling (HPLC tolerance band 97.0–100.5 %)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

Number = Union[int, float]
ArrayLike = Sequence[Number]

COA_PURITY_MIN_PCT = 97.0
COA_PURITY_MAX_PCT = 100.5
TEMP_QUANTIZATION_C = 0.1


@dataclass
class OrnsteinUhlenbeckNoise:
    """
    Mean-reverting OU process with exact discrete AR(1) transition.

        dX = -θ X dt + σ dW
        X_{k+1} = φ X_k + σ_eff · ε ,  φ = exp(-θ Δt)
    """

    theta: float = 0.05
    sigma: float = 0.15
    x0: float = 0.0
    seed: Optional[int] = None
    _state: float = field(init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.theta < 0.0 or self.sigma < 0.0:
            raise ValueError("theta and sigma must be >= 0")
        self._state = float(self.x0)
        self._rng = random.Random(self.seed)

    def reset(self, x0: Optional[float] = None) -> None:
        self._state = float(self.x0 if x0 is None else x0)

    def step(self, dt: float = 1.0) -> float:
        if dt <= 0.0:
            return self._state
        phi = math.exp(-self.theta * dt)
        if self.theta <= 1e-12:
            sig = self.sigma * math.sqrt(dt)
        else:
            sig = self.sigma * math.sqrt(max((1.0 - phi * phi) / (2.0 * self.theta), 0.0))
        self._state = phi * self._state + sig * self._rng.gauss(0.0, 1.0)
        return self._state

    def generate(self, n: int, dt: float = 1.0) -> List[float]:
        self.reset()
        return [self.step(dt) for _ in range(n)]


def sample_coa_purity(
    rng: Optional[random.Random] = None,
    low_pct: float = COA_PURITY_MIN_PCT,
    high_pct: float = COA_PURITY_MAX_PCT,
) -> float:
    """Sample CoA purity uniformly in [97.0, 100.5] % (HPLC tolerance)."""
    r = rng if rng is not None else random
    return float(r.uniform(low_pct, high_pct))


def _quantize_value(value: Number, resolution: float) -> float:
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0")
    bins = round(float(value) / resolution)
    decimals = max(0, int(round(-math.log10(resolution)))) if resolution < 1.0 else 0
    return round(bins * resolution, decimals)


def quantize_temperature(
    values: Union[Number, Sequence[Number]],
    resolution_C: float = TEMP_QUANTIZATION_C,
) -> Union[float, List[float]]:
    """Round temperatures to industrial DCS / PLC historian resolution (0.1 °C)."""
    if isinstance(values, (int, float)):
        return _quantize_value(values, resolution_C)
    return [_quantize_value(v, resolution_C) for v in values]


def quantize_series(values: Sequence[Number], resolution: float) -> List[float]:
    return [_quantize_value(v, resolution) for v in values]


@dataclass
class ProcessNoiseConfig:
    """OU amplitudes for active thermal / pressure historian channels."""

    T_reactor_theta: float = 0.03
    T_reactor_sigma: float = 0.06  # °C
    T_jacket_theta: float = 0.05
    T_jacket_sigma: float = 0.08
    pressure_theta: float = 0.10
    pressure_sigma: float = 0.025  # bar
    torque_theta: float = 0.20
    torque_sigma: float = 1.8  # N·m
    temp_resolution_C: float = TEMP_QUANTIZATION_C
    pressure_resolution_bar: float = 0.01
    torque_resolution_Nm: float = 0.1
    seed: Optional[int] = None


def apply_process_noise(
    time_series_dict: Mapping[str, object],
    coa_purity: Optional[float] = None,
    *,
    config: Optional[ProcessNoiseConfig] = None,
    rng_seed: Optional[int] = None,
    active_mask: Optional[Sequence[bool]] = None,
) -> MutableMapping[str, object]:
    """
    Overlay OU noise + PLC quantization on clean step-machine streams.

    ``active_mask`` (optional) gates noise onto rows where thermal/pressure
    channels are physically live (False ⇒ near-zero noise, e.g. empty tare).
    """
    cfg = config or ProcessNoiseConfig()
    seed = cfg.seed if rng_seed is None else rng_seed
    rng = random.Random(seed)

    out: dict = {k: (list(v) if isinstance(v, list) else v) for k, v in time_series_dict.items()}
    purity = float(coa_purity) if coa_purity is not None else sample_coa_purity(rng)
    out["raw_material_purity_coa"] = purity
    out["coa_assay_factor"] = purity / 100.0

    def _get(key: str) -> Optional[List[float]]:
        if key not in out or not isinstance(out[key], list):
            return None
        return [float(x) for x in out[key]]

    T_rx = _get("T_reactor")
    T_jkt = _get("T_jacket")
    P = _get("pressure") or _get("headspace_pressure")
    torque = _get("agitator_torque")
    n = 0
    for series in (T_rx, T_jkt, P, torque):
        if series:
            n = max(n, len(series))
    if n == 0:
        out["process_noise_applied"] = False
        return out

    mask = list(active_mask) if active_mask is not None else [True] * n
    if len(mask) < n:
        mask = mask + [mask[-1] if mask else True] * (n - len(mask))

    def _ou(theta: float, sigma: float, offset: int) -> List[float]:
        ou = OrnsteinUhlenbeckNoise(
            theta=theta, sigma=sigma, seed=None if seed is None else seed + offset
        )
        path = ou.generate(n, dt=1.0)
        return [path[i] if mask[i] else 0.0 for i in range(n)]

    n_Tr = _ou(cfg.T_reactor_theta, cfg.T_reactor_sigma, 1)
    n_Tj = _ou(cfg.T_jacket_theta, cfg.T_jacket_sigma, 2)
    n_P = _ou(cfg.pressure_theta, cfg.pressure_sigma, 3)
    n_tau = _ou(cfg.torque_theta, cfg.torque_sigma, 4)

    if T_rx is not None:
        noisy = [T_rx[i] + n_Tr[i] for i in range(n)]
        out["T_reactor"] = quantize_temperature(noisy, cfg.temp_resolution_C)
    if T_jkt is not None:
        noisy = [T_jkt[i] + n_Tj[i] for i in range(n)]
        out["T_jacket"] = quantize_temperature(noisy, cfg.temp_resolution_C)
    if P is not None:
        key = "headspace_pressure" if "headspace_pressure" in out else "pressure"
        noisy = [max(0.0, P[i] + n_P[i]) for i in range(n)]
        out[key] = quantize_series(noisy, cfg.pressure_resolution_bar)
        if key != "pressure" and "pressure" in out:
            out["pressure"] = out[key]
    if torque is not None:
        noisy = [max(0.0, torque[i] + n_tau[i]) for i in range(n)]
        # Keep exact zero when agitator commanded OFF
        out["agitator_torque"] = [
            0.0 if torque[i] <= 1e-9 else _quantize_value(noisy[i], cfg.torque_resolution_Nm)
            for i in range(n)
        ]

    out["process_noise_applied"] = True
    return out


__all__ = [
    "OrnsteinUhlenbeckNoise",
    "ProcessNoiseConfig",
    "apply_process_noise",
    "sample_coa_purity",
    "quantize_temperature",
    "quantize_series",
    "COA_PURITY_MIN_PCT",
    "COA_PURITY_MAX_PCT",
    "TEMP_QUANTIZATION_C",
]
