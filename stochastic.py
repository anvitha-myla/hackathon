"""
Stochastic process & sensor artifact layer for EQ-STBR-5000L SCADA streams.

Transforms clean deterministic ODE physics outputs into realistic noisy
historian / DCS telemetry by stacking:

1. Ornstein–Uhlenbeck (AR(1)) process noise (thermal inertia / fluctuation)
2. Cross-sensor coupling (e.g. pressure ripple → H2 mass-flow)
3. Discrete PLC quantization (0.1 °C temperature resolution)
4. Raw-material CoA purity sampling (HPLC analytical tolerance)
5. Dynamic agitator torque from conversion / viscosity around the
   180 RPM / 11.2 kW nominal load point
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from equipment import ReactorEquipmentPackage

Number = Union[int, float]
ArrayLike = Sequence[Number]
TimeSeriesDict = Mapping[str, Any]
MutableTimeSeries = dict[str, Any]

# ---------------------------------------------------------------------------
# Canonical SCADA / ODE key aliases
# ---------------------------------------------------------------------------
_T_REACTOR_KEYS = ("T_reactor", "T_rx", "T_rx_C", "temperature", "temp_reactor")
_T_JACKET_KEYS = ("T_jacket", "T_jkt", "T_jkt_C", "temp_jacket", "jacket_temp")
_PRESSURE_KEYS = ("pressure", "P", "P_H2", "P_bar", "headspace_pressure_bar")
_H2_FLOW_KEYS = (
    "h2_flow",
    "H2_flow",
    "m_dot_H2",
    "h2_flow_kg_min",
    "H2_feed_kg_min",
)
_CONVERSION_KEYS = ("conversion", "X", "X_liquid", "liquid_conversion")
_VISCOSITY_KEYS = ("viscosity", "mu", "mu_Pa_s", "viscosity_Pa_s")
_TORQUE_KEYS = ("agitator_torque", "torque", "torque_Nm", "agitator_torque_Nm")
_RPM_KEYS = ("agitator_rpm", "rpm", "N_rpm", "speed_rpm")
_TIME_KEYS = ("time", "t", "time_s", "timestamp")


def _first_present(data: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    return None


def _as_float_list(values: Iterable[Number]) -> list[float]:
    return [float(v) for v in values]


def _ensure_length(values: Sequence[Number], n: int, fill: float = 0.0) -> list[float]:
    seq = _as_float_list(values)
    if len(seq) == n:
        return seq
    if len(seq) == 0:
        return [fill] * n
    if len(seq) == 1:
        return [seq[0]] * n
    raise ValueError(f"Expected length {n}, got {len(seq)}")


# ===========================================================================
# Ornstein–Uhlenbeck / AR(1) generator
# ===========================================================================


@dataclass
class OrnsteinUhlenbeckNoise:
    """
    Mean-reverting Ornstein–Uhlenbeck process (discrete AR(1) form).

    Continuous SDE::

        dX_t = -θ X_t dt + σ dW_t

    Exact discrete transition over step ``dt``::

        X_{k+1} = φ X_k + σ_eff · ε_k ,   ε ~ N(0,1)
        φ = exp(-θ · dt)
        σ_eff = σ · sqrt((1 - φ²) / (2θ))   (θ > 0)

    which preserves the stationary variance ``σ² / (2θ)``.
    """

    theta: float = 0.05  # mean-reversion rate [1/s]; small → long thermal memory
    sigma: float = 0.15  # diffusion intensity (engineering units / sqrt(s))
    x0: float = 0.0
    seed: Optional[int] = None
    _state: float = field(init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.theta < 0.0:
            raise ValueError("theta must be >= 0")
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self._state = float(self.x0)
        self._rng = random.Random(self.seed)

    def reset(self, x0: Optional[float] = None) -> None:
        self._state = float(self.x0 if x0 is None else x0)

    def _phi(self, dt: float) -> float:
        if dt <= 0.0:
            return 1.0
        return math.exp(-self.theta * dt)

    def _sigma_eff(self, dt: float, phi: float) -> float:
        if self.sigma == 0.0:
            return 0.0
        if self.theta <= 1e-12:
            # Brownian limit: σ √dt
            return self.sigma * math.sqrt(max(dt, 0.0))
        # Exact OU increment std for stationary variance σ²/(2θ)
        return self.sigma * math.sqrt(max((1.0 - phi * phi) / (2.0 * self.theta), 0.0))

    def step(self, dt: float = 1.0) -> float:
        """Advance one step and return the new noise state."""
        phi = self._phi(dt)
        sig = self._sigma_eff(dt, phi)
        eps = self._rng.gauss(0.0, 1.0)
        self._state = phi * self._state + sig * eps
        return self._state

    def generate(self, n: int, dt: float = 1.0) -> list[float]:
        """Generate ``n`` autocorrelated samples (resets from ``x0``)."""
        self.reset()
        return [self.step(dt) for _ in range(n)]

    def generate_for_times(self, times: Sequence[Number]) -> list[float]:
        """Generate OU path on a (possibly irregular) time grid."""
        t = _as_float_list(times)
        if not t:
            return []
        self.reset()
        out = [self.step(0.0)]  # initial draw at t0
        out[0] = self._state
        for i in range(1, len(t)):
            dt = max(0.0, t[i] - t[i - 1])
            out.append(self.step(dt))
        return out


# ===========================================================================
# CoA purity & PLC quantization
# ===========================================================================

COA_PURITY_MIN_PCT = 97.0
COA_PURITY_MAX_PCT = 100.5  # HPLC analytical tolerance band
TEMP_QUANTIZATION_C = 0.1


def sample_coa_purity(
    rng: Optional[random.Random] = None,
    low_pct: float = COA_PURITY_MIN_PCT,
    high_pct: float = COA_PURITY_MAX_PCT,
) -> float:
    """
    Sample raw-material CoA purity uniformly in [97.0, 100.5] %.

    The upper bound above 100% reflects HPLC analytical testing tolerance
    commonly seen on Certificates of Analysis.
    """
    r = rng if rng is not None else random
    return float(r.uniform(low_pct, high_pct))


def _quantize_value(value: Number, resolution: float) -> float:
    """Round to the nearest quantization bin without binary float dust."""
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0")
    bins = round(float(value) / resolution)
    # Decimal places implied by resolution (0.1 → 1, 0.01 → 2, …)
    decimals = max(0, int(round(-math.log10(resolution)))) if resolution < 1.0 else 0
    return round(bins * resolution, decimals)


def quantize_temperature(
    values: Union[Number, Sequence[Number]],
    resolution_C: float = TEMP_QUANTIZATION_C,
) -> Union[float, list[float]]:
    """
    Round temperatures to industrial DCS / PLC historian resolution (0.1 °C).
    """
    if isinstance(values, (int, float)):
        return _quantize_value(values, resolution_C)
    return [_quantize_value(v, resolution_C) for v in values]


def quantize_series(
    values: Sequence[Number],
    resolution: float,
) -> list[float]:
    """Generic PLC quantization for any engineering unit."""
    return [_quantize_value(v, resolution) for v in values]


# ===========================================================================
# Dynamic agitator torque (conversion / viscosity)
# ===========================================================================


@dataclass
class AgitatorTorqueModel:
    """
    Dynamic shaft torque centered on the 180 RPM / 11.2 kW nominal load.

    τ(X, μ) = τ_nom · (N/N_nom)² · μ_ratio(X)^n · (1 + α · X) · derate

    where viscosity ratio grows mildly with liquid conversion (slurry densifies
    / product mixture thickens during nitro-aromatic hydrogenation).
    """

    equipment: ReactorEquipmentPackage = field(
        default_factory=ReactorEquipmentPackage.default
    )
    conversion_viscosity_gain: float = 0.35  # μ/μ_ref ≈ 1 + gain·X
    conversion_load_gain: float = 0.08  # additional load fraction vs X
    viscosity_exponent: Optional[float] = None

    @property
    def nominal_torque_Nm(self) -> float:
        return self.equipment.agitator_nominal_torque_Nm

    @property
    def nominal_power_kW(self) -> float:
        return self.equipment.agitator_nominal_power_kW

    @property
    def nominal_rpm(self) -> float:
        return self.equipment.agitator_speed_rpm

    def viscosity_ratio(
        self,
        conversion: float,
        viscosity_Pa_s: Optional[float] = None,
    ) -> float:
        if viscosity_Pa_s is not None:
            return max(float(viscosity_Pa_s) / self.equipment.mu_ref_Pa_s, 1e-9)
        x = min(max(float(conversion), 0.0), 1.5)
        return max(1.0 + self.conversion_viscosity_gain * x, 1e-9)

    def torque_Nm(
        self,
        conversion: float = 0.0,
        *,
        viscosity_Pa_s: Optional[float] = None,
        speed_rpm: Optional[float] = None,
        density_kg_m3: Optional[float] = None,
    ) -> float:
        eq = self.equipment
        N = eq.agitator_speed_rpm if speed_rpm is None else float(speed_rpm)
        n_exp = (
            eq.agitator_viscosity_exponent
            if self.viscosity_exponent is None
            else float(self.viscosity_exponent)
        )
        mu_ratio = self.viscosity_ratio(conversion, viscosity_Pa_s)
        x = min(max(float(conversion), 0.0), 1.5)
        rho = eq.rho_kg_m3 if density_kg_m3 is None else float(density_kg_m3)
        density_ratio = rho / eq.rho_kg_m3
        speed_ratio = N / eq.agitator_speed_rpm if eq.agitator_speed_rpm else 0.0

        return (
            eq.agitator_nominal_torque_Nm
            * density_ratio
            * (speed_ratio**2)
            * (mu_ratio**n_exp)
            * (1.0 + self.conversion_load_gain * x)
            * eq.agitator_power_derate
        )

    def power_kW(
        self,
        conversion: float = 0.0,
        *,
        viscosity_Pa_s: Optional[float] = None,
        speed_rpm: Optional[float] = None,
        density_kg_m3: Optional[float] = None,
    ) -> float:
        N = (
            self.equipment.agitator_speed_rpm
            if speed_rpm is None
            else float(speed_rpm)
        )
        tau = self.torque_Nm(
            conversion,
            viscosity_Pa_s=viscosity_Pa_s,
            speed_rpm=N,
            density_kg_m3=density_kg_m3,
        )
        omega = N * (2.0 * math.pi / 60.0)
        return (tau * omega) / 1000.0

    def series(
        self,
        conversion: Sequence[Number],
        *,
        viscosity: Optional[Sequence[Number]] = None,
        speed_rpm: Optional[Sequence[Number]] = None,
    ) -> tuple[list[float], list[float]]:
        """Return (torque_Nm, power_kW) arrays along a conversion trajectory."""
        x = _as_float_list(conversion)
        n = len(x)
        mu = (
            _ensure_length(viscosity, n)
            if viscosity is not None
            else [None] * n  # type: ignore[list-item]
        )
        rpm = (
            _ensure_length(speed_rpm, n, fill=self.nominal_rpm)
            if speed_rpm is not None
            else [self.nominal_rpm] * n
        )
        torques: list[float] = []
        powers: list[float] = []
        for i in range(n):
            torques.append(
                self.torque_Nm(
                    x[i],
                    viscosity_Pa_s=None if mu[i] is None else float(mu[i]),  # type: ignore[arg-type]
                    speed_rpm=rpm[i],
                )
            )
            powers.append(
                self.power_kW(
                    x[i],
                    viscosity_Pa_s=None if mu[i] is None else float(mu[i]),  # type: ignore[arg-type]
                    speed_rpm=rpm[i],
                )
            )
        return torques, powers


# ===========================================================================
# Cross-sensor coupling & noise configuration
# ===========================================================================


@dataclass
class ProcessNoiseConfig:
    """Tunable amplitudes / correlation times for SCADA artifact injection."""

    # OU parameters (theta [1/s], sigma [eng. unit / sqrt(s)])
    T_reactor_theta: float = 0.02
    T_reactor_sigma: float = 0.08  # °C
    T_jacket_theta: float = 0.04
    T_jacket_sigma: float = 0.12  # °C
    pressure_theta: float = 0.08
    pressure_sigma: float = 0.035  # bar
    torque_theta: float = 0.15
    torque_sigma: float = 2.5  # N·m
    h2_flow_theta: float = 0.20
    h2_flow_sigma: float = 0.04  # kg/min

    # Cross-sensor: ΔP [bar] → additive H2 flow bias [kg/min per bar]
    pressure_to_h2_flow_gain: float = 0.08
    # Weak thermal cross-talk: reactor temp noise bleeds into jacket reading
    T_reactor_to_jacket_gain: float = 0.05

    temp_resolution_C: float = TEMP_QUANTIZATION_C
    pressure_resolution_bar: float = 0.01
    torque_resolution_Nm: float = 0.1
    h2_flow_resolution_kg_min: float = 0.01

    default_dt_s: float = 1.0
    seed: Optional[int] = None


def _infer_dt(times: Optional[Sequence[Number]], default_dt: float) -> float:
    if times is None or len(times) < 2:
        return default_dt
    dts = [float(times[i] - times[i - 1]) for i in range(1, len(times))]
    positive = [d for d in dts if d > 0.0]
    if not positive:
        return default_dt
    return sum(positive) / len(positive)


def _noise_path(
    n: int,
    times: Optional[Sequence[Number]],
    theta: float,
    sigma: float,
    dt_default: float,
    seed: Optional[int],
) -> list[float]:
    ou = OrnsteinUhlenbeckNoise(theta=theta, sigma=sigma, seed=seed)
    if times is not None and len(times) == n:
        return ou.generate_for_times(times)
    return ou.generate(n, dt=dt_default)


# ===========================================================================
# Public API: apply_process_noise
# ===========================================================================


def apply_process_noise(
    time_series_dict: TimeSeriesDict,
    coa_purity: Optional[float] = None,
    *,
    config: Optional[ProcessNoiseConfig] = None,
    equipment: Optional[ReactorEquipmentPackage] = None,
    rng_seed: Optional[int] = None,
) -> MutableTimeSeries:
    """
    Map clean ODE physics streams → realistic noisy SCADA telemetry.

    Parameters
    ----------
    time_series_dict
        Deterministic ODE outputs. Recognized keys (first match wins):

        - time: ``time`` / ``t`` / ``time_s``
        - reactor temp: ``T_reactor`` / ``T_rx`` / …
        - jacket temp: ``T_jacket`` / ``T_jkt`` / …
        - pressure: ``pressure`` / ``P_H2`` / …
        - H2 flow: ``h2_flow`` / ``H2_feed_kg_min`` / …
        - conversion: ``conversion`` / ``X`` / …
        - viscosity: ``viscosity`` / ``mu_Pa_s`` / …
        - torque: ``agitator_torque`` / ``torque_Nm`` / …
        - rpm: ``agitator_rpm`` / ``rpm`` / …

    coa_purity
        Raw-material CoA purity [%]. If ``None``, sampled uniformly from
        [97.0, 100.5] to reflect HPLC analytical tolerance.

    config
        Optional ``ProcessNoiseConfig`` for OU amplitudes / quantization.

    equipment
        Optional ``ReactorEquipmentPackage`` for nominal agitator scaling.

    rng_seed
        Optional seed overriding ``config.seed`` for reproducibility.

    Returns
    -------
    dict
        Copy of the input streams with noisy / quantized SCADA fields plus
        metadata: ``raw_material_purity_coa``, ``*_noise`` residuals, and
        derived ``agitator_power_kW``.
    """
    cfg = config if config is not None else ProcessNoiseConfig()
    seed = cfg.seed if rng_seed is None else rng_seed
    rng = random.Random(seed)
    eq = equipment if equipment is not None else ReactorEquipmentPackage.default()
    torque_model = AgitatorTorqueModel(equipment=eq)

    # --- CoA purity ---------------------------------------------------------
    if coa_purity is None:
        purity = sample_coa_purity(rng)
    else:
        purity = float(coa_purity)
        if not (COA_PURITY_MIN_PCT <= purity <= COA_PURITY_MAX_PCT):
            # Allow out-of-band values but keep a soft warning flag
            pass

    out: MutableTimeSeries = {k: v for k, v in time_series_dict.items()}
    out["raw_material_purity_coa"] = purity
    # Effective nitroxylene assay factor relative to 100% label claim
    out["coa_assay_factor"] = purity / 100.0

    # --- Resolve series -----------------------------------------------------
    t_key = _first_present(out, _TIME_KEYS)
    times = _as_float_list(out[t_key]) if t_key is not None else None

    def _series_from(keys: Sequence[str]) -> tuple[Optional[str], list[float]]:
        key = _first_present(out, keys)
        if key is None:
            return None, []
        return key, _as_float_list(out[key])

    tr_key, T_rx = _series_from(_T_REACTOR_KEYS)
    tj_key, T_jkt = _series_from(_T_JACKET_KEYS)
    p_key, P = _series_from(_PRESSURE_KEYS)
    f_key, F = _series_from(_H2_FLOW_KEYS)
    x_key, X = _series_from(_CONVERSION_KEYS)
    mu_key, mu = _series_from(_VISCOSITY_KEYS)
    tau_key, tau_in = _series_from(_TORQUE_KEYS)
    rpm_key, rpm = _series_from(_RPM_KEYS)

    n = 0
    for seq in (T_rx, T_jkt, P, F, X, tau_in, times or []):
        if seq:
            n = max(n, len(seq))
    if n == 0:
        out["process_noise_applied"] = False
        return out

    dt = _infer_dt(times, cfg.default_dt_s)
    if times is None:
        times = [i * dt for i in range(n)]
        out["time"] = times
        t_key = "time"

    # Pad missing optional series
    if not X:
        X = [0.0] * n
    else:
        X = _ensure_length(X, n)
    if rpm:
        rpm = _ensure_length(rpm, n, fill=eq.agitator_speed_rpm)
    else:
        rpm = [eq.agitator_speed_rpm] * n
        out["agitator_rpm"] = rpm
        rpm_key = "agitator_rpm"

    # Seed offsets so channels are correlated but not identical
    def _channel_seed(offset: int) -> Optional[int]:
        return None if seed is None else int(seed) + offset

    # --- OU process noise ---------------------------------------------------
    n_Tr = (
        _noise_path(n, times, cfg.T_reactor_theta, cfg.T_reactor_sigma, dt, _channel_seed(1))
        if T_rx
        else [0.0] * n
    )
    n_Tj = (
        _noise_path(n, times, cfg.T_jacket_theta, cfg.T_jacket_sigma, dt, _channel_seed(2))
        if T_jkt
        else [0.0] * n
    )
    n_P = (
        _noise_path(n, times, cfg.pressure_theta, cfg.pressure_sigma, dt, _channel_seed(3))
        if P
        else [0.0] * n
    )
    n_tau = _noise_path(
        n, times, cfg.torque_theta, cfg.torque_sigma, dt, _channel_seed(4)
    )
    n_F = (
        _noise_path(n, times, cfg.h2_flow_theta, cfg.h2_flow_sigma, dt, _channel_seed(5))
        if F
        else [0.0] * n
    )

    # --- Dynamic torque from conversion / viscosity -------------------------
    if mu:
        mu = _ensure_length(mu, n)
        tau_phys, pwr_phys = torque_model.series(X, viscosity=mu, speed_rpm=rpm)
    else:
        tau_phys, pwr_phys = torque_model.series(X, speed_rpm=rpm)

    # If ODE already supplied torque, blend physical model as a soft prior
    if tau_in:
        tau_in = _ensure_length(tau_in, n)
        tau_base = [
            0.5 * tau_in[i] + 0.5 * tau_phys[i] for i in range(n)
        ]
    else:
        tau_base = tau_phys

    # --- Assemble noisy streams + cross-sensor coupling ---------------------
    T_rx_n: list[float] = []
    T_jkt_n: list[float] = []
    P_n: list[float] = []
    F_n: list[float] = []
    tau_n: list[float] = []
    pwr_n: list[float] = []

    P_clean = _ensure_length(P, n, fill=eq.headspace_pressure_target_bar) if P else (
        [eq.headspace_pressure_target_bar] * n
    )
    F_clean = _ensure_length(F, n, fill=0.0) if F else [0.0] * n
    T_rx_clean = _ensure_length(T_rx, n) if T_rx else [0.0] * n
    T_jkt_clean = _ensure_length(T_jkt, n) if T_jkt else [0.0] * n

    for i in range(n):
        # Temperatures with weak reactor→jacket coupling
        tr = T_rx_clean[i] + n_Tr[i] if T_rx else float("nan")
        tj = (
            T_jkt_clean[i] + n_Tj[i] + cfg.T_reactor_to_jacket_gain * n_Tr[i]
            if T_jkt
            else float("nan")
        )

        # Pressure OU
        p = P_clean[i] + n_P[i]

        # Cross-sensor: pressure fluctuation modulates apparent H2 flow
        # (MFC / DP-cell coupling & header ripple)
        dP = p - P_clean[i]
        f = F_clean[i] + n_F[i] + cfg.pressure_to_h2_flow_gain * dP
        f = max(0.0, min(eq.mfc_max_h2_feed_kg_min, f))

        # Torque: physical conversion model + OU shaft ripple
        tau = max(0.0, tau_base[i] + n_tau[i])
        omega = rpm[i] * (2.0 * math.pi / 60.0)
        pwr = (tau * omega) / 1000.0

        T_rx_n.append(tr)
        T_jkt_n.append(tj)
        P_n.append(p)
        F_n.append(f)
        tau_n.append(tau)
        pwr_n.append(pwr)

    # --- PLC / DCS quantization ---------------------------------------------
    if T_rx:
        T_rx_q = quantize_temperature(T_rx_n, cfg.temp_resolution_C)
        assert isinstance(T_rx_q, list)
        out[tr_key or "T_reactor"] = T_rx_q
        out["T_reactor_noise"] = n_Tr
    if T_jkt:
        T_jkt_q = quantize_temperature(T_jkt_n, cfg.temp_resolution_C)
        assert isinstance(T_jkt_q, list)
        out[tj_key or "T_jacket"] = T_jkt_q
        out["T_jacket_noise"] = [
            n_Tj[i] + cfg.T_reactor_to_jacket_gain * n_Tr[i] for i in range(n)
        ]
    if P or p_key:
        out[p_key or "pressure"] = quantize_series(P_n, cfg.pressure_resolution_bar)
        out["pressure_noise"] = n_P
    if F or f_key:
        out[f_key or "h2_flow"] = quantize_series(F_n, cfg.h2_flow_resolution_kg_min)
        out["h2_flow_noise"] = [
            F_n[i] - F_clean[i] for i in range(n)
        ]
    out[tau_key or "agitator_torque"] = quantize_series(
        tau_n, cfg.torque_resolution_Nm
    )
    out["agitator_power_kW"] = [round(p, 3) for p in pwr_n]
    out["agitator_torque_noise"] = n_tau
    out["agitator_torque_physics"] = tau_phys
    out["agitator_power_physics_kW"] = [round(p, 3) for p in pwr_phys]

    # Preserve conversion (optionally scaled by CoA assay — informational)
    if x_key is not None:
        out["conversion_coa_adjusted"] = [
            min(1.0, float(X[i]) * (purity / 100.0)) for i in range(n)
        ]

    out["process_noise_applied"] = True
    out["noise_config"] = {
        "T_reactor_theta": cfg.T_reactor_theta,
        "T_reactor_sigma": cfg.T_reactor_sigma,
        "T_jacket_theta": cfg.T_jacket_theta,
        "T_jacket_sigma": cfg.T_jacket_sigma,
        "pressure_theta": cfg.pressure_theta,
        "pressure_sigma": cfg.pressure_sigma,
        "torque_theta": cfg.torque_theta,
        "torque_sigma": cfg.torque_sigma,
        "pressure_to_h2_flow_gain": cfg.pressure_to_h2_flow_gain,
        "temp_resolution_C": cfg.temp_resolution_C,
        "coa_purity_range_pct": [COA_PURITY_MIN_PCT, COA_PURITY_MAX_PCT],
        "seed": seed,
    }
    return out


__all__ = [
    "OrnsteinUhlenbeckNoise",
    "ProcessNoiseConfig",
    "AgitatorTorqueModel",
    "apply_process_noise",
    "sample_coa_purity",
    "quantize_temperature",
    "quantize_series",
    "COA_PURITY_MIN_PCT",
    "COA_PURITY_MAX_PCT",
    "TEMP_QUANTIZATION_C",
]
