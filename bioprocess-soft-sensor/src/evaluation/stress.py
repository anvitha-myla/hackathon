"""Robustness / stress-testing environment (Prompt 11).

Separate from demo inference. Perturbations are applied to **copies** in memory.
Clean final-test files are never overwritten; artifacts go under ``results/stress``.

Hybrid/OOD implementations may still be stubs. This module therefore owns the
locked attenuation math and a stub predictor so robustness tests can run now:

    beta_trust = exp(-kappa * max(0, D_M - D_threshold))
    X_hybrid   = X_mechanistic + beta_trust * delta_X

When ``src.inference.ood`` exposes Mahalanobis scoring (Prompt 9), stress uses
it. Otherwise a local Mahalanobis stub keeps the same locked equations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_yaml

DEFAULT_CONFIG_NAME = "stress.yaml"
STRESS_OUTPUT_DIR = PROJECT_ROOT / "results" / "stress"

FAULT_TYPES: tuple[str, ...] = (
    "normal",
    "random_noise",
    "spikes",
    "missing_values",
    "impossible_values",
    "sensor_drift",
    "distribution_shift",
    "feed_interruption",
    "temperature_disturbance",
    "combined_faults",
    "severe_ood",
)

RECORD_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "timestamp",
    "fault_type",
    "fault_magnitude",
    "reference",
    "mechanistic_prediction",
    "nn_prediction",
    "hybrid_prediction",
    "error",
    "OOD_distance",
    "beta_trust",
    "ML_contribution",
    "physics_contribution",
    "delta_X_raw",
    "phase_progress",
    "phase_growth",
    "phase_production",
    "phase_autolysis",
)

FORBIDDEN_LIVE_COLUMNS = frozenset(
    {
        "X_reference",
        "x_reference",
        "biomass_reference",
        "reference_biomass",
        "reference",
        "Penicillin_biomass",
        "biomass",
    }
)

DEFAULT_SENSOR_COLUMNS: tuple[str, ...] = (
    "pH",
    "DO",
    "T_vessel",
    "T_jacket",
    "agitation",
    "feed_rate",
    "gas_flow",
    "CO2",
    "O2",
)

IDENTITY_COLUMNS = ("batch_id", "timestamp", "relative_time")


def load_stress_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return load_yaml(DEFAULT_CONFIG_NAME)
    return load_yaml(path)


def _ood_beta_scalar(d_m: float, d_threshold: float, kappa: float) -> float:
    try:
        from src.inference.ood import beta_trust_from_distance

        return float(
            beta_trust_from_distance(float(d_m), kappa=float(kappa), d_threshold=float(d_threshold))
        )
    except Exception:
        if not np.isfinite(d_m) or not np.isfinite(kappa) or not np.isfinite(d_threshold):
            return 0.0
        excess = max(0.0, float(d_m) - float(d_threshold))
        return float(np.clip(np.exp(-float(kappa) * excess), 0.0, 1.0))


def beta_trust(
    d_m: float | np.ndarray,
    d_threshold: float,
    kappa: float,
) -> float | np.ndarray:
    """Locked OOD trust: exp(-kappa * max(0, D_M - D_threshold))."""
    d_m_arr = np.asarray(d_m, dtype=float)
    if d_m_arr.ndim == 0:
        return _ood_beta_scalar(float(d_m_arr), d_threshold, kappa)
    return np.asarray(
        [_ood_beta_scalar(float(v), d_threshold, kappa) for v in d_m_arr.reshape(-1)],
        dtype=float,
    ).reshape(d_m_arr.shape)


def combine_hybrid(
    x_mechanistic: float | np.ndarray,
    delta_x: float | np.ndarray,
    beta: float | np.ndarray,
) -> float | np.ndarray:
    """X_hybrid = X_mechanistic + beta_trust * delta_X."""
    if np.isscalar(x_mechanistic) and np.isscalar(delta_x) and np.isscalar(beta):
        try:
            from src.inference.ood import apply_hybrid

            return float(
                apply_hybrid(float(x_mechanistic), float(delta_x), float(beta)).X_hybrid
            )
        except Exception:
            return float(x_mechanistic) + float(beta) * float(delta_x)
    x = np.asarray(x_mechanistic, dtype=float)
    d = np.asarray(delta_x, dtype=float)
    b = np.asarray(beta, dtype=float)
    return x + b * d


def ml_contribution(beta: float | np.ndarray, delta_x: float | np.ndarray) -> float | np.ndarray:
    """ml_correction = beta_trust * delta_X_pred (docs/07)."""
    out = np.asarray(beta, dtype=float) * np.asarray(delta_x, dtype=float)
    if np.isscalar(beta) and np.isscalar(delta_x):
        return float(out)
    return out


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """D_M = sqrt((phi - mean)^T Sigma^{-1} (phi - mean))."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    delta = x - mean
    # Numerically stable quadratic form; clamp tiny negatives from roundoff.
    quad = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    return np.sqrt(np.maximum(quad, 0.0))


def fit_mahalanobis(
    train_features: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit mean and regularized inverse covariance on training features only."""
    phi = np.asarray(train_features, dtype=float)
    if phi.ndim != 2 or phi.shape[0] < 2:
        raise ValueError("Need at least two training feature rows to fit OOD.")
    finite = np.isfinite(phi).all(axis=1)
    phi = phi[finite]
    if phi.shape[0] < 2:
        raise ValueError("Need at least two finite training rows to fit OOD.")
    mean = phi.mean(axis=0)
    centered = phi - mean
    n = centered.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)
    dim = cov.shape[0]
    cov = cov + float(ridge) * np.eye(dim)
    inv_cov = np.linalg.pinv(cov)
    return mean, cov, inv_cov


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deep copy used before any perturbation."""
    return df.copy(deep=True)


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(None if seed is None else int(seed))


def _sensor_cols(df: pd.DataFrame, sensor_columns: Sequence[str] | None) -> list[str]:
    cols = list(sensor_columns or DEFAULT_SENSOR_COLUMNS)
    return [c for c in cols if c in df.columns]


def _window(n: int, start_frac: float, duration_frac: float) -> slice:
    start = int(max(0, min(n - 1, round(start_frac * n))))
    length = max(1, int(round(duration_frac * n)))
    end = min(n, start + length)
    return slice(start, end)


def apply_random_noise(
    df: pd.DataFrame,
    *,
    relative_std: float = 0.05,
    sensor_columns: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    out = copy_frame(df)
    rng = rng or _rng(None)
    for col in _sensor_cols(out, sensor_columns):
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0.0:
            scale = max(abs(float(np.nanmean(values))), 1.0) * 0.05
        noise = rng.normal(0.0, relative_std * scale, size=len(out))
        out[col] = values + noise
    return out


def apply_spikes(
    df: pd.DataFrame,
    *,
    n_spikes: int = 4,
    magnitude_std: float = 12.0,
    sensor_columns: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    out = copy_frame(df)
    rng = rng or _rng(None)
    cols = _sensor_cols(out, sensor_columns)
    n = len(out)
    if n == 0 or not cols:
        return out
    n_spikes = min(int(n_spikes), n)
    idx = rng.choice(n, size=n_spikes, replace=False)
    for i in idx:
        col = cols[int(rng.integers(0, len(cols)))]
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        sign = 1.0 if rng.random() > 0.5 else -1.0
        values[i] = values[i] + sign * magnitude_std * scale
        out[col] = values
    return out


def apply_missing_values(
    df: pd.DataFrame,
    *,
    fraction: float = 0.12,
    consecutive: bool = True,
    sensor_columns: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    out = copy_frame(df)
    rng = rng or _rng(None)
    cols = _sensor_cols(out, sensor_columns)
    n = len(out)
    if n == 0 or not cols:
        return out
    n_miss = max(1, int(round(fraction * n)))
    if consecutive:
        start = int(rng.integers(0, max(n - n_miss + 1, 1)))
        sl = slice(start, start + n_miss)
        for col in cols[: max(1, len(cols) // 2 + 1)]:
            out.loc[out.index[sl], col] = np.nan
    else:
        idx = rng.choice(n, size=min(n_miss, n), replace=False)
        for i in idx:
            col = cols[int(rng.integers(0, len(cols)))]
            out.iat[int(i), out.columns.get_loc(col)] = np.nan
    return out


def apply_impossible_values(
    df: pd.DataFrame,
    *,
    replacements: Mapping[str, float] | None = None,
    fraction: float = 0.08,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    out = copy_frame(df)
    rng = rng or _rng(None)
    defaults = {"pH": 19.0, "DO": -40.0, "T_vessel": 220.0, "feed_rate": -80.0}
    replacements = dict(defaults if replacements is None else replacements)
    n = len(out)
    if n == 0:
        return out
    n_bad = max(1, int(round(fraction * n)))
    idx = rng.choice(n, size=min(n_bad, n), replace=False)
    for col, value in replacements.items():
        if col in out.columns:
            out.loc[out.index[idx], col] = float(value)
    return out


def apply_sensor_drift(
    df: pd.DataFrame,
    *,
    end_bias_std: float = 8.0,
    sensor_columns: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    out = copy_frame(df)
    rng = rng or _rng(None)
    n = len(out)
    if n == 0:
        return out
    ramp = np.linspace(0.0, 1.0, n)
    for col in _sensor_cols(out, sensor_columns):
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        bias = float(rng.normal(0.0, 1.0)) * end_bias_std * scale
        out[col] = values + ramp * bias
    return out


def apply_distribution_shift(
    df: pd.DataFrame,
    *,
    offset_std: float = 25.0,
    sensor_columns: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    out = copy_frame(df)
    rng = rng or _rng(None)
    for col in _sensor_cols(out, sensor_columns):
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        offset = float(rng.normal(0.0, 1.0)) * offset_std * scale
        # Keep the shift clearly OOD even if the draw is near zero.
        if abs(offset) < 5.0 * scale:
            offset = math.copysign(5.0 * scale * offset_std / max(offset_std, 1.0), offset or 1.0)
        out[col] = values + offset
    return out


def apply_feed_interruption(
    df: pd.DataFrame,
    *,
    start_frac: float = 0.35,
    duration_frac: float = 0.25,
    feed_column: str = "feed_rate",
) -> pd.DataFrame:
    out = copy_frame(df)
    if feed_column not in out.columns:
        return out
    sl = _window(len(out), start_frac, duration_frac)
    values = pd.to_numeric(out[feed_column], errors="coerce").to_numpy(dtype=float)
    values[sl] = 0.0
    out[feed_column] = values
    return out


def apply_temperature_disturbance(
    df: pd.DataFrame,
    *,
    delta_C: float = 18.0,
    start_frac: float = 0.2,
    duration_frac: float = 0.4,
    temperature_column: str = "T_vessel",
) -> pd.DataFrame:
    out = copy_frame(df)
    if temperature_column not in out.columns:
        return out
    n = len(out)
    sl = _window(n, start_frac, duration_frac)
    values = pd.to_numeric(out[temperature_column], errors="coerce").to_numpy(dtype=float)
    width = max(sl.stop - sl.start, 1)
    ramp = np.sin(np.linspace(0.0, np.pi, width))
    values[sl] = values[sl] + float(delta_C) * ramp
    out[temperature_column] = values
    if "T_jacket" in out.columns:
        jacket = pd.to_numeric(out["T_jacket"], errors="coerce").to_numpy(dtype=float)
        jacket[sl] = jacket[sl] + 0.4 * float(delta_C) * ramp
        out["T_jacket"] = jacket
    return out


PERTURBATION_FNS: dict[str, Callable[..., pd.DataFrame]] = {
    "random_noise": apply_random_noise,
    "spikes": apply_spikes,
    "missing_values": apply_missing_values,
    "impossible_values": apply_impossible_values,
    "sensor_drift": apply_sensor_drift,
    "distribution_shift": apply_distribution_shift,
    "feed_interruption": apply_feed_interruption,
    "temperature_disturbance": apply_temperature_disturbance,
}


def apply_perturbation(
    df: pd.DataFrame,
    fault_type: str,
    *,
    config: Mapping[str, Any] | None = None,
    rng: np.random.Generator | int | None = None,
    sensor_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a perturbed **copy**. The input frame is not modified."""
    if df is None:
        raise ValueError("df is required")
    original_id = id(df)
    cfg = dict(config or {})
    pert_cfg = dict(cfg.get("perturbations", cfg))
    sensors = list(sensor_columns or cfg.get("sensor_columns") or DEFAULT_SENSOR_COLUMNS)
    gen = _rng(rng)
    fault = str(fault_type)
    meta: dict[str, Any] = {"fault_type": fault, "fault_magnitude": 0.0}

    if fault in {"normal", "none", ""}:
        out = copy_frame(df)
        meta["fault_magnitude"] = 0.0
    elif fault == "combined_faults":
        out = copy_frame(df)
        include = list(
            pert_cfg.get("combined_faults", {}).get(
                "include",
                ["random_noise", "spikes", "sensor_drift", "feed_interruption", "temperature_disturbance"],
            )
        )
        mag = 0.0
        for name in include:
            if name in {"normal", "combined_faults", "severe_ood"}:
                continue
            out, inner = apply_perturbation(
                out, name, config=cfg, rng=gen, sensor_columns=sensors
            )
            mag += float(inner.get("fault_magnitude", 1.0))
        meta["fault_magnitude"] = mag
        meta["included"] = include
    elif fault == "severe_ood":
        params = dict(pert_cfg.get("severe_ood", {}))
        offset = float(params.get("offset_std", 80.0))
        out = apply_distribution_shift(
            df, offset_std=offset, sensor_columns=sensors, rng=gen
        )
        meta["fault_magnitude"] = offset
    elif fault in PERTURBATION_FNS:
        params = dict(pert_cfg.get(fault, {}))
        fn = PERTURBATION_FNS[fault]
        kwargs: dict[str, Any] = {}
        # Map config keys onto each applicator.
        if fault == "random_noise":
            kwargs["relative_std"] = float(params.get("relative_std", 0.05))
            meta["fault_magnitude"] = kwargs["relative_std"]
        elif fault == "spikes":
            kwargs["n_spikes"] = int(params.get("n_spikes", 4))
            kwargs["magnitude_std"] = float(params.get("magnitude_std", 12.0))
            meta["fault_magnitude"] = kwargs["magnitude_std"]
        elif fault == "missing_values":
            kwargs["fraction"] = float(params.get("fraction", 0.12))
            kwargs["consecutive"] = bool(params.get("consecutive", True))
            meta["fault_magnitude"] = kwargs["fraction"]
        elif fault == "impossible_values":
            keys = ("pH", "DO", "T_vessel", "feed_rate")
            kwargs["replacements"] = {k: float(params[k]) for k in keys if k in params}
            meta["fault_magnitude"] = 1.0
        elif fault == "sensor_drift":
            kwargs["end_bias_std"] = float(params.get("end_bias_std", 8.0))
            meta["fault_magnitude"] = kwargs["end_bias_std"]
        elif fault == "distribution_shift":
            kwargs["offset_std"] = float(params.get("offset_std", 25.0))
            meta["fault_magnitude"] = kwargs["offset_std"]
        elif fault == "feed_interruption":
            kwargs["start_frac"] = float(params.get("start_frac", 0.35))
            kwargs["duration_frac"] = float(params.get("duration_frac", 0.25))
            meta["fault_magnitude"] = kwargs["duration_frac"]
        elif fault == "temperature_disturbance":
            kwargs["delta_C"] = float(params.get("delta_C", 18.0))
            kwargs["start_frac"] = float(params.get("start_frac", 0.2))
            kwargs["duration_frac"] = float(params.get("duration_frac", 0.4))
            meta["fault_magnitude"] = kwargs["delta_C"]
        if fault not in {"feed_interruption", "temperature_disturbance", "impossible_values"}:
            kwargs["sensor_columns"] = sensors
        if fault not in {"feed_interruption", "temperature_disturbance"}:
            kwargs["rng"] = gen
        out = fn(df, **kwargs)
    else:
        raise ValueError(f"Unknown fault_type {fault!r}. Expected one of {FAULT_TYPES}")

    if id(out) == original_id:
        raise RuntimeError("Perturbation must not return the original frame object.")
    return out, meta


def soft_phase_indicators(n: int) -> pd.DataFrame:
    """Continuous phase indicators (no hard NN switching)."""
    if n <= 0:
        return pd.DataFrame(
            columns=["phase_progress", "phase_growth", "phase_production", "phase_autolysis"]
        )
    progress = np.linspace(0.0, 1.0, n)
    growth = np.clip(1.0 - progress / 0.45, 0.0, 1.0)
    auto = np.clip((progress - 0.75) / 0.25, 0.0, 1.0)
    production = np.clip(1.0 - growth - auto, 0.0, 1.0)
    total = np.maximum(growth + production + auto, 1e-12)
    return pd.DataFrame(
        {
            "phase_progress": progress,
            "phase_growth": growth / total,
            "phase_production": production / total,
            "phase_autolysis": auto / total,
        }
    )


def make_synthetic_batch(
    *,
    batch_id: int = 1,
    n: int = 80,
    seed: int = 0,
    process_noise: float = 0.02,
) -> pd.DataFrame:
    """In-memory IndPenSim-like observables. Not written to data/."""
    rng = _rng(int(seed) + int(batch_id) * 17)
    t = np.arange(n, dtype=float)
    progress = t / max(n - 1, 1)
    feed = np.where(progress < 0.15, 0.02, 0.08 + 0.04 * np.sin(progress * 4.0))
    do = 70.0 - 25.0 * progress + rng.normal(0.0, process_noise * 8.0, n)
    ph = 6.5 + 0.15 * np.sin(progress * 6.0) + rng.normal(0.0, process_noise, n)
    t_vessel = 25.0 + 0.4 * np.sin(progress * 3.0) + rng.normal(0.0, process_noise, n)
    t_jacket = t_vessel - 1.5 + rng.normal(0.0, process_noise, n)
    agitation = 200.0 + 40.0 * progress + rng.normal(0.0, process_noise * 5.0, n)
    gas = 6.0 + 0.5 * np.sin(progress * 2.0)
    co2 = 0.4 + 1.8 * progress
    o2 = 20.5 - 1.2 * progress
    # Reduced-physics-like biomass plus a systematic residual the stub ML "knows".
    x_mech = 0.2 + 12.0 * (1.0 - np.exp(-2.2 * progress)) * (0.4 + 0.6 * feed / 0.12)
    residual = 1.5 * np.sin(np.pi * progress) + 0.4
    x_ref = np.maximum(x_mech + residual, 0.0)
    return pd.DataFrame(
        {
            "batch_id": batch_id,
            "timestamp": t,
            "relative_time": t,
            "pH": ph,
            "DO": do,
            "T_vessel": t_vessel,
            "T_jacket": t_jacket,
            "agitation": agitation,
            "feed_rate": feed,
            "gas_flow": gas,
            "CO2": co2,
            "O2": o2,
            "X_reference": x_ref,
            "X_mechanistic": x_mech,
        }
    )


def feature_matrix(
    df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
) -> np.ndarray:
    cols = _sensor_cols(df, sensor_columns)
    if not cols:
        raise ValueError("No sensor columns present for OOD features.")
    if any(c in FORBIDDEN_LIVE_COLUMNS for c in cols):
        raise ValueError("Reference biomass must not enter the OOD feature vector.")
    return df[cols].to_numpy(dtype=float)


@dataclass
class FallbackMahalanobisOOD:
    """Local Mahalanobis OOD (used when Prompt 9 cannot be imported)."""

    mean: np.ndarray
    inv_cov: np.ndarray
    d_threshold: float
    kappa: float
    missing_is_ood: bool = True

    def score_matrix(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = np.asarray(phi, dtype=float)
        if phi.ndim == 1:
            phi = phi.reshape(1, -1)
        finite = np.isfinite(phi)
        sent = self.mean + 50.0 * (1.0 + np.abs(self.mean))
        filled = np.where(finite, phi, sent if self.missing_is_ood else self.mean)
        d_m = mahalanobis_distance(filled, self.mean, self.inv_cov)
        if self.missing_is_ood:
            row_missing = ~finite.all(axis=1)
            d_m = np.where(row_missing, np.maximum(d_m, self.d_threshold + 20.0), d_m)
        beta = np.asarray(beta_trust(d_m, self.d_threshold, self.kappa), dtype=float)
        return d_m, np.clip(beta, 0.0, 1.0)

    def hybrid_at(
        self,
        *,
        x_mechanistic: float,
        delta_x_pred: float,
        phi: np.ndarray,
    ) -> tuple[float, float, float]:
        d_m, beta = self.score_matrix(np.asarray(phi, dtype=float).reshape(1, -1))
        b = float(beta[0])
        hybrid = combine_hybrid(float(x_mechanistic), float(delta_x_pred), b)
        return float(hybrid), float(d_m[0]), b


def fit_ood_from_frames(
    frames: Sequence[pd.DataFrame],
    *,
    sensor_columns: Sequence[str] | None = None,
    kappa: float = 1.0,
    d_threshold: float | None = None,
    ridge: float = 1e-6,
    missing_is_ood: bool = True,
) -> Any:
    mats = [feature_matrix(f, sensor_columns) for f in frames]
    train = np.vstack(mats)
    try:
        from src.inference.ood import MahalanobisOOD as RealOOD

        det = RealOOD(kappa=kappa, d_threshold=d_threshold, ridge=ridge)
        det.fit(train, split="train")
        return det
    except Exception:
        mean, _cov, inv_cov = fit_mahalanobis(train, ridge=ridge)
        finite = train[np.isfinite(train).all(axis=1)]
        d_train = mahalanobis_distance(finite, mean, inv_cov)
        thresh = float(np.quantile(d_train, 0.95)) if d_threshold is None else float(d_threshold)
        return FallbackMahalanobisOOD(
            mean=mean,
            inv_cov=inv_cov,
            d_threshold=thresh,
            kappa=float(kappa),
            missing_is_ood=missing_is_ood,
        )


def _score_row(ood: Any, x_mech: float, delta: float, phi_row: np.ndarray) -> tuple[float, float, float]:
    """Return (hybrid, D_M, beta) using Prompt 9 when present."""
    if hasattr(ood, "hybrid_at") and type(ood).__name__ != "FallbackMahalanobisOOD":
        hybrid_obj, ood_score = ood.hybrid_at(
            x_mechanistic=float(x_mech),
            delta_x_pred=float(delta),
            phi=phi_row,
        )
        d_m = float(ood_score.d_m) if np.isfinite(ood_score.d_m) else float("nan")
        return float(hybrid_obj.X_hybrid), d_m, float(ood_score.beta_trust)
    hybrid, d_m, beta = ood.hybrid_at(
        x_mechanistic=float(x_mech),
        delta_x_pred=float(delta),
        phi=phi_row,
    )
    return float(hybrid), float(d_m), float(beta)


@dataclass
class StubHybridPredictor:
    """Stress predictor: physics column + residual stub + OOD trust gating.

    Mechanistic trajectory is taken from ``X_mechanistic`` (physics output, not
    live reference biomass). Residual ``delta_X`` is a constant learned from
    train residuals so in-distribution correction is active and OOD attenuation
    is testable even before a trained residual MLP is wired in.
    """

    ood: Any
    delta_x: float
    sensor_columns: tuple[str, ...] = DEFAULT_SENSOR_COLUMNS

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        phases = soft_phase_indicators(n)
        x_mech = pd.to_numeric(df.get("X_mechanistic", pd.Series(np.zeros(n))), errors="coerce")
        x_mech_v = np.nan_to_num(x_mech.to_numpy(dtype=float), nan=0.0)
        try:
            phi = feature_matrix(df, self.sensor_columns)
        except ValueError:
            phi = np.zeros((n, 1))
        delta = float(self.delta_x)
        hybrids = np.empty(n, dtype=float)
        d_ms = np.empty(n, dtype=float)
        betas = np.empty(n, dtype=float)
        for i in range(n):
            h, d_m, b = _score_row(self.ood, x_mech_v[i], delta, phi[i])
            if not np.isfinite(h):
                h = float(x_mech_v[i])
            hybrids[i] = h
            d_ms[i] = d_m
            betas[i] = b if np.isfinite(b) else 0.0
        ml = ml_contribution(betas, np.full(n, delta))
        x_nn = x_mech_v + delta
        ref = pd.to_numeric(df.get("X_reference", pd.Series(np.full(n, np.nan))), errors="coerce")
        ref_v = ref.to_numpy(dtype=float)
        err = hybrids - ref_v
        ts = df["timestamp"] if "timestamp" in df.columns else pd.Series(np.arange(n))
        bid = df["batch_id"] if "batch_id" in df.columns else pd.Series(np.zeros(n, dtype=int))
        return pd.DataFrame(
            {
                "batch_id": bid.to_numpy(),
                "timestamp": ts.to_numpy(),
                "mechanistic_prediction": x_mech_v,
                "nn_prediction": np.asarray(x_nn, dtype=float),
                "hybrid_prediction": hybrids,
                "reference": ref_v,
                "error": np.asarray(err, dtype=float),
                "OOD_distance": d_ms,
                "beta_trust": betas,
                "ML_contribution": np.asarray(ml, dtype=float),
                "physics_contribution": x_mech_v,
                "delta_X_raw": np.full(n, delta),
                "phase_progress": phases["phase_progress"].to_numpy(),
                "phase_growth": phases["phase_growth"].to_numpy(),
                "phase_production": phases["phase_production"].to_numpy(),
                "phase_autolysis": phases["phase_autolysis"].to_numpy(),
            }
        )


def _learn_stub_delta(train_frames: Sequence[pd.DataFrame]) -> float:
    deltas: list[float] = []
    for frame in train_frames:
        if "X_reference" in frame.columns and "X_mechanistic" in frame.columns:
            d = (
                pd.to_numeric(frame["X_reference"], errors="coerce")
                - pd.to_numeric(frame["X_mechanistic"], errors="coerce")
            )
            deltas.append(float(np.nanmean(d.to_numpy(dtype=float))))
    if not deltas:
        return 1.0
    return float(np.nanmean(np.asarray(deltas, dtype=float)))


def load_clean_for_stress(
    source: pd.DataFrame | Sequence[pd.DataFrame] | Path,
    *,
    allow_write: bool = False,
) -> list[pd.DataFrame]:
    """Load clean batches as copies. Refuses to write to the source path."""
    if allow_write:
        raise ValueError("Writing to clean test data is forbidden in the stress environment.")
    if isinstance(source, Path) or (isinstance(source, str)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".csv", ".txt"}:
            frames = [pd.read_csv(path)]
        elif path.suffix.lower() in {".parquet"}:
            frames = [pd.read_parquet(path)]
        else:
            raise ValueError(f"Unsupported clean-data suffix: {path.suffix}")
        return [copy_frame(f) for f in frames]
    if isinstance(source, pd.DataFrame):
        return [copy_frame(source)]
    return [copy_frame(f) for f in source]


def score_perturbed(
    clean: pd.DataFrame,
    fault_type: str,
    predictor: StubHybridPredictor,
    *,
    config: Mapping[str, Any] | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    perturbed, meta = apply_perturbation(clean, fault_type, config=config, rng=rng)
    rec = predictor.predict_frame(perturbed)
    rec["fault_type"] = meta["fault_type"]
    rec["fault_magnitude"] = float(meta.get("fault_magnitude", 0.0))
    # Guarantee spec columns even if a future real engine omits some.
    for col in RECORD_COLUMNS:
        if col not in rec.columns:
            rec[col] = np.nan
    return rec.loc[:, list(RECORD_COLUMNS)]


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    rows = []
    for fault, part in records.groupby("fault_type", sort=True):
        err = pd.to_numeric(part["error"], errors="coerce")
        hybrid = pd.to_numeric(part["hybrid_prediction"], errors="coerce")
        mech = pd.to_numeric(part["mechanistic_prediction"], errors="coerce")
        beta = pd.to_numeric(part["beta_trust"], errors="coerce")
        ood = pd.to_numeric(part["OOD_distance"], errors="coerce")
        ml = pd.to_numeric(part["ML_contribution"], errors="coerce")
        ref = pd.to_numeric(part["reference"], errors="coerce")
        rows.append(
            {
                "fault_type": fault,
                "n": int(len(part)),
                "rmse": float(np.sqrt(np.nanmean(np.square(err)))),
                "mae": float(np.nanmean(np.abs(err))),
                "mean_beta_trust": float(np.nanmean(beta)),
                "median_beta_trust": float(np.nanmedian(beta)),
                "mean_OOD_distance": float(np.nanmean(ood)),
                "mean_abs_ML_contribution": float(np.nanmean(np.abs(ml))),
                "mean_abs_hybrid_minus_mech": float(np.nanmean(np.abs(hybrid - mech))),
                "mean_physics_contribution": float(np.nanmean(mech)),
                "mean_reference": float(np.nanmean(ref)),
                "fault_magnitude": float(np.nanmean(pd.to_numeric(part["fault_magnitude"], errors="coerce"))),
            }
        )
    return pd.DataFrame(rows).sort_values("fault_type").reset_index(drop=True)


def write_summary_tables(summary: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robustness_summary.csv"
    json_path = output_dir / "robustness_summary.json"
    summary.to_csv(csv_path, index=False)
    summary.to_json(json_path, orient="records", indent=2)
    return {"csv": csv_path, "json": json_path}


def write_robustness_plots(records: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary = summarize_records(records)

    def _save(fig: plt.Figure, name: str) -> Path:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
        return path

    if not records.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(
            pd.to_numeric(records["OOD_distance"], errors="coerce"),
            pd.to_numeric(records["beta_trust"], errors="coerce"),
            s=8,
            alpha=0.5,
            c="steelblue",
        )
        ax.set_xlabel("OOD distance D_M")
        ax.set_ylabel("beta_trust")
        ax.set_title("Trust factor vs Mahalanobis distance")
        _save(fig, "beta_vs_ood.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        if not summary.empty:
            ax.bar(summary["fault_type"], summary["mean_abs_ML_contribution"], color="darkorange")
            ax.set_ylabel("|ML contribution|")
            ax.set_title("ML contribution by fault type")
            ax.tick_params(axis="x", rotation=40)
        _save(fig, "ml_contribution_by_fault.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        if not summary.empty:
            ax.bar(summary["fault_type"], summary["mean_beta_trust"], color="seagreen")
            ax.set_ylabel("mean beta_trust")
            ax.set_ylim(0.0, 1.05)
            ax.set_title("Mean trust by fault type")
            ax.tick_params(axis="x", rotation=40)
        _save(fig, "beta_by_fault.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        if not summary.empty:
            ax.bar(summary["fault_type"], summary["rmse"], color="slategray")
            ax.set_ylabel("RMSE")
            ax.set_title("Hybrid RMSE by fault type")
            ax.tick_params(axis="x", rotation=40)
        _save(fig, "rmse_by_fault.png")

        severe = records[records["fault_type"].isin(["severe_ood", "distribution_shift", "impossible_values"])]
        normal = records[records["fault_type"] == "normal"]
        fig, ax = plt.subplots(figsize=(7, 4))
        if not normal.empty:
            ax.plot(normal["timestamp"], normal["hybrid_prediction"], label="hybrid (normal)", color="C0")
            ax.plot(normal["timestamp"], normal["mechanistic_prediction"], label="mechanistic", color="C2", ls="--")
        if not severe.empty:
            s = severe[severe["fault_type"] == severe["fault_type"].iloc[0]]
            ax.plot(s["timestamp"], s["hybrid_prediction"], label="hybrid (severe OOD)", color="C3")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("biomass X")
        ax.set_title("Severe OOD: hybrid moves toward mechanistic")
        ax.legend(fontsize=8)
        _save(fig, "hybrid_vs_mechanistic_severe_ood.png")

    return written


@dataclass
class StressResult:
    records: pd.DataFrame
    summary: pd.DataFrame
    plot_paths: list[Path] = field(default_factory=list)
    table_paths: dict[str, Path] = field(default_factory=dict)
    output_dir: Path | None = None
    notes: list[str] = field(default_factory=list)


def run_stress_environment(
    *,
    train_frames: Sequence[pd.DataFrame] | None = None,
    test_frames: Sequence[pd.DataFrame] | None = None,
    fault_types: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int | None = None,
    write_artifacts: bool = True,
) -> StressResult:
    """Run all controlled perturbations on in-memory copies of clean batches."""
    cfg = dict(load_stress_config() if config is None else config)
    sensors = tuple(cfg.get("sensor_columns") or DEFAULT_SENSOR_COLUMNS)
    ood_cfg = dict(cfg.get("ood", {}))
    rng_seed = int(cfg.get("seed", 42) if seed is None else seed)

    if train_frames is None:
        train_frames = [make_synthetic_batch(batch_id=i, n=60, seed=rng_seed) for i in range(1, 4)]
    if test_frames is None:
        test_frames = [make_synthetic_batch(batch_id=99, n=80, seed=rng_seed + 99)]

    train_copies = [copy_frame(f) for f in train_frames]
    test_copies = [copy_frame(f) for f in test_frames]

    ood = fit_ood_from_frames(
        train_copies,
        sensor_columns=sensors,
        kappa=float(ood_cfg.get("kappa") or 1.0),
        d_threshold=ood_cfg.get("d_threshold"),
        ridge=float(ood_cfg.get("ridge") or 1e-6),
        missing_is_ood=bool(ood_cfg.get("missing_is_ood", True)),
    )
    delta = _learn_stub_delta(train_copies)
    predictor = StubHybridPredictor(ood=ood, delta_x=delta, sensor_columns=sensors)

    faults = list(fault_types or [f for f in FAULT_TYPES])
    records_list: list[pd.DataFrame] = []
    gen = _rng(rng_seed)
    for clean in test_copies:
        for fault in faults:
            rec = score_perturbed(clean, fault, predictor, config=cfg, rng=gen)
            records_list.append(rec)

    records = pd.concat(records_list, ignore_index=True) if records_list else pd.DataFrame(columns=list(RECORD_COLUMNS))
    summary = summarize_records(records)

    notes = [
        "Clean frames were copied in memory; sources were not written.",
        "Hybrid/OOD use local stubs unless monitoring.ood.beta_trust is importable.",
        f"Stub residual delta_X={delta:.4f}; OOD threshold={ood.d_threshold:.4f}; kappa={ood.kappa}.",
    ]

    out_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / str(cfg.get("output_dir", "results/stress"))
    plots: list[Path] = []
    tables: dict[str, Path] = {}
    if write_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        records_path = out_dir / "stress_records.csv"
        records.to_csv(records_path, index=False)
        tables = write_summary_tables(summary, out_dir)
        tables["records"] = records_path
        plots = write_robustness_plots(records, out_dir)
        meta_path = out_dir / "stress_run_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "fault_types": faults,
                    "n_records": int(len(records)),
                    "delta_X": delta,
                    "d_threshold": ood.d_threshold,
                    "kappa": ood.kappa,
                    "notes": notes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tables["meta"] = meta_path

    return StressResult(
        records=records,
        summary=summary,
        plot_paths=plots,
        table_paths=tables,
        output_dir=out_dir if write_artifacts else None,
        notes=notes,
    )


def severe_ood_falls_back_to_physics(records: pd.DataFrame) -> dict[str, float]:
    """Measured fallback: severe OOD should drop beta and ML contribution."""
    normal = records[records["fault_type"] == "normal"]
    severe = records[records["fault_type"] == "severe_ood"]
    if normal.empty or severe.empty:
        raise ValueError("Need both normal and severe_ood rows.")
    stats = {
        "normal_mean_beta": float(np.nanmean(normal["beta_trust"])),
        "severe_mean_beta": float(np.nanmean(severe["beta_trust"])),
        "normal_mean_abs_ml": float(np.nanmean(np.abs(normal["ML_contribution"]))),
        "severe_mean_abs_ml": float(np.nanmean(np.abs(severe["ML_contribution"]))),
        "severe_mean_abs_hybrid_minus_mech": float(
            np.nanmean(np.abs(severe["hybrid_prediction"] - severe["mechanistic_prediction"]))
        ),
        "normal_mean_abs_hybrid_minus_mech": float(
            np.nanmean(np.abs(normal["hybrid_prediction"] - normal["mechanistic_prediction"]))
        ),
    }
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    result = run_stress_environment(write_artifacts=True)
    print(result.summary.to_string(index=False))
    print("output_dir:", result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
