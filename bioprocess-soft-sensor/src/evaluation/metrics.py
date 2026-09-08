"""Primary and secondary evaluation metrics for mechanistic / NN-only / hybrid.

Do not assume the hybrid model wins. Metrics are computed from supplied
predictions versus IndPenSim *reference* biomass (simulator, not physical GT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

MODEL_MECHANISTIC = "mechanistic"
MODEL_NN_ONLY = "nn_only"
MODEL_HYBRID = "hybrid"
MODEL_ORDER: tuple[str, str, str] = (
    MODEL_MECHANISTIC,
    MODEL_NN_ONLY,
    MODEL_HYBRID,
)

DEFAULT_EARLY_FRACTION = 0.25
DEFAULT_LATE_FRACTION = 0.25


def _as_1d(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


def aligned_finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Element-wise finite mask across equally shaped 1-D arrays."""
    if not arrays:
        return np.array([], dtype=bool)
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        if arr.shape != arrays[0].shape:
            raise ValueError("Metric arrays must share the same shape")
        mask &= np.isfinite(arr)
    return mask


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    mask = aligned_finite_mask(yt, yp)
    if not np.any(mask):
        return float("nan")
    err = yp[mask] - yt[mask]
    return float(np.sqrt(np.mean(err * err)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    mask = aligned_finite_mask(yt, yp)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(yp[mask] - yt[mask])))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Undefined (NaN) if reference variance is 0."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    mask = aligned_finite_mask(yt, yp)
    if not np.any(mask):
        return float("nan")
    ytm = yt[mask]
    ypm = yp[mask]
    ss_res = float(np.sum((ytm - ypm) ** 2))
    ss_tot = float(np.sum((ytm - np.mean(ytm)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def primary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n": float(int(np.sum(aligned_finite_mask(_as_1d(y_true), _as_1d(y_pred))))),
    }


def _fraction_slice(n: int, fraction: float, *, which: str) -> slice:
    if n < 1:
        return slice(0, 0)
    k = max(1, int(np.ceil(n * float(fraction))))
    k = min(k, n)
    if which == "early":
        return slice(0, k)
    if which == "late":
        return slice(n - k, n)
    raise ValueError(f"Unknown stage {which!r}")


def stage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fraction: float,
    which: str,
) -> dict[str, float]:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    sl = _fraction_slice(yt.size, fraction, which=which)
    return primary_metrics(yt[sl], yp[sl])


def phase_wise_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phase_labels: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    if phase_labels is None:
        return {}
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    phases = np.asarray(phase_labels, dtype=object).reshape(-1)
    if phases.size != yt.size:
        raise ValueError("phase_labels length must match the trajectory")
    out: dict[str, dict[str, float]] = {}
    for label in list(dict.fromkeys(str(p) for p in phases if p is not None and str(p) != "nan")):
        mask = np.array([str(p) == label for p in phases], dtype=bool)
        if not np.any(mask):
            continue
        out[label] = primary_metrics(yt[mask], yp[mask])
    return out


def trajectory_stability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Smoothness vs reference first differences; jerk of the prediction.

    ``delta_rmse`` is RMSE(Δŷ, Δy_ref). Lower means the predicted trajectory
    tracks reference increments more stably. This is not a ranking claim.
    """
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    mask = aligned_finite_mask(yt, yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size < 2:
        return {
            "delta_rmse": float("nan"),
            "delta_mae": float("nan"),
            "pred_jerk_mae": float("nan"),
            "n_deltas": 0.0,
        }
    d_true = np.diff(yt)
    d_pred = np.diff(yp)
    jerk = np.diff(d_pred) if d_pred.size >= 2 else np.array([], dtype=np.float64)
    return {
        "delta_rmse": rmse(d_true, d_pred),
        "delta_mae": mae(d_true, d_pred),
        "pred_jerk_mae": float(np.mean(np.abs(jerk))) if jerk.size else float("nan"),
        "n_deltas": float(d_true.size),
    }


def ood_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ood_mask: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Error stratified by OOD vs in-distribution. Empty if no OOD labels."""
    if ood_mask is None:
        return {}
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    ood = np.asarray(ood_mask, dtype=bool).reshape(-1)
    if ood.size != yt.size:
        raise ValueError("ood_mask length must match the trajectory")
    finite = aligned_finite_mask(yt, yp)
    in_d = finite & ~ood
    out_d = finite & ood
    return {
        "in_distribution": primary_metrics(yt[in_d], yp[in_d]),
        "ood": primary_metrics(yt[out_d], yp[out_d]),
    }


def latency_summary(latency_s: np.ndarray | None) -> dict[str, float]:
    if latency_s is None:
        return {
            "mean_s": float("nan"),
            "p50_s": float("nan"),
            "p95_s": float("nan"),
            "max_s": float("nan"),
            "n": 0.0,
        }
    lat = _as_1d(latency_s)
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return latency_summary(None)
    return {
        "mean_s": float(np.mean(lat)),
        "p50_s": float(np.percentile(lat, 50)),
        "p95_s": float(np.percentile(lat, 95)),
        "max_s": float(np.max(lat)),
        "n": float(lat.size),
    }


@dataclass
class ModelBatchMetrics:
    model: str
    batch_id: Any
    rmse: float
    mae: float
    r2: float
    n: float
    early: dict[str, float] = field(default_factory=dict)
    late: dict[str, float] = field(default_factory=dict)
    phase_wise: dict[str, dict[str, float]] = field(default_factory=dict)
    stability: dict[str, float] = field(default_factory=dict)
    ood: dict[str, dict[str, float]] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "batch_id": self.batch_id,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "n": self.n,
            "early": dict(self.early),
            "late": dict(self.late),
            "phase_wise": {k: dict(v) for k, v in self.phase_wise.items()},
            "stability": dict(self.stability),
            "ood": {k: dict(v) for k, v in self.ood.items()},
            "latency": dict(self.latency),
        }


def score_model_on_batch(
    *,
    model: str,
    batch_id: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    phase_labels: np.ndarray | None = None,
    ood_mask: np.ndarray | None = None,
    latency_s: np.ndarray | None = None,
    early_fraction: float = DEFAULT_EARLY_FRACTION,
    late_fraction: float = DEFAULT_LATE_FRACTION,
) -> ModelBatchMetrics:
    if model not in MODEL_ORDER:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_ORDER}")
    prim = primary_metrics(y_true, y_pred)
    return ModelBatchMetrics(
        model=model,
        batch_id=batch_id,
        rmse=prim["rmse"],
        mae=prim["mae"],
        r2=prim["r2"],
        n=prim["n"],
        early=stage_error(y_true, y_pred, fraction=early_fraction, which="early"),
        late=stage_error(y_true, y_pred, fraction=late_fraction, which="late"),
        phase_wise=phase_wise_errors(y_true, y_pred, phase_labels),
        stability=trajectory_stability(y_true, y_pred),
        ood=ood_performance(y_true, y_pred, ood_mask),
        latency=latency_summary(latency_s),
    )


def mean_ignore_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def aggregate_model_metrics(batch_rows: list[ModelBatchMetrics]) -> dict[str, Any]:
    """Unweighted mean of per-batch primary metrics (each batch counts once)."""
    if not batch_rows:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_batches": 0}
    return {
        "rmse": mean_ignore_nan([r.rmse for r in batch_rows]),
        "mae": mean_ignore_nan([r.mae for r in batch_rows]),
        "r2": mean_ignore_nan([r.r2 for r in batch_rows]),
        "early_rmse": mean_ignore_nan([r.early.get("rmse", float("nan")) for r in batch_rows]),
        "late_rmse": mean_ignore_nan([r.late.get("rmse", float("nan")) for r in batch_rows]),
        "stability_delta_rmse": mean_ignore_nan(
            [r.stability.get("delta_rmse", float("nan")) for r in batch_rows]
        ),
        "ood_rmse": mean_ignore_nan(
            [r.ood.get("ood", {}).get("rmse", float("nan")) for r in batch_rows]
        ),
        "in_distribution_rmse": mean_ignore_nan(
            [r.ood.get("in_distribution", {}).get("rmse", float("nan")) for r in batch_rows]
        ),
        "latency_mean_s": mean_ignore_nan(
            [r.latency.get("mean_s", float("nan")) for r in batch_rows]
        ),
        "n_batches": float(len(batch_rows)),
        "n_samples": float(sum(r.n for r in batch_rows if np.isfinite(r.n))),
    }


def rank_models_by_metric(
    aggregate_by_model: Mapping[str, Mapping[str, Any]],
    metric: str = "rmse",
    *,
    lower_is_better: bool = True,
) -> list[str]:
    """Sort model names by an aggregate metric. Not a research conclusion."""
    scored: list[tuple[float, str]] = []
    for name in MODEL_ORDER:
        if name not in aggregate_by_model:
            continue
        value = aggregate_by_model[name].get(metric, float("nan"))
        if not np.isfinite(value):
            continue
        scored.append((float(value), name))
    reverse = not lower_is_better
    scored.sort(key=lambda item: item[0], reverse=reverse)
    return [name for _, name in scored]
