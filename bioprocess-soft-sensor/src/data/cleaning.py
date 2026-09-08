"""Causal process-telemetry cleaning (docs/03_CLEANING_SPEC.md).

At timestamp t every transform may use x[0], …, x[t] only — never x[t+1].

Live inference forbids two-sided filters, full-batch smoothers, and
future-aware gap filling. Constants live in ``configs/cleaning.yaml`` and
are not estimated from test batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.config import load_yaml
from src.data.quality import (
    QualityFlag,
    apply_physical_flags,
    bounds_for_column,
    check_physical_plausibility,
    non_finite_mask,
)

ArrayF = npt.NDArray[np.floating]
ArrayB = npt.NDArray[np.bool_]
ArrayU = npt.NDArray[np.uint32]

CLEANING_CONFIG_NAME = "cleaning.yaml"
BATCH_ID_COL = "batch_id"


@dataclass(frozen=True)
class CleaningParams:
    """Train-safe constants. Construct via ``load_cleaning_params`` / YAML."""

    target_grid_minutes: float = 1.0
    regrid_when_median_step_minutes_at_most: float = 1.01
    numeric_time_unit: str = "hours"
    time_column_candidates: tuple[str, ...] = ("timestamp", "Time", "time", "t")
    identity_columns: tuple[str, ...] = (
        "batch_id",
        "timestamp",
        "Time",
        "time",
        "t",
        "relative_time",
        "source",
        "dataset",
        "source_dataset",
        "split",
    )
    forward_fill_max_minutes: float = 5.0
    add_missingness_indicators: bool = True
    hampel_window_minutes: float = 5.0
    hampel_min_window_samples: int = 5
    hampel_n_sigmas: float = 3.0
    hampel_mad_scale: float = 1.4826
    hampel_min_periods: int = 3
    smooth_method: str = "ema"
    ema_alpha: float = 0.3
    ma_window_minutes: float = 5.0
    physical_bounds: Mapping[str, Any] = field(default_factory=dict)
    source: str = CLEANING_CONFIG_NAME


@dataclass
class CleanedSeries:
    raw: ArrayF
    cleaned: ArrayF
    quality: ArrayU
    missing: ArrayB
    metadata: dict[str, Any]


@dataclass
class CleanedBatch:
    frame: pd.DataFrame
    metadata: dict[str, Any]
    signal_columns: tuple[str, ...]


def load_cleaning_config(path: str | None = None) -> dict[str, Any]:
    return load_yaml(path or CLEANING_CONFIG_NAME)


def params_from_config(cfg: Mapping[str, Any] | None = None) -> CleaningParams:
    """Map YAML to params. Does not read batch values (no test-set fit)."""
    if cfg is None:
        cfg = load_cleaning_config()
    sampling = cfg.get("sampling") or {}
    missing = cfg.get("missing") or {}
    hampel = cfg.get("hampel") or {}
    smoothing = cfg.get("smoothing") or {}
    identity = tuple(cfg.get("identity_columns") or CleaningParams.identity_columns)
    time_cands = tuple(
        sampling.get("time_column_candidates")
        or CleaningParams.time_column_candidates
    )
    return CleaningParams(
        target_grid_minutes=float(sampling.get("target_grid_minutes", 1.0)),
        regrid_when_median_step_minutes_at_most=float(
            sampling.get("regrid_when_median_step_minutes_at_most", 1.01)
        ),
        numeric_time_unit=str(sampling.get("numeric_time_unit", "hours")),
        time_column_candidates=time_cands,
        identity_columns=identity,
        forward_fill_max_minutes=float(missing.get("forward_fill_max_minutes", 5.0)),
        add_missingness_indicators=bool(
            missing.get("add_missingness_indicators", True)
        ),
        hampel_window_minutes=float(hampel.get("window_minutes", 5.0)),
        hampel_min_window_samples=int(hampel.get("min_window_samples", 5)),
        hampel_n_sigmas=float(hampel.get("n_sigmas", 3.0)),
        hampel_mad_scale=float(hampel.get("mad_scale", 1.4826)),
        hampel_min_periods=int(hampel.get("min_periods", 3)),
        smooth_method=str(smoothing.get("method", "ema")),
        ema_alpha=float(smoothing.get("ema_alpha", 0.3)),
        ma_window_minutes=float(smoothing.get("ma_window_minutes", 5.0)),
        physical_bounds=dict(cfg.get("physical_bounds") or {}),
        source=CLEANING_CONFIG_NAME,
    )


def load_cleaning_params() -> CleaningParams:
    """YAML constants only — never computed from held-out test batches."""
    return params_from_config(load_cleaning_config())


def fit_cleaning_params(
    train_batches: Sequence[pd.DataFrame] | None = None,
) -> CleaningParams:
    """Return YAML params. ``train_batches`` is accepted for the train-only API
    and is not inspected (no statistics are fit in this revision).
    """
    del train_batches
    return load_cleaning_params()


def minutes_to_steps(window_minutes: float, step_minutes: float) -> int:
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    return max(1, int(round(float(window_minutes) / float(step_minutes))))


def infer_step_minutes(
    times: npt.ArrayLike,
    *,
    numeric_unit: str = "hours",
) -> float:
    t = np.asarray(times)
    if t.size < 2:
        if numeric_unit == "hours":
            return 60.0
        if np.issubdtype(t.dtype, np.datetime64):
            return 1.0
        return 1.0
    if np.issubdtype(t.dtype, np.datetime64):
        delta = np.diff(t.astype("datetime64[ns]").astype(np.int64))
        median_ns = float(np.median(delta[np.isfinite(delta.astype(float))]))
        return max(median_ns / 1e9 / 60.0, 1e-9)
    t = t.astype(float)
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0 if numeric_unit != "hours" else 60.0
    median = float(np.median(diffs))
    if numeric_unit == "hours":
        return median * 60.0
    if numeric_unit == "minutes":
        return median
    if numeric_unit == "seconds":
        return median / 60.0
    return median


def bounded_forward_fill(
    values: npt.ArrayLike,
    max_steps: int,
) -> tuple[ArrayF, ArrayB, ArrayB]:
    """Carry the last finite value forward at most ``max_steps`` samples.

    Uses only past samples. A later finite value never fills an earlier gap.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    x = np.asarray(values, dtype=float).copy()
    n = x.size
    filled = np.zeros(n, dtype=bool)
    expired = np.zeros(n, dtype=bool)
    last_val = np.nan
    last_i = -1
    for i in range(n):
        if np.isfinite(x[i]):
            last_val = x[i]
            last_i = i
            continue
        if last_i >= 0 and (i - last_i) <= max_steps:
            x[i] = last_val
            filled[i] = True
        elif last_i >= 0:
            expired[i] = True
        else:
            expired[i] = True
    return x, filled, expired


def causal_hampel(
    values: npt.ArrayLike,
    window: int,
    *,
    n_sigmas: float = 3.0,
    mad_scale: float = 1.4826,
    min_periods: int = 3,
) -> tuple[ArrayF, ArrayB]:
    """Backward-looking Hampel: window is ``x[t-window+1 : t+1]`` (includes t).

    Robust location/scale use only that backward window. Spikes are replaced
    with the window median. Future samples are never read.
    """
    if window < 1:
        raise ValueError("Hampel window must be >= 1")
    x = np.asarray(values, dtype=float).copy()
    n = x.size
    is_spike = np.zeros(n, dtype=bool)
    out = x.copy()
    for t in range(n):
        if not np.isfinite(x[t]):
            continue
        lo = max(0, t - window + 1)
        chunk = x[lo : t + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < min_periods:
            continue
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        sigma = mad_scale * mad
        if sigma <= 0.0:
            deviant = abs(x[t] - med) > 0.0
        else:
            deviant = abs(x[t] - med) > n_sigmas * sigma
        if deviant:
            is_spike[t] = True
            out[t] = med
    return out, is_spike


def causal_ema(values: npt.ArrayLike, alpha: float) -> ArrayF:
    """One-sided EMA: y[t] = alpha * x[t] + (1-alpha) * y[t-1]."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("ema alpha must be in (0, 1]")
    x = np.asarray(values, dtype=float)
    y = np.full(x.shape, np.nan, dtype=float)
    prev = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        if not np.isfinite(prev):
            y[i] = v
        else:
            y[i] = alpha * v + (1.0 - alpha) * prev
        prev = y[i]
    return y


def causal_one_sided_ma(values: npt.ArrayLike, window: int) -> ArrayF:
    """Mean of ``x[t-window+1 : t+1]`` using finite samples only."""
    if window < 1:
        raise ValueError("MA window must be >= 1")
    x = np.asarray(values, dtype=float)
    n = x.size
    y = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - window + 1)
        finite = x[lo : t + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y[t] = float(np.mean(finite))
    return y


def _hampel_window_samples(params: CleaningParams, step_minutes: float) -> int:
    from_minutes = minutes_to_steps(params.hampel_window_minutes, step_minutes)
    return max(int(params.hampel_min_window_samples), from_minutes)


def _ma_window_samples(params: CleaningParams, step_minutes: float) -> int:
    from_minutes = minutes_to_steps(params.ma_window_minutes, step_minutes)
    return max(int(params.hampel_min_window_samples), from_minutes)


def _fill_max_steps(params: CleaningParams, step_minutes: float) -> int:
    return minutes_to_steps(params.forward_fill_max_minutes, step_minutes)


def clean_series(
    values: npt.ArrayLike,
    params: CleaningParams | None = None,
    *,
    step_minutes: float = 1.0,
    column: str | None = None,
) -> CleanedSeries:
    """Clean one 1-D signal causally. ``values[t]`` never sees ``values[t+k]``."""
    params = params or load_cleaning_params()
    raw = np.asarray(values, dtype=float).copy()
    missing = non_finite_mask(raw)
    quality = np.zeros(raw.shape, dtype=np.uint32)

    lo, hi = bounds_for_column(column or "", params.physical_bounds)
    quality = apply_physical_flags(raw, quality, lo, hi)
    implausible = check_physical_plausibility(raw, lo, hi) | np.isinf(raw)

    work = raw.copy()
    work[missing | implausible] = np.nan

    max_steps = _fill_max_steps(params, step_minutes)
    filled, used_fill, expired = bounded_forward_fill(work, max_steps)
    quality[used_fill] |= np.uint32(QualityFlag.FORWARD_FILLED)
    still_gap = ~np.isfinite(filled)
    quality[still_gap] |= np.uint32(QualityFlag.FILL_LIMIT_EXCEEDED)

    h_window = _hampel_window_samples(params, step_minutes)
    despiked, is_spike = causal_hampel(
        filled,
        h_window,
        n_sigmas=params.hampel_n_sigmas,
        mad_scale=params.hampel_mad_scale,
        min_periods=params.hampel_min_periods,
    )
    quality[is_spike] |= np.uint32(QualityFlag.SPIKE)

    method = params.smooth_method.lower()
    if method == "ema":
        cleaned = causal_ema(despiked, params.ema_alpha)
    elif method in {"one_sided_ma", "causal_ma", "ma"}:
        cleaned = causal_one_sided_ma(despiked, _ma_window_samples(params, step_minutes))
    elif method in {"none", "off"}:
        cleaned = despiked
    else:
        raise ValueError(f"Unknown smoothing method {params.smooth_method!r}")

    metadata = {
        "column": column,
        "step_minutes": float(step_minutes),
        "fill_max_steps": int(max_steps),
        "hampel_window_samples": int(h_window),
        "smooth_method": params.smooth_method,
        "ema_alpha": params.ema_alpha,
        "physical_bounds": {"min": lo, "max": hi},
        "params_source": params.source,
        "causal": True,
        "looks_ahead": False,
    }
    return CleanedSeries(
        raw=raw,
        cleaned=cleaned,
        quality=quality,
        missing=missing,
        metadata=metadata,
    )


def _norm_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def signal_columns(
    frame: pd.DataFrame,
    params: CleaningParams,
    extra_exclude: Iterable[str] = (),
) -> list[str]:
    skip = {_norm_col(c) for c in params.identity_columns}
    skip.update(_norm_col(c) for c in extra_exclude)
    cols: list[str] = []
    for col in frame.columns:
        if _norm_col(col) in skip:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        cols.append(str(col))
    return cols


def resolve_time_column(
    frame: pd.DataFrame,
    params: CleaningParams,
    time_col: str | None = None,
) -> str | None:
    if time_col is not None:
        if time_col not in frame.columns:
            raise KeyError(f"time column {time_col!r} not in frame")
        return time_col
    for cand in params.time_column_candidates:
        if cand in frame.columns:
            return cand
    return None


def _ceil_to_grid(times: ArrayF, t0: float, step: float) -> ArrayF:
    n = np.ceil((times - t0) / step - 1e-12)
    n = np.maximum(n, 0)
    return t0 + n * step


def align_to_minute_grid(
    frame: pd.DataFrame,
    time_col: str,
    params: CleaningParams,
    *,
    time_unit: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Place rows on a 1-minute grid without pulling future samples backward.

    Each sample at time tau is attached to the first grid point >= tau.
    Gaps stay NaN until ``bounded_forward_fill`` (later). No two-sided join.
    """
    unit = time_unit or params.numeric_time_unit
    work = frame.sort_values(time_col).copy()
    times = work[time_col].to_numpy()
    step_native = infer_step_minutes(times, numeric_unit=unit)
    meta = {
        "reindexed": False,
        "native_step_minutes": step_native,
        "target_grid_minutes": params.target_grid_minutes,
        "time_column": time_col,
        "time_unit": unit,
    }
    if step_native > params.regrid_when_median_step_minutes_at_most:
        return work, meta

    if np.issubdtype(work[time_col].dtype, np.datetime64):
        t0 = pd.Timestamp(times.min()).floor("min")
        t1 = pd.Timestamp(times.max()).ceil("min")
        grid = pd.date_range(t0, t1, freq=f"{int(params.target_grid_minutes)}min")
        snapped = pd.to_datetime(times).ceil("min")
        work = work.copy()
        work["_grid_time"] = snapped
        work = work.drop_duplicates("_grid_time", keep="first")
        grid_df = pd.DataFrame({time_col: grid})
        merged = grid_df.merge(
            work.drop(columns=[time_col]).rename(columns={"_grid_time": time_col}),
            on=time_col,
            how="left",
        )
        meta["reindexed"] = True
        meta["grid_length"] = int(len(merged))
        return merged, meta

    step = float(params.target_grid_minutes)
    if unit == "hours":
        step = float(params.target_grid_minutes) / 60.0
    t_num = times.astype(float)
    t0 = float(np.nanmin(t_num))
    t1 = float(np.nanmax(t_num))
    grid = np.arange(t0, t1 + 0.5 * step, step)
    snapped = _ceil_to_grid(t_num, t0, step)
    work = work.copy()
    work["_grid_time"] = snapped
    work = work.drop_duplicates("_grid_time", keep="first")
    grid_df = pd.DataFrame({time_col: grid})
    placed = work.drop(columns=[time_col]).rename(columns={"_grid_time": time_col})
    merged = grid_df.merge(placed, on=time_col, how="left")
    meta["reindexed"] = True
    meta["grid_length"] = int(len(merged))
    return merged, meta


def _clean_single_batch_frame(
    frame: pd.DataFrame,
    params: CleaningParams,
    *,
    time_col: str | None,
    time_unit: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    grid_meta: dict[str, Any] = {"reindexed": False}
    if time_col is not None and time_col in out.columns:
        out, grid_meta = align_to_minute_grid(
            out, time_col, params, time_unit=time_unit
        )
        times = out[time_col].to_numpy()
        unit = time_unit or params.numeric_time_unit
        step_minutes = infer_step_minutes(times, numeric_unit=unit)
        if grid_meta.get("reindexed"):
            step_minutes = float(params.target_grid_minutes)
    else:
        step_minutes = float(params.target_grid_minutes)

    cols = signal_columns(out, params, extra_exclude=[time_col] if time_col else ())
    audits: dict[str, Any] = {}
    for col in cols:
        series = clean_series(
            out[col].to_numpy(dtype=float),
            params,
            step_minutes=step_minutes,
            column=col,
        )
        out[f"{col}_raw"] = series.raw
        out[f"{col}_cleaned"] = series.cleaned
        out[f"{col}_quality"] = series.quality
        if params.add_missingness_indicators:
            out[f"{col}_missing"] = series.missing
        audits[col] = series.metadata
    meta = {
        **grid_meta,
        "step_minutes": step_minutes,
        "signal_columns": cols,
        "audits": audits,
        "params_source": params.source,
    }
    return out, meta


def clean_batch(
    frame: pd.DataFrame,
    params: CleaningParams | None = None,
    *,
    time_col: str | None = None,
    time_unit: str | None = None,
    batch_col: str = BATCH_ID_COL,
) -> CleanedBatch:
    """Clean a table causally, independently per ``batch_id`` when present.

    Original signal columns are left unchanged (raw). Cleaned values, quality
    flags, and missingness indicators are added alongside.
    """
    params = params or load_cleaning_params()
    if frame.empty:
        return CleanedBatch(
            frame=frame.copy(),
            metadata={"empty": True, "params_source": params.source},
            signal_columns=(),
        )
    resolved_time = resolve_time_column(frame, params, time_col)
    pieces: list[pd.DataFrame] = []
    metas: list[dict[str, Any]] = []
    if batch_col in frame.columns:
        for batch_id, grp in frame.groupby(batch_col, sort=False):
            cleaned, meta = _clean_single_batch_frame(
                grp, params, time_col=resolved_time, time_unit=time_unit
            )
            meta["batch_id"] = batch_id
            pieces.append(cleaned)
            metas.append(meta)
        out = pd.concat(pieces, axis=0).sort_index()
    else:
        out, meta = _clean_single_batch_frame(
            frame, params, time_col=resolved_time, time_unit=time_unit
        )
        metas.append(meta)
    signal_cols = tuple(metas[0]["signal_columns"]) if metas else ()
    return CleanedBatch(
        frame=out,
        metadata={
            "n_groups": len(metas),
            "groups": metas,
            "params_source": params.source,
            "causal": True,
        },
        signal_columns=signal_cols,
    )


def with_smoothing_method(params: CleaningParams, method: str) -> CleaningParams:
    return replace(params, smooth_method=method)
