"""Causal (backward-looking) derivatives, lags, and rolling statistics.

At index t, every statistic uses only samples 0..t inclusive. No two-sided
filters and no future samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=float)


def backward_derivative(
    values: np.ndarray | pd.Series,
    time_h: np.ndarray | pd.Series,
) -> np.ndarray:
    """First-order backward difference: (x[t] - x[t-1]) / (time[t] - time[t-1])."""
    x = _as_float(values)
    t = _as_float(time_h)
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    out[0] = 0.0
    dt = np.diff(t)
    dx = np.diff(x)
    valid = np.abs(dt) > 0
    out[1:] = np.where(valid, dx / dt, np.nan)
    return out


def causal_lag(values: np.ndarray | pd.Series, lag: int) -> np.ndarray:
    """Shift by ``lag`` samples toward the past. Positions without history are NaN."""
    if lag < 1:
        raise ValueError("lag must be a positive integer")
    x = _as_float(values)
    out = np.full_like(x, np.nan, dtype=float)
    if lag < x.size:
        out[lag:] = x[:-lag]
    return out


def _window_slice(i: int, window: int) -> slice:
    start = max(0, i - window + 1)
    return slice(start, i + 1)


def causal_rolling_mean(values: np.ndarray | pd.Series, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be a positive integer")
    x = _as_float(values)
    out = np.empty_like(x)
    for i in range(x.size):
        sl = x[_window_slice(i, window)]
        out[i] = float(np.nanmean(sl)) if sl.size else np.nan
    return out


def causal_rolling_std(values: np.ndarray | pd.Series, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be a positive integer")
    x = _as_float(values)
    out = np.empty_like(x)
    for i in range(x.size):
        sl = x[_window_slice(i, window)]
        finite = sl[np.isfinite(sl)]
        if finite.size < 2:
            out[i] = 0.0 if finite.size == 1 else np.nan
        else:
            out[i] = float(np.std(finite, ddof=0))
    return out


def causal_rolling_slope(
    values: np.ndarray | pd.Series,
    time_h: np.ndarray | pd.Series,
    window: int,
) -> np.ndarray:
    """Ordinary least-squares slope of x vs time on the backward window ending at t."""
    if window < 1:
        raise ValueError("window must be a positive integer")
    x = _as_float(values)
    t = _as_float(time_h)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.size):
        sl = _window_slice(i, window)
        tw = t[sl]
        xw = x[sl]
        mask = np.isfinite(tw) & np.isfinite(xw)
        if mask.sum() < 2:
            out[i] = 0.0 if mask.sum() == 1 else np.nan
            continue
        tw = tw[mask]
        xw = xw[mask]
        t0 = tw - tw.mean()
        denom = float(np.dot(t0, t0))
        if denom <= 0:
            out[i] = 0.0
            continue
        x0 = xw - xw.mean()
        out[i] = float(np.dot(t0, x0) / denom)
    return out


def causal_cumulative_trapz(
    rate: np.ndarray | pd.Series,
    time_h: np.ndarray | pd.Series,
) -> np.ndarray:
    """Causal trapezoidal integral of a rate vs time. Integral at t=0 is 0."""
    r = _as_float(rate)
    t = _as_float(time_h)
    n = r.size
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt == 0:
            out[i] = out[i - 1]
            continue
        left = r[i - 1]
        right = r[i]
        if not np.isfinite(left):
            left = 0.0 if not np.isfinite(right) else right
        if not np.isfinite(right):
            right = left
        out[i] = out[i - 1] + 0.5 * (left + right) * dt
    return out
