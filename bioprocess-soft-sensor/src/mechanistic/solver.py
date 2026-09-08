"""scipy.integrate.solve_ivp wrapper (BDF / Radau) for sequential steps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

ALLOWED_METHODS = ("BDF", "Radau")


@dataclass
class SolverOutcome:
    y: np.ndarray
    t: float
    success: bool
    status: str
    message: str
    latency_s: float
    method_used: str
    nfev: int | None = None


def _classify_y(
    y: np.ndarray,
    *,
    negative_tol: float,
    maxima: Sequence[float],
) -> str | None:
    if not np.all(np.isfinite(y)):
        return "nan"
    if np.any(np.abs(y) > np.asarray(maxima, dtype=float)):
        return "divergence"
    if np.any(y < -abs(negative_tol)):
        return "negative_state"
    return None


def integrate_step(
    fun: Callable,
    t0: float,
    y0: np.ndarray,
    dt: float,
    *,
    method: str = "BDF",
    fallback_method: str = "Radau",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float | None = None,
    timeout_s: float = 1.0,
    negative_tol: float = 1e-12,
    maxima: Sequence[float] | None = None,
    args: tuple = (),
) -> SolverOutcome:
    """Integrate one causal interval [t0, t0+dt] with BDF, falling back to Radau."""
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    maxima_arr = maxima if maxima is not None else (1e6,) * y0.size
    t0 = float(t0)
    dt = float(dt)
    method = str(method)
    if method not in ALLOWED_METHODS:
        raise ValueError(f"method must be BDF or Radau, got {method!r}")

    pre = _classify_y(y0, negative_tol=negative_tol, maxima=maxima_arr)
    if dt == 0.0:
        return SolverOutcome(
            y=y0.copy(),
            t=t0,
            success=pre is None,
            status="success" if pre is None else pre,
            message="zero dt; state unchanged",
            latency_s=0.0,
            method_used=method,
            nfev=0,
        )
    if dt < 0.0:
        return SolverOutcome(
            y=y0.copy(),
            t=t0,
            success=False,
            status="failed",
            message="dt must be non-negative for sequential causal updates",
            latency_s=0.0,
            method_used=method,
            nfev=0,
        )
    if pre == "nan":
        return SolverOutcome(
            y=y0.copy(),
            t=t0,
            success=False,
            status="nan",
            message="NaN/Inf in state before integrate",
            latency_s=0.0,
            method_used=method,
            nfev=0,
        )

    kwargs: dict = {"rtol": rtol, "atol": atol, "dense_output": False}
    if max_step is not None:
        kwargs["max_step"] = float(max_step)

    methods_try = [method]
    if fallback_method and fallback_method != method:
        methods_try.append(str(fallback_method))

    last: SolverOutcome | None = None
    t1 = t0 + dt
    for meth in methods_try:
        t_start = time.perf_counter()
        try:
            sol = solve_ivp(fun, (t0, t1), y0, method=meth, args=args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — surface solver exceptions as status
            latency = time.perf_counter() - t_start
            last = SolverOutcome(
                y=y0.copy(),
                t=t0,
                success=False,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
                latency_s=latency,
                method_used=meth,
            )
            continue
        latency = time.perf_counter() - t_start
        y_end = np.asarray(sol.y[:, -1], dtype=float) if sol.y.size else y0.copy()
        t_end = float(sol.t[-1]) if sol.t.size else t0
        nfev = int(getattr(sol, "nfev", 0) or 0)
        flags: list[str] = []
        if latency > timeout_s:
            flags.append("excessive_runtime")
        classif = _classify_y(y_end, negative_tol=negative_tol, maxima=maxima_arr)
        if classif:
            flags.append(classif)
        if not sol.success:
            flags.append("failed")
        if not flags:
            return SolverOutcome(
                y=y_end,
                t=t_end,
                success=True,
                status="success",
                message=str(sol.message or "success"),
                latency_s=latency,
                method_used=meth,
                nfev=nfev,
            )
        status = "+".join(flags)
        last = SolverOutcome(
            y=y_end,
            t=t_end,
            success=False,
            status=status,
            message=str(sol.message or status),
            latency_s=latency,
            method_used=meth,
            nfev=nfev,
        )
        # Retry fallback only on integrator failure, not on physical flags.
        if "failed" not in flags:
            return last
    assert last is not None
    return last
