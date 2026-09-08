"""CPU, RAM, model size, parameter count, and monitored sequential inference.

Uses psutil for process/system resource snapshots and time.perf_counter
(via ``src.monitoring.timing``) for stage latencies. Sub-150 ms is a
benchmark *target* for later comparison, not a hardcoded result.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn

from src.monitoring.timing import (
    PIPELINE_STAGES,
    STAGE_CLEANING,
    STAGE_FEATURE_ENGINEERING,
    STAGE_MECHANISTIC_SOLVER,
    STAGE_NN,
    STAGE_OOD,
    STAGE_TOTAL_INFERENCE,
    TimingAccumulator,
    compare_to_benchmark,
    measure_stage,
    time_callable,
)

PROCESS = psutil.Process()

# Warm CPU counters so later interval=None samples are defined.
PROCESS.cpu_percent(interval=None)
psutil.cpu_percent(interval=None)


def snapshot_resources(*, cpu_interval_s: float | None = None) -> dict[str, float]:
    """Live CPU and RAM from psutil (not canned numbers)."""
    vm = psutil.virtual_memory()
    mem = PROCESS.memory_info()
    if cpu_interval_s is None:
        proc_cpu = float(PROCESS.cpu_percent(interval=None))
        sys_cpu = float(psutil.cpu_percent(interval=None))
    else:
        proc_cpu = float(PROCESS.cpu_percent(interval=float(cpu_interval_s)))
        sys_cpu = float(psutil.cpu_percent(interval=None))
    return {
        "cpu_percent": proc_cpu,
        "system_cpu_percent": sys_cpu,
        "ram_used_bytes": float(vm.used),
        "ram_available_bytes": float(vm.available),
        "ram_percent": float(vm.percent),
        "rss_bytes": float(mem.rss),
        "vms_bytes": float(mem.vms),
    }


def count_parameters(module: nn.Module | None) -> int | None:
    """Trainable + non-trainable parameter count from an actual ``nn.Module``."""
    if module is None:
        return None
    if not isinstance(module, nn.Module):
        raise TypeError(f"expected torch.nn.Module, got {type(module)!r}")
    return int(sum(int(p.numel()) for p in module.parameters()))


def model_size_bytes(
    module: nn.Module | None = None,
    path: str | Path | None = None,
) -> int | None:
    """Serialized parameter tensor bytes, or on-disk file size if ``path`` is set."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            return int(p.stat().st_size)
        return None
    if module is None:
        return None
    if not isinstance(module, nn.Module):
        raise TypeError(f"expected torch.nn.Module, got {type(module)!r}")
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, buf)
    return int(buf.tell())


def _default_clean(raw: Mapping[str, float], prev: Mapping[str, float] | None) -> dict[str, float]:
    """Causal cleaner: current sample only; fill NaN from previous (no future)."""
    out: dict[str, float] = {}
    for k, v in raw.items():
        x = float(v)
        if not np.isfinite(x):
            if prev is not None and k in prev and np.isfinite(prev[k]):
                x = float(prev[k])
            else:
                x = 0.0
        out[k] = x
    if "pH" in out:
        out["pH"] = float(np.clip(out["pH"], 0.0, 14.0))
    if "DO" in out:
        out["DO"] = float(max(out["DO"], 0.0))
    return out


def _default_features(
    clean: Mapping[str, float],
    prev_clean: Mapping[str, float] | None,
    dt: float,
    cumulative_feed: float,
) -> tuple[np.ndarray, float]:
    """Causal derived features from current/previous cleaned observables."""
    t_v = float(clean.get("T_vessel", clean.get("T", 298.0)))
    t_j = float(clean.get("T_jacket", t_v))
    delta_t = t_v - t_j
    do = float(clean.get("DO", 0.0))
    co2 = float(clean.get("outlet_CO2", 0.0))
    o2 = float(clean.get("outlet_O2", 0.0))
    feed = float(clean.get("F", clean.get("feed_rate", 0.0)))
    dt_safe = float(dt) if dt > 0.0 else 1.0
    if prev_clean is not None:
        ddo_dt = (do - float(prev_clean.get("DO", do))) / dt_safe
    else:
        ddo_dt = 0.0
    our = max(0.0, 21.0 - o2)
    cer = max(0.0, co2)
    rq = cer / our if our > 1e-12 else 0.0
    cumulative_feed = float(cumulative_feed) + feed * dt_safe
    vec = np.array(
        [
            float(clean.get("pH", 7.0)),
            do,
            t_v,
            t_j,
            float(clean.get("agitation", 0.0)),
            feed,
            float(clean.get("gas_flow", 0.0)),
            co2,
            o2,
            our,
            cer,
            rq,
            delta_t,
            ddo_dt,
            cumulative_feed,
        ],
        dtype=np.float64,
    )
    return vec, cumulative_feed


def _try_build_mechanistic() -> Any | None:
    try:
        from src.mechanistic.model import ReducedMechanisticModel

        return ReducedMechanisticModel()
    except Exception:
        return None


def _fallback_mechanistic_step(
    y: np.ndarray,
    obs: Mapping[str, float],
    dt: float,
) -> np.ndarray:
    """Minimal Euler biomass/substrate/volume step if the ODE solver is unavailable."""
    x, s, v = float(y[0]), float(y[1]), max(float(y[2]), 1e-6)
    f = float(obs.get("F", 0.0))
    do = max(float(obs.get("DO", 0.0)), 0.0)
    mu = 0.12 * (s / (0.5 + s)) * (do / (2.0 + do))
    dx = (mu - 0.01 - f / v) * x
    ds = (f * 500.0) / v - (mu / 0.45) * x - 0.03 * x - (f / v) * s
    dv = f
    y2 = y.copy()
    y2[0] = max(x + dt * dx, 0.0)
    y2[1] = max(s + dt * ds, 0.0)
    y2[2] = max(v + dt * dv, 1e-6)
    return y2


def _mahalanobis(phi: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    d = phi - mean
    val = float(d @ precision @ d)
    return float(np.sqrt(max(val, 0.0)))


@dataclass
class StepMeasurement:
    cpu_percent: float
    ram_percent: float
    rss_bytes: float
    cleaning_latency_s: float
    feature_engineering_latency_s: float
    mechanistic_solver_latency_s: float
    nn_latency_s: float
    ood_latency_s: float
    total_inference_latency_s: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "rss_bytes": self.rss_bytes,
            "cleaning_latency_s": self.cleaning_latency_s,
            "feature_engineering_latency_s": self.feature_engineering_latency_s,
            "mechanistic_solver_latency_s": self.mechanistic_solver_latency_s,
            "nn_latency_s": self.nn_latency_s,
            "ood_latency_s": self.ood_latency_s,
            "total_inference_latency_s": self.total_inference_latency_s,
        }


@dataclass
class MonitoringReport:
    """Session-level computational monitor (current + history)."""

    n_samples: int
    session_elapsed_s: float
    throughput_hz: float
    cpu_percent: float
    ram_percent: float
    rss_bytes: float
    ram_used_bytes: float
    cleaning_latency_s: float
    feature_engineering_latency_s: float
    mechanistic_solver_latency_s: float
    nn_latency_s: float
    ood_latency_s: float
    total_inference_latency_s: float
    model_size_bytes: int | None
    parameter_count: int | None
    mechanistic_backend: str
    vs_benchmark: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "session_elapsed_s": self.session_elapsed_s,
            "throughput_hz": self.throughput_hz,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "rss_bytes": self.rss_bytes,
            "ram_used_bytes": self.ram_used_bytes,
            "cleaning_latency_s": self.cleaning_latency_s,
            "feature_engineering_latency_s": self.feature_engineering_latency_s,
            "mechanistic_solver_latency_s": self.mechanistic_solver_latency_s,
            "nn_latency_s": self.nn_latency_s,
            "ood_latency_s": self.ood_latency_s,
            "total_inference_latency_s": self.total_inference_latency_s,
            "model_size_bytes": self.model_size_bytes,
            "parameter_count": self.parameter_count,
            "mechanistic_backend": self.mechanistic_backend,
            "vs_benchmark": dict(self.vs_benchmark),
            "history": list(self.history),
        }


def _synthetic_raw(t_index: int, rng: np.random.Generator) -> dict[str, float]:
    """Causal synthetic telemetry for a short measurement loop (not IndPenSim)."""
    phase = t_index * 0.05
    return {
        "pH": 6.5 + 0.1 * np.sin(phase) + 0.01 * float(rng.normal()),
        "DO": 40.0 + 5.0 * np.sin(phase * 0.7) + float(rng.normal()),
        "T_vessel": 298.0 + 0.2 * np.sin(phase * 0.3),
        "T_jacket": 297.0 + 0.1 * np.cos(phase * 0.3),
        "agitation": 200.0,
        "F": 0.02 + 0.005 * max(0.0, np.sin(phase)),
        "gas_flow": 1.0,
        "outlet_CO2": 1.5 + 0.2 * np.sin(phase),
        "outlet_O2": 19.0 - 0.3 * np.sin(phase),
    }


def run_monitored_inference_loop(
    n_steps: int = 16,
    *,
    dt: float = 0.2,
    module: nn.Module | None = None,
    model_path: str | Path | None = None,
    seed: int = 0,
    clean_fn: Callable[[Mapping[str, float], Mapping[str, float] | None], Mapping[str, float]]
    | None = None,
    feature_fn: Callable[..., tuple[np.ndarray, float]] | None = None,
    samples: Sequence[Mapping[str, float]] | None = None,
) -> MonitoringReport:
    """Run a sequential (causal) inference stub/loop while recording measurements.

    Cleaning / feature-engineering layers may still be stubs; this loop still
    executes real work and times each stage. If ``ReducedMechanisticModel`` is
    importable it is used for the solver stage; otherwise a local Euler fallback
    keeps the monitor independent of unfinished Prompt 5 files.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    if module is None:
        from src.models.mlp import SmallMLP

        module = SmallMLP(n_in=15, n_out=1)
    module.eval()

    param_count = count_parameters(module)
    size_b = model_size_bytes(module, path=model_path)

    mech = _try_build_mechanistic()
    backend = "reduced_mechanistic_model" if mech is not None else "euler_fallback"
    y = np.array([0.15, 15.0, 100.0], dtype=np.float64)

    clean = clean_fn or _default_clean
    feats = feature_fn or _default_features

    rng = np.random.default_rng(seed)
    acc = TimingAccumulator()
    history: list[dict[str, float]] = []
    prev_clean: dict[str, float] | None = None
    cum_feed = 0.0

    mean = np.zeros(15, dtype=np.float64)
    precision = np.eye(15, dtype=np.float64)

    # Prime NN so first timed forward is not dominated by one-off init.
    with torch.no_grad():
        module(torch.zeros(1, 15, dtype=torch.float32))

    t_session0 = time.perf_counter()
    last_total = 0.0
    last_stages = {s: 0.0 for s in PIPELINE_STAGES}

    for i in range(n_steps):
        raw = dict(samples[i]) if samples is not None else _synthetic_raw(i, rng)

        with measure_stage(STAGE_TOTAL_INFERENCE, acc) as total_slot:
            cleaned, t_clean = time_callable(clean, raw, prev_clean)
            acc.add(STAGE_CLEANING, t_clean)
            cleaned_d = dict(cleaned)

            feat_vec, t_feat = time_callable(feats, cleaned_d, prev_clean, dt, cum_feed)
            if isinstance(feat_vec, tuple):
                phi, cum_feed = feat_vec
            else:
                phi = feat_vec
            acc.add(STAGE_FEATURE_ENGINEERING, t_feat)
            phi = np.asarray(phi, dtype=np.float64).reshape(-1)

            def _mech_step() -> float:
                nonlocal y
                obs = {
                    "F": float(cleaned_d.get("F", 0.0)),
                    "DO": float(cleaned_d.get("DO", 0.0)),
                }
                if mech is not None:
                    res = mech.step(dt, obs)
                    return float(res.X_mechanistic)
                y = _fallback_mechanistic_step(y, obs, dt)
                return float(y[0])

            x_mech, t_mech = time_callable(_mech_step)
            acc.add(STAGE_MECHANISTIC_SOLVER, t_mech)

            x_in = torch.as_tensor(phi[:15], dtype=torch.float32).reshape(1, -1)
            if x_in.shape[1] < 15:
                x_in = torch.nn.functional.pad(x_in, (0, 15 - x_in.shape[1]))
            elif x_in.shape[1] > 15:
                x_in = x_in[:, :15]

            def _nn_step() -> float:
                with torch.no_grad():
                    return float(module(x_in).reshape(-1)[0].item())

            delta, t_nn = time_callable(_nn_step)
            acc.add(STAGE_NN, t_nn)

            def _ood_step() -> float:
                d_m = _mahalanobis(phi[:15] if phi.size >= 15 else np.pad(phi, (0, 15 - phi.size)), mean, precision)
                kappa = 1.0
                d_thr = 3.0
                return float(np.exp(-kappa * max(0.0, d_m - d_thr)))

            _beta, t_ood = time_callable(_ood_step)
            acc.add(STAGE_OOD, t_ood)
            del _beta, x_mech, delta

        last_total = float(total_slot[0])
        last_stages = {
            STAGE_CLEANING: t_clean,
            STAGE_FEATURE_ENGINEERING: t_feat,
            STAGE_MECHANISTIC_SOLVER: t_mech,
            STAGE_NN: t_nn,
            STAGE_OOD: t_ood,
            STAGE_TOTAL_INFERENCE: last_total,
        }
        res = snapshot_resources()
        step = StepMeasurement(
            cpu_percent=res["cpu_percent"],
            ram_percent=res["ram_percent"],
            rss_bytes=res["rss_bytes"],
            cleaning_latency_s=t_clean,
            feature_engineering_latency_s=t_feat,
            mechanistic_solver_latency_s=t_mech,
            nn_latency_s=t_nn,
            ood_latency_s=t_ood,
            total_inference_latency_s=last_total,
        )
        history.append(step.as_dict())
        prev_clean = cleaned_d

    session_elapsed = time.perf_counter() - t_session0
    throughput = float(n_steps) / session_elapsed if session_elapsed > 0.0 else 0.0
    res_end = snapshot_resources()

    mean_or_last = {
        STAGE_CLEANING: float(acc.mean(STAGE_CLEANING) or last_stages[STAGE_CLEANING]),
        STAGE_FEATURE_ENGINEERING: float(
            acc.mean(STAGE_FEATURE_ENGINEERING) or last_stages[STAGE_FEATURE_ENGINEERING]
        ),
        STAGE_MECHANISTIC_SOLVER: float(
            acc.mean(STAGE_MECHANISTIC_SOLVER) or last_stages[STAGE_MECHANISTIC_SOLVER]
        ),
        STAGE_NN: float(acc.mean(STAGE_NN) or last_stages[STAGE_NN]),
        STAGE_OOD: float(acc.mean(STAGE_OOD) or last_stages[STAGE_OOD]),
        STAGE_TOTAL_INFERENCE: float(acc.mean(STAGE_TOTAL_INFERENCE) or last_total),
    }

    return MonitoringReport(
        n_samples=n_steps,
        session_elapsed_s=float(session_elapsed),
        throughput_hz=throughput,
        cpu_percent=res_end["cpu_percent"],
        ram_percent=res_end["ram_percent"],
        rss_bytes=res_end["rss_bytes"],
        ram_used_bytes=res_end["ram_used_bytes"],
        cleaning_latency_s=mean_or_last[STAGE_CLEANING],
        feature_engineering_latency_s=mean_or_last[STAGE_FEATURE_ENGINEERING],
        mechanistic_solver_latency_s=mean_or_last[STAGE_MECHANISTIC_SOLVER],
        nn_latency_s=mean_or_last[STAGE_NN],
        ood_latency_s=mean_or_last[STAGE_OOD],
        total_inference_latency_s=mean_or_last[STAGE_TOTAL_INFERENCE],
        model_size_bytes=size_b,
        parameter_count=param_count,
        mechanistic_backend=backend,
        vs_benchmark=compare_to_benchmark(mean_or_last[STAGE_MECHANISTIC_SOLVER]),
        history=history,
    )
