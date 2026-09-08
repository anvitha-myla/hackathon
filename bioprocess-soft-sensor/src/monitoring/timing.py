"""Stage latency measurement via time.perf_counter (Prompt 12).

Values are measured at runtime. Do not treat the sub-150 ms mechanistic-loop
figure as an achieved latency; it is only a benchmark *target* to compare
against after measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, TypeVar

import time

# Spec 11 / master spec: proposed execution target to *measure against*, not a claim.
MECHANISTIC_LOOP_BENCHMARK_TARGET_S = 0.150

STAGE_CLEANING = "cleaning"
STAGE_FEATURE_ENGINEERING = "feature_engineering"
STAGE_MECHANISTIC_SOLVER = "mechanistic_solver"
STAGE_NN = "nn"
STAGE_OOD = "ood"
STAGE_TOTAL_INFERENCE = "total_inference"

PIPELINE_STAGES: tuple[str, ...] = (
    STAGE_CLEANING,
    STAGE_FEATURE_ENGINEERING,
    STAGE_MECHANISTIC_SOLVER,
    STAGE_NN,
    STAGE_OOD,
    STAGE_TOTAL_INFERENCE,
)

T = TypeVar("T")


@dataclass
class StageTiming:
    """One timed call. ``elapsed_s`` comes from perf_counter, not a constant."""

    stage: str
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "elapsed_s": float(self.elapsed_s)}


@dataclass
class TimingAccumulator:
    """Session history of per-stage latencies (seconds)."""

    records: list[StageTiming] = field(default_factory=list)

    def add(self, stage: str, elapsed_s: float) -> StageTiming:
        rec = StageTiming(stage=stage, elapsed_s=float(elapsed_s))
        self.records.append(rec)
        return rec

    def for_stage(self, stage: str) -> list[float]:
        return [r.elapsed_s for r in self.records if r.stage == stage]

    def last(self, stage: str) -> float | None:
        vals = self.for_stage(stage)
        return vals[-1] if vals else None

    def mean(self, stage: str) -> float | None:
        vals = self.for_stage(stage)
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def as_history(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.records]


@contextmanager
def measure_stage(stage: str, accumulator: TimingAccumulator | None = None) -> Iterator[list[float]]:
    """Context manager: wall time of the enclosed block in seconds."""
    slot: list[float] = []
    t0 = time.perf_counter()
    try:
        yield slot
    finally:
        elapsed = time.perf_counter() - t0
        slot.append(elapsed)
        if accumulator is not None:
            accumulator.add(stage, elapsed)


def time_callable(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)`` from perf_counter."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def compare_to_benchmark(
    measured_s: float,
    *,
    target_s: float = MECHANISTIC_LOOP_BENCHMARK_TARGET_S,
) -> dict[str, Any]:
    """Report measured latency vs the spec target. Does not claim the target is met."""
    measured = float(measured_s)
    target = float(target_s)
    return {
        "measured_s": measured,
        "benchmark_target_s": target,
        "below_target": measured < target,
        "claim": "measurement_only",
    }


def latencies_are_valid(values: Mapping[str, float] | list[float]) -> bool:
    seq = values.values() if isinstance(values, Mapping) else values
    return all(_finite_nonneg(v) for v in seq)


def _finite_nonneg(v: float) -> bool:
    x = float(v)
    return x == x and x != float("inf") and x != float("-inf") and x >= 0.0
