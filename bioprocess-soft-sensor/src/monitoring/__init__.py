"""Mahalanobis OOD and computational monitoring (Prompt 9, 12)."""

from src.monitoring.system import (
    MonitoringReport,
    count_parameters,
    model_size_bytes,
    run_monitored_inference_loop,
    snapshot_resources,
)
from src.monitoring.timing import (
    MECHANISTIC_LOOP_BENCHMARK_TARGET_S,
    TimingAccumulator,
    compare_to_benchmark,
    measure_stage,
    time_callable,
)

__version__ = "0.0.0"

__all__ = [
    "MECHANISTIC_LOOP_BENCHMARK_TARGET_S",
    "MonitoringReport",
    "TimingAccumulator",
    "compare_to_benchmark",
    "count_parameters",
    "measure_stage",
    "model_size_bytes",
    "run_monitored_inference_loop",
    "snapshot_resources",
    "time_callable",
]
