"""Prompt 12: computational monitoring uses real measurements, not canned numbers."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from src.models.mlp import SmallMLP
from src.monitoring.system import (
    count_parameters,
    model_size_bytes,
    run_monitored_inference_loop,
    snapshot_resources,
)
from src.monitoring.timing import (
    MECHANISTIC_LOOP_BENCHMARK_TARGET_S,
    compare_to_benchmark,
    time_callable,
)

LATENCY_KEYS = (
    "cleaning_latency_s",
    "feature_engineering_latency_s",
    "mechanistic_solver_latency_s",
    "nn_latency_s",
    "ood_latency_s",
    "total_inference_latency_s",
    "session_elapsed_s",
)


def _finite_nonneg(x: float) -> bool:
    return math.isfinite(x) and x >= 0.0


def test_time_callable_not_hardcoded_constant() -> None:
    def light() -> int:
        s = 0
        for i in range(200):
            s += i
        return s

    def heavy() -> int:
        s = 0
        for i in range(80_000):
            s += i * i
        return s

    _, t_light = time_callable(light)
    _, t_heavy = time_callable(heavy)
    assert _finite_nonneg(t_light)
    assert _finite_nonneg(t_heavy)
    assert t_light != MECHANISTIC_LOOP_BENCHMARK_TARGET_S
    assert t_heavy != MECHANISTIC_LOOP_BENCHMARK_TARGET_S
    assert t_heavy > t_light


def test_snapshot_resources_from_psutil() -> None:
    snap = snapshot_resources()
    for key in (
        "cpu_percent",
        "system_cpu_percent",
        "ram_used_bytes",
        "ram_percent",
        "rss_bytes",
    ):
        assert key in snap
        assert _finite_nonneg(float(snap[key]))
    assert snap["rss_bytes"] > 0.0
    assert snap["ram_used_bytes"] > 0.0


def test_parameter_count_from_actual_module() -> None:
    net = SmallMLP(n_in=15, n_out=1)
    counted = count_parameters(net)
    expected = sum(int(p.numel()) for p in net.parameters())
    assert counted == expected
    assert counted is not None and counted > 0
    # Different architecture must not share a hardcoded count.
    tiny = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    assert count_parameters(tiny) == sum(int(p.numel()) for p in tiny.parameters())
    assert count_parameters(tiny) != counted


def test_model_size_bytes_from_state_dict_and_file(tmp_path: Path) -> None:
    net = SmallMLP(n_in=15, n_out=1)
    size_mem = model_size_bytes(net)
    assert size_mem is not None and size_mem > 0
    path = tmp_path / "mlp.pt"
    torch.save(net.state_dict(), path)
    size_disk = model_size_bytes(path=path)
    assert size_disk == path.stat().st_size
    assert size_disk > 0


def test_monitored_loop_latencies_finite_nonnegative() -> None:
    report = run_monitored_inference_loop(n_steps=6, dt=0.2, seed=1)
    d = report.as_dict()
    for key in LATENCY_KEYS:
        assert _finite_nonneg(float(d[key])), key
    assert d["throughput_hz"] >= 0.0 and math.isfinite(d["throughput_hz"])
    expected_tp = d["n_samples"] / d["session_elapsed_s"]
    assert math.isclose(d["throughput_hz"], expected_tp, rel_tol=1e-9, abs_tol=1e-12)
    for key in LATENCY_KEYS:
        assert d[key] != MECHANISTIC_LOOP_BENCHMARK_TARGET_S
    assert len(report.history) == 6
    for row in report.history:
        for key in (
            "cleaning_latency_s",
            "feature_engineering_latency_s",
            "mechanistic_solver_latency_s",
            "nn_latency_s",
            "ood_latency_s",
            "total_inference_latency_s",
        ):
            assert _finite_nonneg(row[key])
            assert row[key] != MECHANISTIC_LOOP_BENCHMARK_TARGET_S
        assert row["rss_bytes"] > 0.0


def test_throughput_scales_with_sample_count_not_a_constant() -> None:
    a = run_monitored_inference_loop(n_steps=3, seed=2)
    b = run_monitored_inference_loop(n_steps=10, seed=2)
    assert a.n_samples == 3
    assert b.n_samples == 10
    assert a.throughput_hz != b.throughput_hz or a.session_elapsed_s != b.session_elapsed_s
    assert b.session_elapsed_s >= a.session_elapsed_s * 0.5


def test_parameter_count_on_loop_module() -> None:
    net = SmallMLP(n_in=15, n_out=1)
    report = run_monitored_inference_loop(n_steps=2, module=net, seed=0)
    assert report.parameter_count == count_parameters(net)
    assert report.parameter_count == sum(int(p.numel()) for p in net.parameters())
    assert report.model_size_bytes is not None and report.model_size_bytes > 0


def test_benchmark_helper_does_not_claim_target() -> None:
    cmp_ = compare_to_benchmark(0.2)
    assert cmp_["benchmark_target_s"] == MECHANISTIC_LOOP_BENCHMARK_TARGET_S
    assert cmp_["claim"] == "measurement_only"
    assert cmp_["measured_s"] == 0.2
    assert cmp_["below_target"] is False
