"""Tests for the reduced-order sequential mechanistic model (Prompt 5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.mechanistic.config import (
    MechanisticConfig,
    load_mechanistic_config,
    parse_mechanistic_config,
)
from src.mechanistic.equations import specific_growth_rate
from src.mechanistic.model import ReducedMechanisticModel
from src.mechanistic.solver import integrate_step


def _obs(F: float = 0.05, DO: float = 80.0, **extra: float) -> dict[str, float]:
    out = {"F": F, "DO": DO}
    out.update(extra)
    return out


def test_config_forbids_full_indpensim() -> None:
    with pytest.raises(ValueError, match="IndPenSim"):
        parse_mechanistic_config({"use_full_indpensim_equations": True})


def test_config_forbids_reference_ic_flag() -> None:
    with pytest.raises(ValueError, match="reference biomass"):
        parse_mechanistic_config({"initialize_from_reference_biomass": True})


def test_yaml_defaults_are_reduced_order() -> None:
    cfg = load_mechanistic_config()
    assert cfg.family == "reduced_order_monod"
    assert cfg.use_full_indpensim_equations is False
    assert cfg.primary_estimate == "biomass"
    assert cfg.initialize_from_reference_biomass is False
    assert cfg.solver.method in ("BDF", "Radau")


def test_monod_limits() -> None:
    cfg = load_mechanistic_config()
    p = cfg.parameters
    mu_hi = specific_growth_rate(1e9, 1e9, p)
    assert mu_hi == pytest.approx(p.mu_max, rel=1e-3)
    mu_zero_s = specific_growth_rate(0.0, 100.0, p)
    mu_zero_do = specific_growth_rate(10.0, 0.0, p)
    assert mu_zero_s == pytest.approx(0.0)
    assert mu_zero_do == pytest.approx(0.0)


def test_sequential_operation_no_future() -> None:
    model = ReducedMechanisticModel.from_yaml()
    x0 = model.X_mechanistic
    results = []
    for k in range(8):
        # Only current observables; no remaining trajectory.
        results.append(model.step(0.5, _obs(F=0.08, DO=70.0 + k)))
    assert len(results) == 8
    assert results[-1].t == pytest.approx(4.0)
    assert results[-1].X_mechanistic == pytest.approx(model.X_mechanistic)
    assert results[-1].X_mechanistic > x0
    assert all(r.solver_status == "success" for r in results)
    assert all(r.solver_latency >= 0.0 for r in results)
    assert all("mu" in r.rates and "growth" in r.rates for r in results)
    # Clock is strictly sequential.
    times = [r.t for r in results]
    assert times == sorted(times)
    assert np.all(np.diff(times) == pytest.approx(0.5))


def test_run_sequential_helper() -> None:
    model = ReducedMechanisticModel.from_yaml()
    steps = [_obs(F=0.02, DO=90.0) for _ in range(5)]
    out = model.run_sequential(0.2, steps)
    assert len(out) == 5
    assert model.t == pytest.approx(1.0)


def test_numerical_stability_long_horizon() -> None:
    model = ReducedMechanisticModel.from_yaml()
    last = None
    for _ in range(200):
        last = model.step(0.25, _obs(F=0.04, DO=60.0))
    assert last is not None
    assert last.success, last.solver_status
    assert last.solver_status == "success"
    assert last.S >= -1e-8
    y = model.state_vector
    assert np.all(np.isfinite(y))
    assert np.all(y >= -1e-12)
    assert last.X_mechanistic < model.config.stability.max_X


def test_negative_states_detected_and_flagged() -> None:
    model = ReducedMechanisticModel.from_yaml()
    model.reset(X=-2.0, S=-1.0, V=100.0, P=0.0)
    result = model.step(0.1, _obs())
    assert "negative_state" in result.solver_status
    assert result.success is False
    assert model.X_mechanistic >= 0.0
    assert model.state_vector[1] >= 0.0


def test_nan_inputs_reported() -> None:
    model = ReducedMechanisticModel.from_yaml()
    result = model.step(0.1, _obs(F=np.nan, DO=80.0))
    assert result.success is False
    assert "nan" in result.solver_status or "failed" in result.solver_status


def test_nan_state_reported() -> None:
    model = ReducedMechanisticModel.from_yaml()
    model._y[0] = np.nan
    result = model.step(0.1, _obs())
    assert result.success is False
    assert "nan" in result.solver_status


def test_solver_failure_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.mechanistic import solver as solver_mod

    def boom(*_a, **_k):
        raise RuntimeError("forced integrator crash")

    monkeypatch.setattr(solver_mod, "solve_ivp", boom)
    model = ReducedMechanisticModel.from_yaml()
    result = model.step(0.5, _obs())
    assert result.success is False
    assert "failed" in result.solver_status
    assert "forced integrator crash" in result.solver_message


def test_solver_success_false_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from src.mechanistic import solver as solver_mod

    def fake_ivp(*_a, **_k):
        y0 = np.array([0.15, 15.0, 100.0, 0.0])
        return SimpleNamespace(
            success=False,
            message="mock BDF failure",
            y=np.stack([y0, y0], axis=1),
            t=np.array([0.0, 0.1]),
            nfev=3,
        )

    monkeypatch.setattr(solver_mod, "solve_ivp", fake_ivp)
    model = ReducedMechanisticModel.from_yaml()
    result = model.step(0.1, _obs())
    assert result.success is False
    assert "failed" in result.solver_status
    assert result.solver_latency >= 0.0


def test_negative_dt_is_solver_failure() -> None:
    model = ReducedMechanisticModel.from_yaml()
    t0 = model.t
    x0 = model.X_mechanistic
    result = model.step(-1.0, _obs())
    assert result.success is False
    assert result.solver_status == "failed"
    assert model.t == pytest.approx(t0)
    assert model.X_mechanistic == pytest.approx(x0)


def test_live_step_rejects_reference_biomass() -> None:
    model = ReducedMechanisticModel.from_yaml()
    with pytest.raises(ValueError, match="Reference"):
        model.step(0.1, {"F": 0.1, "DO": 80.0, "X_reference": 12.3})
    with pytest.raises(ValueError, match="Reference"):
        model.step(0.1, F=0.1, DO=80.0, biomass_reference=1.0)
    with pytest.raises(ValueError, match="reference biomass"):
        model.reset(reference_biomass=5.0)


def test_measured_volume_hold() -> None:
    model = ReducedMechanisticModel.from_yaml()
    r = model.step(1.0, _obs(F=0.5, DO=80.0, V=120.0))
    assert r.V == pytest.approx(120.0)
    assert r.success


def test_oxygen_and_substrate_limitation_change_growth() -> None:
    cfg = load_mechanistic_config()
    p = cfg.parameters
    mu_rich = specific_growth_rate(20.0, 100.0, p)
    mu_poor_s = specific_growth_rate(0.05, 100.0, p)
    mu_poor_do = specific_growth_rate(20.0, 0.2, p)
    assert mu_poor_s < mu_rich
    assert mu_poor_do < mu_rich


def test_outputs_include_required_fields() -> None:
    model = ReducedMechanisticModel.from_yaml()
    r = model.step(0.2, _obs())
    d = r.as_dict()
    assert "X_mechanistic" in d
    assert "rates" in d
    assert "solver_status" in d
    assert "solver_latency" in d
    assert r.P is not None


def test_integrate_step_zero_dt() -> None:
    y0 = np.array([0.15, 15.0, 100.0, 0.0])
    out = integrate_step(lambda t, y: y, 0.0, y0, 0.0, method="BDF")
    assert out.status == "success"
    np.testing.assert_allclose(out.y, y0)


def test_radau_method_runs() -> None:
    cfg = load_mechanistic_config()
    cfg = replace(cfg, solver=replace(cfg.solver, method="Radau", fallback_method="BDF"))
    model = ReducedMechanisticModel(cfg)
    r = model.step(0.3, _obs())
    assert r.success
    assert r.method_used == "Radau"
