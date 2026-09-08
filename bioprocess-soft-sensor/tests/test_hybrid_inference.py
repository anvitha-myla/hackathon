"""Prompt 8: sequential hybrid inference — causality, no reference biomass, hybrid math."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from typing import Any

import pytest

from src.inference.components import MechanisticView, strip_reference_biomass
from src.inference.pipeline import SequentialHybridEngine
from src.inference.result import (
    HybridInferenceResult,
    attach_reference_for_evaluation,
    combine_hybrid,
)
from src.inference.state import InferenceState
from src.inference.trust import PlaceholderTrustHook, TrustAssessment


class RecordingCleaner:
    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self.payloads: list[dict[str, Any]] = []

    def step(self, data: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, Any], None]:
        del state
        self.timestamps.append(float(data["timestamp"]))
        self.payloads.append(dict(data))
        return dict(data), None


class RecordingFeatures:
    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self.payloads: list[dict[str, Any]] = []

    def step(self, data: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, float], None]:
        del state
        self.timestamps.append(float(data["timestamp"]))
        self.payloads.append(dict(data))
        out = {str(k): float(v) for k, v in data.items() if _is_number(v)}
        out.setdefault("F", 1.0)
        out.setdefault("DO", 80.0)
        return out, None


class CarryMechanistic:
    """X_{t} = X_{t-1} + increment. Proves sequential state, not a full ODE."""

    def __init__(self, increment: float = 0.25, X0: float = 1.0) -> None:
        self.increment = increment
        self.X0 = X0
        self._x = X0
        self.dts: list[float] = []
        self.payloads: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._x = self.X0
        self.dts.clear()
        self.payloads.clear()

    def step(self, dt: float, observables: Mapping[str, Any], state: InferenceState) -> MechanisticView:
        self.dts.append(float(dt))
        self.payloads.append(dict(observables))
        if state.n_steps == 0:
            self._x = self.X0
        else:
            self._x = float(state.X_mechanistic) + self.increment
        return MechanisticView(
            X_mechanistic=self._x,
            solver_status="ok",
            rates={"mu": 0.05},
            extras={"mu": 0.05},
        )


class ConstResidual:
    def __init__(self, delta: float) -> None:
        self.delta = delta
        self.calls: list[dict[str, Any]] = []

    def predict_delta(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        phase_indicators: Mapping[str, float],
    ) -> float:
        self.calls.append(
            {
                "features": dict(features),
                "X_mechanistic": X_mechanistic,
                "phase_indicators": dict(phase_indicators),
            }
        )
        return float(self.delta)


class ScriptedTrustHook:
    def __init__(self, beta: float, distance: float = 3.0) -> None:
        self.beta = beta
        self.distance = distance
        self.n_calls = 0

    def assess(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        delta_X_raw: float,
        state: InferenceState,
    ) -> TrustAssessment:
        del features, X_mechanistic, delta_X_raw, state
        self.n_calls += 1
        return TrustAssessment(beta_trust=self.beta, ood_distance=self.distance)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _obs(t: float, **extra: Any) -> dict[str, Any]:
    row = {
        "batch_id": "batch-1",
        "timestamp": t,
        "pH": 6.5,
        "DO": 80.0,
        "F": 0.1,
        "T_vessel": 25.0,
        "T_jacket": 20.0,
    }
    row.update(extra)
    return row


def _engine(**kwargs: Any) -> SequentialHybridEngine:
    return SequentialHybridEngine(
        cleaner=kwargs.pop("cleaner", RecordingCleaner()),
        feature_engine=kwargs.pop("feature_engine", RecordingFeatures()),
        mechanistic=kwargs.pop("mechanistic", CarryMechanistic()),
        residual=kwargs.pop("residual", ConstResidual(0.0)),
        trust_hook=kwargs.pop("trust_hook", PlaceholderTrustHook()),
        **kwargs,
    )


def test_step_signature_has_no_reference_biomass() -> None:
    params = inspect.signature(SequentialHybridEngine.step).parameters
    assert list(params) == ["self", "observation"]
    assert "X_reference" not in params
    assert "reference_biomass" not in params
    assert "biomass" not in params


def test_reference_biomass_rejected_as_kwarg() -> None:
    engine = _engine()
    with pytest.raises(TypeError):
        engine.step(_obs(0.0), X_reference=4.2)  # type: ignore[call-arg]


def test_reference_biomass_stripped_and_not_forwarded() -> None:
    cleaner = RecordingCleaner()
    features = RecordingFeatures()
    mech = CarryMechanistic()
    residual = ConstResidual(0.1)
    engine = SequentialHybridEngine(
        cleaner=cleaner,
        feature_engine=features,
        mechanistic=mech,
        residual=residual,
    )
    result = engine.step(_obs(0.0, X_reference=12.3, reference_biomass=12.3, Offline_Biomass_X_offline=9.9))
    assert not hasattr(result, "X_reference") or "X_reference" not in result.as_dict()
    assert "X_reference" not in result.as_dict()
    for payload in cleaner.payloads + features.payloads + mech.payloads:
        assert "X_reference" not in payload
        assert "reference_biomass" not in payload
        assert "Offline_Biomass_X_offline" not in payload
    for call in residual.calls:
        assert "X_reference" not in call["features"]
        assert "reference_biomass" not in call["features"]


def test_evaluation_may_attach_reference_afterward() -> None:
    engine = _engine(residual=ConstResidual(0.2))
    result = engine.step(_obs(0.0))
    attached = attach_reference_for_evaluation(result, 5.5)
    assert attached["X_reference"] == 5.5
    assert "X_reference" not in result.as_dict()


def test_future_data_not_required_generator() -> None:
    cleaner = RecordingCleaner()
    engine = _engine(cleaner=cleaner)

    def only_present() -> Any:
        yield _obs(0.0)
        yield _obs(1.0)

    results = engine.run_sequential(only_present())
    assert len(results) == 2
    assert cleaner.timestamps == [0.0, 1.0]
    assert [r.timestamp for r in results] == [0.0, 1.0]


def test_future_row_never_seen_when_withheld() -> None:
    cleaner = RecordingCleaner()
    features = RecordingFeatures()
    future_batch = [_obs(0.0), _obs(1.0), _obs(2.0)]
    engine = _engine(cleaner=cleaner, feature_engine=features)
    engine.step(future_batch[0])
    engine.step(future_batch[1])
    assert 2.0 not in cleaner.timestamps
    assert 2.0 not in features.timestamps
    assert engine.state.last_timestamp == 1.0


def test_step_rejects_full_trajectory_object() -> None:
    engine = _engine()
    with pytest.raises(TypeError):
        engine.step([_obs(0.0), _obs(1.0)])  # type: ignore[arg-type]


def test_state_carries_mechanistic_x_from_t_to_t1() -> None:
    mech = CarryMechanistic(increment=0.5, X0=2.0)
    engine = _engine(mechanistic=mech)
    r0 = engine.step(_obs(0.0))
    r1 = engine.step(_obs(1.0))
    r2 = engine.step(_obs(2.0))
    assert r0.X_mechanistic == pytest.approx(2.0)
    assert r1.X_mechanistic == pytest.approx(2.5)
    assert r2.X_mechanistic == pytest.approx(3.0)
    assert engine.state.X_mechanistic == pytest.approx(3.0)
    assert engine.state.n_steps == 3
    assert engine.state.last_timestamp == 2.0
    assert mech.dts[0] == pytest.approx(0.0)
    assert mech.dts[1] == pytest.approx(1.0)


def test_hybrid_equation_beta_one() -> None:
    engine = _engine(mechanistic=CarryMechanistic(X0=3.0, increment=0.0), residual=ConstResidual(0.4))
    result = engine.step(_obs(0.0))
    assert result.beta_trust == pytest.approx(1.0)
    assert result.delta_X_raw == pytest.approx(0.4)
    assert result.delta_X_applied == pytest.approx(0.4)
    assert result.X_hybrid == pytest.approx(3.4)
    assert result.X_hybrid == pytest.approx(result.X_mechanistic + result.delta_X_raw)
    assert result.physics_contribution == pytest.approx(3.0)
    assert result.ml_correction == pytest.approx(0.4)


def test_hybrid_equation_with_trust_hook_without_pipeline_rewrite() -> None:
    hook = ScriptedTrustHook(beta=0.5, distance=4.2)
    engine = _engine(
        mechanistic=CarryMechanistic(X0=10.0, increment=0.0),
        residual=ConstResidual(-2.0),
        trust_hook=hook,
    )
    result = engine.step(_obs(0.0))
    assert hook.n_calls == 1
    expected, applied = combine_hybrid(10.0, -2.0, 0.5)
    assert result.beta_trust == pytest.approx(0.5)
    assert result.ood_distance == pytest.approx(4.2)
    assert result.delta_X_applied == pytest.approx(applied)
    assert result.X_hybrid == pytest.approx(expected)
    assert result.X_hybrid == pytest.approx(10.0 + 0.5 * (-2.0))


def test_placeholder_trust_leaves_ood_nan() -> None:
    engine = _engine()
    result = engine.step(_obs(0.0))
    assert result.beta_trust == pytest.approx(1.0)
    assert math.isnan(result.ood_distance)


def test_result_fields_from_spec() -> None:
    engine = _engine()
    result = engine.step(_obs(1.5))
    assert isinstance(result, HybridInferenceResult)
    data = result.as_dict()
    for key in (
        "batch_id",
        "timestamp",
        "X_mechanistic",
        "delta_X_raw",
        "beta_trust",
        "delta_X_applied",
        "X_hybrid",
        "phase_indicators",
        "ood_distance",
        "latency",
        "solver_status",
    ):
        assert key in data
    assert result.batch_id == "batch-1"
    assert result.timestamp == pytest.approx(1.5)
    assert result.latency >= 0.0
    assert "growth" in result.phase_indicators
    assert "production" in result.phase_indicators
    assert "autolysis" in result.phase_indicators
    assert result.solver_status == "ok"


def test_strip_reference_helper() -> None:
    cleaned = strip_reference_biomass({"pH": 6.5, "X_reference": 1.0, "biomass_ref": 2.0, "DO": 40.0})
    assert cleaned == {"pH": 6.5, "DO": 40.0}


def test_default_engine_binds_without_future_or_reference() -> None:
    """Live bind to real mechanistic if present; still one-step causal."""
    engine = SequentialHybridEngine()
    r0 = engine.step(_obs(0.0))
    r1 = engine.step(_obs(1.0))
    assert r1.timestamp == pytest.approx(1.0)
    assert engine.state.n_steps == 2
    assert "X_reference" not in r0.as_dict()
    assert r0.X_hybrid == pytest.approx(r0.X_mechanistic + r0.beta_trust * r0.delta_X_raw)
