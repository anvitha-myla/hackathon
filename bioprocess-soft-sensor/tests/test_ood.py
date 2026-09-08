"""Mahalanobis OOD, beta_trust, and sequential hybrid combine tests (Prompt 9)."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_default
from src.inference.components import MechanisticView
from src.inference.result import combine_hybrid as combine_hybrid_tuple
from src.inference.ood import (
    FALLBACK_PHYSICS,
    OOD_IN_DISTRIBUTION,
    OOD_INVALID,
    OOD_MODERATE,
    OOD_SEVERE,
    MahalanobisOOD,
    apply_hybrid,
    beta_trust_from_distance,
)
from src.inference.pipeline import SequentialHybridEngine
from src.inference.state import InferenceState
from src.inference.trust import MahalanobisTrustHook
from src.monitoring import ood as monitoring_ood


def _fitted_isotropic(n: int = 400, d: int = 4, seed: int = 0) -> tuple[MahalanobisOOD, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = rng.normal(loc=0.0, scale=1.0, size=(n, d))
    det = MahalanobisOOD(kappa=1.0)
    det.fit(train, split="train")
    return det, train


def _point_at_distance(det: MahalanobisOOD, target_dm: float) -> np.ndarray:
    """Construct phi with D_M ≈ target_dm using the stored precision matrix."""
    assert det.precision_ is not None and det.mean_ is not None
    evals, evecs = np.linalg.eigh(det.precision_)
    direction = evecs[:, -1]
    lam = max(float(evals[-1]), 1e-12)
    scale = target_dm / np.sqrt(lam)
    return det.mean_ + scale * direction


def test_monitoring_ood_is_inference_ood() -> None:
    assert monitoring_ood.MahalanobisOOD is MahalanobisOOD
    assert monitoring_ood.apply_hybrid is apply_hybrid
    from src.inference.ood import beta_trust

    assert monitoring_ood.beta_trust is beta_trust


def test_config_method_is_mahalanobis() -> None:
    cfg = load_default()
    assert cfg["ood"]["method"] == "mahalanobis"
    assert "exp(-kappa * max(0, D_M - D_threshold))" in cfg["ood"]["beta_trust"]


def test_beta_formula_and_bounds() -> None:
    assert beta_trust_from_distance(0.0, kappa=1.0, d_threshold=2.0) == pytest.approx(1.0)
    assert beta_trust_from_distance(2.0, kappa=1.0, d_threshold=2.0) == pytest.approx(1.0)
    mid = beta_trust_from_distance(3.0, kappa=1.0, d_threshold=2.0)
    assert mid == pytest.approx(np.exp(-1.0))
    assert 0.0 < mid < 1.0
    far = beta_trust_from_distance(50.0, kappa=1.0, d_threshold=2.0)
    assert 0.0 <= far < 1e-10
    nan_beta = beta_trust_from_distance(float("nan"), kappa=1.0, d_threshold=2.0)
    assert nan_beta == 0.0


def test_hybrid_equation() -> None:
    full = apply_hybrid(10.0, 5.0, 1.0)
    assert full.X_hybrid == pytest.approx(15.0)
    assert full.ml_correction == pytest.approx(5.0)
    none = apply_hybrid(10.0, 5.0, 0.0)
    assert none.X_hybrid == pytest.approx(10.0)
    half_x, half_applied = combine_hybrid_tuple(10.0, 4.0, 0.5)
    assert half_x == pytest.approx(12.0)
    assert half_applied == pytest.approx(2.0)
    clipped = apply_hybrid(1.0, 1.0, 2.0)
    assert clipped.beta_trust == 1.0
    assert clipped.X_hybrid == pytest.approx(2.0)


def test_normal_input_retains_ml_correction() -> None:
    det, train = _fitted_isotropic()
    phi = train.mean(axis=0)
    score = det.score(phi)
    assert score.valid
    assert score.d_m <= score.d_threshold
    assert score.beta_trust == pytest.approx(1.0)
    assert score.ood_state == OOD_IN_DISTRIBUTION
    hybrid, _ = det.hybrid_at(x_mechanistic=2.0, delta_x_pred=3.0, phi=phi)
    assert hybrid.beta_trust == pytest.approx(1.0)
    assert hybrid.X_hybrid == pytest.approx(5.0)
    assert 0.0 <= score.beta_trust <= 1.0


def test_moderate_ood_attenuates() -> None:
    det, _ = _fitted_isotropic()
    phi = _point_at_distance(det, det.d_threshold + 0.75)
    score = det.score(phi)
    assert score.valid
    assert score.d_m > score.d_threshold
    assert score.ood_state == OOD_MODERATE
    assert 0.0 < score.beta_trust < 1.0
    hybrid, _ = det.hybrid_at(x_mechanistic=8.0, delta_x_pred=10.0, phi=phi)
    assert hybrid.X_hybrid == pytest.approx(8.0 + score.beta_trust * 10.0)
    assert abs(hybrid.ml_correction) < abs(10.0)


def test_severe_ood_falls_back_to_physics() -> None:
    det, _ = _fitted_isotropic()
    phi = _point_at_distance(det, det.d_threshold + 20.0)
    score = det.score(phi)
    assert score.valid
    assert score.ood_state == OOD_SEVERE
    assert score.fallback_state == FALLBACK_PHYSICS
    assert score.beta_trust < 1e-3
    hybrid, _ = det.hybrid_at(x_mechanistic=4.0, delta_x_pred=100.0, phi=phi)
    assert hybrid.X_hybrid == pytest.approx(4.0, abs=0.2)
    assert abs(hybrid.ml_correction) < 0.2


def test_covariance_singularity_is_stable() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(80, 1))
    noise = rng.normal(scale=0.01, size=(80, 1))
    # Rank-deficient: columns 0,1,2 are copies of the same latent
    train = np.hstack([base, base, base, noise])
    det = MahalanobisOOD(kappa=1.0)
    det.fit(train, split="train")
    assert det.is_fitted
    assert np.isfinite(det.covariance_).all()
    assert np.isfinite(det.precision_).all()
    score = det.score(train[0])
    assert score.valid
    assert np.isfinite(score.d_m)
    assert 0.0 <= score.beta_trust <= 1.0
    # Duplicate-column query must not raise
    far = np.array([100.0, 100.0, 100.0, 0.0])
    far_score = det.score(far)
    assert far_score.beta_trust <= 1.0
    assert np.isfinite(far_score.d_m) or not far_score.valid


def test_nan_and_invalid_features_fallback() -> None:
    det, _ = _fitted_isotropic()
    nan_score = det.score([np.nan, 0.0, 0.0, 0.0])
    assert nan_score.valid is False
    assert nan_score.ood_state == OOD_INVALID
    assert nan_score.beta_trust == 0.0
    assert nan_score.fallback_state == FALLBACK_PHYSICS
    inf_score = det.score([np.inf, 0.0, 0.0, 0.0])
    assert inf_score.beta_trust == 0.0
    wrong_dim = det.score([0.0, 0.0])
    assert wrong_dim.valid is False
    assert wrong_dim.beta_trust == 0.0
    hybrid, ood = det.hybrid_at(
        x_mechanistic=7.0,
        delta_x_pred=9.0,
        phi=[np.nan, 1.0, 1.0, 1.0],
    )
    assert ood.beta_trust == 0.0
    assert hybrid.X_hybrid == pytest.approx(7.0)


def test_fit_rejects_test_split() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 3))
    det = MahalanobisOOD()
    with pytest.raises(ValueError, match="training data only"):
        det.fit(x, split="test")
    with pytest.raises(ValueError, match="training data only"):
        det.fit(x, split="validation")


def test_fit_ignores_non_train_batches() -> None:
    rng = np.random.default_rng(5)
    train = rng.normal(loc=0.0, size=(50, 3))
    test = rng.normal(loc=80.0, size=(50, 3))
    stacked = np.vstack([train, test])
    batch_ids = ["tr"] * 50 + ["te"] * 50
    det_filtered = MahalanobisOOD(kappa=1.0).fit(
        stacked,
        split="train",
        batch_ids=batch_ids,
        train_batch_ids=["tr"],
    )
    det_train = MahalanobisOOD(kappa=1.0).fit(train, split="train")
    assert np.allclose(det_filtered.mean_, det_train.mean_)
    assert np.allclose(det_filtered.covariance_, det_train.covariance_)
    mean_before = det_filtered.mean_.copy()
    det_filtered.score(test[0])
    assert np.allclose(det_filtered.mean_, mean_before)


def test_sequential_hybrid_step_does_not_need_future() -> None:
    """MahalanobisTrustHook plugs into SequentialHybridEngine.step without future rows."""
    det, train = _fitted_isotropic()
    names = ("f0", "f1", "f2", "f3")
    det.feature_names_ = names
    phi_t = train[0]
    phi_t1 = train[1]
    phi_t2 = _point_at_distance(det, float(det.d_threshold) + 15.0)

    class _PhiFeatures:
        def __init__(self) -> None:
            self._queue = [phi_t, phi_t1, phi_t2]

        def step(self, data: dict, state: InferenceState) -> tuple[dict[str, float], None]:
            del state
            phi = self._queue.pop(0)
            out = {names[i]: float(phi[i]) for i in range(4)}
            out["F"] = 0.1
            out["DO"] = 80.0
            out["timestamp"] = float(data["timestamp"])
            return out, None

    class _HoldMech:
        def reset(self) -> None:
            return None

        def step(self, dt: float, observables: dict, state: InferenceState) -> MechanisticView:
            del dt, observables, state
            return MechanisticView(X_mechanistic=4.0, solver_status="ok", rates={}, extras={})

    class _ConstResidual:
        def predict_delta(self, features, *, X_mechanistic, phase_indicators) -> float:
            del features, X_mechanistic, phase_indicators
            return 10.0

    class _PassClean:
        def step(self, data, state):
            del state
            return dict(data), None

    engine = SequentialHybridEngine(
        cleaner=_PassClean(),
        feature_engine=_PhiFeatures(),
        mechanistic=_HoldMech(),
        residual=_ConstResidual(),
        trust_hook=MahalanobisTrustHook(det, feature_order=names),
        batch_id="b1",
    )
    r0 = engine.step({"batch_id": "b1", "timestamp": 0.0, "DO": 80.0, "F": 0.1})
    r1 = engine.step({"batch_id": "b1", "timestamp": 1.0, "DO": 80.0, "F": 0.1})
    r2 = engine.step({"batch_id": "b1", "timestamp": 2.0, "DO": 80.0, "F": 0.1})
    assert r0.timestamp == pytest.approx(0.0)
    assert r1.timestamp == pytest.approx(1.0)
    assert r2.timestamp == pytest.approx(2.0)
    assert r0.X_hybrid == pytest.approx(r0.X_mechanistic + r0.beta_trust * r0.delta_X_raw)
    assert r1.X_hybrid == pytest.approx(r1.X_mechanistic + r1.beta_trust * r1.delta_X_raw)
    assert r2.X_hybrid == pytest.approx(r2.X_mechanistic + r2.beta_trust * r2.delta_X_raw)
    assert r2.beta_trust < r0.beta_trust
    assert 0.0 <= r0.beta_trust <= 1.0
    assert 0.0 <= r1.beta_trust <= 1.0
    assert 0.0 <= r2.beta_trust <= 1.0
    assert r2.X_hybrid == pytest.approx(r2.X_mechanistic, abs=0.3)
