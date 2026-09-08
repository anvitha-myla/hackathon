"""Sequential hybrid inference engine (Prompt 8).

raw telemetry → causal cleaning → causal features → mechanistic model
→ phase indicators → residual NN → trust hook → hybrid prediction

One timestamp per ``step``. Future rows are never required.
Reference biomass is stripped and never forwarded to any live stage.

OOD: inject ``MahalanobisTrustHook`` (Prompt 9). Default remains
``PlaceholderTrustHook`` (beta_trust=1.0) so unfitted OOD cannot leak. The
hybrid equation is always X_hybrid = X_mechanistic + beta_trust * delta_X_pred.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

from src.inference.components import (
    bind_cleaner,
    bind_feature_engine,
    bind_mechanistic,
    bind_residual,
    default_phase_indicators,
    extract_batch_id,
    extract_timestamp,
    invoke_clean,
    invoke_features,
    is_reference_biomass_key,
    strip_reference_biomass,
)
from src.inference.result import HybridInferenceResult, combine_hybrid, result_from_parts
from src.inference.state import InferenceState
from src.inference.trust import (
    MahalanobisTrustHook,
    PlaceholderTrustHook,
    TrustAssessment,
    TrustHook,
    clamp_beta,
)


class SequentialHybridEngine:
    """Causal, one-sample hybrid estimator."""

    def __init__(
        self,
        *,
        cleaner: Any | None = None,
        feature_engine: Any | None = None,
        mechanistic: Any | None = None,
        residual: Any | None = None,
        trust_hook: TrustHook | None = None,
        batch_id: str | None = None,
    ) -> None:
        self.cleaner = cleaner if cleaner is not None else bind_cleaner()
        self.feature_engine = feature_engine if feature_engine is not None else bind_feature_engine()
        self.mechanistic = mechanistic if mechanistic is not None else bind_mechanistic()
        self.residual = residual if residual is not None else bind_residual()
        self.trust_hook: TrustHook = trust_hook if trust_hook is not None else PlaceholderTrustHook()
        self.state = InferenceState(batch_id=batch_id)

    def reset(self, batch_id: str | None = None) -> None:
        self.state.reset(batch_id=batch_id)
        reset_m = getattr(self.mechanistic, "reset", None)
        if callable(reset_m):
            reset_m()
        for component in (self.cleaner, self.feature_engine, self.residual):
            reset_c = getattr(component, "reset", None)
            if callable(reset_c):
                reset_c()

    def step(self, observation: Mapping[str, Any]) -> HybridInferenceResult:
        """Advance one timestamp. ``observation`` must not be a future window.

        Reference biomass keys are dropped. Do not pass X_reference as an argument.
        """
        if not isinstance(observation, Mapping):
            raise TypeError("step() takes a single-timestamp mapping, not a batch trajectory")
        t0 = time.perf_counter()

        raw = strip_reference_biomass(observation)
        leaked = [k for k in observation if is_reference_biomass_key(str(k))]
        if leaked:
            # Dropped, never forwarded. Do not raise: callers may hold mixed eval rows.
            pass

        timestamp = extract_timestamp(raw)
        batch_id = extract_batch_id(raw, self.state.batch_id)
        if self.state.batch_id is None:
            self.state.batch_id = batch_id
        elif batch_id is not None and batch_id != self.state.batch_id:
            raise ValueError(
                f"batch_id changed from {self.state.batch_id!r} to {batch_id!r}; "
                "call reset() for a new batch"
            )

        if self.state.last_timestamp is not None and timestamp < self.state.last_timestamp:
            raise ValueError("Timestamps must be non-decreasing for causal sequential inference")

        cleaned, clean_state = invoke_clean(self.cleaner, raw, self.state)
        cleaned = strip_reference_biomass(cleaned)
        if clean_state is not None:
            self.state.cleaner_state = clean_state

        features, feat_state = invoke_features(self.feature_engine, cleaned, self.state)
        if feat_state is not None:
            self.state.feature_state = feat_state
            if isinstance(feat_state, Mapping) and "cumulative_feed" in feat_state:
                self.state.accumulated["cumulative_feed"] = float(feat_state["cumulative_feed"])
        elif "cumulative_feed" in features:
            self.state.accumulated["cumulative_feed"] = float(features["cumulative_feed"])

        dt = 0.0
        if self.state.last_timestamp is not None:
            dt = max(timestamp - self.state.last_timestamp, 0.0)
        else:
            mech_t = getattr(self.mechanistic, "model", None)
            t_prev = getattr(mech_t, "t", None) if mech_t is not None else None
            if t_prev is not None:
                dt = max(timestamp - float(t_prev), 0.0)

        mech_obs = {**cleaned, **features}
        mech_view = self.mechanistic.step(dt, mech_obs, self.state)
        x_mech = float(mech_view.X_mechanistic)
        solver_status = str(mech_view.solver_status)
        rates = dict(getattr(mech_view, "rates", {}) or {})
        extras = dict(getattr(mech_view, "extras", {}) or {})
        if extras.get("mu") is not None:
            features["mu_mechanistic"] = float(extras["mu"])
        elif "mu" in rates:
            features["mu_mechanistic"] = float(rates["mu"])
        features["X_mechanistic"] = x_mech
        if extras.get("S") is not None:
            features["S_mechanistic"] = float(extras["S"])
        if extras.get("V") is not None:
            features["volume"] = float(extras["V"])
        if extras.get("P") is not None:
            features["P_mechanistic"] = float(extras["P"])
        if "product_rate" in rates:
            features["product_rate_mechanistic"] = float(rates["product_rate"])

        phase = default_phase_indicators(timestamp, features)
        phase_fn = getattr(self.feature_engine, "phase_indicators", None)
        if callable(phase_fn):
            custom = phase_fn(features, self.state)
            if isinstance(custom, Mapping):
                phase = {str(k): float(v) for k, v in custom.items()}
        self.state.last_phase_indicators = dict(phase)

        delta_raw = float(
            self.residual.predict_delta(
                features,
                X_mechanistic=x_mech,
                phase_indicators=phase,
            )
        )

        assessment = self.trust_hook.assess(
            features,
            X_mechanistic=x_mech,
            delta_X_raw=delta_raw,
            state=self.state,
        )
        if not isinstance(assessment, TrustAssessment):
            raise TypeError("trust_hook.assess must return TrustAssessment")
        beta = clamp_beta(float(assessment.beta_trust))
        ood = float(assessment.ood_distance)

        latency = time.perf_counter() - t0
        result = result_from_parts(
            batch_id=self.state.batch_id,
            timestamp=timestamp,
            X_mechanistic=x_mech,
            delta_X_raw=delta_raw,
            beta_trust=beta,
            phase_indicators=phase,
            ood_distance=ood,
            latency=latency,
            solver_status=solver_status,
        )

        self.state.last_timestamp = timestamp
        self.state.n_steps += 1
        self.state.X_mechanistic = x_mech
        self.state.last_cleaned = dict(cleaned)
        self.state.last_features = dict(features)
        self.state.last_delta_X_raw = delta_raw
        self.state.last_solver_status = solver_status
        return result

    def run_sequential(
        self,
        observations: Iterable[Mapping[str, Any]],
    ) -> list[HybridInferenceResult]:
        """Iterate ``step`` only. Does not prefetch future items from a sequence."""
        results: list[HybridInferenceResult] = []
        for observation in observations:
            results.append(self.step(observation))
        return results


# Public aliases
HybridInferenceEngine = SequentialHybridEngine

__all__ = [
    "HybridInferenceEngine",
    "MahalanobisTrustHook",
    "PlaceholderTrustHook",
    "SequentialHybridEngine",
    "combine_hybrid",
]
