"""Adapters for cleaning / features / mechanistic / residual.

Wires to sibling modules when they expose a sequential ``step`` / residual
``predict`` API. Otherwise uses causal stubs so the hybrid engine can run
before Prompts 3–4 (and untrained residual weights) are finished.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from src.inference.state import InferenceState

REFERENCE_BIOMASS_KEYS = frozenset(
    {
        "x_reference",
        "x_ref",
        "reference_biomass",
        "biomass_reference",
        "biomass_ref",
        "offline_biomass",
        "x_offline",
        "penicillin_biomass",
        "x_true",
        "true_biomass",
        "ground_truth_biomass",
    }
)

# Keys the mechanistic live step rejects (see mechanistic config).
MECHANISTIC_FORBIDDEN = frozenset(
    {
        "X_reference",
        "x_reference",
        "biomass_reference",
        "reference_biomass",
        "Penicillin_biomass",
        "biomass",
    }
)

FEED_ALIASES = (
    "F",
    "feed_rate",
    "feed",
    "Fs",
    "sugar_feed_rate",
    "Sugar feed rate(Fs)",
    "Sugar feed rate",
    "Sugar feed rate(Fs:L/h)",
)
DO_ALIASES = (
    "DO",
    "do",
    "dissolved_oxygen",
    "Dissolved oxygen concentration(DO)",
    "Dissolved oxygen concentration",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "DO2",
)
VOLUME_ALIASES = (
    "V",
    "volume",
    "Volume(V)",
    "Vessel Volume(V)",
    "Vessel Volume(V:L)",
    "reactor_volume",
    "vessel_volume",
)
TIMESTAMP_ALIASES = (
    "timestamp",
    "timestamp_h",
    "t",
    "Time (h)",
    "Time(h)",
    "time",
    "relative_time_h",
    "time_h",
)
BATCH_ALIASES = ("batch_id", "Batch ID", "batch")

DEFAULT_PHASE = {"growth": 1.0, "production": 0.0, "autolysis": 0.0}


def _norm_key(key: str) -> str:
    return str(key).strip().lower().replace(" ", "_")


def is_reference_biomass_key(key: str) -> bool:
    k = _norm_key(key)
    if k in REFERENCE_BIOMASS_KEYS:
        return True
    if "reference" in k and "biomass" in k:
        return True
    if "offline" in k and "biomass" in k:
        return True
    if "x_offline" in k:
        return True
    return False


def strip_reference_biomass(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if not is_reference_biomass_key(str(k))}


def lookup_alias(payload: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    lower = {str(k).lower(): v for k, v in payload.items()}
    for name in aliases:
        if name in payload and payload[name] is not None:
            return payload[name]
        if name.lower() in lower and lower[name.lower()] is not None:
            return lower[name.lower()]
    return None


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def extract_timestamp(observation: Mapping[str, Any]) -> float:
    raw = lookup_alias(observation, TIMESTAMP_ALIASES)
    t = as_float(raw, None)
    if t is None:
        raise ValueError("Observation must include a finite timestamp")
    return t


def extract_batch_id(observation: Mapping[str, Any], fallback: str | None) -> str | None:
    raw = lookup_alias(observation, BATCH_ALIASES)
    if raw is None:
        return fallback
    return str(raw)


def _unwrap_step(out: Any) -> tuple[Any, Any]:
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, None


@runtime_checkable
class SequentialComponent(Protocol):
    def step(self, data: Mapping[str, Any], state: InferenceState) -> Any: ...


@runtime_checkable
class ResidualPredictor(Protocol):
    def predict_delta(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        phase_indicators: Mapping[str, float],
    ) -> float: ...


@runtime_checkable
class MechanisticPredictor(Protocol):
    def step(
        self,
        dt: float,
        observables: Mapping[str, Any],
        state: InferenceState,
    ) -> "MechanisticView": ...

    def reset(self) -> None: ...


@dataclass
class MechanisticView:
    X_mechanistic: float
    solver_status: str
    rates: dict[str, float]
    extras: dict[str, Any]


class IdentityCausalCleaner:
    """Pass-through with causal forward-fill of NaNs from previous cleaned values."""

    def step(self, data: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, Any], Any]:
        cleaned = dict(data)
        prev = state.last_cleaned
        for key, value in list(cleaned.items()):
            if key in TIMESTAMP_ALIASES or key in BATCH_ALIASES:
                continue
            num = as_float(value, None)
            if value is None or (isinstance(value, float) and not math.isfinite(value)) or (
                num is None and value != value  # NaN
            ):
                if key in prev:
                    cleaned[key] = prev[key]
        return cleaned, None


class CausalFeatureStub:
    """Minimal causal features until Prompt 4 publishes a registry."""

    def step(self, cleaned: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, float], Any]:
        features: dict[str, float] = {}
        for key, value in cleaned.items():
            if is_reference_biomass_key(str(key)):
                continue
            num = as_float(value, None)
            if num is not None:
                features[str(key)] = num
        feed = as_float(lookup_alias(cleaned, FEED_ALIASES), 0.0) or 0.0
        do = as_float(lookup_alias(cleaned, DO_ALIASES), None)
        if do is not None:
            features["DO"] = do
        features["F"] = feed
        features["feed_rate"] = feed
        t_vessel = as_float(cleaned.get("T_vessel", cleaned.get("vessel_temperature")), None)
        t_jacket = as_float(cleaned.get("T_jacket", cleaned.get("jacket_temperature")), None)
        if t_vessel is not None:
            features["T_vessel"] = t_vessel
        if t_jacket is not None:
            features["T_jacket"] = t_jacket
        if t_vessel is not None and t_jacket is not None:
            features["delta_T"] = t_vessel - t_jacket
        dt = 0.0
        ts = as_float(lookup_alias(cleaned, TIMESTAMP_ALIASES), None)
        if ts is not None and state.last_timestamp is not None:
            dt = max(ts - state.last_timestamp, 0.0)
        cum = float(state.accumulated.get("cumulative_feed", 0.0)) + feed * dt
        features["cumulative_feed"] = cum
        vol = as_float(lookup_alias(cleaned, VOLUME_ALIASES), None)
        if vol is not None:
            features["volume"] = vol
            features["V"] = vol
        progress = ts if ts is not None else float(state.n_steps)
        features["progress"] = progress
        return features, {"cumulative_feed": cum}


def default_phase_indicators(timestamp: float, features: Mapping[str, float]) -> dict[str, float]:
    """Soft phase scores from causal time/progress only (no future batch length)."""
    t = float(features.get("progress", timestamp))
    # Fixed logistic gates; not batch-end DTW.
    growth = 1.0 / (1.0 + math.exp(t - 40.0))
    autolysis = 1.0 / (1.0 + math.exp(-(t - 180.0)))
    production = max(0.0, 1.0 - growth - autolysis)
    total = growth + production + autolysis
    if total <= 0.0:
        return dict(DEFAULT_PHASE)
    return {
        "growth": growth / total,
        "production": production / total,
        "autolysis": autolysis / total,
        "phase_growth": growth / total,
        "phase_production": production / total,
        "phase_autolysis": autolysis / total,
    }


class HoldMechanisticStub:
    """Constant X carry when the reduced-order model is not importable."""

    def __init__(self, X0: float = 0.15) -> None:
        self._X0 = float(X0)
        self._X = float(X0)

    def reset(self) -> None:
        self._X = self._X0

    def step(
        self,
        dt: float,
        observables: Mapping[str, Any],
        state: InferenceState,
    ) -> MechanisticView:
        del dt, observables
        x = float(state.X_mechanistic) if state.n_steps else self._X
        self._X = x
        return MechanisticView(
            X_mechanistic=x,
            solver_status="stub",
            rates={"mu": 0.0},
            extras={},
        )


class ReducedMechanisticAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def step(
        self,
        dt: float,
        observables: Mapping[str, Any],
        state: InferenceState,
    ) -> MechanisticView:
        del state
        payload = {
            k: v
            for k, v in observables.items()
            if str(k) not in MECHANISTIC_FORBIDDEN and not is_reference_biomass_key(str(k))
        }
        f = as_float(lookup_alias(payload, FEED_ALIASES), None)
        do = as_float(lookup_alias(payload, DO_ALIASES), None)
        if f is None or do is None:
            x = float(getattr(self.model, "X_mechanistic", 0.0))
            return MechanisticView(
                X_mechanistic=x,
                solver_status="missing_inputs",
                rates={},
                extras={},
            )
        live = {"F": f, "DO": do}
        vol = as_float(lookup_alias(payload, VOLUME_ALIASES), None)
        if vol is not None:
            live["V"] = vol
        sf = as_float(payload.get("S_f"), None)
        if sf is not None:
            live["S_f"] = sf
        result = self.model.step(float(dt), live)
        rates = dict(getattr(result, "rates", {}) or {})
        status = str(getattr(result, "solver_status", "unknown"))
        x = float(getattr(result, "X_mechanistic"))
        extras = {
            "S": getattr(result, "S", None),
            "V": getattr(result, "V", None),
            "P": getattr(result, "P", None),
            "mu": rates.get("mu"),
        }
        return MechanisticView(
            X_mechanistic=x,
            solver_status=status,
            rates=rates,
            extras=extras,
        )


class ZeroResidualStub:
    def predict_delta(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        phase_indicators: Mapping[str, float],
    ) -> float:
        del features, X_mechanistic, phase_indicators
        return 0.0


class ResidualMLPAdapter:
    def __init__(self, model: Any, scaler: Any, feature_order: list[str]) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_order = list(feature_order)

    def predict_delta(
        self,
        features: Mapping[str, float],
        *,
        X_mechanistic: float,
        phase_indicators: Mapping[str, float],
    ) -> float:
        import numpy as np

        from src.models.residual_nn import predict_delta

        merged = dict(features)
        merged["X_mechanistic"] = float(X_mechanistic)
        merged["mu_mechanistic"] = float(features.get("mu_mechanistic", 0.0))
        for key, value in phase_indicators.items():
            merged[str(key)] = float(value)
        row = [float(merged.get(name, 0.0)) for name in self.feature_order]
        pred = predict_delta(self.model, self.scaler, np.asarray([row], dtype=np.float64))
        return float(pred[0])


def _module_attr(module: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        obj = getattr(module, name, None)
        if obj is not None:
            return obj
    return None


class SequentialCleanerAdapter:
    """Apply ``clean_series`` to past+current samples only (no future rows, no regrid)."""

    def __init__(self) -> None:
        from src.data.cleaning import load_cleaning_params

        self.params = load_cleaning_params()
        self._history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._history = []

    def step(self, data: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, Any], Any]:
        del state
        from src.data.cleaning import clean_series, infer_step_minutes

        self._history.append(dict(data))
        import pandas as pd

        frame = pd.DataFrame(self._history)
        out = dict(data)
        times = None
        time_col = None
        for alias in TIMESTAMP_ALIASES:
            if alias in frame.columns:
                time_col = alias
                break
        if time_col is not None:
            times = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)
        try:
            step_minutes = infer_step_minutes(times, numeric_unit="hours") if times is not None else 60.0
        except Exception:
            step_minutes = 60.0
        identity = set(self.params.identity_columns) | set(TIMESTAMP_ALIASES) | set(BATCH_ALIASES)
        for col in frame.columns:
            if col in identity or is_reference_biomass_key(str(col)):
                continue
            try:
                raw = frame[col].to_numpy(dtype=float)
            except (TypeError, ValueError):
                continue
            series = clean_series(raw, self.params, step_minutes=step_minutes, column=str(col))
            out[col] = float(series.cleaned[-1])
            out[f"{col}_quality"] = int(series.quality[-1])
        return out, {"n_history": len(self._history)}


class SequentialFeatureAdapter:
    """Causal FeatureEngine: transform only the accumulated past+current rows."""

    def __init__(self, engine: Any | None = None) -> None:
        if engine is None:
            from src.features.feature_engine import FeatureEngine

            engine = FeatureEngine.from_config()
        self.engine = engine
        self._history: list[dict[str, Any]] = []
        self._fallback = CausalFeatureStub()

    def reset(self) -> None:
        self._history = []

    def step(self, cleaned: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, float], Any]:
        import pandas as pd

        self._history.append(dict(cleaned))
        try:
            frame = pd.DataFrame(self._history)
            out_df = self.engine.transform(frame)
            last = out_df.iloc[-1]
            features: dict[str, float] = {}
            for key, value in last.items():
                num = as_float(value, None)
                if num is not None:
                    features[str(key)] = num
            feed = as_float(lookup_alias(cleaned, FEED_ALIASES), None)
            do = as_float(lookup_alias(cleaned, DO_ALIASES), None)
            if feed is not None:
                features.setdefault("F", feed)
                features.setdefault("feed_rate", feed)
            if do is not None:
                features.setdefault("DO", do)
            return features, {"n_history": len(self._history)}
        except Exception:
            return self._fallback.step(cleaned, state)

    def phase_indicators(self, features: Mapping[str, float], state: InferenceState) -> dict[str, float]:
        del state
        g = as_float(features.get("phase_growth"), None)
        p = as_float(features.get("phase_production"), None)
        a = as_float(features.get("phase_autolysis"), None)
        if g is None or p is None or a is None:
            return default_phase_indicators(float(features.get("progress", 0.0)), features)
        return {
            "growth": g,
            "production": p,
            "autolysis": a,
            "phase_growth": g,
            "phase_production": p,
            "phase_autolysis": a,
        }


def bind_cleaner() -> Any:
    try:
        from src.data import cleaning as cleaning_mod
    except Exception:
        return IdentityCausalCleaner()
    cls = _module_attr(
        cleaning_mod,
        ("CausalCleaner", "CleaningEngine", "CausalCleaningPipeline", "Cleaner"),
    )
    if cls is not None:
        try:
            inst = cls() if isinstance(cls, type) else cls
            if hasattr(inst, "step"):
                return inst
        except Exception:
            pass
    if callable(getattr(cleaning_mod, "step", None)):
        return _FnCleaner(cleaning_mod.step)
    if callable(getattr(cleaning_mod, "clean_series", None)):
        try:
            return SequentialCleanerAdapter()
        except Exception:
            pass
    return IdentityCausalCleaner()


def bind_feature_engine() -> Any:
    try:
        from src.features import engineering as feat_mod
    except Exception:
        return CausalFeatureStub()
    cls = _module_attr(
        feat_mod,
        ("CausalFeatureEngine", "FeatureEngine", "FeatureEngineering"),
    )
    if cls is not None:
        try:
            inst = cls() if isinstance(cls, type) else cls
            if hasattr(inst, "step"):
                return inst
            return SequentialFeatureAdapter(inst if not isinstance(cls, type) else inst)
        except Exception:
            pass
    if callable(getattr(feat_mod, "step", None)):
        return _FnFeatures(feat_mod.step)
    return CausalFeatureStub()


def bind_mechanistic() -> Any:
    try:
        from src.mechanistic.model import ReducedMechanisticModel

        return ReducedMechanisticAdapter(ReducedMechanisticModel())
    except Exception:
        return HoldMechanisticStub()


def bind_residual() -> Any:
    try:
        from src.models.residual_nn import default_artifact_dir, load_residual_artifacts
    except Exception:
        return ZeroResidualStub()
    artifact_dir = default_artifact_dir()
    if not Path(artifact_dir).is_dir():
        return ZeroResidualStub()
    required = (
        Path(artifact_dir) / "residual_weights.pt",
        Path(artifact_dir) / "scaler.joblib",
        Path(artifact_dir) / "feature_order.json",
    )
    if not all(p.is_file() for p in required):
        return ZeroResidualStub()
    try:
        fit = load_residual_artifacts(artifact_dir)
        return ResidualMLPAdapter(fit.model, fit.scaler, fit.feature_order)
    except Exception:
        return ZeroResidualStub()


class _FnCleaner:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def step(self, data: Mapping[str, Any], state: InferenceState) -> Any:
        return self.fn(data, state)


class _FnFeatures:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def step(self, data: Mapping[str, Any], state: InferenceState) -> Any:
        return self.fn(data, state)


def invoke_clean(cleaner: Any, data: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, Any], Any]:
    if hasattr(cleaner, "step"):
        out, extra = _unwrap_step(cleaner.step(data, state))
    else:
        out, extra = dict(data), None
    if not isinstance(out, Mapping):
        raise TypeError("Cleaner.step must return a mapping (or (mapping, state))")
    return dict(out), extra


def invoke_features(engine: Any, data: Mapping[str, Any], state: InferenceState) -> tuple[dict[str, float], Any]:
    if hasattr(engine, "step"):
        out, extra = _unwrap_step(engine.step(data, state))
    else:
        out, extra = CausalFeatureStub().step(data, state)
    if isinstance(out, tuple) and len(out) >= 1:
        out = out[0]
    if not isinstance(out, Mapping):
        raise TypeError("Feature engine must return a mapping")
    features: dict[str, float] = {}
    for key, value in out.items():
        num = as_float(value, None)
        if num is not None:
            features[str(key)] = num
    return features, extra
