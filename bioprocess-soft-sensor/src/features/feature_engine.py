"""Causal feature engine for IndPenSim batches.

Definitions are fixed in ``src.features.registry``. This module only evaluates
those formulas on available columns. No future timestamps, no full-batch DTW,
no reference biomass in the live feature vector, and no ML training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import load_yaml

try:
    from src.data.schema import (
        IDENTITY_BATCH_ID,
        IDENTITY_RELATIVE_TIME,
        IDENTITY_TIMESTAMP,
        map_roles,
    )
except ImportError:  # Prompt 1 schema may land in parallel
    IDENTITY_BATCH_ID = "batch_id"
    IDENTITY_TIMESTAMP = "timestamp_h"
    IDENTITY_RELATIVE_TIME = "relative_time_h"

    def map_roles(columns):  # type: ignore[misc]
        return {}

from src.features.derivatives import (
    backward_derivative,
    causal_cumulative_trapz,
    causal_lag,
    causal_rolling_mean,
    causal_rolling_slope,
    causal_rolling_std,
)
from src.features.registry import (
    FEATURE_REGISTRY,
    unimplemented_due_to_missing_columns,
)
from src.features.stoichiometry import (
    RQ_OUR_EPSILON,
    dataset_our_cer_to_mol_h,
    offgas_our_cer_mol_h,
    respiratory_quotient,
)

# Published Goldrick 100-batch headers and common loader aliases.
# Matching is exact after normalization (lowercase, strip punctuation).
_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "time": (
        "time (h)",
        "time",
        "time_h",
        "timestamp",
        "timestamp_h",
        "relative_time",
        "relative_time_h",
        "t",
    ),
    "batch_id": (
        "batch id",
        "batch_id",
        "batch reference(batch_ref:batch ref)",
        "batch_ref",
    ),
    "aeration_rate": (
        "aeration rate(fg:l/h)",
        "aeration rate",
        "fg",
        "aeration_rate",
        "gas_flow",
    ),
    "agitator_rpm": (
        "agitator rpm(rpm:rpm)",
        "agitator rpm",
        "rpm",
        "agitator_rpm",
    ),
    "sugar_feed_rate": (
        "sugar feed rate(fs:l/h)",
        "sugar feed rate",
        "fs",
        "sugar_feed_rate",
        "feed_rate",
        "substrate_feed_rate",
    ),
    "acid_flow_rate": ("acid flow rate(fa:l/h)", "acid flow rate", "fa", "acid_flow_rate"),
    "base_flow_rate": ("base flow rate(fb:l/h)", "base flow rate", "fb", "base_flow_rate"),
    "cooling_water_flow_rate": (
        "heating/cooling water flow rate(fc:l/h)",
        "heating/cooling water flow rate",
        "fc",
        "cooling_water_flow_rate",
    ),
    "heating_water_flow_rate": (
        "heating water flow rate(fh:l/h)",
        "heating water flow rate",
        "fh",
        "heating_water_flow_rate",
    ),
    "wfi_flow_rate": (
        "water for injection/dilution(fw:l/h)",
        "water for injection/dilution",
        "fw",
        "wfi_flow_rate",
    ),
    "air_head_pressure": (
        "air head pressure(pressure:bar)",
        "air head pressure",
        "pressure",
        "air_head_pressure",
    ),
    "dumped_broth_flow": (
        "dumped broth flow(fremoved:l/h)",
        "dumped broth flow",
        "fremoved",
        "dumped_broth_flow",
    ),
    "do": (
        "dissolved oxygen concentration(do2:mg/l)",
        "dissolved oxygen concentration",
        "do2",
        "do",
        "dissolved_oxygen",
    ),
    "vessel_volume": (
        "vessel volume(v:l)",
        "vessel volume",
        "v",
        "vessel_volume",
        "volume",
    ),
    "ph": ("ph(ph:ph)", "ph"),
    "temperature": (
        "temperature(t:k)",
        "temperature",
        "vessel_temperature",
        "t_vessel",
    ),
    "generated_heat": ("generated heat(q:kj)", "generated heat", "q", "generated_heat"),
    "co2_offgas": (
        "carbon dioxide percent in off-gas(co2outgas:%)",
        "carbon dioxide percent in off-gas(co2offgas:%)",
        "carbon dioxide percent in off-gas(co2:%)",
        "carbon dioxide percent in off-gas",
        "co2_offgas",
        "outlet_co2",
    ),
    "o2_offgas": (
        "oxygen in percent in off-gas(o2:o2  (%))",
        "oxygen percent in off-gas(o2:%)",
        "oxygen in percent in off-gas(o2:o2 (%))",
        "oxygen percent in off-gas",
        "oxygen in percent in off-gas",
        "o2_offgas",
        "outlet_o2",
    ),
    "oil_flow": (
        "oil flow(foil:l/hr)",
        "oil flow(foil:l/h)",
        "oil flow",
        "foil",
        "oil_flow",
    ),
    "our_dataset": (
        "oxygen uptake rate(our:(g min^{-1}))",
        "oxygen uptake rate(our)",
        "oxygen uptake rate(our:g min−1)",
        "oxygen uptake rate(our:g/min)",
        "oxygen uptake rate",
        "our_dataset",
        "vourx",
    ),
    "cer_dataset": (
        "carbon evolution rate(cer)",
        "carbon evolution rate(cer:g/h)",
        "carbon evolution rate",
        "cer_dataset",
        "vcerx",
    ),
    "paa_flow": ("paa flow(fpaa:paa/h)", "paa flow", "fpaa", "paa_flow"),
    "biomass": (
        "offline biomass concentratio(x_offline:x(g l^{-1}))",
        "offline biomass concentration(x_offline:x(g l^{-1}))",
        "biomass concentration(x:g/l)",
        "offline biomass concentration(xoffline:g/l)",
        "biomass",
        "x",
        "x_reference",
    ),
    "penicillin": (
        "penicillin concentration(p:g/l)",
        "penicillin",
        "p",
    ),
    "substrate": (
        "substrate concentration(s:g/l)",
        "substrate",
        "s",
    ),
}

_HIDDEN_CANONICAL = frozenset({"biomass", "penicillin", "substrate"})
_RAMAN_NAME = re.compile(r"^\d{2,3}$")

_UNIT_BY_BASE = {
    "do": "mg/L",
    "ph": "pH",
    "temperature": "K",
    "our": "mol/h",
    "cer": "mol/h",
    "sugar_feed_rate": "L/h",
    "agitator_rpm": "RPM",
}


def _norm(name: str) -> str:
    s = str(name).lower().strip()
    s = s.replace("−", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _header_is_percent(name: str) -> bool | None:
    if not name:
        return None
    lower = name.lower()
    if "%" in name or "percent" in lower:
        return True
    return None


def _build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in _CANONICAL_ALIASES.items():
        lookup[_norm(canonical)] = canonical
        for alias in aliases:
            lookup[_norm(alias)] = canonical
    return lookup


_ALIAS_LOOKUP = _build_alias_lookup()

# Loader schema roles (src.data.schema.ROLE_ALIASES) → engine canonical names.
_SCHEMA_ROLE_TO_CANONICAL: dict[str, str] = {
    "time": "time",
    "aeration": "aeration_rate",
    "agitation": "agitator_rpm",
    "sugar_feed": "sugar_feed_rate",
    "acid_flow": "acid_flow_rate",
    "base_flow": "base_flow_rate",
    "heating_cooling_water": "cooling_water_flow_rate",
    "heating_water": "heating_water_flow_rate",
    "dilution_water": "wfi_flow_rate",
    "air_head_pressure": "air_head_pressure",
    "dumped_broth": "dumped_broth_flow",
    "dissolved_oxygen": "do",
    "vessel_volume": "vessel_volume",
    "ph": "ph",
    "vessel_temperature": "temperature",
    "outlet_co2": "co2_offgas",
    "outlet_o2": "o2_offgas",
    "oil_flow": "oil_flow",
    "our": "our_dataset",
    "cer": "cer_dataset",
    "biomass_reference": "biomass",
    "penicillin": "penicillin",
    "substrate": "substrate",
}


def resolve_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map canonical names to the first matching frame column (original spelling).

    Prefers standardized identity fields from the Prompt 1 loader, then
    ``src.data.schema.map_roles``, then published-header aliases.
    """
    cols = list(columns)
    resolved: dict[str, str] = {}
    if IDENTITY_BATCH_ID in cols:
        resolved["batch_id"] = IDENTITY_BATCH_ID
    if IDENTITY_TIMESTAMP in cols:
        resolved["time"] = IDENTITY_TIMESTAMP
    elif IDENTITY_RELATIVE_TIME in cols:
        resolved["time"] = IDENTITY_RELATIVE_TIME
    for role, source_col in map_roles(cols).items():
        canonical = _SCHEMA_ROLE_TO_CANONICAL.get(role)
        if canonical is None or canonical in resolved:
            continue
        resolved[canonical] = source_col
    for col in cols:
        canonical = _ALIAS_LOOKUP.get(_norm(col))
        if canonical is None or canonical in resolved:
            continue
        resolved[canonical] = col
    return resolved


def _series(frame: pd.DataFrame, col: str | None, n: int) -> np.ndarray:
    if col is None or col not in frame.columns:
        return np.full(n, np.nan, dtype=float)
    return np.asarray(frame[col], dtype=float)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    exp[~np.isfinite(exp)] = 0.0
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp / denom


def load_feature_config() -> dict[str, Any]:
    try:
        return load_yaml("features.yaml")
    except FileNotFoundError:
        return {}


@dataclass
class FeatureEngine:
    """Evaluate the fixed registry on one or more batches."""

    lags: tuple[int, ...] = (1, 5, 10)
    rolling_windows: tuple[int, ...] = (5, 10)
    rq_our_epsilon: float = RQ_OUR_EPSILON
    default_dt_h: float = 0.2
    history_signals: tuple[str, ...] = (
        "do",
        "ph",
        "temperature",
        "our",
        "cer",
        "sugar_feed_rate",
        "agitator_rpm",
    )
    y_o2_in: float = 0.2095
    y_co2_in: float = 0.0004
    r_l_bar_mol_k: float = 0.08314462618
    t_production_h: float = 80.0
    t_autolysis_h: float = 180.0
    our_scale_mol_h: float = 50.0
    last_history_names: tuple[str, ...] = field(default_factory=tuple, init=False)
    last_skipped: tuple[str, ...] = field(default_factory=tuple, init=False)
    last_our_source: str = field(default="unavailable", init=False)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> FeatureEngine:
        if cfg is None:
            cfg = load_feature_config()
        air = cfg.get("air") or {}
        phase = cfg.get("phase") or {}
        return cls(
            lags=tuple(int(x) for x in cfg.get("lags", (1, 5, 10))),
            rolling_windows=tuple(int(x) for x in cfg.get("rolling_windows", (5, 10))),
            rq_our_epsilon=float(cfg.get("rq_our_epsilon", RQ_OUR_EPSILON)),
            default_dt_h=float(cfg.get("default_dt_h", 0.2)),
            history_signals=tuple(cfg.get("history_signals") or cls.history_signals),
            y_o2_in=float(air.get("y_o2_in", 0.2095)),
            y_co2_in=float(air.get("y_co2_in", 0.0004)),
            r_l_bar_mol_k=float(air.get("r_l_bar_mol_k", 0.08314462618)),
            t_production_h=float(phase.get("t_production_h", 80.0)),
            t_autolysis_h=float(phase.get("t_autolysis_h", 180.0)),
            our_scale_mol_h=float(phase.get("our_scale_mol_h", 50.0)),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        raman_cols = [c for c in frame.columns if _RAMAN_NAME.match(str(c).strip())]
        work = frame.drop(columns=raman_cols, errors="ignore")
        mapping = resolve_columns(work.columns)
        batch_col = mapping.get("batch_id")
        if batch_col is None:
            return self._transform_one(work, mapping)
        pieces: list[pd.DataFrame] = []
        for _, group in work.groupby(batch_col, sort=False):
            pieces.append(self._transform_one(group, mapping))
        return pd.concat(pieces, axis=0).sort_index()

    def _transform_one(
        self,
        frame: pd.DataFrame,
        mapping: dict[str, str],
    ) -> pd.DataFrame:
        n = len(frame)
        time_col = mapping.get("time")
        if time_col is not None:
            time_h = np.asarray(frame[time_col], dtype=float)
        else:
            time_h = np.arange(n, dtype=float) * self.default_dt_h
        order = np.argsort(time_h, kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        sorted_frame = frame.iloc[order]
        time_sorted = time_h[order]
        elapsed = time_sorted - time_sorted[0] if n else time_sorted

        def col(canonical: str) -> np.ndarray:
            return _series(sorted_frame, mapping.get(canonical), n)

        signals: dict[str, np.ndarray] = {
            "elapsed_time_h": elapsed,
            "ph": col("ph"),
            "do": col("do"),
            "temperature": col("temperature"),
            "agitator_rpm": col("agitator_rpm"),
            "sugar_feed_rate": col("sugar_feed_rate"),
            "aeration_rate": col("aeration_rate"),
            "o2_offgas": col("o2_offgas"),
            "co2_offgas": col("co2_offgas"),
        }

        our, cer, our_source = self._our_cer(col, mapping, n)
        signals["our"] = our
        signals["cer"] = cer
        signals["rq"] = respiratory_quotient(cer, our, epsilon=self.rq_our_epsilon)
        signals["d_do_dt"] = backward_derivative(signals["do"], time_sorted)

        sugar = col("sugar_feed_rate")
        if mapping.get("sugar_feed_rate") is not None:
            signals["cumulative_sugar_feed"] = causal_cumulative_trapz(sugar, time_sorted)
            signals["progress_coordinate"] = signals["cumulative_sugar_feed"].copy()
        else:
            signals["progress_coordinate"] = elapsed.copy()

        if mapping.get("oil_flow") is not None:
            signals["cumulative_oil_feed"] = causal_cumulative_trapz(
                col("oil_flow"), time_sorted
            )

        if mapping.get("vessel_volume") is not None:
            signals["reactor_volume"] = col("vessel_volume")

        inflow_keys = (
            "sugar_feed_rate",
            "oil_flow",
            "wfi_flow_rate",
            "acid_flow_rate",
            "base_flow_rate",
        )
        if any(mapping.get(k) is not None for k in inflow_keys + ("dumped_broth_flow",)):
            net = np.zeros(n, dtype=float)
            for key in inflow_keys:
                if mapping.get(key) is not None:
                    part = col(key)
                    part = np.where(np.isfinite(part), part, 0.0)
                    net = net + part
            if mapping.get("dumped_broth_flow") is not None:
                dumped = col("dumped_broth_flow")
                dumped = np.where(np.isfinite(dumped), dumped, 0.0)
                net = net - dumped
            delta_v = causal_cumulative_trapz(net, time_sorted)
            v0 = signals["reactor_volume"][0] if "reactor_volume" in signals else 0.0
            if not np.isfinite(v0):
                v0 = 0.0
            signals["reactor_volume_integrated"] = v0 + delta_v

        growth, production, autolysis = self._soft_phases(our, elapsed)
        signals["phase_growth"] = growth
        signals["phase_production"] = production
        signals["phase_autolysis"] = autolysis

        history_names: list[str] = []
        for base in self.history_signals:
            series = signals.get(base)
            if series is None or not np.isfinite(series).any():
                continue
            unit = _UNIT_BY_BASE.get(base, "1")
            for lag in self.lags:
                name = f"{base}_lag_{lag}"
                signals[name] = causal_lag(series, lag)
                history_names.append(name)
            for window in self.rolling_windows:
                mean_name = f"{base}_roll_mean_{window}"
                std_name = f"{base}_roll_std_{window}"
                slope_name = f"{base}_roll_slope_{window}"
                signals[mean_name] = causal_rolling_mean(series, window)
                signals[std_name] = causal_rolling_std(series, window)
                signals[slope_name] = causal_rolling_slope(series, time_sorted, window)
                history_names.extend([mean_name, std_name, slope_name])

        skipped = []
        for spec in FEATURE_REGISTRY:
            if spec.implemented and spec.name not in signals:
                if spec.name in {
                    "cumulative_oil_feed",
                    "reactor_volume",
                    "reactor_volume_integrated",
                    "cumulative_sugar_feed",
                }:
                    skipped.append(spec.name)
        self.last_history_names = tuple(history_names)
        self.last_skipped = tuple(skipped)
        self.last_our_source = our_source

        out = pd.DataFrame({name: values[inv] for name, values in signals.items()}, index=frame.index)
        if mapping.get("batch_id") is not None:
            out.insert(0, "batch_id", np.asarray(frame[mapping["batch_id"]]))
        out.insert(1 if "batch_id" in out.columns else 0, "time_h", time_h)
        out.attrs["our_source"] = our_source
        out.attrs["history_names"] = tuple(history_names)
        return out

    def _our_cer(self, col, mapping: dict[str, str], n: int) -> tuple[np.ndarray, np.ndarray, str]:
        o2 = col("o2_offgas")
        co2 = col("co2_offgas")
        fg = col("aeration_rate")
        have_offgas = np.isfinite(o2).any() and np.isfinite(co2).any() and np.isfinite(fg).any()
        if have_offgas:
            o2_name = mapping.get("o2_offgas") or ""
            co2_name = mapping.get("co2_offgas") or ""
            our, cer = offgas_our_cer_mol_h(
                fg,
                o2,
                co2,
                temperature_k=col("temperature"),
                pressure_bar=col("air_head_pressure"),
                y_o2_in=self.y_o2_in,
                y_co2_in=self.y_co2_in,
                r_l_bar_mol_k=self.r_l_bar_mol_k,
                o2_as_percent=_header_is_percent(o2_name),
                co2_as_percent=_header_is_percent(co2_name),
            )
            return our, cer, "offgas_balance"
        our_ds = col("our_dataset")
        cer_ds = col("cer_dataset")
        if np.isfinite(our_ds).any() or np.isfinite(cer_ds).any():
            our, cer = dataset_our_cer_to_mol_h(our_ds, cer_ds)
            return our, cer, "dataset_our_cer"
        return (
            np.full(n, np.nan, dtype=float),
            np.full(n, np.nan, dtype=float),
            "unavailable",
        )

    def _soft_phases(
        self, our: np.ndarray, elapsed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        our_n = np.where(np.isfinite(our), our / self.our_scale_mol_h, 0.0)
        t_n = elapsed
        # Fixed formula: three logits, softmax → indicators sum to 1.
        z_growth = our_n - (t_n / self.t_production_h)
        z_prod = -((t_n - self.t_production_h) / 40.0) ** 2
        z_auto = (t_n / self.t_autolysis_h) - our_n
        scores = np.column_stack([z_growth, z_prod, z_auto])
        probs = _softmax(scores)
        return probs[:, 0], probs[:, 1], probs[:, 2]

    def live_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Column names that are valid live-inference inputs (no hidden refs)."""
        engineered = self.transform(frame)
        forbidden = set(_HIDDEN_CANONICAL)
        return [c for c in engineered.columns if c not in forbidden and c not in {"batch_id"}]


def engineer_batch(frame: pd.DataFrame, engine: FeatureEngine | None = None) -> pd.DataFrame:
    """Public entry point. Source of truth for Prompt 4 feature engineering."""
    if engine is None:
        engine = FeatureEngine.from_config()
    return engine.transform(frame)


def missing_column_report() -> list[dict[str, str]]:
    return [
        {
            "feature": spec.name,
            "reason": spec.source,
            "formula": spec.formula,
        }
        for spec in unimplemented_due_to_missing_columns()
    ]


def assert_no_hidden_targets(columns: Iterable[str]) -> None:
    """Fail if biomass / penicillin / substrate leaked into the live feature table."""
    emitted = set(columns) & _HIDDEN_CANONICAL
    if emitted:
        raise AssertionError(f"Hidden reference channels in live features: {sorted(emitted)}")
