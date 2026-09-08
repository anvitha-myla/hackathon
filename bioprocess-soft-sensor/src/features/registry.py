"""Fixed feature definitions for the causal feature engine.

Each row is a definition (name, source, formula, unit, causal status, model
usage, missing-data behavior). Numerical values are produced at runtime by
``src.features.feature_engine`` and change over the batch; formulas do not.

IndPenSim column availability (Goldrick 100-batch Mendeley dump / published
headers) is recorded here so the engine does not invent sensors.

Not present in that dump (requested by the master spec, left unimplemented):
- jacket temperature — only heating/cooling *water flow* exists, not jacket T
- shaft torque — only agitator RPM exists (typically held at 100 RPM)
- evaporation flow as a measured column — not in the 100-batch headers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CausalStatus = Literal["causal", "unimplemented_missing_column"]
ModelUsage = Literal[
    "nn_only+residual_mlp",
    "residual_mlp",
    "analysis",
    "not_used",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    formula: str
    unit: str
    causal_status: CausalStatus
    model_usage: ModelUsage
    missing_data_behavior: str
    implemented: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def causal(self) -> bool:
        return self.causal_status == "causal"


# --- Implemented process / derived features ---------------------------------

FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="ph",
        source="pH(pH:pH)",
        formula="passthrough of measured pH",
        unit="pH",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent or sample missing",
        tags=("observable",),
    ),
    FeatureSpec(
        name="do",
        source="Dissolved oxygen concentration(DO2:mg/L)",
        formula="passthrough of measured dissolved oxygen",
        unit="mg/L",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent or sample missing",
        tags=("observable",),
    ),
    FeatureSpec(
        name="temperature",
        source="Temperature(T:K)",
        formula="passthrough of vessel temperature",
        unit="K",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent or sample missing",
        tags=("observable",),
    ),
    FeatureSpec(
        name="agitator_rpm",
        source="Agitator RPM(RPM:RPM)",
        formula="passthrough of agitator speed (not shaft torque)",
        unit="RPM",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent",
        tags=("observable",),
    ),
    FeatureSpec(
        name="sugar_feed_rate",
        source="Sugar feed rate(Fs:L/h)",
        formula="passthrough of sugar feed volumetric rate",
        unit="L/h",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent",
        tags=("observable",),
    ),
    FeatureSpec(
        name="aeration_rate",
        source="Aeration rate(Fg:L/h)",
        formula="passthrough of aeration / gas flow",
        unit="L/h",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent",
        tags=("observable",),
    ),
    FeatureSpec(
        name="o2_offgas",
        source="Oxygen in percent in off-gas(O2:O2  (%))",
        formula="passthrough of outlet O2 (percent or fraction)",
        unit="%",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent",
        tags=("observable",),
    ),
    FeatureSpec(
        name="co2_offgas",
        source="carbon dioxide percent in off-gas(CO2outgas:%)",
        formula="passthrough of outlet CO2 (percent or fraction)",
        unit="%",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if column absent",
        tags=("observable",),
    ),
    FeatureSpec(
        name="our",
        source="Fg, outlet O2, outlet CO2, T, P (inert-balance off-gas)",
        formula=(
            "OUR = n_in*yO2_in - n_out*yO2_out, "
            "n_out = n_in*(1-yO2_in-yCO2_in)/(1-yO2_out-yCO2_out), "
            "n_in = P*Fg/(R*T). Fallback: dataset OUR g/min → mol/h."
        ),
        unit="mol/h",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if neither off-gas nor dataset OUR is available",
        tags=("derived", "stoichiometry"),
    ),
    FeatureSpec(
        name="cer",
        source="Fg, outlet O2, outlet CO2, T, P (inert-balance off-gas)",
        formula=(
            "CER = n_out*yCO2_out - n_in*yCO2_in with the same n_in/n_out as OUR. "
            "Fallback: dataset CER g/h → mol/h."
        ),
        unit="mol/h",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if neither off-gas nor dataset CER is available",
        tags=("derived", "stoichiometry"),
    ),
    FeatureSpec(
        name="rq",
        source="cer, our",
        formula="RQ = CER/OUR if |OUR| >= epsilon else NaN (no division by ~0)",
        unit="1",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN when OUR is missing or |OUR| < epsilon",
        tags=("derived", "stoichiometry"),
    ),
    FeatureSpec(
        name="d_do_dt",
        source="do, Time (h)",
        formula="backward difference (DO[t]-DO[t-1])/(t-t-1); 0 at first sample",
        unit="mg/L/h",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if DO or time missing",
        tags=("derived", "derivative"),
    ),
    FeatureSpec(
        name="cumulative_sugar_feed",
        source="Sugar feed rate(Fs:L/h), Time (h)",
        formula="causal trapezoidal integral of Fs dt (volume of sugar feed)",
        unit="L",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if feed rate missing; else 0 at t0",
        tags=("derived", "cumulative"),
    ),
    FeatureSpec(
        name="cumulative_oil_feed",
        source="Oil flow(Foil:L/h), Time (h)",
        formula="causal trapezoidal integral of oil feed (second carbon source)",
        unit="L",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="omitted if oil flow column absent",
        tags=("derived", "cumulative"),
    ),
    FeatureSpec(
        name="reactor_volume",
        source="Vessel Volume(V:L)",
        formula="passthrough of measured broth volume (preferred over integration)",
        unit="L",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="NaN if volume column absent",
        tags=("observable", "derived"),
    ),
    FeatureSpec(
        name="reactor_volume_integrated",
        source="Fs, Foil, Fw, Fa, Fb, Fremoved, V0",
        formula=(
            "V(t) = V(0) + ∫(Fs+Foil+Fw+Fa+Fb-Fremoved) dt; "
            "no evaporation term (Fevp is not a measured 100-batch column)"
        ),
        unit="L",
        causal_status="causal",
        model_usage="analysis",
        missing_data_behavior="omitted if no feed/volume columns; not a substitute for missing Fevp",
        tags=("derived", "cumulative"),
    ),
    FeatureSpec(
        name="progress_coordinate",
        source="cumulative sugar feed (causal integral)",
        formula=(
            "causal cumulative sugar-feed volume; not full-batch DTW and not "
            "normalized by this batch's final duration or final feed total"
        ),
        unit="L",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="falls back to elapsed time (h) if feed missing",
        tags=("derived", "progress"),
    ),
    FeatureSpec(
        name="elapsed_time_h",
        source="Time (h)",
        formula="t - t0 within the batch (causal clock)",
        unit="h",
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior="index * default_dt_h if time column missing",
        tags=("derived", "progress"),
    ),
    FeatureSpec(
        name="phase_growth",
        source="our, elapsed_time_h (soft indicator)",
        formula=(
            "softmax component: high OUR and earlier elapsed time; "
            "continuous indicator for the single residual MLP, not a separate NN"
        ),
        unit="1",
        causal_status="causal",
        model_usage="residual_mlp",
        missing_data_behavior="uniform 1/3 if OUR missing",
        tags=("phase", "soft"),
    ),
    FeatureSpec(
        name="phase_production",
        source="our, elapsed_time_h (soft indicator)",
        formula="softmax component: mid elapsed time; not a separate production NN",
        unit="1",
        causal_status="causal",
        model_usage="residual_mlp",
        missing_data_behavior="uniform 1/3 if OUR missing",
        tags=("phase", "soft"),
    ),
    FeatureSpec(
        name="phase_autolysis",
        source="our, elapsed_time_h (soft indicator)",
        formula="softmax component: later elapsed time and lower OUR; not a separate NN",
        unit="1",
        causal_status="causal",
        model_usage="residual_mlp",
        missing_data_behavior="uniform 1/3 if OUR missing",
        tags=("phase", "soft"),
    ),
    FeatureSpec(
        name="delta_T",
        source="vessel temperature − jacket temperature",
        formula="T_vessel - T_jacket",
        unit="K",
        causal_status="unimplemented_missing_column",
        model_usage="not_used",
        missing_data_behavior="not computed",
        implemented=False,
        tags=("derived", "missing_column"),
    ),
    FeatureSpec(
        name="d_tau_dt",
        source="shaft torque τ",
        formula="backward derivative of torque (dτ/dt)",
        unit="N·m/h",
        causal_status="unimplemented_missing_column",
        model_usage="not_used",
        missing_data_behavior="not computed; agitator RPM is not torque",
        implemented=False,
        tags=("derived", "missing_column"),
    ),
    FeatureSpec(
        name="mechanistic_biomass",
        source="reduced-order mechanistic model (Prompt 5)",
        formula="ODE state X; labeled model-derived when Prompt 5 lands",
        unit="g/L",
        causal_status="unimplemented_missing_column",
        model_usage="not_used",
        missing_data_behavior="not computed in Prompt 4 (no mechanistic solve)",
        implemented=False,
        tags=("model_derived", "deferred"),
    ),
    FeatureSpec(
        name="mechanistic_growth_rate",
        source="reduced-order mechanistic model (Prompt 5)",
        formula="μ(S, DO) X from the reduced model; not full IndPenSim kinetics",
        unit="g/L/h",
        causal_status="unimplemented_missing_column",
        model_usage="not_used",
        missing_data_behavior="not computed in Prompt 4",
        implemented=False,
        tags=("model_derived", "deferred"),
    ),
)


HISTORY_TEMPLATE = FeatureSpec(
    name="{base}_{kind}_{n}",
    source="{base} history (backward window / lag only)",
    formula="lag k / rolling mean / rolling std / rolling OLS slope on past+current samples",
    unit="inherits from base",
    causal_status="causal",
    model_usage="nn_only+residual_mlp",
    missing_data_behavior="NaN until enough past samples exist (lags); rolling uses available past",
    tags=("history",),
)


def implemented_features() -> tuple[FeatureSpec, ...]:
    return tuple(spec for spec in FEATURE_REGISTRY if spec.implemented)


def unimplemented_features() -> tuple[FeatureSpec, ...]:
    return tuple(spec for spec in FEATURE_REGISTRY if not spec.implemented)


def unimplemented_due_to_missing_columns() -> tuple[FeatureSpec, ...]:
    """Requested features that cannot be built from IndPenSim measured columns.

    Mechanistic outputs are deferred to Prompt 5 (no model yet), not missing sensors.
    """
    return tuple(
        spec
        for spec in FEATURE_REGISTRY
        if not spec.implemented and "missing_column" in spec.tags
    )


def registry_by_name() -> dict[str, FeatureSpec]:
    return {spec.name: spec for spec in FEATURE_REGISTRY}


def history_spec(base: str, kind: str, n: int, unit: str) -> FeatureSpec:
    return FeatureSpec(
        name=f"{base}_{kind}_{n}",
        source=f"{base} (causal {kind} {n})",
        formula=HISTORY_TEMPLATE.formula,
        unit=unit,
        causal_status="causal",
        model_usage="nn_only+residual_mlp",
        missing_data_behavior=HISTORY_TEMPLATE.missing_data_behavior,
        tags=("history",),
    )


def registry_records(history_names: tuple[str, ...] = ()) -> list[dict[str, str | bool]]:
    """Rows suitable for the Feature Engineering UI table."""
    rows: list[dict[str, str | bool]] = []
    for spec in FEATURE_REGISTRY:
        rows.append(
            {
                "feature": spec.name,
                "source": spec.source,
                "formula": spec.formula,
                "unit": spec.unit,
                "causal": spec.causal,
                "causal_status": spec.causal_status,
                "model_usage": spec.model_usage,
                "missing_data_behavior": spec.missing_data_behavior,
                "implemented": spec.implemented,
            }
        )
    for name in history_names:
        rows.append(
            {
                "feature": name,
                "source": HISTORY_TEMPLATE.source,
                "formula": HISTORY_TEMPLATE.formula,
                "unit": HISTORY_TEMPLATE.unit,
                "causal": True,
                "causal_status": "causal",
                "model_usage": HISTORY_TEMPLATE.model_usage,
                "missing_data_behavior": HISTORY_TEMPLATE.missing_data_behavior,
                "implemented": True,
            }
        )
    return rows
