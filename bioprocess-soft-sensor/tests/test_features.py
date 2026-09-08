"""Causal feature engine: formulas, RQ guard, registry, and no-lookahead tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.derivatives import (
    backward_derivative,
    causal_cumulative_trapz,
    causal_lag,
    causal_rolling_mean,
    causal_rolling_slope,
)
from src.features.engineering import engineer
from src.features.feature_engine import FeatureEngine, assert_no_hidden_targets
from src.features.registry import (
    FEATURE_REGISTRY,
    unimplemented_due_to_missing_columns,
)
from src.features.stoichiometry import (
    offgas_our_cer_mol_h,
    respiratory_quotient,
)

# Goldrick 100-batch header spellings (do not invent jacket T or torque).
DO_COL = "Dissolved oxygen concentration(DO2:mg/L)"
T_COL = "Temperature(T:K)"
PH_COL = "pH(pH:pH)"
FG_COL = "Aeration rate(Fg:L/h)"
FS_COL = "Sugar feed rate(Fs:L/h)"
RPM_COL = "Agitator RPM(RPM:RPM)"
V_COL = "Vessel Volume(V:L)"
O2_COL = "Oxygen percent in off-gas(O2:%)"
CO2_COL = "carbon dioxide percent in off-gas(CO2offgas:%)"
P_COL = "Air head pressure(pressure:bar)"
TIME_COL = "Time (h)"
BATCH_COL = "Batch ID"
X_COL = "Offline Biomass concentration(Xoffline:g/L)"
PENC_COL = "Penicillin concentration(P:g/L)"


def _mini_batch(n: int = 25, batch_id: int = 1) -> pd.DataFrame:
    t = np.arange(n, dtype=float) * 0.2
    do = 8.0 + 0.05 * np.arange(n)
    return pd.DataFrame(
        {
            TIME_COL: t,
            BATCH_COL: batch_id,
            DO_COL: do,
            T_COL: np.full(n, 298.15),
            PH_COL: 6.5 + 0.01 * np.sin(t),
            FG_COL: np.full(n, 60000.0),
            FS_COL: 8.0 + 40.0 * t,
            RPM_COL: np.full(n, 100.0),
            V_COL: 58000.0 + 2.0 * np.arange(n),
            O2_COL: 20.5 - 0.02 * np.arange(n),
            CO2_COL: 0.4 + 0.03 * np.arange(n),
            P_COL: np.full(n, 0.9),
            X_COL: 1.0 + 0.1 * np.arange(n),
            PENC_COL: 0.01 * np.arange(n),
        }
    )


def test_registry_has_required_fields() -> None:
    required = {
        "name",
        "source",
        "formula",
        "unit",
        "causal_status",
        "model_usage",
        "missing_data_behavior",
    }
    for spec in FEATURE_REGISTRY:
        for field in required:
            assert getattr(spec, field), spec.name
        assert spec.causal is (spec.causal_status == "causal")


def test_unimplemented_missing_columns_are_jacket_t_and_torque() -> None:
    names = {spec.name for spec in unimplemented_due_to_missing_columns()}
    assert names == {"delta_T", "d_tau_dt"}


def test_offgas_our_cer_inert_balance() -> None:
    fg = np.array([22400.0])
    t = np.array([298.15])
    p = np.array([1.0])
    o2 = np.array([19.0])
    co2 = np.array([2.0])
    our, cer = offgas_our_cer_mol_h(fg, o2, co2, temperature_k=t, pressure_bar=p)
    n_in = 1.0 * 22400.0 / (0.08314462618 * 298.15)
    y_o2_out, y_co2_out = 0.19, 0.02
    n_out = n_in * (1 - 0.2095 - 0.0004) / (1 - y_o2_out - y_co2_out)
    assert our[0] == pytest.approx(n_in * 0.2095 - n_out * y_o2_out)
    assert cer[0] == pytest.approx(n_out * y_co2_out - n_in * 0.0004)
    assert our[0] > 0
    assert cer[0] > 0


def test_rq_near_zero_our_is_nan_not_inf() -> None:
    rq = respiratory_quotient(np.array([1.0, 1.0, 0.0]), np.array([2.0, 1e-12, 0.0]), epsilon=1e-8)
    assert rq[0] == pytest.approx(0.5)
    assert np.isnan(rq[1])
    assert np.isnan(rq[2])
    assert np.isfinite(rq[0])


def test_backward_derivative_uses_only_past() -> None:
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 2.0, 5.0])
    d = backward_derivative(x, t)
    assert d[0] == 0.0
    assert d[1] == pytest.approx(2.0)
    assert d[2] == pytest.approx(3.0)


def test_causal_lag_and_rolling() -> None:
    x = np.arange(10, dtype=float)
    t = np.arange(10, dtype=float)
    lag1 = causal_lag(x, 1)
    assert np.isnan(lag1[0])
    assert lag1[1] == 0.0
    mean5 = causal_rolling_mean(x, 5)
    assert mean5[0] == 0.0
    assert mean5[4] == pytest.approx(2.0)
    slope = causal_rolling_slope(x, t, 5)
    assert slope[4] == pytest.approx(1.0)


def test_cumulative_feed_is_causal_integral() -> None:
    t = np.array([0.0, 1.0, 2.0])
    rate = np.array([10.0, 10.0, 10.0])
    cum = causal_cumulative_trapz(rate, t)
    assert cum[0] == 0.0
    assert cum[1] == pytest.approx(10.0)
    assert cum[2] == pytest.approx(20.0)


def test_engine_emits_requested_features_and_not_hidden_refs() -> None:
    engine = FeatureEngine.from_config()
    out = engine.transform(_mini_batch())
    for name in (
        "our",
        "cer",
        "rq",
        "d_do_dt",
        "cumulative_sugar_feed",
        "reactor_volume",
        "progress_coordinate",
        "phase_growth",
        "phase_production",
        "phase_autolysis",
        "do_lag_1",
        "do_lag_5",
        "do_lag_10",
        "do_roll_mean_5",
        "do_roll_std_5",
        "do_roll_slope_5",
    ):
        assert name in out.columns, name
    assert "delta_T" not in out.columns
    assert "d_tau_dt" not in out.columns
    assert_no_hidden_targets(out.columns)
    assert X_COL not in out.columns
    phase = out["phase_growth"] + out["phase_production"] + out["phase_autolysis"]
    assert np.allclose(phase, 1.0, atol=1e-6)
    assert engine.last_our_source == "offgas_balance"


def test_values_change_over_time_definitions_do_not() -> None:
    out = FeatureEngine.from_config().transform(_mini_batch())
    assert not np.allclose(out["cumulative_sugar_feed"].iloc[0], out["cumulative_sugar_feed"].iloc[-1])
    assert not np.allclose(out["progress_coordinate"].iloc[0], out["progress_coordinate"].iloc[-1])
    names = {spec.name for spec in FEATURE_REGISTRY}
    assert "our" in names and "rq" in names


def test_progress_is_not_normalized_by_batch_end() -> None:
    short = FeatureEngine.from_config().transform(_mini_batch(n=10))
    long = FeatureEngine.from_config().transform(_mini_batch(n=25))
    # Same prefix time/feed → same progress at that clock; no DTW / no /T_final.
    assert short["progress_coordinate"].iloc[5] == pytest.approx(long["progress_coordinate"].iloc[5])


def test_engineering_wrapper_matches_engine() -> None:
    frame = _mini_batch()
    a = FeatureEngine.from_config().transform(frame)
    b = engineer(frame)
    pd.testing.assert_frame_equal(a, b)


def test_engine_on_published_fixture_headers() -> None:
    from src.data.loader import load_fixture

    try:
        loaded = load_fixture()
    except Exception as exc:
        pytest.skip(f"IndPenSim fixture/loader not ready: {exc}")
    if loaded.frame.empty:
        pytest.skip("empty fixture")
    engine = FeatureEngine.from_config()
    out = engine.transform(loaded.frame)
    assert len(out) == len(loaded.frame)
    assert np.isfinite(out["our"]).any()
    assert np.isfinite(out["cer"]).any()
    assert "d_do_dt" in out.columns
    assert "progress_coordinate" in out.columns
    assert_no_hidden_targets(out.columns)
    assert "jacket_temperature" not in loaded.column_map
    names = {spec.name for spec in unimplemented_due_to_missing_columns()}
    assert names == {"delta_T", "d_tau_dt"}


def test_two_batches_are_isolated() -> None:
    a = _mini_batch(n=12, batch_id=1)
    b = _mini_batch(n=12, batch_id=2)
    b[DO_COL] = b[DO_COL] + 50.0
    stacked = pd.concat([a, b], ignore_index=True)
    out = FeatureEngine.from_config().transform(stacked)
    only_a = FeatureEngine.from_config().transform(a)
    merged_a = out[out["batch_id"] == 1].reset_index(drop=True)
    pd.testing.assert_series_equal(
        merged_a["do"].reset_index(drop=True),
        only_a["do"].reset_index(drop=True),
        check_names=False,
    )
    assert out.loc[out["batch_id"] == 1, "do"].max() < 20
    assert out.loc[out["batch_id"] == 2, "do"].min() > 40
