"""Causal leakage: mutating a future raw sample must not change features at t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engine import FeatureEngine
from tests.test_features import CO2_COL, DO_COL, FG_COL, FS_COL, O2_COL, _mini_batch


def _numeric_feature_cols(frame: pd.DataFrame) -> list[str]:
    skip = {"batch_id"}
    return [c for c in frame.columns if c not in skip and np.issubdtype(frame[c].dtype, np.number)]


def test_mutate_future_raw_signal_does_not_change_features_at_t() -> None:
    engine = FeatureEngine.from_config()
    raw = _mini_batch(n=30)
    t_check = 12
    future = 20
    baseline = engine.transform(raw)

    mutated = raw.copy()
    mutated.loc[future, DO_COL] = 999.0
    mutated.loc[future, FG_COL] = 1.0
    mutated.loc[future, FS_COL] = 0.0
    mutated.loc[future, O2_COL] = 1.0
    mutated.loc[future, CO2_COL] = 50.0
    after = engine.transform(mutated)

    cols = _numeric_feature_cols(baseline)
    # Strict causality: every row before the mutated future index is unchanged.
    for i in range(t_check, future):
        for col in cols:
            b = baseline.loc[i, col]
            a = after.loc[i, col]
            if np.isnan(b) and np.isnan(a):
                continue
            assert a == b or np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True), (
                f"leakage: feature {col!r} at row {i} changed after mutating row {future}"
            )


def test_mutate_last_row_only_affects_last_row_features() -> None:
    engine = FeatureEngine.from_config()
    raw = _mini_batch(n=16)
    baseline = engine.transform(raw)
    mutated = raw.copy()
    mutated.loc[15, DO_COL] = -1.0
    after = engine.transform(mutated)
    cols = _numeric_feature_cols(baseline)
    for i in range(15):
        for col in cols:
            b = baseline.loc[i, col]
            a = after.loc[i, col]
            if np.isnan(b) and np.isnan(a):
                continue
            assert np.allclose(a, b, equal_nan=True), col
    assert after.loc[15, "do"] != baseline.loc[15, "do"]
    assert after.loc[15, "d_do_dt"] != baseline.loc[15, "d_do_dt"]


def test_np_gradient_would_leak_but_engine_does_not() -> None:
    """Sanity: two-sided np.gradient at t depends on t+1; the engine must not."""
    raw = _mini_batch(n=10)
    do = raw[DO_COL].to_numpy()
    t = raw["Time (h)"].to_numpy()
    two_sided = np.gradient(do, t)
    mutated = do.copy()
    mutated[-1] = do[-1] + 100.0
    two_sided_mut = np.gradient(mutated, t)
    # Two-sided derivative at the second-to-last sample changes.
    assert two_sided[-2] != two_sided_mut[-2]

    engine = FeatureEngine.from_config()
    feat = engine.transform(raw)
    raw2 = raw.copy()
    raw2.loc[9, DO_COL] = raw.loc[9, DO_COL] + 100.0
    feat2 = engine.transform(raw2)
    assert feat.loc[8, "d_do_dt"] == feat2.loc[8, "d_do_dt"]
