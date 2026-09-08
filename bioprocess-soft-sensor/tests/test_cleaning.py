"""Causal leakage, quality, and preservation tests for Prompt 3 cleaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    CleaningParams,
    align_to_minute_grid,
    bounded_forward_fill,
    causal_ema,
    causal_hampel,
    causal_one_sided_ma,
    clean_batch,
    clean_series,
    fit_cleaning_params,
    load_cleaning_params,
    minutes_to_steps,
)
from src.data.quality import QualityFlag, has_flag

CLEANING_SRC = Path(__file__).resolve().parents[1] / "src" / "data" / "cleaning.py"
QUALITY_SRC = Path(__file__).resolve().parents[1] / "src" / "data" / "quality.py"


def _params(**overrides: object) -> CleaningParams:
    base = load_cleaning_params()
    return CleaningParams(
        **{**base.__dict__, **overrides}  # type: ignore[arg-type]
    )


def test_source_has_no_two_sided_savgol_or_future_fill() -> None:
    text = CLEANING_SRC.read_text(encoding="utf-8").lower()
    quality_text = QUALITY_SRC.read_text(encoding="utf-8").lower()
    combined = text + "\n" + quality_text
    assert "savgol" not in combined
    assert "savitzky" not in combined
    assert "savgol_filter" not in combined
    assert ".interpolate(" not in text
    assert ".bfill" not in text
    assert "backfill" not in text


def test_future_value_perturbation_leaves_cleaned_t_unchanged() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=0.2, size=120)
    t = 40
    k = 7
    x_future = x.copy()
    x_future[t + k] = 1.0e6
    params = load_cleaning_params()
    a = clean_series(x, params, step_minutes=1.0, column="pH")
    b = clean_series(x_future, params, step_minutes=1.0, column="pH")
    np.testing.assert_allclose(a.cleaned[: t + 1], b.cleaned[: t + 1], equal_nan=True)
    np.testing.assert_array_equal(a.quality[: t + 1], b.quality[: t + 1])


def test_prefix_equals_full_series_through_t() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    t = 25
    params = load_cleaning_params()
    full = clean_series(x, params, step_minutes=1.0)
    prefix = clean_series(x[: t + 1], params, step_minutes=1.0)
    np.testing.assert_allclose(full.cleaned[: t + 1], prefix.cleaned, equal_nan=True)


def test_bounded_forward_fill_does_not_look_ahead() -> None:
    x = np.array([1.0, np.nan, np.nan, 9.0])
    filled, used, expired = bounded_forward_fill(x, max_steps=10)
    assert filled[1] == 1.0
    assert filled[2] == 1.0
    assert filled[1] != 9.0
    assert filled[1] != 5.0
    assert used[1] and used[2]
    assert not expired[1]

    gap = np.array([np.nan, np.nan, 5.0])
    filled_b, _, expired_b = bounded_forward_fill(gap, max_steps=10)
    assert np.isnan(filled_b[0]) and np.isnan(filled_b[1])
    assert filled_b[2] == 5.0
    assert expired_b[0] and expired_b[1]


def test_bounded_forward_fill_respects_max_steps() -> None:
    x = np.array([1.0, np.nan, np.nan, np.nan])
    filled, used, expired = bounded_forward_fill(x, max_steps=1)
    assert filled[1] == 1.0
    assert np.isnan(filled[2]) and np.isnan(filled[3])
    assert used[1]
    assert expired[2] and expired[3]


def test_hampel_and_ema_are_causal_under_future_spike() -> None:
    x = np.ones(30)
    x[10] = 50.0
    y = x.copy()
    y[20] = 80.0
    despike_a, _ = causal_hampel(x, window=5, min_periods=3)
    despike_b, _ = causal_hampel(y, window=5, min_periods=3)
    np.testing.assert_allclose(despike_a[:11], despike_b[:11])
    ema_a = causal_ema(x, 0.3)
    ema_b = causal_ema(y, 0.3)
    np.testing.assert_allclose(ema_a[:20], ema_b[:20])
    ma_a = causal_one_sided_ma(x, 5)
    ma_b = causal_one_sided_ma(y, 5)
    np.testing.assert_allclose(ma_a[:20], ma_b[:20])


def test_quality_flags_present_and_raw_preserved() -> None:
    x = np.array([7.0, np.nan, 7.1, 99.0, 7.2], dtype=float)
    params = load_cleaning_params()
    result = clean_series(x, params, step_minutes=1.0, column="pH")
    assert result.raw[0] == 7.0
    assert np.isnan(result.raw[1])
    assert result.raw[3] == 99.0
    assert result.missing[1]
    assert has_flag(int(result.quality[1]), QualityFlag.MISSING)
    assert has_flag(int(result.quality[1]), QualityFlag.FORWARD_FILLED)
    assert has_flag(int(result.quality[3]), QualityFlag.IMPLAUSIBLE)
    assert result.cleaned.shape == result.raw.shape
    assert result.quality.dtype == np.uint32
    # Implausible pH is not silently dropped from the audit trail.
    assert result.raw[3] == 99.0


def test_clean_batch_adds_raw_cleaned_quality_columns() -> None:
    df = pd.DataFrame(
        {
            "batch_id": [1] * 8,
            "timestamp": np.arange(8, dtype=float),
            "pH": [6.5, 6.6, np.nan, 6.7, 6.5, 6.6, 6.4, 6.5],
        }
    )
    params = load_cleaning_params()
    params = CleaningParams(**{**params.__dict__, "numeric_time_unit": "minutes"})
    out = clean_batch(df, params, time_col="timestamp", time_unit="minutes")
    frame = out.frame
    assert "pH_raw" in frame.columns
    assert "pH_cleaned" in frame.columns
    assert "pH_quality" in frame.columns
    assert "pH_missing" in frame.columns
    np.testing.assert_allclose(frame["pH_raw"].to_numpy(), df["pH"].to_numpy(), equal_nan=True)
    assert out.signal_columns == ("pH",)


def test_future_perturbation_on_batch_frame() -> None:
    n = 40
    df = pd.DataFrame(
        {
            "batch_id": np.ones(n, dtype=int),
            "timestamp": np.arange(n, dtype=float),
            "DO": np.linspace(80.0, 70.0, n),
        }
    )
    t, k = 10, 5
    df2 = df.copy()
    df2.loc[t + k, "DO"] = -50.0
    params = CleaningParams(
        **{**load_cleaning_params().__dict__, "numeric_time_unit": "minutes"}
    )
    a = clean_batch(df, params, time_col="timestamp", time_unit="minutes")
    b = clean_batch(df2, params, time_col="timestamp", time_unit="minutes")
    np.testing.assert_allclose(
        a.frame["DO_cleaned"].to_numpy()[: t + 1],
        b.frame["DO_cleaned"].to_numpy()[: t + 1],
        equal_nan=True,
    )


def test_no_fill_across_batches() -> None:
    df = pd.DataFrame(
        {
            "batch_id": [1, 1, 2, 2],
            "timestamp": [0.0, 1.0, 0.0, 1.0],
            "pH": [7.0, np.nan, np.nan, 8.0],
        }
    )
    params = CleaningParams(
        **{**load_cleaning_params().__dict__, "numeric_time_unit": "minutes"}
    )
    out = clean_batch(df, params, time_col="timestamp", time_unit="minutes")
    b2 = out.frame.loc[out.frame["batch_id"] == 2].sort_values("timestamp")
    # First sample of batch 2 is missing: must not inherit 7.0 from batch 1.
    assert bool(b2.iloc[0]["pH_missing"])
    assert not has_flag(int(b2.iloc[0]["pH_quality"]), QualityFlag.FORWARD_FILLED)
    assert np.isnan(b2.iloc[0]["pH_cleaned"])


def test_align_to_minute_grid_does_not_use_future_samples() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 2.0, 3.0],
            "pH": [1.0, 2.0, 3.0],
        }
    )
    params = CleaningParams(
        **{
            **load_cleaning_params().__dict__,
            "numeric_time_unit": "minutes",
            "target_grid_minutes": 1.0,
            "regrid_when_median_step_minutes_at_most": 5.0,
        }
    )
    aligned, meta = align_to_minute_grid(
        df, "timestamp", params, time_unit="minutes"
    )
    assert meta["reindexed"]
    # Grid t=1 has no sample with time <= 1 after ceil-to-next-grid placement
    # of 0 -> 0 and 2 -> 2. Value at 1 must not be 2.0 (the future sample).
    row1 = aligned.loc[aligned["timestamp"] == 1.0, "pH"]
    assert len(row1) == 1
    assert pd.isna(row1.iloc[0])

    df_f = df.copy()
    df_f.loc[df_f["timestamp"] == 3.0, "pH"] = 99.0
    aligned_f, _ = align_to_minute_grid(
        df_f, "timestamp", params, time_unit="minutes"
    )
    left = aligned.loc[aligned["timestamp"] <= 2.0, "pH"].to_numpy()
    right = aligned_f.loc[aligned_f["timestamp"] <= 2.0, "pH"].to_numpy()
    np.testing.assert_allclose(left, right, equal_nan=True)


def test_fit_cleaning_params_ignores_frames_and_matches_yaml() -> None:
    yaml_params = load_cleaning_params()
    fake_test = [pd.DataFrame({"pH": np.arange(1000) * 1e6})]
    fitted = fit_cleaning_params(train_batches=fake_test)
    assert fitted.ema_alpha == yaml_params.ema_alpha
    assert fitted.hampel_window_minutes == 5.0
    assert fitted.forward_fill_max_minutes == 5.0
    assert fitted.source == "cleaning.yaml"


def test_minutes_to_steps_and_empty_batch() -> None:
    assert minutes_to_steps(5.0, 1.0) == 5
    empty = clean_batch(pd.DataFrame())
    assert empty.frame.empty
    assert empty.metadata["empty"] is True


def test_causal_ema_ignores_future_nan_replacement() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x.copy()
    y[3] = np.nan
    np.testing.assert_allclose(causal_ema(x, 0.5)[:3], causal_ema(y, 0.5)[:3])


def test_unknown_smooth_method_raises() -> None:
    params = CleaningParams(**{**load_cleaning_params().__dict__, "smooth_method": "nope"})
    with pytest.raises(ValueError, match="Unknown smoothing"):
        clean_series(np.array([1.0, 2.0]), params)
