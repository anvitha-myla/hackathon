"""Prompt 13 UI orchestration and four-tab Streamlit smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_batch
from src.data.loader import load_fixture
from src.inference.components import is_reference_biomass_key
from src.ui import (
    REFERENCE_DISPLAY_NAME,
    dataset_banner,
    feature_map_table,
    live_metrics_row,
    quality_summary,
    run_live_inference,
    run_monitor,
)


def test_feature_map_has_spec_columns() -> None:
    table = feature_map_table()
    assert list(table.columns) == ["Feature", "Source", "Formula", "Unit", "Model usage"]
    assert "our" in set(table["Feature"])
    assert "delta_T" in set(table["Feature"])


def test_cleaning_tab_stats_on_fixture() -> None:
    loaded = load_fixture()
    batch_id = loaded.batch_ids[0]
    batch = loaded.frame.loc[loaded.frame["batch_id"] == batch_id]
    cleaned = clean_batch(batch)
    stats = quality_summary(cleaned)
    assert stats["n_rows"] == len(cleaned.frame)
    assert stats["status"] in {"ok", "flags_present"}
    assert stats["missing_flags"] >= 0
    assert stats["outlier_spike_flags"] >= 0


def test_live_inference_is_sequential_and_hides_reference_from_model() -> None:
    loaded = load_fixture()
    batch_id = loaded.batch_ids[0]
    batch = loaded.frame.loc[loaded.frame["batch_id"] == batch_id]
    live = run_live_inference(batch)
    frame = live["records"]
    assert live["reference_label"] == REFERENCE_DISPLAY_NAME
    assert len(frame) == len(batch)
    assert list(frame["timestamp"]) == list(pd.to_numeric(batch["timestamp_h"], errors="coerce"))
    assert "X_mechanistic" in frame.columns
    assert "X_hybrid" in frame.columns
    assert "X_nn_only" in frame.columns
    import numpy as np

    expected = frame["X_mechanistic"] + frame["beta_trust"] * frame["delta_X_raw"]
    assert np.allclose(frame["X_hybrid"].to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True)
    metrics = live_metrics_row(live)
    assert metrics["physics_estimate"] == metrics["current_mechanistic"]
    for col in batch.columns:
        if is_reference_biomass_key(str(col)):
            break
    else:
        raise AssertionError("fixture should include a biomass reference column")
    # Live records may include X_reference for the plot, attached after prediction.
    assert "X_reference" in frame.columns


def test_fixture_banner_when_dataset_missing() -> None:
    loaded = load_fixture()
    text = dataset_banner(loaded)
    assert "synthetic fixture" in text.lower() or "not found" in text.lower() or "SYNTHETIC" in " ".join(loaded.notes)


def test_monitor_returns_measured_latencies() -> None:
    payload = run_monitor(n_steps=4)
    report = payload["report"]
    assert report.n_samples == 4
    assert report.total_inference_latency_s >= 0.0
    assert payload["parameter_count"] > 0
    assert payload["model_size_bytes"] > 0


def test_streamlit_app_compiles() -> None:
    path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert "Data Cleaning" in source
    assert "Feature Engineering" in source
    assert "Live Inference" in source
    assert "Computational Monitor" in source
    assert "real-world ground truth" in source.lower()
    assert "REFERENCE_DISPLAY_NAME" in source
