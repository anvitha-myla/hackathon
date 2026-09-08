"""Four-tab Streamlit prototype (docs/12_UI_SPEC.md).

Business logic lives in ``src``. This file only renders.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.cleaning import clean_batch
from src.features.feature_engine import FeatureEngine
from src.ui import (
    REFERENCE_DISPLAY_NAME,
    batch_frame,
    dataset_banner,
    feature_map_table,
    live_metrics_row,
    load_workspace_dataset,
    preview_signals,
    quality_summary,
    run_live_inference,
    run_monitor,
    time_column,
)


def _fmt_s(seconds: float) -> str:
    return f"{seconds * 1e3:.2f} ms" if seconds < 1.0 else f"{seconds:.4f} s"


def main() -> None:
    st.set_page_config(page_title="Bioprocess soft sensor", layout="wide")
    st.title("Bioprocess hybrid soft sensor")
    st.caption(
        "Prototype on IndPenSim simulated batches. "
        "Live inference does not use reference biomass as an input."
    )

    loaded = load_workspace_dataset()
    st.info(dataset_banner(loaded))
    for note in loaded.notes[:2]:
        st.caption(note)

    batch_ids = loaded.batch_ids
    if not batch_ids:
        st.error("No batches available.")
        return
    selected = st.sidebar.selectbox("Batch", batch_ids)
    batch = batch_frame(loaded, selected)

    tab_clean, tab_feat, tab_live, tab_mon = st.tabs(
        [
            "Data Cleaning",
            "Feature Engineering",
            "Live Inference",
            "Computational Monitor",
        ]
    )

    with tab_clean:
        _render_cleaning(batch, selected)
    with tab_feat:
        _render_features(batch)
    with tab_live:
        _render_live(batch)
    with tab_mon:
        _render_monitor()


def _render_cleaning(batch: pd.DataFrame, batch_id: object) -> None:
    st.subheader("Data cleaning")
    st.write(f"Selected batch: `{batch_id}`")
    cleaned = clean_batch(batch)
    stats = quality_summary(cleaned)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality status", stats["status"])
    c2.metric("Missing flags", stats["missing_flags"])
    c3.metric("Outlier (spike) flags", stats["outlier_spike_flags"])
    c4.metric("Implausible flags", stats["implausible_flags"])
    tcol = time_column(cleaned.frame)
    signals = preview_signals(cleaned)
    if tcol is None or not signals:
        st.dataframe(cleaned.frame.head(50), use_container_width=True)
        return
    signal = st.selectbox("Signal", signals)
    raw_col = f"{signal}_raw" if f"{signal}_raw" in cleaned.frame.columns else signal
    clean_col = f"{signal}_cleaned" if f"{signal}_cleaned" in cleaned.frame.columns else signal
    plot_df = cleaned.frame[[tcol, raw_col, clean_col]].rename(
        columns={raw_col: "raw", clean_col: "cleaned", tcol: "time"}
    )
    st.line_chart(plot_df.set_index("time")[["raw", "cleaned"]])
    st.caption("Raw vs cleaned (causal). Original values are retained.")


def _render_features(batch: pd.DataFrame) -> None:
    st.subheader("Feature map")
    st.dataframe(feature_map_table(), use_container_width=True, hide_index=True)
    if batch.empty:
        return
    engine = FeatureEngine.from_config()
    feats = engine.transform(batch)
    st.caption("Values on the selected batch (formulas are fixed; values change over time).")
    show_cols = [c for c in ("time_h", "our", "cer", "rq", "d_do_dt", "progress_coordinate") if c in feats.columns]
    if show_cols:
        st.line_chart(feats.set_index(show_cols[0])[show_cols[1:]] if len(show_cols) > 1 else feats[show_cols])


def _render_live(batch: pd.DataFrame) -> None:
    st.subheader("Live inference")
    live = run_live_inference(batch)
    frame = live["records"]
    if frame.empty:
        st.warning("No rows to run.")
        return
    metrics = live_metrics_row(live)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hybrid biomass", f"{metrics['current_hybrid']:.4g}")
    m2.metric("Physics estimate", f"{metrics['physics_estimate']:.4g}")
    err = metrics["prediction_error"]
    m3.metric("Prediction error", "n/a" if err is None else f"{err:.4g}")
    m4.metric("beta trust", f"{metrics['beta_trust']:.3f}")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Phase", str(metrics["phase"]))
    ood = metrics["ood_distance"]
    n2.metric("OOD distance", "unscored" if ood is None else f"{ood:.3g}")
    n3.metric("ML correction", f"{metrics['ml_correction']:.4g}")
    nn = metrics["current_nn_only"]
    n4.metric("NN-only", "not trained" if nn is None else f"{nn:.4g}")
    if not live["nn_only_available"]:
        st.caption(live["nn_status"].get("note", "NN-only weights are not present."))
    plot = pd.DataFrame(
        {
            REFERENCE_DISPLAY_NAME: frame["X_reference"],
            "Mechanistic": frame["X_mechanistic"],
            "NN-only": frame["X_nn_only"],
            "Hybrid": frame["X_hybrid"],
        },
        index=frame["timestamp"],
    )
    st.line_chart(plot)
    st.caption(
        f"{REFERENCE_DISPLAY_NAME} is a simulator benchmark, not real-world ground truth. "
        "It is attached after prediction for plotting only."
    )


def _render_monitor() -> None:
    st.subheader("Computational monitor")
    n_steps = st.slider("Timed samples", min_value=4, max_value=32, value=12)
    if st.button("Measure pipeline", type="primary"):
        st.session_state["monitor"] = run_monitor(n_steps=n_steps)
    payload = st.session_state.get("monitor")
    if payload is None:
        st.caption("Run a measurement to record CPU, RAM, and stage latency.")
        return
    report = payload["report"]
    snap = payload["snapshot"]
    a, b, c, d = st.columns(4)
    a.metric("CPU (process %)", f"{snap['cpu_percent']:.1f}")
    b.metric("RAM %", f"{snap['ram_percent']:.1f}")
    c.metric("Latency (mean total)", _fmt_s(report.total_inference_latency_s))
    d.metric("Throughput", f"{report.throughput_hz:.2f} samples/s")
    st.write("Pipeline breakdown (mean)")
    breakdown = pd.DataFrame(
        {
            "stage": ["Cleaning", "Features", "Physics", "NN", "OOD", "Total"],
            "latency_s": [
                report.cleaning_latency_s,
                report.feature_engineering_latency_s,
                report.mechanistic_solver_latency_s,
                report.nn_latency_s,
                report.ood_latency_s,
                report.total_inference_latency_s,
            ],
        }
    )
    st.dataframe(breakdown, hide_index=True, use_container_width=True)
    st.metric("Model size (bytes)", payload["model_size_bytes"] or 0)
    st.metric("Parameter count", payload["parameter_count"] or 0)
    hist = pd.DataFrame(report.history)
    if not hist.empty and "total_inference_latency_s" in hist:
        st.line_chart(hist[["total_inference_latency_s", "cpu_percent", "ram_percent"]])
    bench = payload["vs_benchmark"]
    st.caption(
        f"Mechanistic-loop measurement {bench['measured_s']*1e3:.2f} ms vs "
        f"{bench['benchmark_target_s']*1e3:.0f} ms target (measurement only, not a claim)."
    )


if __name__ == "__main__":
    main()
