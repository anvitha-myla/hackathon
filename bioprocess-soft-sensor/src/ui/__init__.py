"""UI orchestration. Calls ``src`` modules; does not reimplement models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.data.cleaning import CleanedBatch, clean_batch
from src.data.loader import load_dataset
from src.data.quality import QualityFlag, has_flag
from src.data.schema import (
    IDENTITY_BATCH_ID,
    IDENTITY_TIMESTAMP,
    LoadedDataset,
    match_role,
)
from src.evaluation.metrics import mae, rmse
from src.features.registry import registry_records
from src.inference.components import is_reference_biomass_key, strip_reference_biomass
from src.inference.pipeline import SequentialHybridEngine
from src.inference.result import attach_reference_for_evaluation
from src.models.nn_only import artifacts_exist, prompt6_status
from src.monitoring.system import (
    count_parameters,
    model_size_bytes,
    run_monitored_inference_loop,
    snapshot_resources,
)
from src.monitoring.timing import compare_to_benchmark

REFERENCE_DISPLAY_NAME = "IndPenSim Reference"

LIVE_DROP_ROLES = (
    "biomass_reference",
    "penicillin",
    "penicillin_offline",
    "substrate",
    "paa_offline",
    "nh3_offline",
    "viscosity_offline",
)


def load_workspace_dataset() -> LoadedDataset:
    """Prefer ``data/raw`` IndPenSim; otherwise the labeled synthetic fixture."""
    return load_dataset(fallback_fixture=True)


def dataset_banner(loaded: LoadedDataset) -> str:
    if loaded.dataset_found:
        return (
            "Loaded IndPenSim files from data/raw. "
            "This is a simulator reference dataset, not physical industrial telemetry."
        )
    return (
        "IndPenSim files were not found under data/raw. "
        "Showing the labeled synthetic fixture (not the official 100-batch dump)."
    )


def batch_frame(loaded: LoadedDataset, batch_id: Any) -> pd.DataFrame:
    frame = loaded.frame
    if frame.empty or IDENTITY_BATCH_ID not in frame.columns:
        return frame.copy()
    return frame.loc[frame[IDENTITY_BATCH_ID] == batch_id].copy()


def time_column(frame: pd.DataFrame) -> str | None:
    if IDENTITY_TIMESTAMP in frame.columns:
        return IDENTITY_TIMESTAMP
    if "Time (h)" in frame.columns:
        return "Time (h)"
    if "timestamp" in frame.columns:
        return "timestamp"
    return None


def quality_summary(cleaned: CleanedBatch) -> dict[str, Any]:
    frame = cleaned.frame
    n_missing = 0
    n_spike = 0
    n_implausible = 0
    n_filled = 0
    n_quality_cols = 0
    for col in frame.columns:
        if not str(col).endswith("_quality"):
            continue
        n_quality_cols += 1
        q = pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        n_missing += int(sum(has_flag(int(v), QualityFlag.MISSING) for v in q))
        n_spike += int(sum(has_flag(int(v), QualityFlag.SPIKE) for v in q))
        n_implausible += int(sum(has_flag(int(v), QualityFlag.IMPLAUSIBLE) for v in q))
        n_filled += int(sum(has_flag(int(v), QualityFlag.FORWARD_FILLED) for v in q))
    status = "ok"
    if n_implausible or n_spike:
        status = "flags_present"
    if frame.empty:
        status = "empty"
    return {
        "status": status,
        "n_rows": int(len(frame)),
        "n_signal_columns": len(cleaned.signal_columns),
        "missing_flags": n_missing,
        "outlier_spike_flags": n_spike,
        "implausible_flags": n_implausible,
        "forward_filled_flags": n_filled,
        "quality_columns": n_quality_cols,
        "causal": bool(cleaned.metadata.get("causal", True)),
    }


def preview_signals(cleaned: CleanedBatch, max_signals: int = 4) -> list[str]:
    return list(cleaned.signal_columns[:max_signals])


def feature_map_table() -> pd.DataFrame:
    rows = registry_records()
    display = []
    for row in rows:
        usage = str(row.get("model_usage", ""))
        if usage == "nn_only+residual_mlp":
            dest = "NN-only + residual MLP"
        elif usage == "residual_mlp":
            dest = "Residual MLP (hybrid)"
        elif usage == "analysis":
            dest = "Analysis"
        else:
            dest = "Not used"
        display.append(
            {
                "Feature": row["feature"],
                "Source": row["source"],
                "Formula": row["formula"],
                "Unit": row["unit"],
                "Model usage": dest,
            }
        )
    return pd.DataFrame(display)


def _reference_value(row: Mapping[str, Any], columns: list[str]) -> float | None:
    col = match_role(columns, "biomass_reference")
    if col is None:
        for key, value in row.items():
            if is_reference_biomass_key(str(key)):
                try:
                    x = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(x):
                    return x
        return None
    try:
        x = float(row[col])
    except (TypeError, ValueError, KeyError):
        return None
    return x if np.isfinite(x) else None


def _observation_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    obs = strip_reference_biomass({str(k): v for k, v in row.items()})
    drop_cols = set()
    cols = list(obs.keys())
    for role in LIVE_DROP_ROLES:
        hit = match_role(cols, role)
        if hit is not None:
            drop_cols.add(hit)
    for key in list(obs):
        if key in drop_cols:
            obs.pop(key, None)
    if "timestamp" not in obs and IDENTITY_TIMESTAMP in obs:
        obs["timestamp"] = obs[IDENTITY_TIMESTAMP]
    return obs


def run_live_inference(batch: pd.DataFrame) -> dict[str, Any]:
    """Sequential hybrid on one batch. Reference biomass is evaluation-only."""
    engine = SequentialHybridEngine()
    nn_status = prompt6_status()
    nn_ready = bool(artifacts_exist())
    records: list[dict[str, Any]] = []
    columns = [str(c) for c in batch.columns]
    if batch.empty:
        return {
            "records": pd.DataFrame(),
            "nn_only_available": nn_ready,
            "nn_status": nn_status,
            "reference_label": REFERENCE_DISPLAY_NAME,
            "n_steps": 0,
        }

    engine.reset(batch_id=str(batch.iloc[0].get(IDENTITY_BATCH_ID, "batch")))
    for _, series in batch.iterrows():
        row = series.to_dict()
        x_ref = _reference_value(row, columns)
        obs = _observation_from_row(row)
        result = engine.step(obs)
        payload = result.as_dict()
        if x_ref is not None:
            payload = attach_reference_for_evaluation(result, x_ref)
            payload["error"] = float(result.X_hybrid) - float(x_ref)
        else:
            payload["X_reference"] = float("nan")
            payload["error"] = float("nan")
        payload["X_nn_only"] = float("nan")
        payload["phase"] = max(
            result.phase_indicators,
            key=lambda k: result.phase_indicators[k],
        )
        records.append(payload)

    frame = pd.DataFrame(records)
    out: dict[str, Any] = {
        "records": frame,
        "nn_only_available": nn_ready,
        "nn_status": nn_status,
        "reference_label": REFERENCE_DISPLAY_NAME,
        "n_steps": int(len(frame)),
    }
    if not frame.empty and np.isfinite(frame["X_reference"]).any():
        y = frame["X_reference"].to_numpy(dtype=float)
        out["rmse_hybrid"] = rmse(y, frame["X_hybrid"].to_numpy(dtype=float))
        out["mae_hybrid"] = mae(y, frame["X_hybrid"].to_numpy(dtype=float))
        out["rmse_mechanistic"] = rmse(y, frame["X_mechanistic"].to_numpy(dtype=float))
        out["mae_mechanistic"] = mae(y, frame["X_mechanistic"].to_numpy(dtype=float))
    return out


def live_metrics_row(live: dict[str, Any]) -> dict[str, Any]:
    frame = live["records"]
    if frame is None or frame.empty:
        return {}
    last = frame.iloc[-1]
    return {
        "current_hybrid": float(last["X_hybrid"]),
        "current_mechanistic": float(last["X_mechanistic"]),
        "current_nn_only": float(last["X_nn_only"]) if np.isfinite(last["X_nn_only"]) else None,
        "prediction_error": float(last["error"]) if np.isfinite(last.get("error", np.nan)) else None,
        "phase": last.get("phase"),
        "ood_distance": float(last["ood_distance"]) if np.isfinite(last["ood_distance"]) else None,
        "beta_trust": float(last["beta_trust"]),
        "ml_correction": float(last["ml_correction"]),
        "physics_estimate": float(last["physics_contribution"]),
    }


def run_monitor(n_steps: int = 16) -> dict[str, Any]:
    from src.models.mlp import SmallMLP

    net = SmallMLP(n_in=15, n_out=1)
    report = run_monitored_inference_loop(n_steps=n_steps, module=net)
    snap = snapshot_resources()
    return {
        "report": report,
        "snapshot": snap,
        "parameter_count": count_parameters(net),
        "model_size_bytes": model_size_bytes(net),
        "vs_benchmark": compare_to_benchmark(report.mechanistic_solver_latency_s),
    }
