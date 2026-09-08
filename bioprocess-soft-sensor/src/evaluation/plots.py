"""Trajectory and aggregate plots. Reference series is IndPenSim Reference."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import MODEL_HYBRID, MODEL_MECHANISTIC, MODEL_NN_ONLY, MODEL_ORDER
from src.evaluation.runner import BatchTrajectory, EvaluationRun

REFERENCE_LABEL = "IndPenSim Reference"
MODEL_LABELS = {
    MODEL_MECHANISTIC: "Mechanistic",
    MODEL_NN_ONLY: "NN-only",
    MODEL_HYBRID: "Hybrid",
}

# Distinct colors; reference is the simulator, not physical ground truth.
_COLORS = {
    "reference": "#111111",
    MODEL_MECHANISTIC: "#1f77b4",
    MODEL_NN_ONLY: "#ff7f0e",
    MODEL_HYBRID: "#2ca02c",
}


def reference_series_label() -> str:
    return REFERENCE_LABEL


def plot_trajectory_comparison(
    trajectory: BatchTrajectory,
    *,
    path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Overlay IndPenSim Reference, Mechanistic, NN-only, Hybrid."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    t = np.asarray(trajectory.time, dtype=np.float64)
    ax.plot(t, trajectory.x_reference, color=_COLORS["reference"], lw=2.2, label=REFERENCE_LABEL)
    ax.plot(
        t,
        trajectory.mechanistic,
        color=_COLORS[MODEL_MECHANISTIC],
        lw=1.6,
        label=MODEL_LABELS[MODEL_MECHANISTIC],
    )
    ax.plot(
        t,
        trajectory.nn_only,
        color=_COLORS[MODEL_NN_ONLY],
        lw=1.6,
        label=MODEL_LABELS[MODEL_NN_ONLY],
    )
    ax.plot(
        t,
        trajectory.hybrid,
        color=_COLORS[MODEL_HYBRID],
        lw=1.6,
        label=MODEL_LABELS[MODEL_HYBRID],
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Biomass X")
    ax.set_title(title or f"Batch {trajectory.batch_id} — simulator reference vs models")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    saved: Path | None = None
    if path is not None:
        saved = Path(path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=120)
    plt.close(fig)
    return saved


def plot_aggregate_metrics(
    run: EvaluationRun,
    *,
    path: str | Path | None = None,
) -> Path | None:
    """Bar chart of mean per-batch RMSE and MAE. No winner banner."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    names = [MODEL_LABELS[m] for m in MODEL_ORDER]
    rmse = [run.aggregate.get(m, {}).get("rmse", np.nan) for m in MODEL_ORDER]
    mae = [run.aggregate.get(m, {}).get("mae", np.nan) for m in MODEL_ORDER]
    colors = [_COLORS[m] for m in MODEL_ORDER]
    axes[0].bar(names, rmse, color=colors)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Mean per-batch RMSE")
    axes[1].bar(names, mae, color=colors)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Mean per-batch MAE")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Model comparison (do not assume hybrid wins)")
    fig.tight_layout()
    saved: Path | None = None
    if path is not None:
        saved = Path(path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=120)
    plt.close(fig)
    return saved


def write_trajectory_plots(
    trajectories: Sequence[BatchTrajectory],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for traj in trajectories:
        dest = output_dir / f"trajectory_batch_{traj.batch_id}.png"
        written = plot_trajectory_comparison(traj, path=dest)
        if written is not None:
            paths.append(written)
    return paths
