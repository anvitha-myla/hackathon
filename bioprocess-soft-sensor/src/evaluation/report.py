"""Per-batch / aggregate tables and markdown reports.

The report never states that hybrid wins. Rankings are numeric sort order only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import MODEL_ORDER, rank_models_by_metric
from src.evaluation.plots import REFERENCE_LABEL
from src.evaluation.runner import EvaluationRun


def per_batch_frame(run: EvaluationRun) -> pd.DataFrame:
    rows = []
    for item in run.per_batch:
        rows.append(
            {
                "batch_id": item.batch_id,
                "model": item.model,
                "rmse": item.rmse,
                "mae": item.mae,
                "r2": item.r2,
                "n": item.n,
                "early_rmse": item.early.get("rmse"),
                "late_rmse": item.late.get("rmse"),
                "stability_delta_rmse": item.stability.get("delta_rmse"),
                "ood_rmse": item.ood.get("ood", {}).get("rmse") if item.ood else float("nan"),
                "latency_mean_s": item.latency.get("mean_s"),
            }
        )
    columns = [
        "batch_id",
        "model",
        "rmse",
        "mae",
        "r2",
        "n",
        "early_rmse",
        "late_rmse",
        "stability_delta_rmse",
        "ood_rmse",
        "latency_mean_s",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_frame(run: EvaluationRun) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        payload = dict(run.aggregate.get(model) or {})
        payload["model"] = model
        rows.append(payload)
    frame = pd.DataFrame(rows)
    if "model" in frame.columns:
        cols = ["model"] + [c for c in frame.columns if c != "model"]
        frame = frame.loc[:, cols]
    return frame


def comparison_table(run: EvaluationRun) -> pd.DataFrame:
    """Wide comparison: one row per model, primary + extra metrics."""
    return aggregate_frame(run)


def write_tables(run: EvaluationRun, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_batch = per_batch_frame(run)
    aggregate = aggregate_frame(run)
    paths = {
        "per_batch_csv": output_dir / "per_batch_metrics.csv",
        "aggregate_csv": output_dir / "aggregate_metrics.csv",
        "comparison_csv": output_dir / "comparison_table.csv",
        "run_json": output_dir / "evaluation_run.json",
    }
    per_batch.to_csv(paths["per_batch_csv"], index=False)
    aggregate.to_csv(paths["aggregate_csv"], index=False)
    comparison_table(run).to_csv(paths["comparison_csv"], index=False)
    paths["run_json"].write_text(json.dumps(run.as_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    return paths


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "—"
    return f"{number:.6g}"


def markdown_report(run: EvaluationRun) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Status: `{run.status}`",
        f"- Production IndPenSim 20-batch claim: **{run.production_claim}**",
        f"- Reference series label: **{REFERENCE_LABEL}** (simulator, not physical ground truth)",
        "- Models: Mechanistic-only, NN-only, Hybrid",
        "- Hybrid is **not** assumed to win. Rankings below are metric sort order only.",
        f"- Reason: {run.reason}",
        f"- Batches scored: {run.n_batches}",
        "",
    ]
    if run.leakage_audit is not None:
        lines.extend(
            [
                "## Leakage audit",
                "",
                f"- Holdout clean: `{run.leakage_audit.ok}`",
                f"- Test ids inspected: {len(run.leakage_audit.test_ids)}",
                f"- Leaked sources: {len(run.leakage_audit.leaked)}",
                "",
            ]
        )
    if run.status == "skipped" or not run.aggregate:
        lines.extend(
            [
                "## Final 20-batch evaluation",
                "",
                "Not run. No production metrics are reported.",
                "",
            ]
        )
        if run.skipped_checks:
            lines.append("```json")
            lines.append(json.dumps(run.skipped_checks, indent=2, default=str))
            lines.append("```")
            lines.append("")
        return "\n".join(lines) + "\n"

    lines.extend(["## Aggregate comparison", "", "| model | RMSE | MAE | R² | early RMSE | late RMSE | Δ-RMSE | OOD RMSE | latency mean (s) |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for model in MODEL_ORDER:
        row = run.aggregate.get(model, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _fmt(row.get("rmse")),
                    _fmt(row.get("mae")),
                    _fmt(row.get("r2")),
                    _fmt(row.get("early_rmse")),
                    _fmt(row.get("late_rmse")),
                    _fmt(row.get("stability_delta_rmse")),
                    _fmt(row.get("ood_rmse")),
                    _fmt(row.get("latency_mean_s")),
                ]
            )
            + " |"
        )
    rmse_rank = rank_models_by_metric(run.aggregate, "rmse", lower_is_better=True)
    lines.extend(
        [
            "",
            f"RMSE sort order (lowest first): {', '.join(rmse_rank) if rmse_rank else 'n/a'}.",
            "This ordering is not a locked research conclusion.",
            "",
            "## Per-batch metrics",
            "",
        ]
    )
    frame = per_batch_frame(run)
    if frame.empty:
        lines.append("(none)")
    else:
        header = list(frame.columns)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for _, rec in frame.iterrows():
            lines.append("| " + " | ".join(_fmt(rec[c]) if c != "batch_id" and c != "model" else str(rec[c]) for c in header) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_markdown_report(run: EvaluationRun, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(run), encoding="utf-8")
    return path
