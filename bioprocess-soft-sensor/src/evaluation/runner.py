"""Evaluate mechanistic-only, NN-only, and hybrid on held-out batches.

``run_final_test()`` is gated: it refuses if test batch_ids appear in train
artifacts (scaler / OOD fit / training metadata) and it will not invent a
20-batch IndPenSim production result when data or trained models are missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

from src.config import PROJECT_ROOT
from src.evaluation.metrics import (
    MODEL_HYBRID,
    MODEL_MECHANISTIC,
    MODEL_NN_ONLY,
    MODEL_ORDER,
    ModelBatchMetrics,
    aggregate_model_metrics,
    score_model_on_batch,
)
from src.models.nn_only import prompt6_status
from src.models.training import TEST_SPLIT_FORBIDDEN, load_split_manifest

FIT_ID_KEYS = (
    "train_batch_ids",
    "training_batch_ids",
    "scaler_batch_ids",
    "ood_fit_batch_ids",
    "ood_train_batch_ids",
    "fit_batch_ids",
    "validation_batch_ids",
)

SKIP_NAMES = {".gitkeep", ".gitignore"}


def _inference_engine_status(project_root: Path) -> str:
    engine = project_root / "src" / "inference" / "engine.py"
    pipeline = project_root / "src" / "inference" / "pipeline.py"
    if pipeline.is_file():
        return "implemented"
    if engine.is_file():
        text = engine.read_text(encoding="utf-8")
        if "Hybrid inference stub" in text:
            return "stub"
        return "implemented"
    return "missing"


class TestSplitLeakageError(RuntimeError):
    """Raised when test batch_ids leaked into train/scaler/OOD artifacts."""


@dataclass
class BatchTrajectory:
    """One batch of IndPenSim *reference* biomass and three model trajectories."""

    batch_id: Any
    time: np.ndarray
    x_reference: np.ndarray
    mechanistic: np.ndarray
    nn_only: np.ndarray
    hybrid: np.ndarray
    phase_labels: np.ndarray | None = None
    ood_mask: np.ndarray | None = None
    latency_s: dict[str, np.ndarray] | None = None

    def prediction(self, model: str) -> np.ndarray:
        if model == MODEL_MECHANISTIC:
            return np.asarray(self.mechanistic, dtype=np.float64)
        if model == MODEL_NN_ONLY:
            return np.asarray(self.nn_only, dtype=np.float64)
        if model == MODEL_HYBRID:
            return np.asarray(self.hybrid, dtype=np.float64)
        raise KeyError(model)


@dataclass
class LeakageAudit:
    ok: bool
    test_ids: list[Any]
    leaked: dict[str, list[Any]] = field(default_factory=dict)
    inspected_files: list[str] = field(default_factory=list)
    test_split_used_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "test_ids": [str(x) for x in self.test_ids],
            "leaked": {k: [str(x) for x in v] for k, v in self.leaked.items()},
            "inspected_files": list(self.inspected_files),
            "test_split_used_flags": list(self.test_split_used_flags),
        }


@dataclass
class EvaluationRun:
    status: str
    reason: str
    production_claim: bool
    leakage_audit: LeakageAudit | None
    per_batch: list[ModelBatchMetrics] = field(default_factory=list)
    aggregate: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_checks: dict[str, Any] = field(default_factory=dict)
    n_batches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "production_claim": self.production_claim,
            "n_batches": self.n_batches,
            "leakage_audit": None if self.leakage_audit is None else self.leakage_audit.as_dict(),
            "per_batch": [row.as_dict() for row in self.per_batch],
            "aggregate": self.aggregate,
            "skipped_checks": dict(self.skipped_checks),
            "models": list(MODEL_ORDER),
            "reference_label": "IndPenSim Reference",
            "hybrid_assumed_winner": False,
        }


def _norm_id(value: Any) -> str:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _walk_ids_from_mapping(payload: Mapping[str, Any], *, source: str) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    for key in FIT_ID_KEYS:
        if key in payload and payload[key] is not None:
            found[f"{source}:{key}"] = list(payload[key])
    for nested_key in ("metadata", "scaler", "ood", "training"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            found.update(_walk_ids_from_mapping(nested, source=f"{source}.{nested_key}"))
    return found


def collect_fit_batch_ids(artifact_roots: Sequence[Path]) -> tuple[dict[str, list[Any]], list[str], list[str]]:
    """Read train/scaler/OOD batch ids from JSON and optional joblib metadata."""
    by_source: dict[str, list[Any]] = {}
    files: list[str] = []
    used_flags: list[str] = []
    for root in artifact_roots:
        root = Path(root)
        if not root.exists():
            continue
        json_files = [root] if root.is_file() and root.suffix == ".json" else list(root.rglob("*.json"))
        for path in json_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            files.append(str(path))
            if not isinstance(payload, dict):
                continue
            if payload.get("test_split_used") is True:
                used_flags.append(str(path))
            by_source.update(_walk_ids_from_mapping(payload, source=str(path)))
        joblibs = [root] if root.is_file() and root.suffix == ".joblib" else list(root.rglob("*.joblib"))
        for path in joblibs:
            try:
                obj = joblib.load(path)
            except Exception:
                continue
            files.append(str(path))
            ids = getattr(obj, "train_batch_ids", None)
            if ids is not None:
                by_source[f"{path}:train_batch_ids"] = list(ids)
            if isinstance(obj, dict):
                by_source.update(_walk_ids_from_mapping(obj, source=str(path)))
    return by_source, files, used_flags


def default_artifact_roots(project_root: Path | None = None) -> list[Path]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return [
        root / "models",
        root / "data" / "processed",
        root / "results",
    ]


def audit_test_holdout(
    test_ids: Iterable[Any],
    artifact_roots: Sequence[Path],
) -> LeakageAudit:
    test_set = {_norm_id(i) for i in test_ids}
    by_source, files, used_flags = collect_fit_batch_ids(artifact_roots)
    leaked: dict[str, list[Any]] = {}
    for source, ids in by_source.items():
        overlap = sorted({_norm_id(i) for i in ids} & test_set)
        if overlap:
            leaked[source] = overlap
    ok = not leaked and not used_flags
    return LeakageAudit(
        ok=ok,
        test_ids=sorted(test_set),
        leaked=leaked,
        inspected_files=files,
        test_split_used_flags=used_flags,
    )


def assert_test_holdout_clean(audit: LeakageAudit) -> None:
    if audit.ok:
        return
    parts = []
    if audit.leaked:
        parts.append(f"test batch_ids in train/scaler/OOD artifacts: {audit.leaked}")
    if audit.test_split_used_flags:
        parts.append(f"test_split_used=true in {audit.test_split_used_flags}")
    raise TestSplitLeakageError(f"{TEST_SPLIT_FORBIDDEN} " + "; ".join(parts))


def _data_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.name not in SKIP_NAMES:
            out.append(path)
    return out


def indpensim_on_disk(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    raw = _data_files(root / "data" / "raw")
    processed = _data_files(root / "data" / "processed")
    present = bool(raw or processed)
    return {
        "present": present,
        "raw_files": [str(p) for p in raw],
        "processed_files": [str(p) for p in processed],
    }


def trained_model_status(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    residual_dir = root / "models" / "residual"
    nn_dir = root / "models" / "nn_only"
    ood_dir = root / "models" / "ood"
    residual_ok = (residual_dir / "residual_weights.pt").is_file() and (
        residual_dir / "scaler.joblib"
    ).is_file()
    nn_ok = (nn_dir / "nn_only_weights.pt").is_file() and (nn_dir / "scaler.joblib").is_file()
    ood_ok = (ood_dir / "ood_stats.joblib").is_file() or (ood_dir / "ood_metadata.json").is_file()
    return {
        "mechanistic_code_present": (root / "src" / "mechanistic" / "model.py").is_file(),
        "nn_only_weights_present": nn_ok,
        "residual_weights_present": residual_ok,
        "ood_fit_present": ood_ok,
        "hybrid_ready": residual_ok and ood_ok,
        "prompt6": prompt6_status(),
        "inference_engine": _inference_engine_status(root),
    }


def evaluate_batches(
    trajectories: Sequence[BatchTrajectory],
    *,
    early_fraction: float = 0.25,
    late_fraction: float = 0.25,
) -> tuple[list[ModelBatchMetrics], dict[str, dict[str, Any]]]:
    """Score supplied trajectories. Does not claim a production 20-batch result."""
    rows: list[ModelBatchMetrics] = []
    for traj in trajectories:
        y_true = np.asarray(traj.x_reference, dtype=np.float64)
        for model in MODEL_ORDER:
            lat = None if traj.latency_s is None else traj.latency_s.get(model)
            rows.append(
                score_model_on_batch(
                    model=model,
                    batch_id=traj.batch_id,
                    y_true=y_true,
                    y_pred=traj.prediction(model),
                    phase_labels=traj.phase_labels,
                    ood_mask=traj.ood_mask,
                    latency_s=lat,
                    early_fraction=early_fraction,
                    late_fraction=late_fraction,
                )
            )
    aggregate: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        aggregate[model] = aggregate_model_metrics([r for r in rows if r.model == model])
    return rows, aggregate


def _load_test_ids(split_path: Path | None, project_root: Path) -> list[Any]:
    if split_path is not None:
        split = load_split_manifest(split_path)
        return list(split["test"])
    default = project_root / "data" / "splits" / "manifest.json"
    alt = project_root / "data" / "splits" / "split.json"
    path = default if default.is_file() else alt
    if not path.is_file():
        return []
    split = load_split_manifest(path)
    return list(split["test"])


def run_final_test(
    *,
    project_root: str | Path | None = None,
    artifact_roots: Sequence[str | Path] | None = None,
    split_path: str | Path | None = None,
    trajectories: Sequence[BatchTrajectory] | None = None,
) -> EvaluationRun:
    """Official hold-out evaluation. Never a fake 20-batch production table.

    Leakage into train/scaler/OOD artifacts raises ``TestSplitLeakageError``.
    Missing IndPenSim data or untrained models returns ``status='skipped'``.
    Passing ``trajectories`` still requires a clean holdout audit and still
    sets ``production_claim=False`` (synthetic or caller-supplied paths).
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    roots = (
        [Path(p) for p in artifact_roots]
        if artifact_roots is not None
        else default_artifact_roots(root)
    )
    split_file = Path(split_path) if split_path is not None else None
    test_ids = _load_test_ids(split_file, root)
    if not test_ids and trajectories is None:
        dummy_test = []
    else:
        dummy_test = test_ids
    audit = audit_test_holdout(dummy_test, roots)
    if dummy_test:
        assert_test_holdout_clean(audit)

    data_info = indpensim_on_disk(root)
    model_info = trained_model_status(root)
    ready = (
        data_info["present"]
        and model_info["nn_only_weights_present"]
        and model_info["residual_weights_present"]
        and model_info["hybrid_ready"]
        and model_info["inference_engine"] == "implemented"
        and len(test_ids) == 20
    )

    if trajectories is not None:
        if dummy_test:
            assert_test_holdout_clean(audit)
        rows, aggregate = evaluate_batches(trajectories)
        return EvaluationRun(
            status="ok_caller_trajectories",
            reason=(
                "Scored caller-supplied trajectories. This is not a production "
                "IndPenSim 20-batch result."
            ),
            production_claim=False,
            leakage_audit=audit,
            per_batch=rows,
            aggregate=aggregate,
            skipped_checks={"indpensim": data_info, "models": model_info},
            n_batches=len(trajectories),
        )

    if not ready:
        reasons = []
        if not data_info["present"]:
            reasons.append("IndPenSim raw/processed files are not on disk")
        if not model_info["nn_only_weights_present"]:
            reasons.append("Prompt 6 NN-only weights are missing (stub interface only)")
        if not model_info["residual_weights_present"]:
            reasons.append("residual MLP weights are missing")
        if not model_info["hybrid_ready"]:
            reasons.append("hybrid/OOD artifacts are not ready")
        if model_info["inference_engine"] in {"stub", "missing"}:
            reasons.append("hybrid inference engine is not available")
        if len(test_ids) != 20:
            reasons.append(
                f"test split has {len(test_ids)} ids (need 20 untouched batches from the manifest)"
            )
        return EvaluationRun(
            status="skipped",
            reason="; ".join(reasons) if reasons else "final evaluation not ready",
            production_claim=False,
            leakage_audit=audit,
            skipped_checks={"indpensim": data_info, "models": model_info, "n_test_ids": len(test_ids)},
            n_batches=0,
        )

    return EvaluationRun(
        status="skipped",
        reason=(
            "Holdout looks clean and artifacts may be present, but this runner "
            "does not walk the official 20 IndPenSim test batches end-to-end. "
            "Refusing to fabricate production metrics."
        ),
        production_claim=False,
        leakage_audit=audit,
        skipped_checks={"indpensim": data_info, "models": model_info},
        n_batches=0,
    )


def make_synthetic_trajectories(n_batches: int = 3, n_steps: int = 40, seed: int = 0) -> list[BatchTrajectory]:
    """Deterministic toy curves for unit tests. Not IndPenSim."""
    rng = np.random.default_rng(seed)
    out: list[BatchTrajectory] = []
    t = np.linspace(0.0, 1.0, n_steps)
    phases = np.array(
        ["growth"] * (n_steps // 3)
        + ["production"] * (n_steps // 3)
        + ["autolysis"] * (n_steps - 2 * (n_steps // 3)),
        dtype=object,
    )
    for i in range(n_batches):
        ref = 0.2 + 4.0 * t + 0.15 * np.sin(6.0 * np.pi * t)
        mechanistic = ref - 0.4 - 0.1 * t
        nn_only = ref + 0.05 * rng.normal(size=n_steps)
        hybrid = 0.5 * mechanistic + 0.5 * nn_only
        ood = np.zeros(n_steps, dtype=bool)
        ood[-max(1, n_steps // 8) :] = True
        lat = {
            MODEL_MECHANISTIC: np.full(n_steps, 0.002),
            MODEL_NN_ONLY: np.full(n_steps, 0.001),
            MODEL_HYBRID: np.full(n_steps, 0.003),
        }
        out.append(
            BatchTrajectory(
                batch_id=f"synth-{i}",
                time=t,
                x_reference=ref,
                mechanistic=mechanistic,
                nn_only=nn_only,
                hybrid=hybrid,
                phase_labels=phases,
                ood_mask=ood,
                latency_s=lat,
            )
        )
    return out


def main() -> None:
    run = run_final_test()
    print(json.dumps(run.as_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
