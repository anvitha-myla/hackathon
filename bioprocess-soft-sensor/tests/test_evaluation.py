"""Unit tests for RMSE/MAE/R² and evaluation leakage gating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import mae, r2_score, rank_models_by_metric, rmse
from src.evaluation.plots import REFERENCE_LABEL, plot_trajectory_comparison, reference_series_label
from src.evaluation.report import markdown_report
from src.evaluation.runner import (
    TestSplitLeakageError,
    audit_test_holdout,
    evaluate_batches,
    make_synthetic_trajectories,
    run_final_test,
)


def test_rmse_mae_perfect_match() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2_score(y, y) == pytest.approx(1.0)


def test_rmse_mae_known_values() -> None:
    y = np.array([0.0, 1.0, 2.0])
    yhat = np.array([1.0, 1.0, 1.0])
    assert mae(y, yhat) == pytest.approx(2.0 / 3.0)
    assert rmse(y, yhat) == pytest.approx(np.sqrt((1.0 + 0.0 + 1.0) / 3.0))


def test_r2_classic_example() -> None:
    y = np.array([3.0, -0.5, 2.0, 7.0])
    yhat = np.array([2.5, 0.0, 2.0, 8.0])
    # ss_res = 0.25 + 0.25 + 0 + 1 = 1.5
    # ymean = 2.875; ss_tot = (0.125^2 + (-3.375)^2 + (-0.875)^2 + 4.125^2)
    expected = 1.0 - 1.5 / float(np.sum((y - np.mean(y)) ** 2))
    assert r2_score(y, yhat) == pytest.approx(expected)


def test_r2_undefined_for_constant_reference() -> None:
    y = np.ones(4)
    yhat = np.array([1.0, 1.1, 0.9, 1.0])
    assert np.isnan(r2_score(y, yhat))


def test_metrics_ignore_nan_pairs() -> None:
    y = np.array([1.0, np.nan, 3.0])
    yhat = np.array([1.0, 2.0, 5.0])
    assert mae(y, yhat) == pytest.approx(1.0)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(2.0))


def test_runner_refuses_when_test_ids_in_train_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "models" / "residual"
    artifact.mkdir(parents=True)
    (artifact / "training_metadata.json").write_text(
        json.dumps(
            {
                "train_batch_ids": [1, 2, 81],
                "validation_batch_ids": [61],
                "test_split_used": False,
            }
        ),
        encoding="utf-8",
    )
    split = {
        "train": list(range(1, 61)),
        "validation": list(range(61, 81)),
        "test": list(range(81, 101)),
    }
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "splits" / "manifest.json").write_text(
        json.dumps(split), encoding="utf-8"
    )
    with pytest.raises(TestSplitLeakageError):
        run_final_test(project_root=tmp_path, artifact_roots=[artifact])


def test_runner_refuses_scaler_batch_id_leak(tmp_path: Path) -> None:
    artifact = tmp_path / "scaler_meta"
    artifact.mkdir()
    (artifact / "scaler_meta.json").write_text(
        json.dumps({"scaler_batch_ids": ["b-test-1", "b-train-1"]}),
        encoding="utf-8",
    )
    audit = audit_test_holdout(["b-test-1"], [artifact])
    assert audit.ok is False
    with pytest.raises(TestSplitLeakageError):
        run_final_test(
            project_root=tmp_path,
            artifact_roots=[artifact],
            split_path=_write_split(tmp_path, test=["b-test-1"], train=["b-train-1"]),
        )


def test_runner_refuses_ood_fit_leak_and_test_split_used_flag(tmp_path: Path) -> None:
    artifact = tmp_path / "ood"
    artifact.mkdir()
    (artifact / "ood_metadata.json").write_text(
        json.dumps({"ood_fit_batch_ids": [99], "test_split_used": True}),
        encoding="utf-8",
    )
    with pytest.raises(TestSplitLeakageError):
        run_final_test(
            project_root=tmp_path,
            artifact_roots=[artifact],
            split_path=_write_split(tmp_path, test=[99], train=[1]),
        )


def test_readme_only_raw_dir_is_not_indpensim(tmp_path: Path) -> None:
    from src.evaluation.runner import indpensim_on_disk

    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "README.md").write_text("place files here\n", encoding="utf-8")
    (tmp_path / "data" / "processed").mkdir(parents=True)
    info = indpensim_on_disk(tmp_path)
    assert info["present"] is False


def test_run_final_test_skips_without_indpensim(tmp_path: Path) -> None:
    _write_split(tmp_path, test=list(range(81, 101)), train=list(range(1, 61)))
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "src" / "inference").mkdir(parents=True)
    (tmp_path / "src" / "mechanistic").mkdir(parents=True)
    (tmp_path / "src" / "inference" / "engine.py").write_text(
        '"""Hybrid inference stub: X_hybrid = X_mech + beta_trust * delta_X. Implemented in Prompt 8."""\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "mechanistic" / "model.py").write_text("# placeholder\n", encoding="utf-8")
    run = run_final_test(project_root=tmp_path, artifact_roots=[tmp_path / "models"])
    assert run.status == "skipped"
    assert run.production_claim is False
    assert run.n_batches == 0
    assert "IndPenSim" in run.reason


def test_evaluate_synthetic_trajectories_does_not_claim_hybrid_wins() -> None:
    trajs = make_synthetic_trajectories(n_batches=3, n_steps=40, seed=1)
    rows, aggregate = evaluate_batches(trajs)
    assert len(rows) == 9
    ranking = rank_models_by_metric(aggregate, "rmse", lower_is_better=True)
    assert set(ranking) == {"mechanistic", "nn_only", "hybrid"}
    from src.evaluation.runner import EvaluationRun

    run = EvaluationRun(
        status="ok_caller_trajectories",
        reason="synthetic",
        production_claim=False,
        leakage_audit=None,
        per_batch=rows,
        aggregate=aggregate,
        n_batches=3,
    )
    text = markdown_report(run)
    assert "assumed to win" in text.lower()
    assert "not" in text.lower()
    assert "IndPenSim Reference" in text
    assert run.production_claim is False
    assert "hybrid wins" not in text.lower()


def test_reference_plot_label() -> None:
    assert reference_series_label() == "IndPenSim Reference"
    assert REFERENCE_LABEL == "IndPenSim Reference"
    assert "ground truth" not in REFERENCE_LABEL.lower()


def test_plot_writes_png(tmp_path: Path) -> None:
    traj = make_synthetic_trajectories(n_batches=1, n_steps=16, seed=0)[0]
    dest = tmp_path / "traj.png"
    written = plot_trajectory_comparison(traj, path=dest)
    assert written is not None and written.is_file()
    assert written.stat().st_size > 0


def _write_split(root: Path, *, test: list, train: list) -> Path:
    path = root / "data" / "splits" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"train": train, "validation": [], "test": test}),
        encoding="utf-8",
    )
    return path
