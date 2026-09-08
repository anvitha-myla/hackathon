"""Prompt 11: robustness environment — perturbations, records, OOD fallback."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stress import (
    FAULT_TYPES,
    RECORD_COLUMNS,
    StubHybridPredictor,
    apply_perturbation,
    beta_trust,
    combine_hybrid,
    copy_frame,
    file_checksum,
    fit_ood_from_frames,
    load_clean_for_stress,
    load_stress_config,
    make_synthetic_batch,
    ml_contribution,
    run_stress_environment,
    score_perturbed,
    severe_ood_falls_back_to_physics,
    summarize_records,
    write_robustness_plots,
    write_summary_tables,
)


def test_stress_config_matches_spec_faults() -> None:
    cfg = load_stress_config()
    assert cfg["clean_data_write"] == "forbidden"
    pert = cfg["perturbations"]
    for name in (
        "random_noise",
        "spikes",
        "missing_values",
        "impossible_values",
        "sensor_drift",
        "distribution_shift",
        "feed_interruption",
        "temperature_disturbance",
        "combined_faults",
    ):
        assert name in pert


def test_beta_trust_formula_in_distribution_and_severe() -> None:
    # In distribution: D_M <= threshold → beta = 1
    assert beta_trust(0.5, d_threshold=1.0, kappa=1.0) == pytest.approx(1.0)
    assert beta_trust(1.0, d_threshold=1.0, kappa=2.5) == pytest.approx(1.0)
    # Severe OOD: excess 20, kappa 1 → exp(-20) ≈ 0
    severe = float(beta_trust(21.0, d_threshold=1.0, kappa=1.0))
    assert severe == pytest.approx(math.exp(-20.0))
    assert severe < 1e-8


def test_hybrid_equation_and_contributions() -> None:
    x_m, delta, beta = 4.0, 2.0, 0.5
    hybrid = combine_hybrid(x_m, delta, beta)
    assert hybrid == pytest.approx(5.0)
    assert ml_contribution(beta, delta) == pytest.approx(1.0)
    # beta → 0 ⇒ hybrid → mechanistic
    assert combine_hybrid(x_m, delta, 0.0) == pytest.approx(x_m)
    # beta = 1 ⇒ full ML correction
    assert combine_hybrid(x_m, delta, 1.0) == pytest.approx(x_m + delta)


def test_perturbations_copy_and_do_not_mutate_clean_frame() -> None:
    clean = make_synthetic_batch(batch_id=1, n=40, seed=3)
    snapshot = copy_frame(clean)
    cfg = load_stress_config()
    for fault in FAULT_TYPES:
        if fault == "normal":
            continue
        out, meta = apply_perturbation(clean, fault, config=cfg, rng=0)
        pd.testing.assert_frame_equal(clean, snapshot)
        assert id(out) != id(clean)
        assert meta["fault_type"] in {fault, "severe_ood"}
        assert not out.equals(clean) or fault == "normal"


def test_each_perturbation_has_expected_signature() -> None:
    clean = make_synthetic_batch(batch_id=2, n=50, seed=4)
    cfg = load_stress_config()

    noisy, _ = apply_perturbation(clean, "random_noise", config=cfg, rng=1)
    assert not np.allclose(noisy["DO"].to_numpy(), clean["DO"].to_numpy())

    spiked, _ = apply_perturbation(clean, "spikes", config=cfg, rng=2)
    delta = np.abs(spiked["DO"] - clean["DO"]) + np.abs(spiked["pH"] - clean["pH"])
    assert float(delta.max()) > 0.0

    missing, _ = apply_perturbation(clean, "missing_values", config=cfg, rng=3)
    assert missing.isna().any().any()

    impossible, _ = apply_perturbation(clean, "impossible_values", config=cfg, rng=4)
    assert (impossible["pH"] > 14).any() or (impossible["DO"] < 0).any()

    drift, _ = apply_perturbation(clean, "sensor_drift", config=cfg, rng=5)
    # End of trajectory should be more biased than the start.
    start = abs(float(drift["DO"].iloc[0] - clean["DO"].iloc[0]))
    end = abs(float(drift["DO"].iloc[-1] - clean["DO"].iloc[-1]))
    assert end >= start

    shifted, _ = apply_perturbation(clean, "distribution_shift", config=cfg, rng=6)
    assert abs(float(shifted["T_vessel"].mean() - clean["T_vessel"].mean())) > 1.0

    feed, _ = apply_perturbation(clean, "feed_interruption", config=cfg, rng=7)
    assert float((feed["feed_rate"] == 0).mean()) > 0.05

    temp, _ = apply_perturbation(clean, "temperature_disturbance", config=cfg, rng=8)
    assert float(np.nanmax(np.abs(temp["T_vessel"] - clean["T_vessel"]))) > 1.0

    combo, _ = apply_perturbation(clean, "combined_faults", config=cfg, rng=9)
    assert not combo.equals(clean)


def test_clean_file_checksum_unchanged_when_stress_reads(tmp_path: Path) -> None:
    clean = make_synthetic_batch(batch_id=7, n=20, seed=1)
    path = tmp_path / "final_test_batch.csv"
    clean.to_csv(path, index=False)
    before = file_checksum(path)
    frames = load_clean_for_stress(path)
    apply_perturbation(frames[0], "random_noise", rng=0)
    after = file_checksum(path)
    assert before == after
    with pytest.raises(ValueError, match="forbidden"):
        load_clean_for_stress(path, allow_write=True)


def test_records_contain_required_columns() -> None:
    train = [make_synthetic_batch(batch_id=i, n=40, seed=0) for i in range(1, 3)]
    test = make_synthetic_batch(batch_id=50, n=40, seed=99)
    ood = fit_ood_from_frames(train, kappa=1.0)
    pred = StubHybridPredictor(ood=ood, delta_x=1.25)
    rec = score_perturbed(test, "normal", pred, rng=0)
    for col in RECORD_COLUMNS:
        assert col in rec.columns
    assert rec["fault_type"].eq("normal").all()


def test_normal_ml_correction_active_severe_ood_falls_back() -> None:
    result = run_stress_environment(
        fault_types=("normal", "severe_ood"),
        output_dir=None,
        write_artifacts=False,
        seed=42,
    )
    stats = severe_ood_falls_back_to_physics(result.records)
    # Normal: ML correction active.
    assert stats["normal_mean_beta"] > 0.7
    assert stats["normal_mean_abs_ml"] > stats["severe_mean_abs_ml"]
    assert stats["normal_mean_abs_hybrid_minus_mech"] > 0.05
    # Severe OOD: beta → 0 and hybrid → mechanistic.
    assert stats["severe_mean_beta"] < 0.05
    assert stats["severe_mean_abs_hybrid_minus_mech"] < 0.05
    assert stats["severe_mean_beta"] < stats["normal_mean_beta"]


def test_no_crash_on_corrupted_sensors() -> None:
    train = [make_synthetic_batch(batch_id=i, n=30, seed=2) for i in range(1, 3)]
    test = make_synthetic_batch(batch_id=8, n=30, seed=5)
    ood = fit_ood_from_frames(train, kappa=1.0)
    pred = StubHybridPredictor(ood=ood, delta_x=0.8)
    wrecked = copy_frame(test)
    wrecked.loc[wrecked.index[0:5], "DO"] = np.nan
    wrecked.loc[wrecked.index[5:8], "pH"] = np.inf
    wrecked.loc[wrecked.index[8:10], "T_vessel"] = -np.inf
    wrecked.loc[wrecked.index[10], "feed_rate"] = 1e300
    rec = pred.predict_frame(wrecked)
    assert len(rec) == len(wrecked)
    assert rec["hybrid_prediction"].notna().all()
    assert np.isfinite(rec["hybrid_prediction"].to_numpy()).all()
    assert rec["beta_trust"].between(0.0, 1.0).all()


@pytest.mark.parametrize(
    "fault",
    [
        "random_noise",
        "spikes",
        "missing_values",
        "impossible_values",
        "sensor_drift",
        "distribution_shift",
        "feed_interruption",
        "temperature_disturbance",
        "combined_faults",
        "severe_ood",
        "normal",
    ],
)
def test_all_faults_run_without_crash(fault: str) -> None:
    result = run_stress_environment(
        fault_types=(fault,),
        write_artifacts=False,
        seed=1,
    )
    assert not result.records.empty
    assert set(RECORD_COLUMNS) <= set(result.records.columns)
    assert np.isfinite(result.records["hybrid_prediction"].to_numpy()).all()


def test_plots_and_summary_tables(tmp_path: Path) -> None:
    result = run_stress_environment(
        fault_types=("normal", "random_noise", "severe_ood", "combined_faults"),
        output_dir=tmp_path,
        write_artifacts=True,
        seed=42,
    )
    summary = summarize_records(result.records)
    assert {"fault_type", "rmse", "mae", "mean_beta_trust", "mean_abs_ML_contribution"} <= set(
        summary.columns
    )
    tables = write_summary_tables(summary, tmp_path)
    plots = write_robustness_plots(result.records, tmp_path)
    assert tables["csv"].exists()
    assert tables["json"].exists()
    names = {p.name for p in plots}
    assert "beta_vs_ood.png" in names
    assert "ml_contribution_by_fault.png" in names
    assert "hybrid_vs_mechanistic_severe_ood.png" in names
    assert (tmp_path / "stress_records.csv").exists()


def test_unknown_fault_raises() -> None:
    df = make_synthetic_batch(n=10, seed=0)
    with pytest.raises(ValueError, match="Unknown fault_type"):
        apply_perturbation(df, "not_a_fault")
