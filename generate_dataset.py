#!/usr/bin/env python3
"""
Generate the EQ-STBR-5000L 120-batch hydrogenation dataset.

Orchestrates deterministic semi-batch physics, stochastic SCADA artifacts,
and single/compound anomaly injection. Partitioning is strictly by
``batch_id`` (GroupKFold-safe): 84 train / 36 test.

Outputs
-------
dataset/raw/batch_XXX.csv
    12 core SCADA columns at 1 min resolution (150–240 rows).
dataset/targets/batch_XXX.csv
    Ground-truth ``reaction_conversion_pct`` per time step.
dataset/batch_summary.csv
    batch_id, split, anomaly_type, total_duration_min, final_conversion_pct,
    time_to_endpoint_min.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from anomalies import (
    AnomalyType,
    CASCADE_FEED_DELAY_MIN,
    COOLING_SPIKE_DURATION_MIN,
    COOLING_UA_DROP,
    FEED_PAUSE_DURATION_MIN,
    FOAMING_TORQUE_DROP,
    SENSOR_DRIFT_BIAS_C,
    generate_shift_note,
)
from equipment import ReactorEquipmentPackage
from stochastic import (
    AgitatorTorqueModel,
    apply_process_noise,
    sample_coa_purity,
)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
N_BATCHES = 120
N_TRAIN = 84
N_TEST = 36
DURATION_MIN_RANGE = (150, 240)
DT_MIN = 1.0  # logging grid: 1 minute per row
ENDPOINT_CONVERSION_PCT = 98.0
DEFAULT_SEED = 20260808

CORE_COLUMNS = [
    "batch_id",
    "time_step_min",
    "raw_material_purity_coa",
    "T_reactor",
    "T_jacket",
    "delta_T",
    "agitator_torque",
    "headspace_pressure",
    "H2_flow_rate",
    "cooling_water_flow",
    "shift_note",
    "anomaly_flag",
]


# ===========================================================================
# Batch plan (GroupKFold by batch_id)
# ===========================================================================


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    split: str  # "train" | "test"
    anomaly: AnomalyType


def build_batch_plan(seed: int = DEFAULT_SEED) -> list[BatchSpec]:
    """
    Construct the 120-batch anomaly distribution matrix.

    Training (84): ~70 NONE + ~14 single (no SENSOR_DRIFT, no compounds).
    Testing  (36): ~16 NONE + ~10 single (incl. SENSOR_DRIFT) + ~10 compound.
    """
    rng = random.Random(seed)

    train_anoms: list[AnomalyType] = (
        [AnomalyType.NONE] * 70
        + [AnomalyType.SURFACE_FOAMING] * 5
        + [AnomalyType.COOLING_SPIKE] * 5
        + [AnomalyType.FEED_PAUSE] * 4
    )
    assert len(train_anoms) == N_TRAIN
    # SENSOR_DRIFT must be 100% excluded from training
    assert AnomalyType.SENSOR_DRIFT not in train_anoms
    assert all(not a.is_compound for a in train_anoms)

    test_anoms: list[AnomalyType] = (
        [AnomalyType.NONE] * 16
        + [AnomalyType.SURFACE_FOAMING] * 2
        + [AnomalyType.COOLING_SPIKE] * 2
        + [AnomalyType.FEED_PAUSE] * 2
        + [AnomalyType.SENSOR_DRIFT] * 4
        + [AnomalyType.COMPOUND_FOAM_COOLING] * 5
        + [AnomalyType.COMPOUND_CASCADE] * 5
    )
    assert len(test_anoms) == N_TEST

    rng.shuffle(train_anoms)
    rng.shuffle(test_anoms)

    # Stable batch_id assignment: 1..84 train, 85..120 test (GroupKFold-safe)
    plan = [
        BatchSpec(batch_id=i + 1, split="train", anomaly=train_anoms[i])
        for i in range(N_TRAIN)
    ]
    plan += [
        BatchSpec(batch_id=N_TRAIN + i + 1, split="test", anomaly=test_anoms[i])
        for i in range(N_TEST)
    ]
    assert len(plan) == N_BATCHES
    return plan


# ===========================================================================
# Fast semi-batch physics (vector-friendly Euler, dt = 1 min)
# ===========================================================================


@dataclass
class PhysicsResult:
    time_min: np.ndarray
    T_reactor: np.ndarray
    T_jacket: np.ndarray
    pressure_bar: np.ndarray
    h2_flow_kg_min: np.ndarray
    cooling_water_m3_h: np.ndarray
    ua_W_K: np.ndarray
    q_cooling_W: np.ndarray
    conversion: np.ndarray  # 0..1
    agitator_torque_Nm: np.ndarray
    agitator_power_kW: np.ndarray


def _anomaly_schedules(
    n: int,
    anomaly: AnomalyType,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Boolean / scale schedules (length ``n``, index = minute) baked into ODE.
    """
    ua_scale = np.ones(n, dtype=np.float64)
    feed_scale = np.ones(n, dtype=np.float64)
    foam_scale = np.ones(n, dtype=np.float64)
    drift_bias = np.zeros(n, dtype=np.float64)

    def _window(start: int, duration: int) -> slice:
        a = int(max(0, min(n - 1, start)))
        b = int(max(a + 1, min(n, a + max(duration, 1))))
        return slice(a, b)

    peak = int(0.35 * n) + int(rng.integers(-10, 11))

    needs_cooling = anomaly in (
        AnomalyType.COOLING_SPIKE,
        AnomalyType.COMPOUND_FOAM_COOLING,
        AnomalyType.COMPOUND_CASCADE,
    )
    needs_foam = anomaly in (
        AnomalyType.SURFACE_FOAMING,
        AnomalyType.COMPOUND_FOAM_COOLING,
    )
    needs_feed = anomaly in (
        AnomalyType.FEED_PAUSE,
        AnomalyType.COMPOUND_CASCADE,
    )
    needs_drift = anomaly in (
        AnomalyType.SENSOR_DRIFT,
        AnomalyType.COMPOUND_CASCADE,
    )

    cool_start = peak
    if needs_cooling:
        dur = int(
            round(
                float(
                    rng.uniform(
                        COOLING_SPIKE_DURATION_MIN[0],
                        COOLING_SPIKE_DURATION_MIN[1],
                    )
                )
            )
        )
        sl = _window(cool_start, dur)
        ua_scale[sl] = 1.0 - COOLING_UA_DROP
        cool_start = sl.start

    if needs_foam:
        # Abrupt foaming near peak (~8–15 min persistence)
        foam_dur = int(rng.integers(8, 16))
        sl = _window(peak, foam_dur)
        foam_scale[sl] = 1.0 - FOAMING_TORQUE_DROP

    if needs_feed:
        if anomaly is AnomalyType.COMPOUND_CASCADE:
            feed_start = cool_start + int(CASCADE_FEED_DELAY_MIN)
        else:
            feed_start = int(0.50 * n) + int(rng.integers(-15, 16))
        sl = _window(feed_start, int(FEED_PAUSE_DURATION_MIN))
        feed_scale[sl] = 0.0

    if needs_drift:
        if anomaly is AnomalyType.COMPOUND_CASCADE:
            drift_start = cool_start
        else:
            drift_start = int(0.45 * n)
        # Linear 0 → +0.5 °C from drift_start to end; hold not needed (to EOS)
        if drift_start < n - 1:
            ramp = np.linspace(0.0, SENSOR_DRIFT_BIAS_C, n - drift_start)
            drift_bias[drift_start:] = ramp

    return {
        "ua_scale": ua_scale,
        "feed_scale": feed_scale,
        "foam_scale": foam_scale,
        "drift_bias": drift_bias,
    }


def simulate_batch_physics(
    duration_min: int,
    coa_purity: float,
    anomaly: AnomalyType,
    equipment: ReactorEquipmentPackage,
    seed: int,
) -> PhysicsResult:
    """
    Semi-batch catalytic hydrogenation Euler integration at 1 min steps.

    States: conversion X, reactor temperature T, headspace pressure P.
    Kinetics are first-order in residual nitroxylene with H2 availability
    and CoA assay factor. Exotherm is removed via η·U·A·ΔT.
    """
    rng = np.random.default_rng(seed)
    eq = equipment
    n = int(duration_min)
    t = np.arange(n, dtype=np.float64)  # minutes

    schedules = _anomaly_schedules(n, anomaly, rng)
    ua_scale = schedules["ua_scale"]
    feed_scale = schedules["feed_scale"]
    foam_scale = schedules["foam_scale"]
    drift_bias = schedules["drift_bias"]

    assay = float(coa_purity) / 100.0
    # Effective initial moles scale with CoA (over-purity → slightly more reactant)
    n_rxn_mol = eq.nitroxylene_moles * assay

    # Stoichiometry: assume ~3 mol H2 / mol nitroxylene (aromatic NO2 → NH2 path)
    h2_per_mol = 3.0
    h2_stoich_kg = n_rxn_mol * h2_per_mol * (eq.h2_molar_mass_kg_kmol / 1000.0)

    # Controllers / setpoints with mild batch-to-batch variation
    T_set = 85.0 + float(rng.normal(0.0, 1.5))
    T_jkt = float(
        np.clip(
            eq.tcu_supply_temp_C + rng.normal(0.0, 1.0),
            eq.tcu_supply_temp_min_C,
            eq.tcu_supply_temp_max_C,
        )
    )
    P_target = eq.headspace_pressure_target_bar + float(rng.normal(0.0, 0.15))
    feed_nom = float(np.clip(3.2 + rng.normal(0.0, 0.30), 2.0, eq.mfc_max_h2_feed_kg_min))

    # Thermal
    m_cp = eq.liquid_charge_mass_kg * eq.cp_mix_J_kgK + eq.c_vessel_J_K  # J/K
    ua0 = eq.ua_W_K  # W/K
    # Net heat of reaction absorbed by batch per kg H2 consumed
    dH_J_per_kg_H2 = 40.0e6  # J/kg H2 (~80 kJ/mol H2)

    # Kinetic rate constant [1/min] at reference T
    k_ref = 0.035 + float(rng.normal(0.0, 0.003))
    E_over_R = 4000.0  # K
    T_ref = 358.15  # K (~85 °C)

    torque_model = AgitatorTorqueModel(equipment=eq)

    X = np.zeros(n, dtype=np.float64)
    T = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=np.float64)
    F = np.zeros(n, dtype=np.float64)
    UA = np.zeros(n, dtype=np.float64)
    Q = np.zeros(n, dtype=np.float64)
    CW = np.zeros(n, dtype=np.float64)
    TAU = np.zeros(n, dtype=np.float64)
    PWR = np.zeros(n, dtype=np.float64)
    Tj = np.full(n, T_jkt, dtype=np.float64)

    T[0] = 55.0 + float(rng.normal(0.0, 1.0))  # heat-up start
    P[0] = 6.0 + float(rng.normal(0.0, 0.2))
    X[0] = 0.0

    dt_s = DT_MIN * 60.0
    cp_water = 4184.0  # J/(kg·K)
    rho_water = 997.0
    dT_cw = 8.0  # °C rise across jacket exchanger (nominal)

    for i in range(n):
        # Heat-up then feed phase
        heatup = 20.0 + float(rng.normal(0.0, 2.0))
        feeding = i >= heatup and X[i] < 0.995

        # Pressure + temperature-aware feed controller (cut back on exotherm)
        if feeding:
            F_cmd = feed_nom * (1.0 + 0.10 * (P_target - P[i]) / max(P_target, 1e-6))
            if T[i] > T_set + 5.0:
                F_cmd *= max(0.45, 1.0 - 0.04 * (T[i] - T_set))
            F_cmd = float(np.clip(F_cmd, 0.0, eq.mfc_max_h2_feed_kg_min))
        else:
            F_cmd = 0.0
        F[i] = F_cmd * feed_scale[i]

        UA[i] = ua0 * ua_scale[i]
        # Jacket tracks TCU supply; colder if reactor runs hot
        T_jkt_cmd = T_jkt - (2.0 if T[i] > T_set + 2.0 else 0.0)
        Tj[i] = float(
            np.clip(
                T_jkt_cmd + 0.3 * math.sin(i / 18.0) + rng.normal(0.0, 0.05),
                eq.tcu_supply_temp_min_C,
                eq.tcu_supply_temp_max_C + 5.0,
            )
        )

        # Kinetics
        T_K = T[i] + 273.15
        k = k_ref * math.exp(-E_over_R * (1.0 / T_K - 1.0 / T_ref))
        p_fac = max(P[i], 0.0) / (max(P[i], 0.0) + 2.0)
        r_kin = k * max(1.0 - X[i], 0.0) * p_fac * max(assay, 0.9)
        dm_h2_kin = r_kin * h2_stoich_kg * DT_MIN
        dm_h2_feed = F[i] * DT_MIN
        dm_h2 = min(dm_h2_kin, dm_h2_feed) if feeding else 0.0
        dm_hs = max(dm_h2_feed - dm_h2, 0.0)

        dX = dm_h2 / max(h2_stoich_kg, 1e-12)
        Q[i] = UA[i] * (T[i] - Tj[i])
        q_rxn_W = (dm_h2 * dH_J_per_kg_H2) / dt_s
        q_ag_W = 11.2e3 * foam_scale[i] * 0.10
        dT = ((q_rxn_W + q_ag_W - Q[i]) * dt_s) / m_cp

        V_hs = eq.headspace_volume_m3
        R = 8.314462618
        dn_hs = dm_hs / eq.h2_molar_mass_kg_kmol
        dP_bar = (dn_hs * 1000.0 * R * T_K / V_hs) / 1.0e5
        dP_ctrl = -0.15 * (P[i] - P_target)
        dP = dP_bar + dP_ctrl

        q_cool = max(Q[i], 0.0)
        m_dot_cw = q_cool / max(cp_water * dT_cw, 1e-6)
        CW[i] = (m_dot_cw / rho_water) * 3600.0

        TAU[i] = torque_model.torque_Nm(float(X[i])) * foam_scale[i]
        PWR[i] = torque_model.power_kW(float(X[i])) * foam_scale[i]

        if i + 1 < n:
            X[i + 1] = float(np.clip(X[i] + dX, 0.0, 1.0))
            T_next = T[i] + dT
            if not feeding:
                # Heat-up or post-reaction temperature hold toward setpoint
                T_next += 0.12 * (T_set - T[i])
            else:
                T_next += 0.03 * (T_set - T[i])
            T[i + 1] = float(np.clip(T_next, 50.0, 115.0))
            P[i + 1] = float(
                np.clip(
                    P[i] + dP,
                    eq.operating_pressure_min_bar * 0.5,
                    eq.design_pressure_max_bar,
                )
            )

    # Apply sensor drift to the *measured* reactor temperature channel later;
    # keep true T here and pass drift schedule out via attribute on result —
    # stored by adding bias only in SCADA assembly.
    T_meas = T + drift_bias

    return PhysicsResult(
        time_min=t,
        T_reactor=T_meas,
        T_jacket=Tj,
        pressure_bar=P,
        h2_flow_kg_min=F,
        cooling_water_m3_h=CW,
        ua_W_K=UA,
        q_cooling_W=Q,
        conversion=X,
        agitator_torque_Nm=TAU,
        agitator_power_kW=PWR,
    )


# ===========================================================================
# SCADA assembly / IO
# ===========================================================================


def _time_to_endpoint_min(conversion: np.ndarray, time_min: np.ndarray) -> float:
    thr = ENDPOINT_CONVERSION_PCT / 100.0
    hit = np.where(conversion >= thr)[0]
    if hit.size == 0:
        return float(time_min[-1])
    return float(time_min[int(hit[0])])


def physics_to_timeseries(phys: PhysicsResult) -> dict:
    """Map physics arrays into the dict schema expected by stochastic/anomalies."""
    return {
        "time": phys.time_min.tolist(),  # minutes (dt=1)
        "T_reactor": phys.T_reactor.tolist(),
        "T_jacket": phys.T_jacket.tolist(),
        "pressure": phys.pressure_bar.tolist(),
        "h2_flow": phys.h2_flow_kg_min.tolist(),
        "conversion": phys.conversion.tolist(),
        "agitator_torque": phys.agitator_torque_Nm.tolist(),
        "agitator_power_kW": phys.agitator_power_kW.tolist(),
        "UA": phys.ua_W_K.tolist(),
        "Q_cooling": phys.q_cooling_W.tolist(),
        "cooling_water_flow": phys.cooling_water_m3_h.tolist(),
    }


def assemble_scada_rows(
    batch_id: int,
    anomaly: AnomalyType,
    coa_purity: float,
    phys: PhysicsResult,
    noisy: dict,
    shift_note: Optional[str],
) -> tuple[list[dict], list[dict], dict]:
    """Build core SCADA rows, target rows, and summary dict."""
    n = len(phys.time_min)
    flag = anomaly.value if anomaly is not AnomalyType.NONE else "NONE"
    note = shift_note or ""

    T_rx = np.asarray(noisy.get("T_reactor", phys.T_reactor), dtype=np.float64)
    T_jk = np.asarray(noisy.get("T_jacket", phys.T_jacket), dtype=np.float64)
    torque = np.asarray(
        noisy.get("agitator_torque", phys.agitator_torque_Nm), dtype=np.float64
    )
    pressure = np.asarray(noisy.get("pressure", phys.pressure_bar), dtype=np.float64)
    h2 = np.asarray(noisy.get("h2_flow", phys.h2_flow_kg_min), dtype=np.float64)
    cw = np.asarray(phys.cooling_water_m3_h, dtype=np.float64)
    # Prefer post-anomaly cooling water if recomputed; else scale with UA schedule
    if "anomaly_ua_scale" in noisy:
        scale = np.asarray(noisy["anomaly_ua_scale"], dtype=np.float64)
        cw = cw * scale

    # Ensure length
    def _fit(a: np.ndarray) -> np.ndarray:
        if len(a) == n:
            return a
        if len(a) > n:
            return a[:n]
        out = np.zeros(n, dtype=np.float64)
        out[: len(a)] = a
        if len(a):
            out[len(a) :] = a[-1]
        return out

    T_rx, T_jk, torque, pressure, h2, cw = map(_fit, (T_rx, T_jk, torque, pressure, h2, cw))
    delta_T = T_rx - T_jk

    rows: list[dict] = []
    targets: list[dict] = []
    for i in range(n):
        rows.append(
            {
                "batch_id": batch_id,
                "time_step_min": int(phys.time_min[i]),
                "raw_material_purity_coa": round(float(coa_purity), 3),
                "T_reactor": round(float(T_rx[i]), 1),
                "T_jacket": round(float(T_jk[i]), 1),
                "delta_T": round(float(delta_T[i]), 1),
                "agitator_torque": round(float(torque[i]), 1),
                "headspace_pressure": round(float(pressure[i]), 2),
                "H2_flow_rate": round(float(h2[i]), 2),
                "cooling_water_flow": round(float(cw[i]), 3),
                "shift_note": note,
                "anomaly_flag": flag,
            }
        )
        targets.append(
            {
                "batch_id": batch_id,
                "time_step_min": int(phys.time_min[i]),
                "reaction_conversion_pct": round(float(phys.conversion[i]) * 100.0, 4),
            }
        )

    summary = {
        "batch_id": batch_id,
        "split": "train" if batch_id <= N_TRAIN else "test",
        "anomaly_type": flag,
        "total_duration_min": int(n),
        "final_conversion_pct": round(float(phys.conversion[-1]) * 100.0, 4),
        "time_to_endpoint_min": round(
            _time_to_endpoint_min(phys.conversion, phys.time_min), 1
        ),
        "raw_material_purity_coa": round(float(coa_purity), 3),
    }
    return rows, targets, summary


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_one_batch(
    spec: BatchSpec,
    out_raw: Path,
    out_targets: Path,
    equipment: ReactorEquipmentPackage,
    seed: int,
) -> dict:
    """Simulate → noise → write CSVs (anomalies embedded in physics schedules)."""
    rng = random.Random(seed + spec.batch_id * 997)
    duration = rng.randint(*DURATION_MIN_RANGE)
    coa = sample_coa_purity(rng)

    phys = simulate_batch_physics(
        duration_min=duration,
        coa_purity=coa,
        anomaly=spec.anomaly,
        equipment=equipment,
        seed=seed + spec.batch_id * 13,
    )
    ts = physics_to_timeseries(phys)

    noisy = apply_process_noise(
        ts,
        coa_purity=coa,
        equipment=equipment,
        rng_seed=seed + spec.batch_id * 29,
    )

    # Shift notes: sparse on normals, denser on labeled anomalies / compounds
    if spec.anomaly is AnomalyType.NONE:
        note_p = 0.05
        always = False
    elif spec.anomaly.is_compound:
        note_p = 1.0
        always = True
    else:
        note_p = 0.40
        always = False
    shift_note = generate_shift_note(
        spec.anomaly,
        always=always,
        probability=note_p,
        seed=seed + spec.batch_id * 41,
    )

    rows, targets, summary = assemble_scada_rows(
        batch_id=spec.batch_id,
        anomaly=spec.anomaly,
        coa_purity=coa,
        phys=phys,
        noisy=noisy,
        shift_note=shift_note,
    )
    summary["split"] = spec.split

    _write_csv(out_raw / f"batch_{spec.batch_id:03d}.csv", rows, CORE_COLUMNS)
    _write_csv(
        out_targets / f"batch_{spec.batch_id:03d}.csv",
        targets,
        ["batch_id", "time_step_min", "reaction_conversion_pct"],
    )
    return summary


def generate_dataset(
    output_dir: Path | str = "dataset",
    seed: int = DEFAULT_SEED,
    n_batches: int = N_BATCHES,
) -> Path:
    """
    Generate the full dataset under ``output_dir``.

    Returns the path to ``batch_summary.csv``.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    tgt_dir = output_dir / "targets"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    plan = build_batch_plan(seed=seed)[:n_batches]
    equipment = ReactorEquipmentPackage.default()
    summaries: list[dict] = []

    t0 = time.perf_counter()
    for spec in plan:
        summaries.append(
            generate_one_batch(
                spec=spec,
                out_raw=raw_dir,
                out_targets=tgt_dir,
                equipment=equipment,
                seed=seed,
            )
        )
    elapsed = time.perf_counter() - t0

    summary_path = output_dir / "batch_summary.csv"
    _write_csv(
        summary_path,
        summaries,
        [
            "batch_id",
            "split",
            "anomaly_type",
            "total_duration_min",
            "final_conversion_pct",
            "time_to_endpoint_min",
            "raw_material_purity_coa",
        ],
    )

    # Write a compact manifest for GroupKFold consumers
    manifest_path = output_dir / "split_manifest.csv"
    _write_csv(
        manifest_path,
        [
            {
                "batch_id": s.batch_id,
                "split": s.split,
                "anomaly_type": s.anomaly.value,
            }
            for s in plan
        ],
        ["batch_id", "split", "anomaly_type"],
    )

    _validate_dataset(output_dir, plan, elapsed)
    return summary_path


def _validate_dataset(output_dir: Path, plan: list[BatchSpec], elapsed_s: float) -> None:
    raw_dir = output_dir / "raw"
    summary_path = output_dir / "batch_summary.csv"
    assert summary_path.exists()

    files = sorted(raw_dir.glob("batch_*.csv"))
    assert len(files) == len(plan), f"expected {len(plan)} batches, found {len(files)}"

    # Distribution checks
    train = [s for s in plan if s.split == "train"]
    test = [s for s in plan if s.split == "test"]
    assert len(train) == min(N_TRAIN, len(plan))
    assert all(s.anomaly is not AnomalyType.SENSOR_DRIFT for s in train)
    assert all(not s.anomaly.is_compound for s in train)
    assert any(s.anomaly is AnomalyType.SENSOR_DRIFT for s in test) or len(test) == 0
    assert any(s.anomaly.is_compound for s in test) or len(test) == 0

    # Spot-check one CSV schema / duration
    with files[0].open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CORE_COLUMNS
        rows = list(reader)
    assert DURATION_MIN_RANGE[0] <= len(rows) <= DURATION_MIN_RANGE[1]

    print(
        f"Generated {len(plan)} batches → {output_dir}/ in {elapsed_s:.2f}s "
        f"(train={sum(1 for s in plan if s.split=='train')}, "
        f"test={sum(1 for s in plan if s.split=='test')})"
    )
    if elapsed_s > 30.0:
        print(f"WARNING: generation exceeded 30s budget ({elapsed_s:.2f}s)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset root directory (default: ./dataset)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-batches",
        type=int,
        default=N_BATCHES,
        help="Number of batches to generate (default 120)",
    )
    args = parser.parse_args(argv)
    generate_dataset(output_dir=args.output_dir, seed=args.seed, n_batches=args.n_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
