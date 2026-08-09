#!/usr/bin/env python3
"""
EQ-STBR-5000L 120-batch dataset generator.

Orchestrates the 8-step process state machine, stiff kinetic ODE for
ACTIVE_HYDROGENATION, stochastic SCADA artifacts, and step-bounded
anomalies. Partitioning is GroupKFold-safe by ``batch_id`` (84 train / 36 test).

Outputs
-------
dataset/raw/batch_XXX.csv
    14 core SCADA columns at 1-minute resolution (~330 rows / batch).
dataset/targets/batch_XXX.csv
    Ground-truth ``reaction_conversion_pct`` (+ ``time_to_endpoint_min`` meta).
dataset/ground_truth/batch_summary.csv
    batch_id, split, anomaly_type, total_duration_min, final_conversion_pct.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from anomalies import AnomalySchedule, AnomalyType, build_anomaly_schedule
from equipment import (
    TOTAL_BATCH_DURATION_MIN,
    ProcessStep,
    ReactorEquipmentPackage,
    STEP_BOUNDS,
    STEP_DURATIONS_MIN,
    STEP_ORDER,
    build_step_constraints,
)
from stochastic import apply_process_noise, sample_coa_purity

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
N_BATCHES = 120
N_TRAIN = 84
N_TEST = 36
ENDPOINT_CONVERSION_PCT = 98.0
DEFAULT_SEED = 20260809
DT_MIN = 1.0

CORE_COLUMNS = [
    "batch_id",
    "time_step_min",
    "current_step_id",
    "raw_material_purity_coa",
    "vessel_weight_kg",
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
# Batch plan
# ===========================================================================


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    split: str
    anomaly: AnomalyType


def build_batch_plan(seed: int = DEFAULT_SEED) -> List[BatchSpec]:
    """
    Train (84): 70 NONE + 14 single (no SENSOR_DRIFT, no compounds).
    Test  (36): 16 NONE + 10 single (incl. SENSOR_DRIFT) + 10 compound.
    """
    rng = random.Random(seed)
    train = (
        [AnomalyType.NONE] * 70
        + [AnomalyType.SURFACE_FOAMING] * 5
        + [AnomalyType.COOLING_SPIKE] * 5
        + [AnomalyType.FEED_PAUSE] * 4
    )
    test = (
        [AnomalyType.NONE] * 16
        + [AnomalyType.SURFACE_FOAMING] * 2
        + [AnomalyType.COOLING_SPIKE] * 2
        + [AnomalyType.FEED_PAUSE] * 2
        + [AnomalyType.SENSOR_DRIFT] * 4
        + [AnomalyType.COMPOUND_FOAM_COOLING] * 5
        + [AnomalyType.COMPOUND_CASCADE] * 5
    )
    assert len(train) == N_TRAIN and len(test) == N_TEST
    assert AnomalyType.SENSOR_DRIFT not in train
    assert all(not a.is_compound for a in train)
    rng.shuffle(train)
    rng.shuffle(test)
    plan = [
        BatchSpec(i + 1, "train", train[i]) for i in range(N_TRAIN)
    ] + [
        BatchSpec(N_TRAIN + i + 1, "test", test[i]) for i in range(N_TEST)
    ]
    return plan


# ===========================================================================
# Batch trajectory buffers
# ===========================================================================


@dataclass
class BatchTrajectory:
    time_min: np.ndarray
    step_id: np.ndarray
    weight_kg: np.ndarray
    T_reactor: np.ndarray
    T_jacket: np.ndarray
    pressure_bar: np.ndarray
    h2_flow: np.ndarray
    cooling_L_min: np.ndarray
    torque_Nm: np.ndarray
    conversion: np.ndarray  # 0..1
    ua_scale: np.ndarray


def _empty_trajectory(n: int) -> BatchTrajectory:
    z = np.zeros(n, dtype=np.float64)
    return BatchTrajectory(
        time_min=np.arange(n, dtype=np.float64),
        step_id=np.zeros(n, dtype=np.int32),
        weight_kg=z.copy(),
        T_reactor=z.copy(),
        T_jacket=z.copy(),
        pressure_bar=z.copy(),
        h2_flow=z.copy(),
        cooling_L_min=z.copy(),
        torque_Nm=z.copy(),
        conversion=z.copy(),
        ua_scale=np.ones(n, dtype=np.float64),
    )


# ===========================================================================
# Step-wise deterministic physics
# ===========================================================================


def _lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def _n2_cycle_pressure(frac: float) -> float:
    """
    N2 leak-check cycle: 1 → 3 → 1 bar over the inerting step.
    Two half-cycles for realism.
    """
    # Triangle over [0,1]: 1→3→1→3→1 compressed into one step
    phase = (frac * 2.0) % 1.0
    if phase < 0.5:
        return _lerp(1.0, 3.0, phase / 0.5)
    return _lerp(3.0, 1.0, (phase - 0.5) / 0.5)


def _simulate_active_hydrogenation_ode(
    eq: ReactorEquipmentPackage,
    n_A0: float,
    T0_C: float,
    P0_bar: float,
    mass0_kg: float,
    duration_min: int,
    ua_scale: np.ndarray,
    feed_scale: np.ndarray,
    foam_scale: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stiff kinetic ODE for Step 5 using ``solve_ivp`` (BDF).

    State y = [X, T_C, P_bar, m_kg]
      X     conversion (0..1)
      T_C   reactor temperature (°C)
      P_bar headspace H2 pressure
      m_kg  vessel liquid mass

    Returns per-minute arrays:
      X, T, P, m, H2_flow, cooling_L_min, torque
    """
    V_L = eq.working_fill_volume_L
    R = 8.314462618
    T_set = 80.0
    P_target = eq.headspace_pressure_target_bar
    feed_nom = float(
        np.clip(
            rng.uniform(eq.h2_feed_min_kg_min, eq.h2_feed_max_kg_min),
            eq.h2_feed_min_kg_min,
            eq.mfc_max_h2_feed_kg_min,
        )
    )
    T_jkt_base = float(
        np.clip(eq.tcu_supply_temp_C + rng.normal(0.0, 1.0), 15.0, 25.0)
    )
    M_H2 = eq.h2_molar_mass_kg_kmol / 1000.0  # kg/mol

    # Piecewise-constant schedules sampled inside RHS via floor(t)
    def _sched(arr: np.ndarray, t: float) -> float:
        i = int(np.clip(math.floor(t), 0, duration_min - 1))
        return float(arr[i])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        X, T_C, P, m = y
        X = float(np.clip(X, 0.0, 1.0))
        C_A = n_A0 * max(1.0 - X, 0.0) / V_L
        ua_s = _sched(ua_scale, t)
        f_s = _sched(feed_scale, t)
        foam_s = _sched(foam_scale, t)

        # Temperature-aware H2 dosing (1.5–3.5 kg/min band)
        F_cmd = feed_nom
        if T_C > T_set + 2.0:
            F_cmd *= max(0.55, 1.0 - 0.04 * (T_C - T_set))
        elif T_C < T_set - 5.0:
            F_cmd = min(eq.h2_feed_max_kg_min, F_cmd * 1.1)
        F = float(np.clip(F_cmd, 0.0, eq.mfc_max_h2_feed_kg_min)) * f_s

        r = eq.reaction_rate_mol_L_min(T_C, C_A, max(P, 0.0))
        r_H2_kg_min = r * V_L * eq.stoich_H2 * M_H2
        cons_kg_min = min(r_H2_kg_min, F) if F > 0.0 else 0.0
        cons_mol_A_min = (cons_kg_min / M_H2) / eq.stoich_H2
        dX_dt = cons_mol_A_min / max(n_A0, 1e-12)

        T_jkt = T_jkt_base - (3.0 if T_C > T_set else 0.0)
        T_jkt = float(np.clip(T_jkt, eq.tcu_supply_temp_min_C, eq.tcu_supply_temp_max_C))
        Q = eq.ua_W_K * ua_s * (T_C - T_jkt)  # W
        r_eff = cons_mol_A_min / V_L
        q_rxn_W = (-eq.delta_H_J_mol) * (r_eff * V_L) / 60.0
        P_ag = eq.agitator_power_W * foam_s
        dT_dt = (q_rxn_W - Q + P_ag) / max(eq.thermal_mass_J_K(m), 1.0) * 60.0
        dT_dt += 0.04 * (T_set - T_C)  # cascade hold toward 80°C

        # Pressure control: strong vent/fill toward 10 bar
        dm_hs = max(F - cons_kg_min, 0.0)
        dn_hs_kmol_min = dm_hs / eq.h2_molar_mass_kg_kmol
        T_K = T_C + 273.15
        dP_fill = (dn_hs_kmol_min * 1000.0 * R * T_K / eq.headspace_volume_m3) / 1.0e5
        dP_ctrl = -0.55 * (P - P_target)
        # Extra vent when above operating max
        if P > eq.operating_pressure_max_bar:
            dP_ctrl -= 0.8 * (P - eq.operating_pressure_max_bar)
        dP_dt = dP_fill + dP_ctrl

        dm_dt = cons_kg_min
        return np.array([dX_dt, dT_dt, dP_dt, dm_dt], dtype=np.float64)

    t_eval = np.arange(0.0, duration_min, DT_MIN)
    y0 = np.array([0.0, T0_C, P0_bar, mass0_kg], dtype=np.float64)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(duration_min - 1) + 1e-9),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
        max_step=1.0,
    )
    if not sol.success or sol.y.shape[1] != duration_min:
        Y = np.zeros((4, duration_min), dtype=np.float64)
        Y[:, 0] = y0
        for i in range(duration_min - 1):
            dy = rhs(float(i), Y[:, i])
            Y[:, i + 1] = Y[:, i] + dy * DT_MIN
            Y[0, i + 1] = np.clip(Y[0, i + 1], 0.0, 1.0)
            Y[1, i + 1] = np.clip(Y[1, i + 1], 60.0, 92.0)
            Y[2, i + 1] = np.clip(Y[2, i + 1], 5.0, eq.operating_pressure_max_bar)
            Y[3, i + 1] = max(Y[3, i + 1], mass0_kg)
        X, T, P, m = Y
    else:
        X = np.clip(sol.y[0], 0.0, 1.0)
        T = np.clip(sol.y[1], 60.0, 92.0)
        P = np.clip(sol.y[2], 5.0, eq.operating_pressure_max_bar)
        m = np.maximum(sol.y[3], mass0_kg)

    # Ensure Step 5 ends at ~98% (digest finishes the rest)
    if X[-1] < 0.98:
        gap = 0.98 - float(X[-1])
        X = np.clip(X + gap * (np.linspace(0.0, 1.0, duration_min) ** 1.15), 0.0, 0.98)

    H2 = np.zeros(duration_min, dtype=np.float64)
    CW = np.zeros(duration_min, dtype=np.float64)
    TAU = np.zeros(duration_min, dtype=np.float64)
    Tj = np.zeros(duration_min, dtype=np.float64)
    for i in range(duration_min):
        Tj[i] = float(
            np.clip(
                T_jkt_base - (3.0 if T[i] > T_set else 0.0),
                eq.tcu_supply_temp_min_C,
                eq.tcu_supply_temp_max_C,
            )
        )
        F_cmd = feed_nom
        if T[i] > T_set + 2.0:
            F_cmd *= max(0.55, 1.0 - 0.04 * (T[i] - T_set))
        H2[i] = float(np.clip(F_cmd, 0.0, eq.mfc_max_h2_feed_kg_min)) * float(feed_scale[i])
        if float(feed_scale[i]) > 0.0:
            H2[i] = float(np.clip(H2[i], eq.h2_feed_min_kg_min, eq.h2_feed_max_kg_min))

        # Cooling water 50–120 L/min from controlled reactor temperature
        # (jacket ΔT is large; flow demand tracks T_rx set-point error / level).
        t_frac = float(np.clip((T[i] - 70.0) / 20.0, 0.0, 1.0))
        CW[i] = eq.cooling_water_min_L_min + t_frac * (
            eq.cooling_water_max_L_min - eq.cooling_water_min_L_min
        )
        CW[i] *= 0.75 + 0.25 * float(ua_scale[i])
        if float(feed_scale[i]) <= 0.0:
            CW[i] = eq.cooling_water_min_L_min + 0.35 * (
                CW[i] - eq.cooling_water_min_L_min
            )
        CW[i] = float(
            np.clip(CW[i], eq.cooling_water_min_L_min, eq.cooling_water_max_L_min)
        )
        TAU[i] = eq.agitator_torque_Nm(
            conversion=float(X[i]), foam_scale=float(foam_scale[i]), mode="full"
        )

    return X, T, P, m, H2, CW, TAU, Tj


def simulate_batch(
    eq: ReactorEquipmentPackage,
    coa_purity: float,
    schedule: AnomalySchedule,
    seed: int,
) -> BatchTrajectory:
    """Run the full 8-step state machine for one batch."""
    rng = np.random.default_rng(seed)
    n = TOTAL_BATCH_DURATION_MIN
    traj = _empty_trajectory(n)
    constraints = build_step_constraints(eq)

    ua_scale = np.asarray(schedule.ua_scale, dtype=np.float64)
    feed_scale = np.asarray(schedule.feed_scale, dtype=np.float64)
    foam_scale = np.asarray(schedule.foam_scale, dtype=np.float64)
    sensor_bias = np.asarray(schedule.sensor_bias_C, dtype=np.float64)
    traj.ua_scale = ua_scale.copy()

    assay = float(coa_purity) / 100.0
    n_A0 = eq.nitroxylene_moles * assay

    # Carry state across steps
    weight = 0.0
    T_rx = 25.0
    P = 1.0
    X = 0.0

    for step in STEP_ORDER:
        a, b = STEP_BOUNDS[step]
        dur = b - a
        cons = constraints[step]
        frac = np.linspace(0.0, 1.0, dur, endpoint=False)

        if step is ProcessStep.ACTIVE_HYDROGENATION:
            X_s, T_s, P_s, m_s, H2_s, CW_s, TAU_s, Tj_s = _simulate_active_hydrogenation_ode(
                eq=eq,
                n_A0=n_A0,
                T0_C=max(T_rx, 68.0),
                P0_bar=max(P, 8.0),
                mass0_kg=max(weight, eq.liquid_charge_mass_kg),
                duration_min=dur,
                ua_scale=ua_scale[a:b],
                feed_scale=feed_scale[a:b],
                foam_scale=foam_scale[a:b],
                rng=rng,
            )
            traj.conversion[a:b] = X_s
            traj.T_reactor[a:b] = T_s
            traj.T_jacket[a:b] = Tj_s
            traj.pressure_bar[a:b] = P_s
            traj.weight_kg[a:b] = m_s
            traj.h2_flow[a:b] = H2_s
            traj.cooling_L_min[a:b] = CW_s
            traj.torque_Nm[a:b] = TAU_s
            traj.step_id[a:b] = step.step_id
            weight, T_rx, P, X = float(m_s[-1]), float(T_s[-1]), float(P_s[-1]), float(X_s[-1])
            continue

        for i, f in enumerate(frac):
            t = a + i
            traj.step_id[t] = step.step_id

            # --- Weight ---------------------------------------------------
            if step is ProcessStep.RAW_MATERIAL_LOADING:
                # Hit full charge by last minute
                f_w = i / max(dur - 1, 1)
                weight = _lerp(0.0, eq.liquid_charge_mass_kg, f_w)
            elif step is ProcessStep.PRODUCT_DISCHARGE:
                w_start = float(traj.weight_kg[a - 1]) if a > 0 else weight
                f_w = i / max(dur - 1, 1)
                weight = _lerp(w_start, 0.0, f_w)
            elif step is ProcessStep.PREPARATION_TARE:
                weight = 0.0
            elif step in (
                ProcessStep.NITROGEN_INERTING,
                ProcessStep.PRE_HEATING,
                ProcessStep.DIGESTION_HOLD,
                ProcessStep.DEGASSING_COOLING,
            ):
                weight = max(weight, eq.liquid_charge_mass_kg * 0.99)

            # --- Temperature / jacket -------------------------------------
            ua_s = float(ua_scale[t])
            if step is ProcessStep.PREPARATION_TARE:
                T_rx = 25.0 + float(rng.normal(0.0, 0.05))
                T_jkt = 25.0
            elif step is ProcessStep.RAW_MATERIAL_LOADING:
                T_rx = 25.0 + 0.3 * math.sin(f * math.pi) + float(rng.normal(0.0, 0.05))
                T_jkt = 25.0
            elif step is ProcessStep.NITROGEN_INERTING:
                T_rx = 25.0 + float(rng.normal(0.0, 0.05))
                T_jkt = 25.0
            elif step is ProcessStep.PRE_HEATING:
                f_r = i / max(dur - 1, 1)
                T_target = _lerp(25.0, 70.0, min(1.0, f_r))
                T_jkt = cons.T_jkt_C
                lag = 0.18 * ua_s
                T_rx = T_rx + lag * (T_target - T_rx)
                T_rx = float(np.clip(T_rx, 25.0, 72.0))
            elif step is ProcessStep.DIGESTION_HOLD:
                T_jkt = 70.0
                T_rx = T_rx + 0.25 * (70.0 - T_rx)
                T_rx = float(np.clip(T_rx, 68.0, 85.0))
            elif step is ProcessStep.DEGASSING_COOLING:
                f_r = i / max(dur - 1, 1)
                T_jkt = eq.jacket_chill_temp_C
                T_target = _lerp(70.0, 30.0, f_r)
                lag = 0.15 * ua_s
                T_rx = T_rx + lag * (T_target - T_rx)
                T_rx = float(np.clip(T_rx, 28.0, 75.0))
            elif step is ProcessStep.PRODUCT_DISCHARGE:
                T_jkt = 25.0
                T_rx = T_rx + 0.08 * (30.0 - T_rx)
            else:
                T_jkt = cons.T_jkt_C

            # --- Pressure -------------------------------------------------
            f_r = i / max(dur - 1, 1)
            if step is ProcessStep.NITROGEN_INERTING:
                P = _n2_cycle_pressure(f_r)
            elif step is ProcessStep.PRE_HEATING:
                P = _lerp(1.0, 10.0, min(1.0, f_r))
            elif step is ProcessStep.DIGESTION_HOLD:
                P = 10.0 + float(rng.normal(0.0, 0.02))
            elif step is ProcessStep.DEGASSING_COOLING:
                P = _lerp(10.0, 1.0, f_r)
            elif step in (
                ProcessStep.PREPARATION_TARE,
                ProcessStep.RAW_MATERIAL_LOADING,
                ProcessStep.PRODUCT_DISCHARGE,
            ):
                P = 1.0
            else:
                P = float(np.clip(P, 1.0, eq.design_pressure_max_bar))

            # --- H2 flow --------------------------------------------------
            if step is ProcessStep.PRE_HEATING:
                h2 = eq.h2_pad_kg_min * float(feed_scale[t])
            elif step is ProcessStep.DIGESTION_HOLD:
                h2 = _lerp(0.4, 0.0, f_r) * float(feed_scale[t])
            else:
                h2 = 0.0

            # --- Cooling water --------------------------------------------
            if step is ProcessStep.DIGESTION_HOLD:
                cw = _lerp(30.0, 25.0, f_r)
            elif step is ProcessStep.DEGASSING_COOLING:
                cw = _lerp(90.0, 70.0, f_r) * (0.7 + 0.3 * ua_s)
            elif step is ProcessStep.PRE_HEATING:
                cw = 0.0
            else:
                cw = 0.0

            # --- Agitator torque ------------------------------------------
            mode = cons.torque_mode
            if mode == "off":
                torque = 0.0
            elif mode == "idle":
                torque = eq.agitator_torque_Nm(mode="idle")
            elif mode == "weight_gated":
                torque = (
                    eq.agitator_torque_Nm(mode="idle")
                    if weight > 800.0
                    else 0.0
                )
            elif mode == "weight_gated_off":
                torque = (
                    eq.agitator_torque_Nm(conversion=X, mode="full")
                    if weight >= 500.0
                    else 0.0
                )
            else:
                torque = eq.agitator_torque_Nm(
                    conversion=X, foam_scale=float(foam_scale[t]), mode="full"
                )

            # --- Conversion -----------------------------------------------
            if step is ProcessStep.DIGESTION_HOLD:
                X = _lerp(max(X, 0.98), 1.0, f_r)
            elif step in (
                ProcessStep.DEGASSING_COOLING,
                ProcessStep.PRODUCT_DISCHARGE,
            ):
                X = max(X, 0.995)
            elif step in (
                ProcessStep.PREPARATION_TARE,
                ProcessStep.RAW_MATERIAL_LOADING,
                ProcessStep.NITROGEN_INERTING,
                ProcessStep.PRE_HEATING,
            ):
                X = 0.0

            traj.weight_kg[t] = weight
            traj.T_reactor[t] = T_rx
            traj.T_jacket[t] = T_jkt
            traj.pressure_bar[t] = P
            traj.h2_flow[t] = h2
            traj.cooling_L_min[t] = cw
            traj.torque_Nm[t] = torque
            traj.conversion[t] = X

        # Update carry-outs from last minute of step
        weight = float(traj.weight_kg[b - 1])
        T_rx = float(traj.T_reactor[b - 1])
        P = float(traj.pressure_bar[b - 1])
        X = float(traj.conversion[b - 1])

    # Apply sensor drift bias to measured reactor temperature
    traj.T_reactor = traj.T_reactor + sensor_bias
    return traj


# ===========================================================================
# Export
# ===========================================================================


def _time_to_endpoint_min(conversion: np.ndarray, time_min: np.ndarray) -> float:
    thr = ENDPOINT_CONVERSION_PCT / 100.0
    hit = np.where(conversion >= thr)[0]
    if hit.size == 0:
        return float(time_min[-1])
    return float(time_min[int(hit[0])])


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def generate_one_batch(
    spec: BatchSpec,
    out_raw: Path,
    out_targets: Path,
    equipment: ReactorEquipmentPackage,
    seed: int,
) -> dict:
    rng = random.Random(seed + spec.batch_id * 997)
    coa = sample_coa_purity(rng)
    n = TOTAL_BATCH_DURATION_MIN

    schedule = build_anomaly_schedule(
        spec.anomaly,
        n_minutes=n,
        rng_seed=seed + spec.batch_id * 17,
        write_shift_note=True,
    )

    traj = simulate_batch(
        eq=equipment,
        coa_purity=coa,
        schedule=schedule,
        seed=seed + spec.batch_id * 13,
    )

    # Noise active when vessel has inventory or pressurized / reacting
    active = [
        bool(traj.weight_kg[i] > 50.0 or traj.step_id[i] >= 3)
        for i in range(n)
    ]
    clean = {
        "time": traj.time_min.tolist(),
        "T_reactor": traj.T_reactor.tolist(),
        "T_jacket": traj.T_jacket.tolist(),
        "headspace_pressure": traj.pressure_bar.tolist(),
        "pressure": traj.pressure_bar.tolist(),
        "agitator_torque": traj.torque_Nm.tolist(),
        "h2_flow": traj.h2_flow.tolist(),
        "conversion": traj.conversion.tolist(),
    }
    noisy = apply_process_noise(
        clean,
        coa_purity=coa,
        rng_seed=seed + spec.batch_id * 29,
        active_mask=active,
    )

    T_rx = np.asarray(noisy["T_reactor"], dtype=np.float64)
    T_jkt = np.asarray(noisy["T_jacket"], dtype=np.float64)
    P = np.asarray(noisy.get("headspace_pressure", traj.pressure_bar), dtype=np.float64)
    torque = np.asarray(noisy["agitator_torque"], dtype=np.float64)
    # Keep H2 / cooling / weight deterministic (anomaly schedules already applied)
    h2 = traj.h2_flow
    cw = traj.cooling_L_min
    weight = traj.weight_kg
    note = schedule.shift_note or ""
    flag = spec.anomaly.value

    tte = _time_to_endpoint_min(traj.conversion, traj.time_min)
    rows: List[dict] = []
    targets: List[dict] = []
    for i in range(n):
        rows.append(
            {
                "batch_id": spec.batch_id,
                "time_step_min": int(traj.time_min[i]),
                "current_step_id": int(traj.step_id[i]),
                "raw_material_purity_coa": round(coa, 3),
                "vessel_weight_kg": round(float(weight[i]), 1),
                "T_reactor": round(float(T_rx[i]), 1),
                "T_jacket": round(float(T_jkt[i]), 1),
                "delta_T": round(float(T_rx[i] - T_jkt[i]), 1),
                "agitator_torque": round(float(torque[i]), 1),
                "headspace_pressure": round(float(P[i]), 2),
                "H2_flow_rate": round(float(h2[i]), 3),
                "cooling_water_flow": round(float(cw[i]), 2),
                "shift_note": note,
                "anomaly_flag": flag,
            }
        )
        targets.append(
            {
                "batch_id": spec.batch_id,
                "time_step_min": int(traj.time_min[i]),
                "reaction_conversion_pct": round(float(traj.conversion[i]) * 100.0, 4),
                "time_to_endpoint_min": round(tte, 1),
            }
        )

    _write_csv(out_raw / f"batch_{spec.batch_id:03d}.csv", rows, CORE_COLUMNS)
    _write_csv(
        out_targets / f"batch_{spec.batch_id:03d}.csv",
        targets,
        [
            "batch_id",
            "time_step_min",
            "reaction_conversion_pct",
            "time_to_endpoint_min",
        ],
    )

    return {
        "batch_id": spec.batch_id,
        "split": spec.split,
        "anomaly_type": flag,
        "total_duration_min": n,
        "final_conversion_pct": round(float(traj.conversion[-1]) * 100.0, 4),
        "time_to_endpoint_min": round(tte, 1),
        "raw_material_purity_coa": round(coa, 3),
    }


def generate_dataset(
    output_dir: Path | str = "dataset",
    seed: int = DEFAULT_SEED,
    n_batches: int = N_BATCHES,
) -> Path:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    tgt_dir = output_dir / "targets"
    gt_dir = output_dir / "ground_truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    plan = build_batch_plan(seed=seed)[:n_batches]
    equipment = ReactorEquipmentPackage.default()
    summaries: List[dict] = []

    t0 = time.perf_counter()
    for spec in plan:
        summaries.append(
            generate_one_batch(spec, raw_dir, tgt_dir, equipment, seed)
        )
    elapsed = time.perf_counter() - t0

    summary_path = gt_dir / "batch_summary.csv"
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
    # Compatibility copy at dataset root
    _write_csv(
        output_dir / "batch_summary.csv",
        summaries,
        [
            "batch_id",
            "split",
            "anomaly_type",
            "total_duration_min",
            "final_conversion_pct",
        ],
    )
    _write_csv(
        output_dir / "split_manifest.csv",
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

    _validate(output_dir, plan, elapsed)
    return summary_path


def _validate(output_dir: Path, plan: List[BatchSpec], elapsed_s: float) -> None:
    raw = sorted((output_dir / "raw").glob("batch_*.csv"))
    assert len(raw) == len(plan)
    train = [s for s in plan if s.split == "train"]
    test = [s for s in plan if s.split == "test"]
    assert all(s.anomaly is not AnomalyType.SENSOR_DRIFT for s in train)
    assert all(not s.anomaly.is_compound for s in train)
    if test:
        assert any(s.anomaly is AnomalyType.SENSOR_DRIFT for s in test)
        assert any(s.anomaly.is_compound for s in test)

    with raw[0].open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert list(reader.fieldnames) == CORE_COLUMNS
        rows = list(reader)
    assert len(rows) == TOTAL_BATCH_DURATION_MIN
    assert rows[0]["current_step_id"] == "1"
    assert rows[-1]["current_step_id"] == "8"

    print(
        f"Generated {len(plan)} batches → {output_dir}/ in {elapsed_s:.2f}s "
        f"(train={len(train)}, test={len(test)}, rows/batch={TOTAL_BATCH_DURATION_MIN})"
    )
    if elapsed_s > 30.0:
        print(f"WARNING: exceeded 30s budget ({elapsed_s:.2f}s)")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output-dir", type=Path, default=Path("dataset"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-batches", type=int, default=N_BATCHES)
    args = p.parse_args(argv)
    generate_dataset(args.output_dir, args.seed, args.n_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
