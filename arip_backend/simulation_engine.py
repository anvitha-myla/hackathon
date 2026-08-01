"""ARIP Digital Twin — Coupled Stiff ODE Simulation Engine (Blueprint).

State vector
------------
y(t) = [C_nitro, C_amine, C_H2_liq, T_reactor, T_jacket, P_headspace]

Core physics
------------
* Arrhenius rate:     r = k0 · exp(-Ea/(R·T)) · C_nitro · C_H2_liq
* H2 mass transfer:   R_transfer = kLa · (C* − C_H2_liq)
* Reactor energy:     dT_r/dt = [r·V·(−ΔH) − UA(T_r−T_j)] / (V·ρ·Cp)
* Jacket energy:      dT_j/dt with coolant flow F_c (algebraic TCU law)

Solver
------
Implicit stiff integrator via ``scipy.integrate.solve_ivp`` (``BDF`` default;
``Radau`` supported). Actuators are memoryless functions of (t, y) so the RHS
stays Lipschitz-friendly for implicit Jacobian estimation.

Phase event sequencer (segmented terminal events)
-------------------------------------------------
1. T_reactor ≥ 80 °C     → open H2 dosing valve (end heat-up)
2. C_nitro ≤ 0.5% of C0  → digestion hold
3. t = t_digest          → dump / cool-down

Run
---
    python -m arip_backend.simulation_engine
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

BACKEND_ROOT = Path(__file__).resolve().parent
R_GAS = 8.314462618  # J/(mol·K)

I_NITRO, I_AMINE, I_H2, I_TR, I_TJ, I_P = range(6)
STATE_NAMES = [
    "C_nitro",
    "C_amine",
    "C_H2_liq",
    "T_reactor",
    "T_jacket",
    "P_headspace",
]


class BatchPhase(str, Enum):
    HEAT_UP = "HEAT_UP"
    HYDROGENATION = "HYDROGENATION"
    DIGESTION = "DIGESTION"
    DUMP_COOL = "DUMP_COOL"
    COMPLETE = "COMPLETE"


def load_json(filepath: str | Path) -> dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def smoothstep(x: float, x0: float, width: float) -> float:
    """Smooth 0→1 transition centred at x0."""
    return float(1.0 / (1.0 + np.exp(-(x - x0) / max(width, 1e-6))))


def project_state(y: np.ndarray, T_coolant_in: float) -> np.ndarray:
    """Hard physical bounds on the state vector."""
    yp = np.asarray(y, dtype=float).copy()
    yp[I_NITRO] = max(yp[I_NITRO], 0.0)
    yp[I_AMINE] = max(yp[I_AMINE], 0.0)
    yp[I_H2] = max(yp[I_H2], 0.0)
    yp[I_TR] = float(np.clip(yp[I_TR], 5.0, 200.0))
    yp[I_TJ] = float(np.clip(yp[I_TJ], T_coolant_in, 200.0))
    yp[I_P] = float(np.clip(yp[I_P], 0.8, 16.0))
    return yp


@dataclass
class PhaseSequencer:
    T_h2_open_c: float = 80.0
    conversion_complete: float = 0.995
    digestion_hold_s: float = 900.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, t: float, event: str, message: str, phase: str, **data: Any) -> None:
        self.events.append(
            {"time_s": float(t), "event": event, "phase": phase, "message": message, **data}
        )


class SimulationEngine:
    """Blueprint coupled ODE engine with stiff solver + phase sequencer."""

    def __init__(
        self,
        batch_path: Path | None = None,
        reaction_master_path: Path | None = None,
        thermo_path: Path | None = None,
        equipment_path: Path | None = None,
        steps_dir: Path | None = None,
    ) -> None:
        self.batch_pkg = load_json(batch_path or BACKEND_ROOT / "batch_packages" / "batch_run_001.json")
        self.rxn_pkg = load_json(
            reaction_master_path
            or BACKEND_ROOT / "reaction_packages" / "nitroxylene_hydrogenation_master.json"
        )
        self.thermo_pkg = load_json(
            thermo_path or BACKEND_ROOT / "reaction_packages" / "thermodynamic_properties.json"
        )
        self.eq_pkg = load_json(
            equipment_path
            or BACKEND_ROOT / "equipment_packages" / "module_1_reactors" / "eq_stbr_5000l.json"
        )

        steps_root = steps_dir or BACKEND_ROOT / "reaction_packages" / "reaction_steps"
        steps = [load_json(steps_root / name) for name in self.rxn_pkg["mechanism_steps"]]

        self.k0 = float(steps[0]["kinetics"]["pre_exponential_k0"])
        self.Ea = float(steps[0]["kinetics"]["activation_energy_ea_kj_mol"]) * 1000.0
        self.dH = -540000.0  # J/mol overall (blueprint)

        mix = self.thermo_pkg["mixture_transport_properties"]
        self.rho = float(mix["bulk_liquid_density_kg_m3"])
        self.Cp = float(mix["bulk_heat_capacity_j_kgk"])
        self.kla = float(mix["kla_base_coefficient_1_s"])
        self.henry_package = float(
            self.thermo_pkg["chemical_species"]["H2"]["henry_law_constant_bar_m3_kmol"]
        )

        thermal = self.eq_pkg["thermal_parameters"]
        self.U = max(float(thermal["overall_heat_transfer_coeff_U_W_m2K"]), 400.0)
        self.A_jacket = float(thermal["jacket_heat_transfer_area_m2"])
        self.A = max(self.A_jacket, 55.0)
        self.V_jacket = float(thermal.get("jacket_volume_m3", 0.45))

        ic = self.batch_pkg["initial_conditions"]
        rt = self.batch_pkg["recipe_targets"]
        self.V = float(ic["initial_liquid_volume_m3"])
        self.c_cat = float(ic["catalyst_loading_kg"]) / self.V
        self.T_sp = float(rt["target_operating_temp_c"])
        self.P_sp = float(rt["target_operating_pressure_bar"])
        self.T_coolant_in = float(rt["coolant_inlet_temp_c"])
        self.henry = min(self.henry_package, self.P_sp / 0.04)

        self.rho_coolant = 1000.0
        self.Cp_coolant = 4180.0
        self.Q_util_max_W = 2.0e6
        self.F_max = 0.05

        self.sequencer = PhaseSequencer()
        self.phase = BatchPhase.HEAT_UP
        self.h2_latched = False
        self.digest_start_t: Optional[float] = None
        self.C_nitro_0 = float(ic["initial_concentrations_kmol_m3"]["2,4-nitroxylene"])

    def reaction_rate(self, C_nitro: float, C_h2: float, T_c: float) -> float:
        T_k = float(np.clip(T_c + 273.15, 273.15, 523.15))
        exponent = float(np.clip(-self.Ea / (R_GAS * T_k), -80.0, 80.0))
        k = self.k0 * np.exp(exponent)
        return float(k * max(C_nitro, 0.0) * max(C_h2, 0.0) * self.c_cat)

    def _tcu_actuators(self, T_r: float) -> tuple[float, float]:
        """Memoryless TCU: returns (F_c [m³/s], Q_util [W]).

        Must be a pure function of temperature so implicit solvers can
        estimate Jacobians safely (no integral / dt state in the RHS).
        """
        if self.phase == BatchPhase.HEAT_UP:
            heat_frac = smoothstep(self.sequencer.T_h2_open_c + 2.0 - T_r, 0.0, 3.0)
            Q_util = self.Q_util_max_W * heat_frac
            # Light cooling only if we overshoot hard during heat-up
            over = max(T_r - (self.sequencer.T_h2_open_c + 8.0), 0.0)
            Fc = self.F_max * min(over / 20.0, 1.0) * (1.0 - heat_frac)
            return Fc, Q_util

        if self.phase in (BatchPhase.HYDROGENATION, BatchPhase.DIGESTION):
            # Cool above SP, heat below SP (smooth band around setpoint)
            cool = smoothstep(T_r, self.T_sp, 1.5)
            heat = 1.0 - smoothstep(T_r, self.T_sp - 1.0, 1.5)
            Fc = self.F_max * cool * min(max((T_r - self.T_sp) / 12.0, 0.0), 1.0)
            # Keep some cooling headroom on large exotherms
            if T_r > self.T_sp + 15.0:
                Fc = max(Fc, 0.45 * self.F_max)
            if T_r > self.T_sp + 25.0:
                Fc = self.F_max
            Q_util = self.Q_util_max_W * heat * min(max((self.T_sp - T_r) / 10.0, 0.0), 1.0)
            if self.phase == BatchPhase.DIGESTION:
                Q_util *= 0.5
            return Fc, Q_util

        # DUMP_COOL: full coolant, no utility
        return self.F_max, 0.0

    def reactor_system_ode(self, t: float, y: np.ndarray) -> list[float]:
        """RHS: dy/dt for the blueprint 6-state vector (pure in y)."""
        y = project_state(y, self.T_coolant_in)
        C_nitro = float(y[I_NITRO])
        C_h2 = float(y[I_H2])
        T_r = float(y[I_TR])
        T_j = float(y[I_TJ])
        P = float(y[I_P])

        if self.phase == BatchPhase.HEAT_UP:
            valve = 0.0
            P_target = 1.0
            r_scale = 0.0
        elif self.phase == BatchPhase.HYDROGENATION:
            valve = 1.0
            P_target = self.P_sp
            r_scale = 1.0
        elif self.phase == BatchPhase.DIGESTION:
            valve = 1.0
            P_target = self.P_sp
            r_scale = 0.25
        else:
            valve = 0.0
            P_target = 1.0
            r_scale = 0.0

        Fc, Q_util_W = self._tcu_actuators(T_r)
        dP = (P_target - P) / 25.0

        C_star = valve * P / self.henry
        R_transfer = self.kla * (C_star - C_h2)
        r_rxn = self.reaction_rate(C_nitro, C_h2, T_r) * r_scale

        dC_nitro = -r_rxn
        dC_amine = +r_rxn
        dC_h2 = R_transfer - 3.0 * r_rxn

        Q_rxn_W = r_rxn * (-self.dH) * 1000.0 * self.V
        Q_trans_W = self.U * self.A * (T_r - T_j)
        Q_coolant_W = Fc * self.rho_coolant * self.Cp_coolant * (self.T_coolant_in - T_j)

        dT_r = (Q_rxn_W - Q_trans_W) / (self.V * self.rho * self.Cp)
        dT_j = (Q_coolant_W + Q_trans_W + Q_util_W) / (
            self.V_jacket * self.rho_coolant * self.Cp_coolant
        )

        return [
            float(np.clip(dC_nitro, -0.5, 0.5)),
            float(np.clip(dC_amine, -0.5, 0.5)),
            float(np.clip(dC_h2, -0.5, 0.5)),
            float(np.clip(dT_r, -2.0, 2.0)),
            float(np.clip(dT_j, -2.0, 2.0)),
            float(np.clip(dP, -0.5, 0.5)),
        ]

    def initial_state(self) -> np.ndarray:
        ic = self.batch_pkg["initial_conditions"]
        c = ic["initial_concentrations_kmol_m3"]
        return project_state(
            np.array(
                [
                    float(c["2,4-nitroxylene"]),
                    float(c.get("2,4-xylidine", 0.0)),
                    float(c.get("H2_dissolved", 0.0)),
                    float(ic["initial_temperature_c"]),
                    float(self.T_coolant_in) + 15.0,
                    float(ic["initial_pressure_bar"]),
                ],
                dtype=float,
            ),
            self.T_coolant_in,
        )

    def _solve_segment(
        self,
        t0: float,
        t1: float,
        y0: np.ndarray,
        method: str,
        events: list | None = None,
        max_step: float = 60.0,
    ):
        y0 = project_state(y0, self.T_coolant_in)
        sol = solve_ivp(
            self.reactor_system_ode,
            (t0, t1),
            y0,
            method=method,
            rtol=1e-4,
            atol=1e-7,
            max_step=max_step,
            events=events,
        )
        if not sol.success:
            raise RuntimeError(
                f"ODE solver failed ({method}) in phase {self.phase.value}: {sol.message}"
            )
        return sol

    def run(
        self,
        t_end_s: float = 18000.0,
        n_eval: int = 600,
        method: str = "BDF",
    ) -> dict[str, Any]:
        """Phase-segmented stiff integration with terminal plant events."""
        y = self.initial_state()
        t = 0.0
        self.h2_latched = False
        self.digest_start_t = None
        self.sequencer = PhaseSequencer()
        self.sequencer.record(
            0.0, "batch_start", "Batch started — heat-up phase", BatchPhase.HEAT_UP.value
        )

        t_hist: list[float] = [t]
        y_hist: list[np.ndarray] = [y.copy()]
        phase_hist: list[str] = [BatchPhase.HEAT_UP.value]

        def append_sol(sol, phase: BatchPhase, skip_first: bool = True) -> None:
            start = 1 if skip_first and len(sol.t) else 0
            for i in range(start, len(sol.t)):
                ti = float(sol.t[i])
                yi = project_state(sol.y[:, i], self.T_coolant_in)
                t_hist.append(ti)
                y_hist.append(yi)
                phase_hist.append(phase.value)

        # ----- Segment 1: HEAT_UP until T_reactor = 80 °C -----
        self.phase = BatchPhase.HEAT_UP

        def event_heatup(_t, yv):  # noqa: ANN001
            return float(yv[I_TR]) - self.sequencer.T_h2_open_c

        event_heatup.direction = 1
        event_heatup.terminal = True

        sol = self._solve_segment(t, t_end_s, y, method, events=[event_heatup], max_step=30.0)
        append_sol(sol, BatchPhase.HEAT_UP, skip_first=True)

        if sol.t_events and len(sol.t_events[0]):
            t = float(sol.t_events[0][-1])
            y = project_state(sol.y_events[0][-1], self.T_coolant_in)
            self.sequencer.record(
                t,
                "h2_valve_open",
                "Heat-up complete — H2 dosing valve opened",
                BatchPhase.HYDROGENATION.value,
                T_trigger_c=self.sequencer.T_h2_open_c,
            )
            self.h2_latched = True
        else:
            t = float(sol.t[-1])
            y = project_state(sol.y[:, -1], self.T_coolant_in)
            return self._finalize(t_hist, y_hist, phase_hist, method, t_end_s, n_eval)

        # ----- Segment 2: HYDROGENATION until 99.5% conversion -----
        self.phase = BatchPhase.HYDROGENATION
        C_thresh = (1.0 - self.sequencer.conversion_complete) * self.C_nitro_0

        def event_conversion(_t, yv):  # noqa: ANN001
            return float(yv[I_NITRO]) - C_thresh

        event_conversion.direction = -1
        event_conversion.terminal = True

        sol = self._solve_segment(t, t_end_s, y, method, events=[event_conversion], max_step=60.0)
        append_sol(sol, BatchPhase.HYDROGENATION, skip_first=True)

        if sol.t_events and len(sol.t_events[0]):
            t = float(sol.t_events[0][-1])
            y = project_state(sol.y_events[0][-1], self.T_coolant_in)
            conv = 1.0 - float(y[I_NITRO]) / self.C_nitro_0
            self.digest_start_t = t
            self.sequencer.record(
                t,
                "digestion_start",
                "99.5% conversion — digestion hold started",
                BatchPhase.DIGESTION.value,
                conversion=conv,
            )
        else:
            t = float(sol.t[-1])
            y = project_state(sol.y[:, -1], self.T_coolant_in)
            return self._finalize(t_hist, y_hist, phase_hist, method, t_end_s, n_eval)

        # ----- Segment 3: DIGESTION for t_digest -----
        self.phase = BatchPhase.DIGESTION
        t_digest_end = min(t + self.sequencer.digestion_hold_s, t_end_s)
        sol = self._solve_segment(t, t_digest_end, y, method, events=None, max_step=60.0)
        append_sol(sol, BatchPhase.DIGESTION, skip_first=True)
        t = float(sol.t[-1])
        y = project_state(sol.y[:, -1], self.T_coolant_in)

        if t + 1e-9 < t_digest_end:
            return self._finalize(t_hist, y_hist, phase_hist, method, t_end_s, n_eval)

        self.sequencer.record(
            t,
            "dump_cool",
            "Digestion hold complete — dump / cool-down triggered",
            BatchPhase.DUMP_COOL.value,
        )

        # ----- Segment 4: DUMP / COOL to horizon -----
        self.phase = BatchPhase.DUMP_COOL
        if t < t_end_s:
            sol = self._solve_segment(t, t_end_s, y, method, events=None, max_step=60.0)
            append_sol(sol, BatchPhase.DUMP_COOL, skip_first=True)

        return self._finalize(t_hist, y_hist, phase_hist, method, t_end_s, n_eval)

    def _finalize(
        self,
        t_hist: list[float],
        y_hist: list[np.ndarray],
        phase_hist: list[str],
        method: str,
        t_end_s: float,
        n_eval: int,
    ) -> dict[str, Any]:
        t_arr = np.asarray(t_hist, dtype=float)
        y_arr = np.column_stack(y_hist) if y_hist else np.zeros((6, 0))

        n_points = max(int(n_eval), 200)
        t_eval = np.linspace(0.0, float(t_arr[-1]) if len(t_arr) else t_end_s, n_points)
        rows = []
        for ti in t_eval:
            i = int(np.searchsorted(t_arr, ti, side="right") - 1)
            i = max(0, min(i, len(t_arr) - 1))
            if i + 1 < len(t_arr) and t_arr[i + 1] > t_arr[i]:
                alpha = (ti - t_arr[i]) / (t_arr[i + 1] - t_arr[i])
                y = (1.0 - alpha) * y_arr[:, i] + alpha * y_arr[:, i + 1]
                phase = phase_hist[i + 1] if alpha > 0.5 else phase_hist[i]
            else:
                y = y_arr[:, i]
                phase = phase_hist[i]
            y = project_state(y, self.T_coolant_in)
            valve = 1.0 if phase in (BatchPhase.HYDROGENATION.value, BatchPhase.DIGESTION.value) else 0.0
            r = self.reaction_rate(float(y[I_NITRO]), float(y[I_H2]), float(y[I_TR]))
            conv = 1.0 - float(y[I_NITRO]) / self.C_nitro_0 if self.C_nitro_0 > 0 else 0.0
            rows.append(
                {
                    "time_s": float(ti),
                    "time_min": float(ti) / 60.0,
                    "C_nitro": float(y[I_NITRO]),
                    "C_amine": float(y[I_AMINE]),
                    "C_H2_liq": float(y[I_H2]),
                    "C_H2_star": float(valve * y[I_P] / self.henry),
                    "T_reactor_c": float(y[I_TR]),
                    "T_jacket_c": float(y[I_TJ]),
                    "P_headspace_bar": float(y[I_P]),
                    "r_rxn": r,
                    "Q_rxn_W": r * (-self.dH) * 1000.0 * self.V,
                    "Q_rem_W": self.U * self.A * (float(y[I_TR]) - float(y[I_TJ])),
                    "nitro_conversion": conv,
                    "phase": phase,
                    "h2_valve_open": valve > 0.5,
                }
            )

        df = pd.DataFrame(rows)
        dumped = any(e["event"] == "dump_cool" for e in self.sequencer.events)
        final_phase = BatchPhase.COMPLETE.value if dumped else (
            phase_hist[-1] if phase_hist else BatchPhase.HEAT_UP.value
        )
        self.sequencer.record(
            float(df["time_s"].iloc[-1]),
            "batch_complete",
            "Simulation horizon complete",
            final_phase,
        )

        return {
            "success": True,
            "solver": method,
            "batch_id": self.batch_pkg["batch_id"],
            "reaction_package_id": self.batch_pkg["reaction_package_id"],
            "equipment_package_id": self.batch_pkg["equipment_package_id"],
            "state_vector": STATE_NAMES,
            "n_time_points": len(df),
            "t_final_s": float(df["time_s"].iloc[-1]),
            "final_phase": final_phase,
            "final_conversion_pct": float(df["nitro_conversion"].iloc[-1] * 100.0),
            "final_temperature_c": float(df["T_reactor_c"].iloc[-1]),
            "final_jacket_c": float(df["T_jacket_c"].iloc[-1]),
            "final_pressure_bar": float(df["P_headspace_bar"].iloc[-1]),
            "max_temperature_c": float(df["T_reactor_c"].max()),
            "min_jacket_c": float(df["T_jacket_c"].min()),
            "max_jacket_c": float(df["T_jacket_c"].max()),
            "delta_H_rxn_J_mol": self.dH,
            "U_W_m2K": self.U,
            "A_m2": self.A,
            "events": self.sequencer.events,
            "timeseries": df,
        }


def main() -> None:
    engine = SimulationEngine()
    result = engine.run(t_end_s=18000.0, n_eval=600, method="BDF")

    out_dir = BACKEND_ROOT / "simulation_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{result['batch_id']}_blueprint_timeseries.csv"
    json_path = out_dir / f"{result['batch_id']}_blueprint_result.json"
    result["timeseries"].to_csv(csv_path, index=False)

    payload = {k: v for k, v in result.items() if k != "timeseries"}
    payload["output_csv"] = str(csv_path)
    payload["timeseries_preview"] = result["timeseries"].head(5).to_dict(orient="records")
    payload["timeseries_tail"] = result["timeseries"].tail(3).to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("Blueprint Simulation Engine — Completed Successfully")
    print(f"Solver:           {result['solver']} (stiff implicit)")
    print(f"State vector:     {result['state_vector']}")
    print(f"Batch:            {result['batch_id']}")
    print(f"Equipment:        {result['equipment_package_id']}  U={result['U_W_m2K']}  A={result['A_m2']}")
    print(f"ΔH_rxn:           {result['delta_H_rxn_J_mol']/1000:.1f} kJ/mol")
    print(f"Time points:      {result['n_time_points']}")
    print(f"t_final:          {result['t_final_s']:.1f} s")
    print(f"Final phase:      {result['final_phase']}")
    print(f"Final conversion: {result['final_conversion_pct']:.2f}%")
    print(f"Final T_reactor:  {result['final_temperature_c']:.2f} °C")
    print(f"Final T_jacket:   {result['final_jacket_c']:.2f} °C")
    print(f"Jacket range:     {result['min_jacket_c']:.2f} … {result['max_jacket_c']:.2f} °C")
    print(f"Final P:          {result['final_pressure_bar']:.2f} bar")
    print(f"Max T_reactor:    {result['max_temperature_c']:.2f} °C")
    print("Events:")
    for ev in result["events"]:
        print(f"  t={ev['time_s']:.1f}s  [{ev.get('event')}]  {ev.get('message')}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
