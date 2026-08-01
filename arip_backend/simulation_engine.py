"""ARIP Digital Twin — Batch ODE Simulation Engine.

Loads Reaction + Equipment + Batch packages, applies closed-loop TCU
temperature and H2 pressure control, and integrates the stiff ODE system
with ``scipy.integrate.solve_ivp(method='BDF')``.

State vector
------------
y = [C_nitro, C_nitroso, C_rha, C_amine, C_H2O, C_h2, T_reactor, P_bar]

Heat removal
------------
Q_rem = U · A · (T_reactor − T_jacket)

Run
---
    python -m arip_backend.simulation_engine
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

BACKEND_ROOT = Path(__file__).resolve().parent
R = 8.314462618  # J/(mol·K)


def load_json(filepath: str | Path) -> dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class PIDState:
    """Simple discrete PID for jacket temperature command."""

    Kp: float = 4.0
    Ki: float = 0.02
    Kd: float = 0.5
    integral: float = 0.0
    prev_error: float = 0.0
    output_min: float = -20.0
    output_max: float = 200.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        err = setpoint - measurement
        self.integral += err * max(dt, 0.0)
        # Anti-windup clamp on integral contribution
        self.integral = float(np.clip(self.integral, -500.0, 500.0))
        deriv = (err - self.prev_error) / dt if dt > 1e-12 else 0.0
        self.prev_error = err
        # Jacket colder than reactor when err < 0 (need cooling)
        raw = setpoint + self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        return float(np.clip(raw, self.output_min, self.output_max))


class SimulationEngine:
    """Closed-loop semi-batch hydrogenation simulator."""

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
        eq_id = self.batch_pkg["equipment_package_id"]
        default_eq = BACKEND_ROOT / "equipment_packages" / "module_1_reactors" / "eq_stbr_5000l.json"
        self.eq_pkg = load_json(equipment_path or default_eq)
        if self.eq_pkg.get("equipment_id") != eq_id:
            # Prefer path matching batch equipment id
            candidate = BACKEND_ROOT / "equipment_packages" / "module_1_reactors" / "eq_stbr_5000l.json"
            if candidate.exists():
                self.eq_pkg = load_json(candidate)

        steps_root = steps_dir or BACKEND_ROOT / "reaction_packages" / "reaction_steps"
        self.steps = [
            load_json(steps_root / name)
            for name in self.rxn_pkg["mechanism_steps"]
        ]
        if len(self.steps) != 3:
            raise ValueError("Expected 3 mechanism steps in reaction master")

        # Kinetic parameters (Ea in J/mol, ΔH in J/mol — match SI / user engine convention)
        self.k0 = [float(s["kinetics"]["pre_exponential_k0"]) for s in self.steps]
        self.Ea = [float(s["kinetics"]["activation_energy_ea_kj_mol"]) * 1000.0 for s in self.steps]
        self.dH = [float(s["thermodynamics"]["reaction_enthalpy_kj_mol"]) * 1000.0 for s in self.steps]  # J/mol

        # Transport / thermo
        mix = self.thermo_pkg["mixture_transport_properties"]
        self.rho = float(mix["bulk_liquid_density_kg_m3"])
        self.Cp = float(mix["bulk_heat_capacity_j_kgk"])
        self.kla = float(mix["kla_base_coefficient_1_s"])
        henry = float(self.thermo_pkg["chemical_species"]["H2"]["henry_law_constant_bar_m3_kmol"])
        self.henry = henry

        # Equipment U·A
        thermal = self.eq_pkg.get("thermal_parameters", {})
        self.U = float(thermal.get("overall_heat_transfer_coeff_U_W_m2K", 350.0))
        self.A = float(thermal.get("jacket_heat_transfer_area_m2", 12.5))

        # Batch recipe
        ic = self.batch_pkg["initial_conditions"]
        rt = self.batch_pkg["recipe_targets"]
        self.V = float(ic["initial_liquid_volume_m3"])
        self.c_cat = float(ic["catalyst_loading_kg"]) / self.V  # kg/m³
        self.T_sp = float(rt["target_operating_temp_c"])
        self.P_sp = float(rt["target_operating_pressure_bar"])
        self.T_coolant_in = float(rt["coolant_inlet_temp_c"])

        self.pid = PIDState(
            Kp=3.5,
            Ki=0.015,
            Kd=0.4,
            output_min=self.T_coolant_in,
            output_max=float(self.eq_pkg["limits"]["max_operating_temp_c"]),
        )
        self._last_t: float | None = None
        self.T_jacket = self.T_sp
        self.history: list[dict[str, Any]] = []

    def _rates(self, C_nitro: float, C_nitroso: float, C_rha: float, C_h2: float, T_c: float) -> tuple[float, float, float]:
        T_k = T_c + 273.15
        # Power-law / LH form: r = k0·exp(-Ea/RT)·C_org·C_H2·C_cat
        k1 = self.k0[0] * np.exp(-self.Ea[0] / (R * T_k))
        k2 = self.k0[1] * np.exp(-self.Ea[1] / (R * T_k))
        k3 = self.k0[2] * np.exp(-self.Ea[2] / (R * T_k))
        r1 = k1 * max(C_nitro, 0.0) * max(C_h2, 0.0) * self.c_cat
        r2 = k2 * max(C_nitroso, 0.0) * max(C_h2, 0.0) * self.c_cat
        r3 = k3 * max(C_rha, 0.0) * max(C_h2, 0.0) * self.c_cat
        return float(r1), float(r2), float(r3)

    def reactor_odes(self, t: float, y: np.ndarray) -> list[float]:
        """System of ODEs with closed-loop TCU and pressure control."""
        C_nitro, C_nitroso, C_rha, C_amine, C_h2o, C_h2, T, P = y

        dt = 1.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        # --- Pressure control: dose H2 to hold headspace setpoint ---
        # First-order pressure tracking; C* from Henry's law at controlled P
        tau_p = 20.0
        dP_dt = (self.P_sp - P) / tau_p
        P_eff = max(P, 0.1)
        C_h2_sat = P_eff / self.henry  # kmol/m³

        r1, r2, r3 = self._rates(C_nitro, C_nitroso, C_rha, C_h2, T)

        # Mass balances (Haber sequence)
        dC_nitro_dt = -r1
        dC_nitroso_dt = r1 - r2
        dC_rha_dt = r2 - r3
        dC_amine_dt = r3
        dC_h2o_dt = r1 + r3
        dC_h2_dt = self.kla * (C_h2_sat - C_h2) - (r1 + r2 + r3)

        # --- Temperature PID → jacket setpoint ---
        self.T_jacket = self.pid.update(self.T_sp, T, dt if dt > 0 else 1.0)

        # Heat generation / removal [W/m³]
        # r [kmol/(m³·s)] · (−ΔH) [J/mol] · 1000 [mol/kmol] → W/m³
        Q_rxn = (r1 * (-self.dH[0]) + r2 * (-self.dH[1]) + r3 * (-self.dH[2])) * 1000.0
        Q_rem = (self.U * self.A / self.V) * (T - self.T_jacket)  # W/m³
        dT_dt = (Q_rxn - Q_rem) / (self.rho * self.Cp)

        return [
            dC_nitro_dt,
            dC_nitroso_dt,
            dC_rha_dt,
            dC_amine_dt,
            dC_h2o_dt,
            dC_h2_dt,
            dT_dt,
            dP_dt,
        ]

    def initial_state(self) -> np.ndarray:
        ic = self.batch_pkg["initial_conditions"]
        c = ic["initial_concentrations_kmol_m3"]
        return np.array(
            [
                float(c.get("2,4-nitroxylene", 0.0)),
                float(c.get("2,4-nitrosoxylene", 0.0)),
                float(c.get("2,4-xylidylhydroxylamine", 0.0)),
                float(c.get("2,4-xylidine", 0.0)),
                float(c.get("H2O", 0.0)),
                float(c.get("H2_dissolved", 0.01)),
                float(ic["initial_temperature_c"]),
                float(ic["initial_pressure_bar"]),
            ],
            dtype=float,
        )

    def run(self, t_end_s: float = 7200.0, n_eval: int = 500) -> dict[str, Any]:
        y0 = self.initial_state()
        t_span = (0.0, t_end_s)
        t_eval = np.linspace(0.0, t_end_s, n_eval)
        self._last_t = 0.0
        self.pid.integral = 0.0
        self.pid.prev_error = 0.0

        # Safety events
        max_T = float(self.eq_pkg["limits"]["max_operating_temp_c"])
        max_P = float(self.eq_pkg["limits"]["max_operating_pressure_bar"])
        max_rha = float(self.rxn_pkg["safety_critical_limits"]["max_rha_accumulation_mole_fraction"])
        c0 = float(y0[0])

        def temp_event(t, y):  # noqa: ANN001
            return max_T - y[6]

        temp_event.terminal = True
        temp_event.direction = -1

        def pressure_event(t, y):  # noqa: ANN001
            return max_P - y[7]

        pressure_event.terminal = True
        pressure_event.direction = -1

        def rha_event(t, y):  # noqa: ANN001
            total = max(sum(max(v, 0.0) for v in y[:6]), 1e-12)
            return max_rha - max(y[2], 0.0) / total

        rha_event.terminal = True
        rha_event.direction = -1

        def conversion_event(t, y):  # noqa: ANN001
            conv = 1.0 - max(y[0], 0.0) / c0 if c0 > 0 else 0.0
            return conv - 0.98

        conversion_event.terminal = True
        conversion_event.direction = 1

        solution = solve_ivp(
            self.reactor_odes,
            t_span,
            y0,
            method="BDF",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
            max_step=30.0,
            events=[temp_event, pressure_event, rha_event, conversion_event],
        )
        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")

        rows = []
        for i, t in enumerate(solution.t):
            y = solution.y[:, i]
            r1, r2, r3 = self._rates(y[0], y[1], y[2], y[5], y[6])
            Q_rxn = (r1 * (-self.dH[0]) + r2 * (-self.dH[1]) + r3 * (-self.dH[2])) * 1000.0
            total = max(sum(max(v, 0.0) for v in y[:6]), 1e-12)
            rows.append(
                {
                    "time_s": float(t),
                    "time_min": float(t) / 60.0,
                    "C_nitroxylene": float(y[0]),
                    "C_nitrosoxylene": float(y[1]),
                    "C_hydroxylamine": float(y[2]),
                    "C_xylidine": float(y[3]),
                    "C_H2O": float(y[4]),
                    "C_H2_dissolved": float(y[5]),
                    "T_c": float(y[6]),
                    "P_bar": float(y[7]),
                    "T_jacket_c": self.T_jacket,
                    "r1": r1,
                    "r2": r2,
                    "r3": r3,
                    "Q_rxn_W_m3": Q_rxn,
                    "nitro_conversion": 1.0 - float(y[0]) / c0 if c0 > 0 else 0.0,
                    "x_rha": float(max(y[2], 0.0) / total),
                }
            )
        df = pd.DataFrame(rows)

        event_labels = ["temperature", "pressure", "rha", "conversion"]
        events = []
        if solution.t_events:
            for idx, times in enumerate(solution.t_events):
                if len(times) > 0:
                    events.append({"time_s": float(times[-1]), "event": event_labels[idx]})

        final_conv = (1.0 - solution.y[0][-1] / y0[0]) * 100.0 if y0[0] > 0 else 0.0
        return {
            "success": True,
            "batch_id": self.batch_pkg["batch_id"],
            "reaction_package_id": self.batch_pkg["reaction_package_id"],
            "equipment_package_id": self.batch_pkg["equipment_package_id"],
            "solver": "BDF",
            "n_time_points": len(solution.t),
            "t_final_s": float(solution.t[-1]),
            "final_conversion_pct": float(final_conv),
            "final_temperature_c": float(solution.y[6][-1]),
            "final_pressure_bar": float(solution.y[7][-1]),
            "max_temperature_c": float(df["T_c"].max()),
            "max_x_rha": float(df["x_rha"].max()),
            "U_W_m2K": self.U,
            "A_m2": self.A,
            "events": events,
            "timeseries": df,
        }


def main() -> None:
    engine = SimulationEngine()
    result = engine.run(t_end_s=7200.0, n_eval=500)

    out_dir = BACKEND_ROOT / "simulation_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{result['batch_id']}_timeseries.csv"
    json_path = out_dir / f"{result['batch_id']}_result.json"
    result["timeseries"].to_csv(csv_path, index=False)
    serializable = {k: v for k, v in result.items() if k != "timeseries"}
    serializable["timeseries_preview"] = result["timeseries"].head(3).to_dict(orient="records")
    serializable["output_csv"] = str(csv_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)

    print("Simulation Completed Successfully!")
    print(f"Batch:            {result['batch_id']}")
    print(f"Equipment:        {result['equipment_package_id']}  (U={result['U_W_m2K']} W/m2K, A={result['A_m2']} m2)")
    print(f"Time points:      {result['n_time_points']}")
    print(f"t_final:          {result['t_final_s']:.1f} s")
    print(f"Final Conversion: {result['final_conversion_pct']:.2f}%")
    print(f"Final Temperature:{result['final_temperature_c']:.2f} °C")
    print(f"Final Pressure:   {result['final_pressure_bar']:.2f} bar")
    print(f"Max x_RHA:        {result['max_x_rha']:.4f}")
    print(f"Events:           {result['events']}")
    print(f"Wrote:            {csv_path}")
    print(f"Wrote:            {json_path}")


if __name__ == "__main__":
    main()
