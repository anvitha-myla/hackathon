"""Numerical ODE solver engine for semi-batch nitroxylene hydrogenation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from arip_backend.schemas.batch import BatchPackage
from arip_backend.schemas.control import ControlPackage
from arip_backend.schemas.reaction import (
    MasterReactionPackage,
    ReactionStepPackage,
    ThermodynamicPropertiesPackage,
    build_rate_law_callable,
    henrys_law_ch2_star,
)
from arip_backend.simulation.control_loops import ProcessControllers
from arip_backend.simulation.events import BatchPhase, EventStateMachine, make_scipy_events

# State vector layout
STATE_NAMES = ["NX", "NSX", "XHA", "XYL", "H2O", "H2", "T_c", "P_bar"]
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}

SPECIES_MAP = {
    "NX": "2,4-nitroxylene",
    "NSX": "2,4-nitrosoxylene",
    "XHA": "2,4-xylidylhydroxylamine",
    "XYL": "2,4-xylidine",
    "H2O": "H2O",
    "H2": "H2_dissolved",
}


@dataclass
class ODEEngineConfig:
    method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = 30.0


class ODEEngine:
    """Stiff ODE engine + event state machine for the digital-twin batch."""

    def __init__(
        self,
        batch: BatchPackage,
        control: ControlPackage,
        master: MasterReactionPackage,
        steps: list[ReactionStepPackage],
        thermo: ThermodynamicPropertiesPackage,
        *,
        config: ODEEngineConfig | None = None,
    ) -> None:
        self.batch = batch
        self.control = control
        self.master = master
        self.steps = sorted(steps, key=lambda s: s.step_id)
        if len(self.steps) != 3:
            raise ValueError("ODE engine expects exactly 3 mechanism steps")
        self.thermo = thermo
        self.config = config or ODEEngineConfig()
        self.controllers = ProcessControllers(control)
        self.state_machine = EventStateMachine(
            max_reactor_temp_c=control.interlocks.max_reactor_temp_c,
            max_rha_mole_fraction=control.interlocks.max_rha_mole_fraction,
            max_pressure_bar=control.interlocks.max_pressure_bar,
        )
        self.rate_fns = [build_rate_law_callable(s) for s in self.steps]
        self.delta_H = [float(s.thermodynamics.reaction_enthalpy_kj_mol) for s in self.steps]
        h2 = thermo.chemical_species["H2"]
        assert h2.henry_law_constant_bar_m3_kmol is not None
        self.henry = float(h2.henry_law_constant_bar_m3_kmol)
        self.kla = float(thermo.mixture_transport_properties.kla_base_coefficient_1_s)
        self.rho = float(thermo.mixture_transport_properties.bulk_liquid_density_kg_m3)
        self.cp = float(thermo.mixture_transport_properties.bulk_heat_capacity_j_kgk)
        self.V = float(batch.initial_charge.volume_m3)
        self.c_cat = float(batch.initial_charge.catalyst_loading_kg_m3)
        self.c0_nitro = float(batch.initial_charge.concentrations_kmol_m3.get("2,4-nitroxylene", 0.0))
        self._last_t: float | None = None

    def _concentrations(self, y: np.ndarray) -> dict[str, float]:
        return {
            SPECIES_MAP["NX"]: max(y[STATE_INDEX["NX"]], 0.0),
            SPECIES_MAP["NSX"]: max(y[STATE_INDEX["NSX"]], 0.0),
            SPECIES_MAP["XHA"]: max(y[STATE_INDEX["XHA"]], 0.0),
            SPECIES_MAP["XYL"]: max(y[STATE_INDEX["XYL"]], 0.0),
            SPECIES_MAP["H2O"]: max(y[STATE_INDEX["H2O"]], 0.0),
            SPECIES_MAP["H2"]: max(y[STATE_INDEX["H2"]], 0.0),
        }

    def _rates(self, y: np.ndarray) -> tuple[float, float, float]:
        conc = self._concentrations(y)
        t_k = y[STATE_INDEX["T_c"]] + 273.15
        r1 = self.rate_fns[0](conc, t_k, self.c_cat)
        r2 = self.rate_fns[1](conc, t_k, self.c_cat)
        r3 = self.rate_fns[2](conc, t_k, self.c_cat)
        return r1, r2, r3

    def rhs(self, t: float, y: np.ndarray) -> list[float]:
        if self.state_machine.phase == BatchPhase.COMPLETE:
            return [0.0] * len(STATE_NAMES)

        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        T = y[STATE_INDEX["T_c"]]
        P = y[STATE_INDEX["P_bar"]]
        # Control loops
        T_j = self.controllers.step_thermal(T, dt if dt > 0 else 1.0)
        P_sp = self.controllers.step_pressure(P, dt if dt > 0 else 1.0)

        r1, r2, r3 = self._rates(y)
        # Stoichiometry (Haber sequence)
        dNX = -r1
        dNSX = r1 - r2
        dXHA = r2 - r3
        dXYL = r3
        dH2O = r1 + r3
        c_star = henrys_law_ch2_star(P_sp, self.henry)
        c_h2 = max(y[STATE_INDEX["H2"]], 0.0)
        dH2 = self.kla * (c_star - c_h2) - (r1 + r2 + r3)

        # Energy balance: rho*Cp*V*dT/dt = Q_rxn*V - UA*(T-Tj)
        # Q_rxn [kJ/(m³·s)] = sum ri*(-dHi); convert to W/m³ via *1000
        q_rxn_kw_m3 = r1 * (-self.delta_H[0]) + r2 * (-self.delta_H[1]) + r3 * (-self.delta_H[2])
        q_rxn_w = q_rxn_kw_m3 * 1000.0 * self.V
        q_removal_w = self.controllers.ua_W_K * (T - T_j)
        dT = (q_rxn_w - q_removal_w) / (self.rho * self.cp * self.V)

        # Pressure tracks controlled setpoint with first-order lag
        tau_p = 15.0
        dP = (P_sp - P) / tau_p

        dydt = [dNX, dNSX, dXHA, dXYL, dH2O, dH2, dT, dP]
        for feed in self.batch.liquid_feeds:
            if feed.start_time_s <= t <= feed.end_time_s and feed.molar_flow_kmol_s > 0:
                key = next((k for k, v in SPECIES_MAP.items() if v == feed.species), None)
                if key is not None:
                    dydt[STATE_INDEX[key]] += feed.molar_flow_kmol_s / self.V

        return dydt

    def initial_state(self) -> np.ndarray:
        ic = self.batch.initial_charge
        c = ic.concentrations_kmol_m3
        return np.array(
            [
                c.get("2,4-nitroxylene", 0.0),
                c.get("2,4-nitrosoxylene", 0.0),
                c.get("2,4-xylidylhydroxylamine", 0.0),
                c.get("2,4-xylidine", 0.0),
                c.get("H2O", 0.0),
                c.get("H2_dissolved", henrys_law_ch2_star(ic.pressure_bar, self.henry)),
                ic.temperature_c,
                ic.pressure_bar,
            ],
            dtype=float,
        )

    def run(self) -> dict[str, Any]:
        """Integrate ODEs and return time-series output + events."""
        y0 = self.initial_state()
        t_end = float(self.batch.batch_time_s)
        t_eval = np.arange(0.0, t_end + 1e-9, float(self.batch.sample_interval_s))
        self.state_machine.start(0.0)
        self._last_t = 0.0

        events = make_scipy_events(self.state_machine, STATE_INDEX, self.c0_nitro)

        sol = solve_ivp(
            fun=self.rhs,
            t_span=(0.0, t_end),
            y0=y0,
            method=self.config.method,
            t_eval=t_eval,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
            events=events,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Finalize event state machine from trajectory (and terminal event cause)
        for i, t in enumerate(sol.t):
            y = sol.y[:, i]
            conc = self._concentrations(y)
            total = sum(conc.values()) or 1.0
            x_rha = conc[SPECIES_MAP["XHA"]] / total
            conv = 1.0 - conc[SPECIES_MAP["NX"]] / self.c0_nitro if self.c0_nitro > 0 else 0.0
            self.state_machine.evaluate(
                float(t),
                temperature_c=float(y[STATE_INDEX["T_c"]]),
                pressure_bar=float(y[STATE_INDEX["P_bar"]]),
                rha_mole_fraction=x_rha,
                nitro_conversion=conv,
            )
            if self.state_machine.tripped or self.state_machine.phase == BatchPhase.COMPLETE:
                break

        # Map solve_ivp terminal event indices: [temp, pressure, rha, conversion]
        event_names = ["temperature_interlock", "pressure_interlock", "rha_interlock", "conversion_complete"]
        if sol.t_events:
            for idx, times in enumerate(sol.t_events):
                if times is not None and len(times) > 0:
                    t_ev = float(times[-1])
                    name = event_names[idx]
                    self.state_machine.record(
                        t_ev,
                        "solver_event",
                        f"solve_ivp terminal event: {name}",
                        event=name,
                    )
                    if name == "conversion_complete":
                        self.state_machine._transition(
                            t_ev,
                            BatchPhase.COMPLETE,
                            "Conversion complete (solver event)",
                        )
                    elif name.endswith("_interlock"):
                        self.state_machine.tripped = True
                        self.state_machine._transition(
                            t_ev,
                            BatchPhase.TRIPPED,
                            f"Interlock trip via solver event ({name})",
                            event=name,
                        )

        rows = []
        for i, t in enumerate(sol.t):
            y = sol.y[:, i]
            r1, r2, r3 = self._rates(y)
            q_rxn = r1 * (-self.delta_H[0]) + r2 * (-self.delta_H[1]) + r3 * (-self.delta_H[2])
            conc = self._concentrations(y)
            total = sum(conc.values()) or 1.0
            x_rha = conc[SPECIES_MAP["XHA"]] / total
            conv = 1.0 - conc[SPECIES_MAP["NX"]] / self.c0_nitro if self.c0_nitro > 0 else 0.0
            phase = self.state_machine.phase.value
            # Reconstruct phase label along trajectory for output clarity
            if self.state_machine.tripped and float(t) >= (self.state_machine.events[-1].time_s if self.state_machine.events else 0):
                # keep tripped only after trip time
                trip_times = [e.time_s for e in self.state_machine.events if e.phase == BatchPhase.TRIPPED]
                if trip_times and float(t) >= trip_times[0]:
                    phase = BatchPhase.TRIPPED.value
                elif float(t) == 0.0:
                    phase = BatchPhase.HEAT_UP.value
                else:
                    phase = BatchPhase.HYDROGENATION.value
            rows.append(
                {
                    "time_s": float(t),
                    "time_min": float(t) / 60.0,
                    "C_nitroxylene": conc[SPECIES_MAP["NX"]],
                    "C_nitrosoxylene": conc[SPECIES_MAP["NSX"]],
                    "C_hydroxylamine": conc[SPECIES_MAP["XHA"]],
                    "C_xylidine": conc[SPECIES_MAP["XYL"]],
                    "C_H2O": conc[SPECIES_MAP["H2O"]],
                    "C_H2_dissolved": conc[SPECIES_MAP["H2"]],
                    "C_H2_star": henrys_law_ch2_star(y[STATE_INDEX["P_bar"]], self.henry),
                    "T_c": float(y[STATE_INDEX["T_c"]]),
                    "P_bar": float(y[STATE_INDEX["P_bar"]]),
                    "T_jacket_c": self.controllers.jacket_temp_c,
                    "r1": r1,
                    "r2": r2,
                    "r3": r3,
                    "Q_rxn_kW_m3": q_rxn,
                    "nitro_conversion": conv,
                    "x_rha": x_rha,
                    "phase": phase,
                }
            )

        df = pd.DataFrame(rows)
        return {
            "success": True,
            "solver": self.config.method,
            "message": sol.message,
            "n_steps": int(getattr(sol, "nfev", 0)),
            "t_final_s": float(sol.t[-1]),
            "events": self.state_machine.as_dicts(),
            "final_phase": self.state_machine.phase.value,
            "timeseries": df,
            "timeseries_records": df.to_dict(orient="records"),
            "summary": {
                "final_conversion": float(df["nitro_conversion"].iloc[-1]),
                "final_T_c": float(df["T_c"].iloc[-1]),
                "final_P_bar": float(df["P_bar"].iloc[-1]),
                "max_T_c": float(df["T_c"].max()),
                "max_x_rha": float(df["x_rha"].max()),
                "max_Q_rxn_kW_m3": float(df["Q_rxn_kW_m3"].max()),
            },
        }
