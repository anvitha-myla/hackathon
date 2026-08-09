"""
EQ-STBR-5000L equipment package and 8-step process state machine.

Semi-batch catalytic hydrogenation:
    2,4-Nitroxylene + 3 H2 → 2,4-Xylidine + 2 H2O
in a 5,000 L Hastelloy stirred-tank reactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class ProcessStep(str, Enum):
    """Eight distinct operational steps for the hydrogenation batch."""

    PREPARATION_TARE = "PREPARATION_TARE"  # Step 1
    RAW_MATERIAL_LOADING = "RAW_MATERIAL_LOADING"  # Step 2
    NITROGEN_INERTING = "NITROGEN_INERTING"  # Step 3
    PRE_HEATING = "PRE_HEATING"  # Step 4
    ACTIVE_HYDROGENATION = "ACTIVE_HYDROGENATION"  # Step 5
    DIGESTION_HOLD = "DIGESTION_HOLD"  # Step 6
    DEGASSING_COOLING = "DEGASSING_COOLING"  # Step 7
    PRODUCT_DISCHARGE = "PRODUCT_DISCHARGE"  # Step 8

    @property
    def step_id(self) -> int:
        return STEP_ORDER.index(self) + 1

    @property
    def duration_min(self) -> int:
        return STEP_DURATIONS_MIN[self]


STEP_ORDER: Tuple[ProcessStep, ...] = (
    ProcessStep.PREPARATION_TARE,
    ProcessStep.RAW_MATERIAL_LOADING,
    ProcessStep.NITROGEN_INERTING,
    ProcessStep.PRE_HEATING,
    ProcessStep.ACTIVE_HYDROGENATION,
    ProcessStep.DIGESTION_HOLD,
    ProcessStep.DEGASSING_COOLING,
    ProcessStep.PRODUCT_DISCHARGE,
)

STEP_DURATIONS_MIN: Dict[ProcessStep, int] = {
    ProcessStep.PREPARATION_TARE: 15,
    ProcessStep.RAW_MATERIAL_LOADING: 45,
    ProcessStep.NITROGEN_INERTING: 30,
    ProcessStep.PRE_HEATING: 45,
    ProcessStep.ACTIVE_HYDROGENATION: 90,
    ProcessStep.DIGESTION_HOLD: 30,
    ProcessStep.DEGASSING_COOLING: 45,
    ProcessStep.PRODUCT_DISCHARGE: 30,
}

TOTAL_BATCH_DURATION_MIN: int = sum(STEP_DURATIONS_MIN.values())  # 330


def step_time_bounds() -> Dict[ProcessStep, Tuple[int, int]]:
    """Absolute batch-minute [start, end) for each process step."""
    bounds: Dict[ProcessStep, Tuple[int, int]] = {}
    t = 0
    for step in STEP_ORDER:
        dur = STEP_DURATIONS_MIN[step]
        bounds[step] = (t, t + dur)
        t += dur
    return bounds


STEP_BOUNDS: Dict[ProcessStep, Tuple[int, int]] = step_time_bounds()


def step_at_minute(t_min: int) -> ProcessStep:
    """Return the process step active at absolute batch minute ``t_min``."""
    for step, (a, b) in STEP_BOUNDS.items():
        if a <= t_min < b:
            return step
    return STEP_ORDER[-1]


@dataclass
class ReactorEquipmentPackage:
    """
    Custom equipment package for EQ-STBR-5000L.

    Holds vessel geometry, charge inventory, thermal network, agitation,
    and H2 pressure / feed hydraulics. Thermal multipliers (``U``, ``A``,
    ``cooling_efficiency``) are mutable for anomaly injection.
    """

    # --- Identity -----------------------------------------------------------
    tag: str = "EQ-STBR-5000L"
    vessel_type: str = "5,000 L Hastelloy Stirred Tank Reactor (STBR)"
    material: str = "Hastelloy"

    # --- Geometry / inventory -----------------------------------------------
    vessel_volume_L: float = 5000.0
    working_fill_volume_L: float = 3500.0  # V_w (70% fill, 30% H2 headspace)
    fill_factor: float = 0.70
    headspace_fraction: float = 0.30

    liquid_charge_mass_kg: float = 3899.0
    nitroxylene_mass_kg: float = 3150.0
    nitroxylene_moles: float = 20838.8  # 2,4-Nitroxylene

    # --- Physical properties ------------------------------------------------
    rho_kg_m3: float = 1114.0
    cp_mix_kJ_kgK: float = 2.1
    c_vessel_kJ_K: float = 850.0
    mu_ref_Pa_s: float = 0.0025

    # --- Kinetics (nitro-aromatic hydrogenation) ----------------------------
    delta_H_kJ_mol: float = -540.0  # exothermic
    E_a_kJ_mol: float = 45.0
    k0_L_mol_min: float = 12500.0  # second-order pre-exponential
    stoich_H2: float = 3.0  # mol H2 / mol nitroxylene
    h2_molar_mass_kg_kmol: float = 2.016
    # Henry-like dissolved H2 scale [mol/L/bar] (effective for r = k·C_A·C_H2)
    henry_H2_mol_L_bar: float = 0.08

    # --- Thermal systems ----------------------------------------------------
    heat_exchange_area_m2: float = 12.5  # A
    overall_heat_transfer_coeff_W_m2K: float = 350.0  # U nominal
    cooling_efficiency: float = 1.0
    tcu_supply_temp_min_C: float = 15.0
    tcu_supply_temp_max_C: float = 25.0
    tcu_supply_temp_C: float = 20.0
    jacket_heat_temp_C: float = 85.0  # Step 4 heating
    jacket_chill_temp_C: float = 15.0  # Step 7 chilling

    # --- Agitation package --------------------------------------------------
    agitator_motor_power_kW: float = 15.0
    agitator_speed_rpm: float = 180.0
    agitator_nominal_power_kW: float = 11.2
    agitator_nominal_torque_Nm: float = 594.2
    agitator_idle_torque_Nm: float = 150.0
    agitator_viscosity_exponent: float = 0.15
    agitator_power_derate: float = 1.0

    # --- Pressure & H2 feed hydraulics --------------------------------------
    headspace_pressure_target_bar: float = 10.0
    operating_pressure_min_bar: float = 5.0
    operating_pressure_max_bar: float = 15.0
    design_pressure_max_bar: float = 20.0
    atmospheric_pressure_bar: float = 1.0
    mfc_max_h2_feed_kg_min: float = 5.0
    h2_feed_min_kg_min: float = 1.5
    h2_feed_max_kg_min: float = 3.5
    h2_pad_kg_min: float = 0.1  # Step 4 pad
    cooling_water_min_L_min: float = 50.0
    cooling_water_max_L_min: float = 120.0

    # --- Runtime state ------------------------------------------------------
    h2_fed_kg: float = 0.0
    liquid_mass_kg: float = field(init=False)
    headspace_pressure_bar: float = field(init=False)

    def __post_init__(self) -> None:
        self.liquid_mass_kg = float(self.liquid_charge_mass_kg)
        self.headspace_pressure_bar = float(self.atmospheric_pressure_bar)

    # =======================================================================
    # Derived properties
    # =======================================================================

    @property
    def vessel_volume_m3(self) -> float:
        return self.vessel_volume_L * 1.0e-3

    @property
    def working_fill_volume_m3(self) -> float:
        return self.working_fill_volume_L * 1.0e-3

    @property
    def headspace_volume_m3(self) -> float:
        return self.vessel_volume_m3 - self.working_fill_volume_m3

    @property
    def ua_W_K(self) -> float:
        return (
            self.overall_heat_transfer_coeff_W_m2K
            * self.heat_exchange_area_m2
            * self.cooling_efficiency
        )

    @property
    def cp_mix_J_kgK(self) -> float:
        return self.cp_mix_kJ_kgK * 1.0e3

    @property
    def c_vessel_J_K(self) -> float:
        return self.c_vessel_kJ_K * 1.0e3

    @property
    def delta_H_J_mol(self) -> float:
        return self.delta_H_kJ_mol * 1.0e3

    @property
    def E_a_J_mol(self) -> float:
        return self.E_a_kJ_mol * 1.0e3

    @property
    def agitator_omega_rad_s(self) -> float:
        return self.agitator_speed_rpm * 2.0 * 3.141592653589793 / 60.0

    @property
    def agitator_power_W(self) -> float:
        return self.agitator_nominal_power_kW * 1000.0 * self.agitator_power_derate

    def thermal_mass_J_K(self, mass_kg: Optional[float] = None) -> float:
        m = self.liquid_mass_kg if mass_kg is None else float(mass_kg)
        return m * self.cp_mix_J_kgK + self.c_vessel_J_K

    def configure(self, **kwargs: float) -> "ReactorEquipmentPackage":
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"No parameter '{key}'")
            setattr(self, key, value)
        return self

    def set_ua(
        self,
        *,
        U: Optional[float] = None,
        A: Optional[float] = None,
        cooling_efficiency: Optional[float] = None,
    ) -> "ReactorEquipmentPackage":
        if U is not None:
            self.overall_heat_transfer_coeff_W_m2K = float(U)
        if A is not None:
            self.heat_exchange_area_m2 = float(A)
        if cooling_efficiency is not None:
            self.cooling_efficiency = float(cooling_efficiency)
        return self

    def heat_transfer_rate_W(
        self,
        T_rx_C: float,
        T_jkt_C: float,
        *,
        ua_scale: float = 1.0,
    ) -> float:
        """Q = η·U·A·(T_rx − T_jkt) [W]. Positive ⇒ heat leaving the batch."""
        return self.ua_W_K * float(ua_scale) * (T_rx_C - T_jkt_C)

    def Q_cooling(
        self, T_rx: float, T_jkt: float, *, ua_scale: float = 1.0
    ) -> float:
        return self.heat_transfer_rate_W(T_rx, T_jkt, ua_scale=ua_scale)

    def agitator_torque_Nm(
        self,
        *,
        conversion: float = 0.0,
        foam_scale: float = 1.0,
        mode: str = "full",
    ) -> float:
        """
        Shaft torque [N·m].

        mode:
          - ``off``  → 0
          - ``idle`` → idle torque (~150 N·m)
          - ``full`` → nominal 594.2 N·m scaled by conversion / foam
        """
        if mode == "off":
            return 0.0
        if mode == "idle":
            return self.agitator_idle_torque_Nm * foam_scale
        x = max(0.0, min(float(conversion), 1.2))
        # Mild load rise with conversion (viscosity / slurry densification)
        load = 1.0 + 0.08 * x
        return (
            self.agitator_nominal_torque_Nm
            * load
            * self.agitator_power_derate
            * foam_scale
        )

    def rate_constant(self, T_C: float) -> float:
        """Arrhenius k(T) [L/(mol·min)]."""
        R = 8.314462618  # J/(mol·K)
        T_K = T_C + 273.15
        return self.k0_L_mol_min * (
            __import__("math").exp(-self.E_a_J_mol / (R * T_K))
        )

    def reaction_rate_mol_L_min(
        self,
        T_C: float,
        C_A: float,
        P_H2_bar: float,
    ) -> float:
        """
        Second-order rate r = k(T) · C_A · C_H2  [mol/(L·min)].

        Dissolved hydrogen approximated as C_H2 = H · P_H2.
        """
        if C_A <= 0.0 or P_H2_bar <= 0.0:
            return 0.0
        C_H2 = self.henry_H2_mol_L_bar * max(P_H2_bar, 0.0)
        return self.rate_constant(T_C) * max(C_A, 0.0) * C_H2

    def reset_mass_balance(self) -> None:
        self.liquid_mass_kg = float(self.liquid_charge_mass_kg)
        self.h2_fed_kg = 0.0
        self.headspace_pressure_bar = float(self.atmospheric_pressure_bar)

    @classmethod
    def default(cls) -> "ReactorEquipmentPackage":
        return cls()


@dataclass
class StepConstraints:
    """Enforced physical envelopes for a single process step."""

    step: ProcessStep
    duration_min: int
    weight_kg: Tuple[float, float]  # (start, end) ramp targets
    T_rx_C: Tuple[float, float]
    T_jkt_C: float
    P_bar: Tuple[float, float]
    H2_flow_kg_min: Tuple[float, float]
    torque_mode: str  # off | idle | full | weight_gated
    cooling_L_min: Tuple[float, float]
    conversion_pct: Tuple[float, float]
    vent_open: bool = False
    notes: str = ""


def build_step_constraints(eq: ReactorEquipmentPackage) -> Dict[ProcessStep, StepConstraints]:
    """Canonical step-specific physical constraints for EQ-STBR-5000L."""
    m = eq.liquid_charge_mass_kg
    return {
        ProcessStep.PREPARATION_TARE: StepConstraints(
            step=ProcessStep.PREPARATION_TARE,
            duration_min=15,
            weight_kg=(0.0, 0.0),
            T_rx_C=(25.0, 25.0),
            T_jkt_C=25.0,
            P_bar=(1.0, 1.0),
            H2_flow_kg_min=(0.0, 0.0),
            torque_mode="off",
            cooling_L_min=(0.0, 0.0),
            conversion_pct=(0.0, 0.0),
            vent_open=True,
            notes="Tare / empty vessel, vent open, agitator OFF",
        ),
        ProcessStep.RAW_MATERIAL_LOADING: StepConstraints(
            step=ProcessStep.RAW_MATERIAL_LOADING,
            duration_min=45,
            weight_kg=(0.0, m),
            T_rx_C=(25.0, 25.0),
            T_jkt_C=25.0,
            P_bar=(1.0, 1.0),
            H2_flow_kg_min=(0.0, 0.0),
            torque_mode="weight_gated",  # OFF until weight > 800 kg → idle
            cooling_L_min=(0.0, 0.0),
            conversion_pct=(0.0, 0.0),
            vent_open=True,
            notes="Charge ramp 0 → 3899 kg; agitator idle after 800 kg",
        ),
        ProcessStep.NITROGEN_INERTING: StepConstraints(
            step=ProcessStep.NITROGEN_INERTING,
            duration_min=30,
            weight_kg=(m, m),
            T_rx_C=(25.0, 25.0),
            T_jkt_C=25.0,
            P_bar=(1.0, 1.0),  # cycles 1→3→1 handled in simulator
            H2_flow_kg_min=(0.0, 0.0),
            torque_mode="idle",
            cooling_L_min=(0.0, 0.0),
            conversion_pct=(0.0, 0.0),
            vent_open=False,
            notes="N2 pressure leak-check cycles 1 → 3 → 1 bar",
        ),
        ProcessStep.PRE_HEATING: StepConstraints(
            step=ProcessStep.PRE_HEATING,
            duration_min=45,
            weight_kg=(m, m),
            T_rx_C=(25.0, 70.0),
            T_jkt_C=eq.jacket_heat_temp_C,
            P_bar=(1.0, 10.0),
            H2_flow_kg_min=(eq.h2_pad_kg_min, eq.h2_pad_kg_min),
            torque_mode="full",
            cooling_L_min=(0.0, 0.0),
            conversion_pct=(0.0, 0.0),
            vent_open=False,
            notes="Jacket heat to 70°C; H2 pad to 10 bar",
        ),
        ProcessStep.ACTIVE_HYDROGENATION: StepConstraints(
            step=ProcessStep.ACTIVE_HYDROGENATION,
            duration_min=90,
            weight_kg=(m, m),  # rises with H2 uptake in ODE
            T_rx_C=(70.0, 90.0),
            T_jkt_C=eq.tcu_supply_temp_C,
            P_bar=(10.0, 10.0),
            H2_flow_kg_min=(eq.h2_feed_min_kg_min, eq.h2_feed_max_kg_min),
            torque_mode="full",
            cooling_L_min=(eq.cooling_water_min_L_min, eq.cooling_water_max_L_min),
            conversion_pct=(0.0, 98.0),
            vent_open=False,
            notes="Stiff kinetic ODE; exotherm under jacket cooling",
        ),
        ProcessStep.DIGESTION_HOLD: StepConstraints(
            step=ProcessStep.DIGESTION_HOLD,
            duration_min=30,
            weight_kg=(m, m),
            T_rx_C=(70.0, 70.0),
            T_jkt_C=70.0,
            P_bar=(10.0, 10.0),
            H2_flow_kg_min=(0.5, 0.0),  # decays to 0
            torque_mode="full",
            cooling_L_min=(20.0, 40.0),
            conversion_pct=(98.0, 100.0),
            vent_open=False,
            notes="Digest to completion; H2 feed decays off",
        ),
        ProcessStep.DEGASSING_COOLING: StepConstraints(
            step=ProcessStep.DEGASSING_COOLING,
            duration_min=45,
            weight_kg=(m, m),
            T_rx_C=(70.0, 30.0),
            T_jkt_C=eq.jacket_chill_temp_C,
            P_bar=(10.0, 1.0),
            H2_flow_kg_min=(0.0, 0.0),
            torque_mode="full",
            cooling_L_min=(60.0, 100.0),
            conversion_pct=(100.0, 100.0),
            vent_open=False,
            notes="Depressurize + chilled jacket cool-down",
        ),
        ProcessStep.PRODUCT_DISCHARGE: StepConstraints(
            step=ProcessStep.PRODUCT_DISCHARGE,
            duration_min=30,
            weight_kg=(m, 0.0),
            T_rx_C=(30.0, 30.0),
            T_jkt_C=25.0,
            P_bar=(1.0, 1.0),
            H2_flow_kg_min=(0.0, 0.0),
            torque_mode="weight_gated_off",  # trips OFF when weight < 500 kg
            cooling_L_min=(0.0, 0.0),
            conversion_pct=(100.0, 100.0),
            vent_open=True,
            notes="Discharge to empty; agitator off below 500 kg",
        ),
    }


__all__ = [
    "ProcessStep",
    "STEP_ORDER",
    "STEP_DURATIONS_MIN",
    "STEP_BOUNDS",
    "TOTAL_BATCH_DURATION_MIN",
    "step_at_minute",
    "step_time_bounds",
    "ReactorEquipmentPackage",
    "StepConstraints",
    "build_step_constraints",
]
