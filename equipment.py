"""
Reactor equipment package for EQ-STBR-5000L.

Standalone custom equipment model for an industrial semi-batch catalytic
hydrogenation vessel (Hastelloy stirred-tank reactor). Parameters are
mutable so UA, cooling efficiency, and related factors can be adjusted
during anomaly-injection studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Optional

# ---------------------------------------------------------------------------
# Unit helpers (SI-facing internals)
# ---------------------------------------------------------------------------
_RPM_TO_RAD_S = 2.0 * 3.141592653589793 / 60.0  # rad/s per RPM
_BAR_TO_PA = 1.0e5
_L_TO_M3 = 1.0e-3
_KW_TO_W = 1.0e3
_KJ_TO_J = 1.0e3
_MIN_TO_S = 60.0


@dataclass
class ReactorEquipmentPackage:
    """
    Custom equipment package for EQ-STBR-5000L semi-batch catalytic
    hydrogenation.

    Holds vessel geometry, charge inventory, thermal network, agitation,
    and H2 pressure / feed hydraulics. All thermal and hydraulic
    multipliers are public attributes so anomaly injectors can retune
    ``U``, ``A``, ``cooling_efficiency``, or agitator factors at runtime.
    """

    # --- Identity -----------------------------------------------------------
    tag: str = "EQ-STBR-5000L"
    vessel_type: str = "5,000 L Hastelloy Stirred Tank Reactor (STBR)"
    material: str = "Hastelloy"

    # --- Geometry / inventory -----------------------------------------------
    vessel_volume_L: float = 5000.0
    working_fill_volume_L: float = 3500.0  # V_w
    fill_factor: float = 0.70
    headspace_fraction: float = 0.30

    liquid_charge_mass_kg: float = 3899.0
    nitroxylene_mass_kg: float = 3150.0
    nitroxylene_moles: float = 20838.8  # 2,4-Nitroxylene

    # --- Physical properties ------------------------------------------------
    rho_kg_m3: float = 1114.0  # average liquid mixture density
    cp_mix_kJ_kgK: float = 2.1
    c_vessel_kJ_K: float = 850.0
    # Reference viscosity for torque scaling (Pa·s); typical organic mixture
    mu_ref_Pa_s: float = 0.0025

    # --- Thermal systems ----------------------------------------------------
    heat_exchange_area_m2: float = 12.5  # A
    overall_heat_transfer_coeff_W_m2K: float = 350.0  # U (nominal)
    cooling_efficiency: float = 1.0  # anomaly-injectable multiplier on UA
    tcu_supply_temp_min_C: float = 15.0
    tcu_supply_temp_max_C: float = 25.0
    tcu_supply_temp_C: float = 20.0  # nominal mid-range setpoint

    # --- Agitation package --------------------------------------------------
    agitator_motor_power_kW: float = 15.0
    agitator_speed_rpm: float = 180.0
    agitator_nominal_power_kW: float = 11.2
    agitator_nominal_torque_Nm: float = 594.2
    # Reynolds exponent for laminar→turbulent blending of viscosity effect
    agitator_viscosity_exponent: float = 0.15
    agitator_power_derate: float = 1.0  # anomaly-injectable

    # --- Pressure & H2 feed hydraulics --------------------------------------
    headspace_pressure_target_bar: float = 10.0
    operating_pressure_min_bar: float = 5.0
    operating_pressure_max_bar: float = 15.0
    design_pressure_max_bar: float = 20.0
    mfc_max_h2_feed_kg_min: float = 5.0
    # Ideal-gas / inventory bookkeeping (optional headspace model aids)
    h2_molar_mass_kg_kmol: float = 2.016
    headspace_temperature_C: float = 25.0

    # --- Runtime mass-balance state (H2 dosing) -----------------------------
    h2_fed_kg: float = 0.0
    liquid_mass_kg: float = field(init=False)
    headspace_pressure_bar: float = field(init=False)

    def __post_init__(self) -> None:
        self.liquid_mass_kg = float(self.liquid_charge_mass_kg)
        self.headspace_pressure_bar = float(self.headspace_pressure_target_bar)

    # =======================================================================
    # Derived geometry / capacity properties
    # =======================================================================

    @property
    def vessel_volume_m3(self) -> float:
        return self.vessel_volume_L * _L_TO_M3

    @property
    def working_fill_volume_m3(self) -> float:
        """V_w in m³ (3.5 m³ at nominal 70% fill)."""
        return self.working_fill_volume_L * _L_TO_M3

    @property
    def headspace_volume_m3(self) -> float:
        return self.vessel_volume_m3 - self.working_fill_volume_m3

    @property
    def ua_W_K(self) -> float:
        """
        Effective UA including cooling-efficiency multiplier.

        Anomaly injectors may lower ``cooling_efficiency`` or retune
        ``overall_heat_transfer_coeff_W_m2K`` / ``heat_exchange_area_m2``.
        """
        return (
            self.overall_heat_transfer_coeff_W_m2K
            * self.heat_exchange_area_m2
            * self.cooling_efficiency
        )

    @property
    def cp_mix_J_kgK(self) -> float:
        return self.cp_mix_kJ_kgK * _KJ_TO_J

    @property
    def c_vessel_J_K(self) -> float:
        return self.c_vessel_kJ_K * _KJ_TO_J

    @property
    def thermal_mass_liquid_J_K(self) -> float:
        """m_liquid * Cp_mix [J/K]."""
        return self.liquid_mass_kg * self.cp_mix_J_kgK

    @property
    def thermal_mass_total_J_K(self) -> float:
        """Liquid + vessel shell thermal capacity [J/K]."""
        return self.thermal_mass_liquid_J_K + self.c_vessel_J_K

    @property
    def agitator_omega_rad_s(self) -> float:
        return self.agitator_speed_rpm * _RPM_TO_RAD_S

    @property
    def mfc_max_h2_feed_kg_s(self) -> float:
        return self.mfc_max_h2_feed_kg_min / _MIN_TO_S

    # =======================================================================
    # Configuration / anomaly injection
    # =======================================================================

    def configure(self, **kwargs: float) -> "ReactorEquipmentPackage":
        """
        Update any numeric equipment parameter in place.

        Typical anomaly keys: ``overall_heat_transfer_coeff_W_m2K``,
        ``heat_exchange_area_m2``, ``cooling_efficiency``,
        ``agitator_power_derate``, ``mfc_max_h2_feed_kg_min``.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"{type(self).__name__} has no configurable parameter '{key}'"
                )
            setattr(self, key, value)
        return self

    def set_ua(
        self,
        *,
        U: Optional[float] = None,
        A: Optional[float] = None,
        cooling_efficiency: Optional[float] = None,
    ) -> "ReactorEquipmentPackage":
        """Convenience setter for heat-transfer network (anomaly injection)."""
        if U is not None:
            self.overall_heat_transfer_coeff_W_m2K = float(U)
        if A is not None:
            self.heat_exchange_area_m2 = float(A)
        if cooling_efficiency is not None:
            self.cooling_efficiency = float(cooling_efficiency)
        return self

    def snapshot(self) -> dict[str, float | str]:
        """Export key constants and live state for logging / replay."""
        return {
            "tag": self.tag,
            "vessel_type": self.vessel_type,
            "V_w_m3": self.working_fill_volume_m3,
            "rho_kg_m3": self.rho_kg_m3,
            "Cp_mix_kJ_kgK": self.cp_mix_kJ_kgK,
            "C_vessel_kJ_K": self.c_vessel_kJ_K,
            "A_m2": self.heat_exchange_area_m2,
            "U_W_m2K": self.overall_heat_transfer_coeff_W_m2K,
            "cooling_efficiency": self.cooling_efficiency,
            "UA_W_K": self.ua_W_K,
            "agitator_rpm": self.agitator_speed_rpm,
            "agitator_nominal_torque_Nm": self.agitator_nominal_torque_Nm,
            "P_target_bar": self.headspace_pressure_target_bar,
            "P_headspace_bar": self.headspace_pressure_bar,
            "mfc_max_kg_min": self.mfc_max_h2_feed_kg_min,
            "liquid_mass_kg": self.liquid_mass_kg,
            "h2_fed_kg": self.h2_fed_kg,
            "nitroxylene_moles": self.nitroxylene_moles,
        }

    # =======================================================================
    # Thermal helpers
    # =======================================================================

    def clamp_jacket_temperature_C(self, T_jkt_C: float) -> float:
        """Clamp jacket / TCU supply temperature to the configured range."""
        return min(
            self.tcu_supply_temp_max_C,
            max(self.tcu_supply_temp_min_C, T_jkt_C),
        )

    def heat_transfer_rate_W(
        self,
        T_rx_C: float,
        T_jkt_C: Optional[float] = None,
        *,
        U: Optional[float] = None,
        A: Optional[float] = None,
        cooling_efficiency: Optional[float] = None,
        clamp_tcu_supply: bool = False,
    ) -> float:
        """
        Instantaneous heat transfer rate Q = η · U · A · (T_rx − T_jkt) [W].

        Positive Q means heat leaving the reaction mass (cooling) when
        T_rx > T_jkt. ``T_jkt_C`` is the jacket-side process temperature and
        is not clamped by default (jacket fluid may warm above TCU supply).
        Pass ``clamp_tcu_supply=True`` only when ``T_jkt_C`` represents a TCU
        supply setpoint that must stay inside 15–25 °C. Optional U / A / η
        overrides allow one-shot anomaly injection without mutating state.
        """
        if T_jkt_C is None:
            T_jkt_C = self.tcu_supply_temp_C
        if clamp_tcu_supply:
            T_jkt_C = self.clamp_jacket_temperature_C(T_jkt_C)

        U_eff = (
            self.overall_heat_transfer_coeff_W_m2K if U is None else float(U)
        )
        A_eff = self.heat_exchange_area_m2 if A is None else float(A)
        eta = (
            self.cooling_efficiency
            if cooling_efficiency is None
            else float(cooling_efficiency)
        )
        return eta * U_eff * A_eff * (T_rx_C - T_jkt_C)

    # Alias matching the process notation Q_cooling
    def Q_cooling(
        self,
        T_rx: float,
        T_jkt: Optional[float] = None,
        **overrides: float,
    ) -> float:
        """Q_cooling = U · A · (T_rx − T_jkt) with optional η / UA overrides [W]."""
        return self.heat_transfer_rate_W(T_rx, T_jkt, **overrides)

    def cooling_duty_kW(
        self,
        T_rx_C: float,
        T_jkt_C: Optional[float] = None,
        **overrides: float,
    ) -> float:
        """Heat removal duty in kW (positive when cooling the batch)."""
        return self.heat_transfer_rate_W(T_rx_C, T_jkt_C, **overrides) / _KW_TO_W

    # =======================================================================
    # Agitation / torque helpers
    # =======================================================================

    def agitator_torque_Nm(
        self,
        *,
        density_kg_m3: Optional[float] = None,
        viscosity_Pa_s: Optional[float] = None,
        speed_rpm: Optional[float] = None,
        power_derate: Optional[float] = None,
    ) -> float:
        """
        Estimate agitator shaft torque from fluid density / viscosity.

        Uses a blended turbulent scaling relative to nominal design point:

            τ / τ_nom ≈ (ρ / ρ_nom) · (N / N_nom)² · (μ / μ_ref)^n · derate

        where ``n = agitator_viscosity_exponent`` (mild viscosity effect in
        the transitional/turbulent regime typical of hydrogenation slurries).
        """
        rho = self.rho_kg_m3 if density_kg_m3 is None else float(density_kg_m3)
        mu = self.mu_ref_Pa_s if viscosity_Pa_s is None else float(viscosity_Pa_s)
        N = self.agitator_speed_rpm if speed_rpm is None else float(speed_rpm)
        derate = (
            self.agitator_power_derate
            if power_derate is None
            else float(power_derate)
        )

        if rho <= 0.0 or N <= 0.0 or mu <= 0.0:
            return 0.0

        density_ratio = rho / self.rho_kg_m3
        speed_ratio = N / self.agitator_speed_rpm
        viscosity_ratio = (mu / self.mu_ref_Pa_s) ** self.agitator_viscosity_exponent

        return (
            self.agitator_nominal_torque_Nm
            * density_ratio
            * (speed_ratio**2)
            * viscosity_ratio
            * derate
        )

    def agitator_power_kW(
        self,
        *,
        density_kg_m3: Optional[float] = None,
        viscosity_Pa_s: Optional[float] = None,
        speed_rpm: Optional[float] = None,
        power_derate: Optional[float] = None,
    ) -> float:
        """Shaft power P = τ · ω [kW] at the requested fluid / speed state."""
        N = self.agitator_speed_rpm if speed_rpm is None else float(speed_rpm)
        torque = self.agitator_torque_Nm(
            density_kg_m3=density_kg_m3,
            viscosity_Pa_s=viscosity_Pa_s,
            speed_rpm=N,
            power_derate=power_derate,
        )
        omega = N * _RPM_TO_RAD_S
        return (torque * omega) / _KW_TO_W

    def agitator_within_motor_rating(
        self,
        *,
        density_kg_m3: Optional[float] = None,
        viscosity_Pa_s: Optional[float] = None,
        speed_rpm: Optional[float] = None,
    ) -> bool:
        """True if estimated shaft power is at or below motor nameplate."""
        return (
            self.agitator_power_kW(
                density_kg_m3=density_kg_m3,
                viscosity_Pa_s=viscosity_Pa_s,
                speed_rpm=speed_rpm,
            )
            <= self.agitator_motor_power_kW
        )

    # =======================================================================
    # Pressure & H2 mass-balance helpers
    # =======================================================================

    def clamp_h2_feed_rate_kg_min(self, feed_kg_min: float) -> float:
        """Clamp requested H2 feed to [0, MFC max]."""
        return min(self.mfc_max_h2_feed_kg_min, max(0.0, feed_kg_min))

    def clamp_headspace_pressure_bar(
        self,
        pressure_bar: float,
        *,
        to_design_limit: bool = True,
    ) -> float:
        """
        Clamp headspace pressure.

        By default clamps to design max (20 bar). Pass
        ``to_design_limit=False`` to clamp to the operating window (5–15 bar).
        """
        lo = 0.0 if to_design_limit else self.operating_pressure_min_bar
        hi = (
            self.design_pressure_max_bar
            if to_design_limit
            else self.operating_pressure_max_bar
        )
        return min(hi, max(lo, pressure_bar))

    def pressure_in_operating_range(
        self, pressure_bar: Optional[float] = None
    ) -> bool:
        P = (
            self.headspace_pressure_bar
            if pressure_bar is None
            else float(pressure_bar)
        )
        return self.operating_pressure_min_bar <= P <= self.operating_pressure_max_bar

    def update_h2_mass_balance(
        self,
        dt_s: float,
        feed_kg_min: float,
        *,
        consumption_kg_s: float = 0.0,
        update_pressure: bool = True,
        T_headspace_C: Optional[float] = None,
    ) -> MutableMapping[str, float]:
        """
        Advance H2 dosing mass balance over ``dt_s`` seconds.

        Mass balance on fed hydrogen inventory:

            m_H2(t+dt) = m_H2(t) + ṁ_feed · dt − ṁ_consumed · dt

        Liquid charge mass is increased by the net H2 retained in the
        slurry (consumed / dissolved contribution approximated by
        ``consumption_kg_s``). Optional ideal-gas headspace pressure update
        uses unconsumed feed accumulation in the headspace volume.

        Returns a dict of instantaneous balance terms (SI-ish mixed units
        documented by key suffix).
        """
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")

        feed_kg_min_clamped = self.clamp_h2_feed_rate_kg_min(feed_kg_min)
        feed_kg_s = feed_kg_min_clamped / _MIN_TO_S
        cons_kg_s = max(0.0, float(consumption_kg_s))

        dm_feed = feed_kg_s * dt_s
        dm_cons = min(cons_kg_s * dt_s, self.h2_fed_kg + dm_feed)
        dm_headspace = dm_feed - dm_cons

        self.h2_fed_kg += dm_feed
        # Consumed H2 reports to liquid/product mass; free H2 stays gaseous.
        self.liquid_mass_kg += dm_cons

        result: dict[str, float] = {
            "dt_s": dt_s,
            "feed_kg_min": feed_kg_min_clamped,
            "feed_kg_s": feed_kg_s,
            "consumption_kg_s": cons_kg_s,
            "dm_feed_kg": dm_feed,
            "dm_consumed_kg": dm_cons,
            "dm_headspace_kg": dm_headspace,
            "h2_fed_kg": self.h2_fed_kg,
            "liquid_mass_kg": self.liquid_mass_kg,
            "headspace_pressure_bar": self.headspace_pressure_bar,
        }

        if update_pressure and self.headspace_volume_m3 > 0.0:
            T_C = (
                self.headspace_temperature_C
                if T_headspace_C is None
                else float(T_headspace_C)
            )
            T_K = T_C + 273.15
            R_kJ_kmolK = 8.314462618  # kJ/(kmol·K) ≡ kPa·m³/(kmol·K)
            # Δn = Δm / M ; ΔP [kPa] = Δn R T / V
            dn_kmol = dm_headspace / self.h2_molar_mass_kg_kmol
            dP_kPa = (dn_kmol * R_kJ_kmolK * T_K) / self.headspace_volume_m3
            dP_bar = dP_kPa / 100.0
            self.headspace_pressure_bar = self.clamp_headspace_pressure_bar(
                self.headspace_pressure_bar + dP_bar,
                to_design_limit=True,
            )
            result["dP_bar"] = dP_bar
            result["headspace_pressure_bar"] = self.headspace_pressure_bar

        return result

    def reset_mass_balance(
        self,
        *,
        liquid_mass_kg: Optional[float] = None,
        h2_fed_kg: float = 0.0,
        headspace_pressure_bar: Optional[float] = None,
    ) -> None:
        """Reset dosing inventory to charge / target pressure conditions."""
        self.liquid_mass_kg = (
            float(self.liquid_charge_mass_kg)
            if liquid_mass_kg is None
            else float(liquid_mass_kg)
        )
        self.h2_fed_kg = float(h2_fed_kg)
        self.headspace_pressure_bar = (
            float(self.headspace_pressure_target_bar)
            if headspace_pressure_bar is None
            else float(headspace_pressure_bar)
        )

    # =======================================================================
    # Factory
    # =======================================================================

    @classmethod
    def default(cls) -> "ReactorEquipmentPackage":
        """Nominal EQ-STBR-5000L package as specified."""
        return cls()

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float]) -> "ReactorEquipmentPackage":
        """Build a package with selected fields overridden at construction."""
        return cls(**dict(overrides))


__all__ = ["ReactorEquipmentPackage"]
