"""Two-Tier Decision Engine — Industrial PSM & Automation.

Tier 1 — Hard Safety Interlocks (deterministic, non-overridable)
    * T_reactor > 105 °C     → ALARM_CRITICAL, jacket 100%, H₂ cutoff
    * P_headspace > 15 bar   → ALARM_HIGH, close H₂ MFC valve
    * C_hydroxyl > 0.08 mol/L → WARNING_ACCUMULATION, restrict dT/dt

Tier 2 — Soft Process Optimizations (advisory)
    * Mass-transfer bottleneck 3r/(k_L a C*) > 0.85 → +30 RPM advice
    * Stage-3 C_nitro < 0.5% of C0 → Stage-4 isothermal digest advice

Run
---
    python -m arip_backend.decision_engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from arip_backend.schemas.decision import DecisionOutput, InterlockAction, SafetyStatus

# --- Tier-1 hard limits (PSM) -------------------------------------------------
T_CRITICAL_C = 105.0
P_HIGH_BAR = 15.0
C_HYDROXYL_WARN_MOL_L = 0.08  # mol/L ≡ kmol/m³ numerically
MAX_DT_DT_ON_ACCUM_C_PER_MIN = 1.0  # restrict heat-up rate when RHA accumulates

# --- Tier-2 soft thresholds ---------------------------------------------------
MT_BOTTLENECK_RATIO = 0.85
NITRO_STAGE3_FRAC = 0.005  # 0.5% of initial charge
AGITATOR_BUMP_RPM = 30.0

# Alarm / interlock tags
ALARM_CRITICAL_TEMP = "ALARM_CRITICAL"
ALARM_HIGH_PRESSURE = "ALARM_HIGH"
WARNING_ACCUMULATION = "WARNING_ACCUMULATION"
IL_TEMP_TRIP = "IL_T_REACTOR_HIGH"
IL_PRESSURE_TRIP = "IL_P_HEADSPACE_HIGH"
IL_RHA_ACCUM = "IL_HYDROXYLAMINE_ACCUMULATION"
REC_AGITATOR = "REC_INCREASE_AGITATOR_RPM"
REC_STAGE4 = "REC_TRANSITION_STAGE4_DIGEST"


@dataclass
class ProcessSnapshot:
    """Minimal live plant / digital-twin state for decision evaluation."""

    T_reactor_c: float
    P_headspace_bar: float
    C_hydroxyl: float = 0.0  # mol/L (= kmol/m³)
    C_nitro: float = 0.0
    C_nitro_0: float = 2.8
    r_rxn: float = 0.0
    kla: float = 0.085
    C_H2_star: float = 0.04
    agitator_rpm: float = 180.0
    stage: int = 3  # batch stage index (3 = hydrogenation / late reaction)
    t_s: float = 0.0
    dT_dt_c_per_min: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProcessSnapshot":
        return cls(
            T_reactor_c=float(data.get("T_reactor_c", data.get("T_reactor", 0.0))),
            P_headspace_bar=float(
                data.get("P_headspace_bar", data.get("P_headspace", 0.0))
            ),
            C_hydroxyl=float(
                data.get("C_hydroxyl", data.get("C_XHA", data.get("hydroxylamine", 0.0)))
            ),
            C_nitro=float(data.get("C_nitro", data.get("C_NX", 0.0))),
            C_nitro_0=float(data.get("C_nitro_0", data.get("C0_nitro", 2.8))),
            r_rxn=float(data.get("r_rxn", data.get("r", 0.0))),
            kla=float(data.get("kla", data.get("kLa", 0.085))),
            C_H2_star=float(data.get("C_H2_star", data.get("C_H2_star_kmol_m3", 0.04))),
            agitator_rpm=float(data.get("agitator_rpm", data.get("RPM", 180.0))),
            stage=int(data.get("stage", data.get("batch_stage", 3))),
            t_s=float(data.get("t_s", data.get("time_s", 0.0))),
            dT_dt_c_per_min=(
                float(data["dT_dt_c_per_min"])
                if data.get("dT_dt_c_per_min") is not None
                else None
            ),
        )

    def mass_transfer_ratio(self, *, stoich_h2: float = 3.0) -> float:
        denom = max(self.kla * max(self.C_H2_star, 1e-12), 1e-12)
        return float(stoich_h2 * self.r_rxn / denom)

    def nitro_fraction_remaining(self) -> float:
        if self.C_nitro_0 <= 0.0:
            return 0.0
        return max(self.C_nitro, 0.0) / self.C_nitro_0


@dataclass
class TwoTierDecisionEngine:
    """PSM two-tier decision engine: hard interlocks + soft optimizations.

    Tier-1 outcomes always win. Soft Tier-2 advice is suppressed while
    ``safety_status == CRITICAL`` so operators focus on trip response.
    """

    T_critical_c: float = T_CRITICAL_C
    P_high_bar: float = P_HIGH_BAR
    C_hydroxyl_warn: float = C_HYDROXYL_WARN_MOL_L
    mt_bottleneck: float = MT_BOTTLENECK_RATIO
    nitro_stage3_frac: float = NITRO_STAGE3_FRAC
    agitator_bump_rpm: float = AGITATOR_BUMP_RPM
    max_dT_dt_on_accum: float = MAX_DT_DT_ON_ACCUM_C_PER_MIN
    suppress_soft_on_critical: bool = True
    history: list[DecisionOutput] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Tier 1 — Hard Safety Interlocks
    # ------------------------------------------------------------------
    def evaluate_tier1(
        self,
        snap: ProcessSnapshot,
    ) -> tuple[SafetyStatus, list[str], list[str], InterlockAction]:
        """Return (status, interlock_ids, alarm_messages, actuator_overrides)."""
        status = SafetyStatus.NOMINAL
        interlocks: list[str] = []
        alarms: list[str] = []
        actions = InterlockAction()

        # Temperature Limit Check
        if snap.T_reactor_c > self.T_critical_c:
            status = SafetyStatus.CRITICAL
            interlocks.append(IL_TEMP_TRIP)
            alarms.append(
                f"{ALARM_CRITICAL_TEMP}: T_reactor={snap.T_reactor_c:.1f}°C "
                f"> {self.T_critical_c:.0f}°C — max cooling + H₂ feed cutoff"
            )
            actions.cooling_jacket_flow_pct = 100.0
            actions.h2_feed_cutoff = True
            actions.h2_mfc_valve_closed = True

        # Overpressure Interlock
        if snap.P_headspace_bar > self.P_high_bar:
            # Escalate to CRITICAL if not already; ALARM_HIGH is the tag
            if status != SafetyStatus.CRITICAL:
                status = SafetyStatus.CRITICAL
            interlocks.append(IL_PRESSURE_TRIP)
            alarms.append(
                f"{ALARM_HIGH_PRESSURE}: P_headspace={snap.P_headspace_bar:.2f} bar "
                f"> {self.P_high_bar:.0f} bar — close H₂ MFC valve"
            )
            actions.h2_mfc_valve_closed = True
            actions.h2_feed_cutoff = True

        # Thermal Runaway Accumulation Hazard (hydroxylamine / RHA)
        if snap.C_hydroxyl > self.C_hydroxyl_warn:
            if status == SafetyStatus.NOMINAL:
                status = SafetyStatus.WARNING
            interlocks.append(IL_RHA_ACCUM)
            alarms.append(
                f"{WARNING_ACCUMULATION}: C_hydroxyl={snap.C_hydroxyl:.4f} mol/L "
                f"> {self.C_hydroxyl_warn:.2f} mol/L — restrict temperature rise rate "
                f"to ≤ {self.max_dT_dt_on_accum:.1f} °C/min"
            )
            actions.max_dT_dt_c_per_min = self.max_dT_dt_on_accum

        return status, interlocks, alarms, actions

    # ------------------------------------------------------------------
    # Tier 2 — Soft Process Optimizations
    # ------------------------------------------------------------------
    def evaluate_tier2(self, snap: ProcessSnapshot) -> list[str]:
        """Advisory recommendations (never override hard interlocks)."""
        recs: list[str] = []

        mt_ratio = snap.mass_transfer_ratio()
        if mt_ratio > self.mt_bottleneck:
            new_rpm = snap.agitator_rpm + self.agitator_bump_rpm
            recs.append(
                f"{REC_AGITATOR}: Mass transfer bottleneck "
                f"(3r/(k_L a · C_H2*)={mt_ratio:.2f} > {self.mt_bottleneck:.2f}). "
                f"Increase Agitator Speed by +{self.agitator_bump_rpm:.0f} RPM "
                f"to enhance k_L a (suggested setpoint {new_rpm:.0f} RPM)."
            )

        # Stage Transition Recommendation (Stage 3 → Stage 4 digest)
        if snap.stage == 3 and snap.nitro_fraction_remaining() < self.nitro_stage3_frac:
            pct = 100.0 * snap.nitro_fraction_remaining()
            recs.append(
                f"{REC_STAGE4}: C_nitro={snap.C_nitro:.4f} kmol/m³ "
                f"({pct:.2f}% of charge) < 0.5% during Stage 3. "
                "Transition batch to Stage 4 (Isothermal Digest)."
            )

        return recs

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        state: ProcessSnapshot | Mapping[str, Any],
    ) -> DecisionOutput:
        """Run Tier-1 then Tier-2; return validated ``DecisionOutput``."""
        snap = state if isinstance(state, ProcessSnapshot) else ProcessSnapshot.from_mapping(state)

        status, interlocks, alarms, actions = self.evaluate_tier1(snap)
        soft = self.evaluate_tier2(snap)

        blocked = False
        if self.suppress_soft_on_critical and status == SafetyStatus.CRITICAL:
            blocked = True
            soft = []

        out = DecisionOutput(
            safety_status=status.value,
            active_interlocks=interlocks,
            optimization_recommendations=soft,
            alarms=alarms,
            actuator_overrides=actions,
            tier1_blocked_optimizations=blocked,
            evaluated_at_s=snap.t_s,
            metadata={
                "T_reactor_c": snap.T_reactor_c,
                "P_headspace_bar": snap.P_headspace_bar,
                "C_hydroxyl_mol_L": snap.C_hydroxyl,
                "mass_transfer_ratio": snap.mass_transfer_ratio(),
                "nitro_frac_remaining": snap.nitro_fraction_remaining(),
                "stage": snap.stage,
                "limits": {
                    "T_critical_c": self.T_critical_c,
                    "P_high_bar": self.P_high_bar,
                    "C_hydroxyl_warn_mol_L": self.C_hydroxyl_warn,
                    "mt_bottleneck": self.mt_bottleneck,
                },
            },
        )
        self.history.append(out)
        return out

    def is_trip_active(self, decision: DecisionOutput | None = None) -> bool:
        d = decision or (self.history[-1] if self.history else None)
        if d is None:
            return False
        return d.safety_status == SafetyStatus.CRITICAL.value


def _demo() -> None:
    engine = TwoTierDecisionEngine()
    print("Two-Tier Decision Engine — PSM / Automation Demo")
    print("=" * 60)

    cases = [
        (
            "Nominal hydrogenation",
            ProcessSnapshot(
                T_reactor_c=88.0,
                P_headspace_bar=10.0,
                C_hydroxyl=0.02,
                C_nitro=1.2,
                C_nitro_0=2.8,
                r_rxn=0.0008,
                kla=0.085,
                C_H2_star=0.04,
                agitator_rpm=180.0,
                stage=3,
                t_s=1800.0,
            ),
        ),
        (
            "Mass-transfer limited",
            ProcessSnapshot(
                T_reactor_c=90.0,
                P_headspace_bar=10.0,
                C_hydroxyl=0.03,
                C_nitro=0.9,
                r_rxn=0.012,
                kla=0.085,
                C_H2_star=0.04,
                agitator_rpm=180.0,
                stage=3,
                t_s=2400.0,
            ),
        ),
        (
            "Hydroxylamine accumulation",
            ProcessSnapshot(
                T_reactor_c=92.0,
                P_headspace_bar=10.0,
                C_hydroxyl=0.095,
                C_nitro=0.5,
                stage=3,
                t_s=3000.0,
            ),
        ),
        (
            "Stage-3 complete → digest",
            ProcessSnapshot(
                T_reactor_c=85.0,
                P_headspace_bar=10.0,
                C_hydroxyl=0.01,
                C_nitro=0.01,
                C_nitro_0=2.8,
                stage=3,
                t_s=7000.0,
            ),
        ),
        (
            "Thermal critical trip",
            ProcessSnapshot(
                T_reactor_c=108.5,
                P_headspace_bar=11.0,
                C_hydroxyl=0.04,
                C_nitro=0.8,
                r_rxn=0.02,
                kla=0.085,
                C_H2_star=0.04,
                stage=3,
                t_s=1500.0,
            ),
        ),
        (
            "Overpressure trip",
            ProcessSnapshot(
                T_reactor_c=95.0,
                P_headspace_bar=15.4,
                C_hydroxyl=0.02,
                C_nitro=1.0,
                stage=3,
                t_s=1600.0,
            ),
        ),
    ]

    for title, snap in cases:
        d = engine.evaluate(snap)
        print(f"\n[{title}]")
        print(f"  safety_status:  {d.safety_status}")
        print(f"  interlocks:     {d.active_interlocks or '—'}")
        print(f"  alarms:         {d.alarms or '—'}")
        print(f"  overrides:      {d.actuator_overrides.model_dump()}")
        print(f"  recommendations:{d.optimization_recommendations or '—'}")
        if d.tier1_blocked_optimizations:
            print("  (Tier-2 advice suppressed under CRITICAL)")


if __name__ == "__main__":
    _demo()
