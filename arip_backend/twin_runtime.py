"""Digital-twin orchestrator — ODE → Residual ML → EKF → Decision → AI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from arip_backend.ai_explainer import LocalAIExplainer
from arip_backend.decision_engine import ProcessSnapshot, TwoTierDecisionEngine
from arip_backend.ekf_estimator import (
    EKF_STATE_NAMES,
    EKFStateEstimator,
    I_H2,
    I_HYDROXYL,
    I_NITRO,
    I_P,
    I_T,
    I_XYLIDINE,
    ProcessModelParams,
)
from arip_backend.residual_ml import ResidualCorrectionEngine
from arip_backend.schemas.residual import ScientificFeatures
from arip_backend.schemas.telemetry import MetricValue, TwinStepRequest, UnifiedTwinFrame

STAGE_NAMES = {
    1: "HEAT_UP",
    2: "PRESSURIZE",
    3: "HYDROGENATION",
    4: "ISOTHERMAL_DIGEST",
    5: "DUMP_COOL",
}


def _metric(
    tag: str,
    value: float,
    unit: str,
    display_name: str,
    category: str,
    precision: int = 2,
    quality: str = "GOOD",
) -> MetricValue:
    return MetricValue(
        tag=tag,
        value=float(value),
        unit=unit,
        display_name=display_name,
        precision=precision,
        quality=quality,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
    )


@dataclass
class TwinOrchestrator:
    """Stateful live twin used by REST and WebSocket servers."""

    batch_id: str = "BATCH-2026-0801-01"
    t_s: float = 0.0
    stage_index: int = 1
    agitator_rpm: float = 180.0
    C_nitro_0: float = 2.8
    T_jacket_c: float = 90.0
    T_sp_c: float = 85.0
    P_sp_bar: float = 10.0
    params: ProcessModelParams = field(default_factory=ProcessModelParams)
    ekf: EKFStateEstimator = field(init=False)
    residual: ResidualCorrectionEngine = field(init=False)
    decision: TwoTierDecisionEngine = field(init=False)
    explainer: LocalAIExplainer = field(init=False)
    x_truth: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        self.params = ProcessModelParams(V=3.2, c_cat=12.5 / 3.2)
        self.ekf = EKFStateEstimator.from_defaults(
            x0=[self.C_nitro_0, 0.0, 0.0, 0.0, 75.0, 1.0],
            params=self.params,
        )
        self.x_truth = self.ekf.state_vector()
        self.residual = ResidualCorrectionEngine()
        self.decision = TwoTierDecisionEngine()
        self.explainer = LocalAIExplainer(timeout_s=1.5)

    def reset(self) -> None:
        self.__post_init__()
        self.t_s = 0.0
        self.stage_index = 1
        self.agitator_rpm = 180.0
        self.T_jacket_c = 90.0
        self.T_sp_c = 85.0
        self.P_sp_bar = 10.0

    def _controls(self, req: TwinStepRequest) -> dict[str, float]:
        T = float(self.x_truth[I_T])
        if req.T_jacket_c is not None:
            self.T_jacket_c = float(req.T_jacket_c)
        elif T < 80.0:
            self.T_jacket_c = 95.0
        else:
            # Track toward setpoint with mild lag for Chart B
            self.T_jacket_c = 0.85 * self.T_jacket_c + 0.15 * self.T_sp_c

        if req.P_sp_bar is not None:
            self.P_sp_bar = float(req.P_sp_bar)

        h2_default = 1.0 if T >= 80.0 and self.stage_index >= 2 else 0.0
        return {
            "T_jacket": float(self.T_jacket_c),
            "P_sp": float(self.P_sp_bar),
            "h2_valve": float(req.h2_valve if req.h2_valve is not None else h2_default),
        }

    def _infer_stage(self, x: np.ndarray, requested: Optional[int]) -> int:
        if requested is not None:
            return int(requested)
        T = float(x[I_T])
        frac = float(x[I_NITRO]) / self.C_nitro_0 if self.C_nitro_0 > 0 else 0.0
        if self.stage_index >= 5:
            return 5
        if T < 80.0 and frac > 0.95:
            return 1
        if T >= 80.0 and float(x[I_P]) < 8.0 and frac > 0.9:
            return 2
        if frac < 0.005:
            return 4 if self.stage_index >= 3 else 3
        return 3

    def _rates(self, x: np.ndarray, u: dict[str, float]) -> dict[str, float]:
        """Evaluate instantaneous kinetics / heat for residual features."""
        # Reuse continuous dynamics internals via a light recompute
        dx = self.ekf.continuous_dynamics(0.0, x, u)
        C_n = max(float(x[I_NITRO]), 0.0)
        C_h2 = max(float(x[I_H2]), 0.0)
        C_oh = max(float(x[I_HYDROXYL]), 0.0)
        T = float(np.clip(x[I_T], 5.0, 200.0))
        P = float(x[I_P])
        p = self.params
        T_k = T + 273.15
        k1 = p.k0_1 * np.exp(-p.Ea_1 / (8.314462618 * T_k)) * p.c_cat
        k3 = p.k0_3 * np.exp(-p.Ea_3 / (8.314462618 * T_k)) * p.c_cat
        r1 = k1 * C_n * C_h2
        r3 = k3 * C_oh * C_h2
        r = r1 + r3
        Q_rxn = (r1 * (-p.dH_1) + r3 * (-p.dH_3)) * p.V
        valve = float(u.get("h2_valve", 1.0))
        C_star = valve * P / p.henry
        mt = ResidualCorrectionEngine.mass_transfer_ratio(r, p.kla, C_star)
        return {
            "r_rxn": float(r),
            "Q_rxn_W": float(Q_rxn),
            "C_H2_star": float(C_star),
            "mass_transfer_ratio": float(mt),
            "dT_dt": float(dx[I_T]),
        }

    async def step(self, req: TwinStepRequest) -> UnifiedTwinFrame:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        if req.agitator_rpm is not None:
            self.agitator_rpm = float(req.agitator_rpm)
        if req.t_s is not None:
            # Allow client to set absolute clock without rewinding physics
            self.t_s = float(req.t_s)

        u = self._controls(req)

        # --- 1) SciPy ODE physics step (truth plant) ---
        t_phys = time.perf_counter()
        self.x_truth = self.ekf.discrete_process(self.x_truth, req.dt_s, u)
        self.t_s += float(req.dt_s)
        self.stage_index = self._infer_stage(self.x_truth, req.stage)
        rates = self._rates(self.x_truth, u)
        timings["ode_ms"] = (time.perf_counter() - t_phys) * 1000.0

        conversion = 1.0 - float(self.x_truth[I_NITRO]) / self.C_nitro_0
        yield_est = float(self.x_truth[I_XYLIDINE]) / self.C_nitro_0

        physics_state = {
            "C_nitro": float(self.x_truth[I_NITRO]),
            "C_H2": float(self.x_truth[I_H2]),
            "C_hydroxyl": float(self.x_truth[I_HYDROXYL]),
            "C_xylidine": float(self.x_truth[I_XYLIDINE]),
            "T_reactor_c": float(self.x_truth[I_T]),
            "P_headspace_bar": float(self.x_truth[I_P]),
            "yield": float(np.clip(yield_est, 0.0, 1.5)),
            "conversion": float(np.clip(conversion, 0.0, 1.5)),
            "r_rxn": rates["r_rxn"],
            "Q_rxn_W": rates["Q_rxn_W"],
        }

        # --- 2) Residual ML correction ---
        t_ml = time.perf_counter()
        features = ScientificFeatures(
            conversion=float(np.clip(conversion, 0.0, 1.0)),
            Q_rxn_W=rates["Q_rxn_W"],
            mass_transfer_ratio=rates["mass_transfer_ratio"],
            T_reactor_c=float(self.x_truth[I_T]),
            agitator_rpm=self.agitator_rpm,
        )
        refined = self.residual.apply_correction(physics_state, features)
        timings["residual_ms"] = (time.perf_counter() - t_ml) * 1000.0

        # --- 3) EKF fuse (sensors) ---
        t_ekf = time.perf_counter()
        z_clean = self.ekf.measurement_model(self.x_truth, u)
        if req.sensors:
            z = np.array(
                [
                    float(req.sensors.get("T_reactor", req.sensors.get("T_reactor_c", z_clean[0]))),
                    float(req.sensors.get("P_headspace", req.sensors.get("P_headspace_bar", z_clean[1]))),
                    float(req.sensors.get("MFC_H2_rate", z_clean[2])),
                ],
                dtype=float,
            )
        elif req.sensor_noise:
            noise = self._rng.normal(0.0, 1.0, size=3) * np.sqrt(np.diag(self.ekf.R))
            z = z_clean + noise
        else:
            z = z_clean

        # Apply decision trip overrides onto u for next EKF predict path
        fused = self.ekf.step(req.dt_s, z, u)
        timings["ekf_ms"] = (time.perf_counter() - t_ekf) * 1000.0

        fused_map = fused.fused_state
        # Prefer refined concentrations / yield for operator KPIs
        T_op = float(refined.refined_state.get("T_reactor_c", fused_map["T_reactor"]))
        C_nitro_op = float(refined.refined_state.get("C_nitro", fused_map["C_nitro"]))
        yield_op = float(refined.refined_state.get("yield", yield_est))

        # --- 4) Decision engine ---
        t_dec = time.perf_counter()
        snap = ProcessSnapshot(
            T_reactor_c=T_op,
            P_headspace_bar=float(fused_map["P_headspace"]),
            C_hydroxyl=float(fused_map["C_hydroxyl"]),
            C_nitro=C_nitro_op,
            C_nitro_0=self.C_nitro_0,
            r_rxn=rates["r_rxn"],
            kla=self.params.kla,
            C_H2_star=rates["C_H2_star"],
            agitator_rpm=self.agitator_rpm,
            stage=min(max(self.stage_index, 1), 5),
            t_s=self.t_s,
            dT_dt_c_per_min=rates["dT_dt"] * 60.0,
        )
        decision = self.decision.evaluate(snap)
        timings["decision_ms"] = (time.perf_counter() - t_dec) * 1000.0

        # Enforce hard overrides on plant controls for subsequent steps
        if decision.actuator_overrides.h2_feed_cutoff or decision.actuator_overrides.h2_mfc_valve_closed:
            # Reflect trip in truth valve immediately
            self.x_truth[I_P] = min(float(self.x_truth[I_P]), 15.0)
        if decision.actuator_overrides.cooling_jacket_flow_pct == 100.0:
            self.T_jacket_c = min(self.T_jacket_c, 25.0)

        # --- 5) AI explainer ---
        ai_payload: Optional[dict[str, Any]] = None
        t_ai = time.perf_counter()
        if req.include_ai_advisory:
            advisory = await self.explainer.generate_operator_advisory(
                stage=STAGE_NAMES.get(self.stage_index, f"STAGE_{self.stage_index}"),
                fused_state={
                    "T_reactor_c": T_op,
                    "P_headspace_bar": float(fused_map["P_headspace"]),
                    "C_nitro": C_nitro_op,
                    "C_xylidine": float(fused_map["C_xylidine"]),
                },
                ekf_confidence=float(fused.confidence_score),
                active_interlocks=decision.active_interlocks,
                optimization_advice=decision.optimization_recommendations,
                safety_status=decision.safety_status,
            )
            ai_payload = advisory.model_dump()
        timings["ai_ms"] = (time.perf_counter() - t_ai) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

        # SCADA badge
        if decision.safety_status == "CRITICAL":
            badge = "TRIP"
        elif decision.safety_status == "WARNING":
            badge = "WARN"
        elif self.stage_index >= 5:
            badge = "HOLD"
        else:
            badge = "RUN"

        # Industrial metric table
        U = self.params.UA / max(self.params.V ** (2.0 / 3.0), 1.0)  # rough display U
        metrics = [
            _metric("BATCH.TIME", self.t_s, "s", "Batch Elapsed Time", "status", 1),
            _metric("RX.T", T_op, "°C", "Reactor Temperature (T_reactor)", "thermal", 2),
            _metric("RX.TJ", float(self.T_jacket_c), "°C", "Jacket Temperature (T_jacket)", "thermal", 2),
            _metric("RX.P", float(fused_map["P_headspace"]), "bar", "Headspace Pressure (P_headspace)", "pressure", 2),
            _metric("RX.C_NITRO", C_nitro_op, "mol/L", "Nitroxylene Concentration", "composition", 4),
            _metric("RX.C_XYL", float(fused_map["C_xylidine"]), "mol/L", "Xylidine Concentration", "composition", 4),
            _metric("PHY.C_NITRO", float(physics_state["C_nitro"]), "mol/L", "Physics Nitroxylene Concentration", "composition", 4),
            _metric("PHY.C_XYL", float(physics_state["C_xylidine"]), "mol/L", "Physics Xylidine Concentration", "composition", 4),
            _metric("PHY.T", float(physics_state["T_reactor_c"]), "°C", "Physics Reactor Temperature", "thermal", 2),
            _metric("EKF.T", float(fused_map["T_reactor"]), "°C", "EKF Fused Reactor Temperature", "thermal", 2),
            _metric("EKF.C_NITRO", float(fused_map["C_nitro"]), "mol/L", "EKF Fused Nitroxylene Concentration", "composition", 4),
            _metric("EKF.C_XYL", float(fused_map["C_xylidine"]), "mol/L", "EKF Fused Xylidine Concentration", "composition", 4),
            _metric("RX.C_H2", float(fused_map["C_H2"]), "mol/L", "Dissolved Hydrogen Concentration", "composition", 5),
            _metric(
                "RX.C_OH",
                float(fused_map["C_hydroxyl"]),
                "mol/L",
                "Intermediate Hydroxylamine Concentration",
                "composition",
                4,
            ),
            _metric("RX.CONV", conversion * 100.0, "%", "Nitroxylene Conversion", "quality", 2),
            _metric("RX.YIELD", yield_op * 100.0, "%", "Xylidine Yield", "quality", 2),
            _metric("RX.Q_RXN", rates["Q_rxn_W"] / 1000.0, "kW", "Instantaneous Heat Release Rate (Q_rxn)", "energy", 2),
            _metric("RX.R", rates["r_rxn"], "mol/L·s", "Chemical Reaction Rate", "kinetics", 6),
            _metric(
                "RX.MT_RATIO",
                rates["mass_transfer_ratio"],
                "—",
                "Mass Transfer Bottleneck Ratio 3r/(k_L a · C_H2*)",
                "kinetics",
                3,
            ),
            _metric("RX.U", float(U), "W/m²K", "Overall Heat Transfer Coefficient (U)", "thermal", 1),
            _metric("AG.RPM", self.agitator_rpm, "RPM", "Agitator Rotational Speed", "mechanical", 0),
            _metric("H2.MFC", float(z[2]), "kg/min", "Hydrogen Mass-Flow Controller Rate", "mechanical", 3),
            _metric(
                "JACKET.FLOW",
                (
                    12.0
                    if decision.actuator_overrides.cooling_jacket_flow_pct == 100.0
                    else float(2.0 + 0.2 * max(T_op - self.T_sp_c, 0.0))
                ),
                "kg/min",
                "Jacket Inlet Flow Rate",
                "mechanical",
                2,
            ),
            _metric("EKF.CONF", float(fused.confidence_score), "%", "EKF State Estimation Confidence", "quality", 1),
            _metric(
                "ML.dT",
                float(refined.residual.delta_T_exotherm_c),
                "°C",
                "ML Residual Temperature Correction (ΔT_exotherm)",
                "quality",
                3,
                quality="SUBSTITUTE" if refined.is_pure_physics else "GOOD",
            ),
        ]

        units_map = {m.tag: m.unit for m in metrics}
        trend_point = {m.tag: m.value for m in metrics}
        trend_point["t_s"] = self.t_s
        trend_point["t_min"] = self.t_s / 60.0

        overlays = {
            "physics": {
                "C_nitro": float(physics_state["C_nitro"]),
                "C_xylidine": float(physics_state["C_xylidine"]),
                "T_reactor_c": float(physics_state["T_reactor_c"]),
                "P_headspace_bar": float(physics_state["P_headspace_bar"]),
            },
            "ekf_fused": {
                "C_nitro": float(fused_map["C_nitro"]),
                "C_xylidine": float(fused_map["C_xylidine"]),
                "T_reactor_c": float(fused_map["T_reactor"]),
                "P_headspace_bar": float(fused_map["P_headspace"]),
            },
            "T_jacket_c": float(self.T_jacket_c),
            "H2_MFC_kg_min": float(z[2]),
        }
        return UnifiedTwinFrame(
            batch_id=self.batch_id,
            t_s=self.t_s,
            t_min=self.t_s / 60.0,
            stage=STAGE_NAMES.get(self.stage_index, f"STAGE_{self.stage_index}"),
            stage_index=self.stage_index,
            safety_status=decision.safety_status,  # type: ignore[arg-type]
            scada_badge=badge,  # type: ignore[arg-type]
            metrics=metrics,
            physics_state=physics_state,
            refined_state={k: float(v) for k, v in refined.refined_state.items()},
            ekf={
                "fused_state": fused.fused_state,
                "sensor_residuals": fused.sensor_residuals,
                "confidence_score": fused.confidence_score,
                "covariance_trace": float(np.trace(np.asarray(fused.covariance_matrix))),
            },
            decision=decision.model_dump(),
            residual={
                "is_pure_physics": refined.is_pure_physics,
                "model_name": refined.model_name,
                "lab_dataset_present": bool(getattr(self.residual, "lab_dataset_present", False)),
                "delta": refined.residual.model_dump(),
                "mode": "pure_physics" if refined.is_pure_physics else "hybrid_ml",
            },
            ai_advisory=ai_payload,
            trend_point=trend_point,
            units_map=units_map,
            pipeline_ms=timings,
            overlays=overlays,
        )

    async def jump_to(
        self,
        t_target_s: float,
        *,
        keyframe_every_s: float = 10.0,
        max_dt_s: float = 2.0,
    ) -> tuple[UnifiedTwinFrame, list[dict[str, float]]]:
        """Mode B: instant time scrub — integrate from t=0 to ``t_target_s``.

        Returns the final unified frame plus sparse keyframes for plot rebuild.
        """
        target = max(0.0, float(t_target_s))
        self.reset()
        keyframes: list[dict[str, float]] = []
        next_kf = 0.0

        # Capture IC keyframe
        ic_frame = await self.step(
            TwinStepRequest(dt_s=1e-6, include_ai_advisory=False, sensor_noise=False)
        )
        keyframes.append(dict(ic_frame.trend_point))

        while self.t_s + 1e-9 < target:
            dt = min(max_dt_s, target - self.t_s)
            frame = await self.step(
                TwinStepRequest(dt_s=dt, include_ai_advisory=False, sensor_noise=False)
            )
            if self.t_s >= next_kf - 1e-9:
                keyframes.append(dict(frame.trend_point))
                next_kf = self.t_s + keyframe_every_s

        final = await self.step(
            TwinStepRequest(dt_s=1e-6, include_ai_advisory=True, sensor_noise=False)
        )
        if not keyframes or abs(keyframes[-1].get("t_s", -1) - final.t_s) > 1e-6:
            keyframes.append(dict(final.trend_point))
        return final, keyframes
