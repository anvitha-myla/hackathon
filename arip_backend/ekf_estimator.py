"""Extended Kalman Filter (EKF) state estimator for the ARIP digital twin.

Fuses SciPy physics predictions with live, noisy plant sensors:

    x = [C_nitro, C_H2, C_hydroxyl, C_xylidine, T_reactor, P_headspace]^T
    z = [T_reactor, P_headspace, MFC_H2_rate]^T

Process model ``f`` is a reduced Haber hydrogenation ODE stepped over ``dt``;
measurement model ``h`` maps the state to the three sensor channels.
Numerical Jacobians ``F = ∂f/∂x`` and ``H = ∂h/∂x`` are used each cycle.

Run demo
--------
    python -m arip_backend.ekf_estimator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.integrate import solve_ivp

R_GAS = 8.314462618  # J/(mol·K)

# EKF state layout
I_NITRO, I_H2, I_HYDROXYL, I_XYLIDINE, I_T, I_P = range(6)
EKF_STATE_NAMES = [
    "C_nitro",
    "C_H2",
    "C_hydroxyl",
    "C_xylidine",
    "T_reactor",
    "P_headspace",
]
N_STATE = 6

# Measurement layout: z = [T, P, MFC H2 rate]
I_Z_T, I_Z_P, I_Z_MFC = range(3)
MEASUREMENT_NAMES = ["T_reactor", "P_headspace", "MFC_H2_rate"]
N_MEAS = 3


class FusedStateEstimate(BaseModel):
    """Posterior EKF estimate after a predict+update cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fused_state: dict[str, float] = Field(
        ...,
        description="Posterior state estimate x̂_k keyed by EKF_STATE_NAMES",
    )
    sensor_residuals: dict[str, float] = Field(
        ...,
        description="Innovation ν = z − h(x̂⁻) for each measurement channel",
    )
    covariance_matrix: list[list[float]] = Field(
        ...,
        description="Posterior error covariance P_k (6×6)",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="0–100% confidence from Tr(P_k)",
    )
    timestamp_s: float = Field(0.0, description="Filter time stamp [s]")
    kalman_gain: Optional[list[list[float]]] = Field(
        default=None,
        description="Optional Kalman gain K_k (6×3) for diagnostics",
    )

    @field_validator("covariance_matrix")
    @classmethod
    def _check_cov_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != N_STATE or any(len(row) != N_STATE for row in v):
            raise ValueError(f"covariance_matrix must be {N_STATE}×{N_STATE}")
        return v


@dataclass
class ProcessModelParams:
    """Reduced-order hydrogenation kinetics + thermal / pressure params."""

    k0_1: float = 85000.0  # nitro → hydroxyl (lumped)
    Ea_1: float = 48500.0  # J/mol
    k0_3: float = 21000.0  # hydroxyl → xylidine
    Ea_3: float = 52300.0  # J/mol
    dH_1: float = -330000.0  # J/mol (nitro → hydroxyl lumped)
    dH_3: float = -210000.0  # J/mol
    kla: float = 0.085
    henry: float = 250.0  # bar·m³/kmol (effective)
    rho: float = 870.0
    Cp: float = 2100.0
    V: float = 3.2
    c_cat: float = 3.90625  # kg/m³
    UA: float = 22000.0  # W/K
    T_jacket: float = 80.0  # °C (held for estimator plant model)
    P_sp: float = 10.0
    tau_p: float = 25.0
    # MFC: converts liquid-side H2 uptake [kmol/(m³·s)] → SLPM-equivalent rate
    mfc_scale_slpm: float = 500.0


@dataclass
class EKFStateEstimator:
    """Extended Kalman Filter fusing ODE physics with noisy sensors.

    Prediction
        x̂⁻ = f(x̂, u, dt)
        F  = ∂f/∂x |_{x̂}   (central finite differences)
        P⁻ = F P Fᵀ + Q

    Update
        ν  = z − h(x̂⁻)
        H  = ∂h/∂x |_{x̂⁻}
        K  = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
        x̂  = x̂⁻ + K ν
        P  = (I − K H) P⁻
    """

    x: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    params: ProcessModelParams = field(default_factory=ProcessModelParams)
    t: float = 0.0
    jac_eps: float = 1.0e-6
    # Trace of P used to map Tr(P) → confidence (set from P0 at init)
    _tr_ref: float = field(default=1.0, repr=False)
    _last_K: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).reshape(N_STATE)
        self.P = np.asarray(self.P, dtype=float).reshape(N_STATE, N_STATE)
        self.Q = np.asarray(self.Q, dtype=float).reshape(N_STATE, N_STATE)
        self.R = np.asarray(self.R, dtype=float).reshape(N_MEAS, N_MEAS)
        self._tr_ref = max(float(np.trace(self.P)), 1.0e-12)
        self.x = self._project_state(self.x)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_defaults(
        cls,
        x0: Sequence[float] | None = None,
        *,
        params: ProcessModelParams | None = None,
    ) -> "EKFStateEstimator":
        """Build an estimator with engineering-default Q, R, and P0."""
        p = params or ProcessModelParams()
        if x0 is None:
            x0 = np.array([2.8, 0.0, 0.0, 0.0, 75.0, 1.0], dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)

        # Process noise: concentrations / T / P random-walk intensity per step
        q_diag = np.array([1e-5, 1e-6, 1e-5, 1e-5, 0.05, 0.01])
        Q = np.diag(q_diag)

        # Measurement noise: T ±0.5 °C, P ±0.05 bar, MFC ±2 SLPM (1σ)
        R = np.diag([0.5**2, 0.05**2, 2.0**2])

        # Initial covariance — loose on unmeasured concentrations
        p_diag = np.array([0.25, 0.01, 0.25, 0.25, 1.0, 0.25])
        P0 = np.diag(p_diag)

        return cls(x=x0, P=P0, Q=Q, R=R, params=p)

    # ------------------------------------------------------------------
    # Continuous physics ḟ(x, u)  (SciPy-compatible RHS)
    # ------------------------------------------------------------------
    def continuous_dynamics(self, t: float, x: np.ndarray, u: dict[str, float] | None = None) -> np.ndarray:
        """Reduced Haber ODE right-hand side for the EKF state."""
        u = u or {}
        p = self.params
        C_n = max(float(x[I_NITRO]), 0.0)
        C_h2 = max(float(x[I_H2]), 0.0)
        C_oh = max(float(x[I_HYDROXYL]), 0.0)
        C_xyl = max(float(x[I_XYLIDINE]), 0.0)
        T = float(np.clip(x[I_T], 5.0, 200.0))
        P = float(np.clip(x[I_P], 0.5, 20.0))

        T_j = float(u.get("T_jacket", p.T_jacket))
        P_sp = float(u.get("P_sp", p.P_sp))
        valve = float(u.get("h2_valve", 1.0 if T >= 80.0 else 0.0))

        T_k = T + 273.15
        k1 = p.k0_1 * np.exp(-p.Ea_1 / (R_GAS * T_k)) * p.c_cat
        k3 = p.k0_3 * np.exp(-p.Ea_3 / (R_GAS * T_k)) * p.c_cat
        r1 = k1 * C_n * C_h2  # nitro → hydroxyl (lumped)
        r3 = k3 * C_oh * C_h2  # hydroxyl → xylidine

        C_star = valve * P / p.henry
        R_transfer = p.kla * (C_star - C_h2)

        # Stoichiometry: ~2 H2 for nitro→OH, ~1 H2 for OH→xylidine (lumped)
        dC_n = -r1
        dC_oh = r1 - r3
        dC_xyl = r3
        dC_h2 = R_transfer - 2.0 * r1 - 1.0 * r3

        # r [kmol/m³·s] · ΔH [J/mol] · V [m³] · 1000 [mol/kmol] → W
        Q_rxn = (r1 * (-p.dH_1) + r3 * (-p.dH_3)) * p.V * 1000.0
        Q_rem = p.UA * (T - T_j)
        dT = (Q_rxn - Q_rem) / (p.V * p.rho * p.Cp)

        dP = (valve * P_sp + (1.0 - valve) * 1.0 - P) / p.tau_p

        return np.array([dC_n, dC_h2, dC_oh, dC_xyl, dT, dP], dtype=float)

    def discrete_process(
        self,
        x: np.ndarray,
        dt: float,
        u: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Propagate state over ``dt`` with SciPy RK45 (physics prediction)."""
        x0 = self._project_state(np.asarray(x, dtype=float))
        if dt <= 0.0:
            return x0.copy()

        def rhs(t: float, y: np.ndarray) -> list[float]:
            return self.continuous_dynamics(t, y, u).tolist()

        sol = solve_ivp(
            rhs,
            (0.0, float(dt)),
            x0,
            method="RK45",
            rtol=1e-5,
            atol=1e-8,
            max_step=max(float(dt) / 5.0, 0.5),
        )
        if not sol.success:
            # Fallback: explicit Euler with clipped rates
            dx = self.continuous_dynamics(0.0, x0, u)
            return self._project_state(x0 + float(dt) * dx)
        return self._project_state(sol.y[:, -1])

    # ------------------------------------------------------------------
    # Measurement model h(x)
    # ------------------------------------------------------------------
    def measurement_model(self, x: np.ndarray, u: dict[str, float] | None = None) -> np.ndarray:
        """Predicted sensors: [T, P, MFC_H2_rate]."""
        u = u or {}
        p = self.params
        T = float(x[I_T])
        P = float(x[I_P])
        C_h2 = max(float(x[I_H2]), 0.0)
        valve = float(u.get("h2_valve", 1.0 if T >= 80.0 else 0.0))
        P_sp = float(u.get("P_sp", p.P_sp))
        C_star = valve * P / p.henry
        R_transfer = p.kla * (C_star - C_h2)
        # Positive uptake → MFC must supply H2; scale to SLPM-like units
        mfc = max(R_transfer, 0.0) * p.V * p.mfc_scale_slpm
        # Mild pressure-error feed-forward (valve chasing P_sp)
        mfc += valve * max(P_sp - P, 0.0) * 5.0
        return np.array([T, P, mfc], dtype=float)

    # ------------------------------------------------------------------
    # Numerical Jacobians
    # ------------------------------------------------------------------
    def process_jacobian(
        self,
        x: np.ndarray,
        dt: float,
        u: dict[str, float] | None = None,
    ) -> np.ndarray:
        """F_k = ∂f/∂x via central differences around the discrete process map."""
        F = np.zeros((N_STATE, N_STATE), dtype=float)
        x = np.asarray(x, dtype=float)
        eps = self.jac_eps
        # Scale eps per state magnitude for better conditioning
        scales = np.maximum(np.abs(x), 1.0)
        for j in range(N_STATE):
            dx = eps * scales[j]
            xp = x.copy()
            xm = x.copy()
            xp[j] += dx
            xm[j] -= dx
            fp = self.discrete_process(xp, dt, u)
            fm = self.discrete_process(xm, dt, u)
            F[:, j] = (fp - fm) / (2.0 * dx)
        return F

    def measurement_jacobian(
        self,
        x: np.ndarray,
        u: dict[str, float] | None = None,
    ) -> np.ndarray:
        """H_k = ∂h/∂x via central differences."""
        H = np.zeros((N_MEAS, N_STATE), dtype=float)
        x = np.asarray(x, dtype=float)
        scales = np.maximum(np.abs(x), 1.0)
        for j in range(N_STATE):
            dx = self.jac_eps * scales[j]
            xp = x.copy()
            xm = x.copy()
            xp[j] += dx
            xm[j] -= dx
            hp = self.measurement_model(xp, u)
            hm = self.measurement_model(xm, u)
            H[:, j] = (hp - hm) / (2.0 * dx)
        return H

    # ------------------------------------------------------------------
    # EKF steps
    # ------------------------------------------------------------------
    def predict(self, dt: float, u: dict[str, float] | None = None) -> np.ndarray:
        """Time-update: propagate x̂ and P through the ODE process model."""
        F = self.process_jacobian(self.x, dt, u)
        self.x = self.discrete_process(self.x, dt, u)
        self.P = F @ self.P @ F.T + self.Q
        # Enforce symmetry / PSD soft-fix
        self.P = 0.5 * (self.P + self.P.T)
        self.t += float(dt)
        return self.x.copy()

    def update(
        self,
        z: Sequence[float],
        u: dict[str, float] | None = None,
        *,
        return_gain: bool = False,
    ) -> FusedStateEstimate:
        """Measurement-update: fuse sensors z into the predicted state."""
        z_arr = np.asarray(z, dtype=float).reshape(N_MEAS)
        H = self.measurement_jacobian(self.x, u)
        z_pred = self.measurement_model(self.x, u)
        innovation = z_arr - z_pred

        S = H @ self.P @ H.T + self.R
        # Solve K = P Hᵀ S⁻¹ without forming S⁻¹ explicitly
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self._project_state(self.x + K @ innovation)
        I = np.eye(N_STATE)
        # Joseph form for numerical robustness
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._last_K = K

        return FusedStateEstimate(
            fused_state={name: float(self.x[i]) for i, name in enumerate(EKF_STATE_NAMES)},
            sensor_residuals={name: float(innovation[i]) for i, name in enumerate(MEASUREMENT_NAMES)},
            covariance_matrix=self.P.tolist(),
            confidence_score=self.confidence_score(),
            timestamp_s=float(self.t),
            kalman_gain=K.tolist() if return_gain else None,
        )

    def step(
        self,
        dt: float,
        z: Sequence[float],
        u: dict[str, float] | None = None,
    ) -> FusedStateEstimate:
        """Full EKF cycle: predict then update."""
        self.predict(dt, u)
        return self.update(z, u)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    def confidence_score(self) -> float:
        """Map Tr(P) → 0–100%. Lower posterior uncertainty ⇒ higher confidence.

        score = 100 · exp(−Tr(P) / Tr(P₀))
        """
        tr = float(np.trace(self.P))
        score = 100.0 * float(np.exp(-tr / self._tr_ref))
        return float(np.clip(score, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _project_state(x: np.ndarray) -> np.ndarray:
        xp = np.asarray(x, dtype=float).copy()
        xp[I_NITRO] = max(xp[I_NITRO], 0.0)
        xp[I_H2] = max(xp[I_H2], 0.0)
        xp[I_HYDROXYL] = max(xp[I_HYDROXYL], 0.0)
        xp[I_XYLIDINE] = max(xp[I_XYLIDINE], 0.0)
        xp[I_T] = float(np.clip(xp[I_T], 5.0, 200.0))
        xp[I_P] = float(np.clip(xp[I_P], 0.5, 20.0))
        return xp

    def state_vector(self) -> np.ndarray:
        return self.x.copy()

    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "x": {n: float(self.x[i]) for i, n in enumerate(EKF_STATE_NAMES)},
            "P_trace": float(np.trace(self.P)),
            "confidence_score": self.confidence_score(),
        }


def _demo() -> None:
    """Simulate a plant trajectory, corrupt sensors, and run the EKF."""
    rng = np.random.default_rng(42)
    params = ProcessModelParams()
    ekf = EKFStateEstimator.from_defaults(
        x0=[2.8, 0.0, 0.0, 0.0, 75.0, 1.0],
        params=params,
    )

    # Truth starts at the same IC
    x_true = ekf.state_vector()
    dt = 5.0
    n_steps = 120  # 10 min

    print("EKF State Estimator — Sensor Fusion Demo")
    print(f"State: {EKF_STATE_NAMES}")
    print(f"Sensors: {MEASUREMENT_NAMES}")
    print(f"{'t':>6} {'T_true':>7} {'T_meas':>7} {'T_ekf':>7} {'P_ekf':>6} {'conf%':>6} {'C_n':>6}")

    last: FusedStateEstimate | None = None
    for k in range(n_steps):
        u = {
            "T_jacket": 85.0 if x_true[I_T] >= 80.0 else 95.0,
            "P_sp": 10.0,
            "h2_valve": 1.0 if x_true[I_T] >= 80.0 else 0.0,
        }
        # Truth propagates with same physics
        x_true = ekf.discrete_process(x_true, dt, u)

        z_clean = ekf.measurement_model(x_true, u)
        noise = rng.normal(0.0, 1.0, size=N_MEAS) * np.sqrt(np.diag(ekf.R))
        z_noisy = z_clean + noise

        last = ekf.step(dt, z_noisy, u)

        if k % 12 == 0 or k == n_steps - 1:
            print(
                f"{ekf.t:6.0f} {x_true[I_T]:7.2f} {z_noisy[I_Z_T]:7.2f} "
                f"{last.fused_state['T_reactor']:7.2f} {last.fused_state['P_headspace']:6.2f} "
                f"{last.confidence_score:6.1f} {last.fused_state['C_nitro']:6.3f}"
            )

    assert last is not None
    print("\nFinal FusedStateEstimate:")
    print(f"  fused_state:       { {k: round(v, 4) for k, v in last.fused_state.items()} }")
    print(f"  sensor_residuals:  { {k: round(v, 4) for k, v in last.sensor_residuals.items()} }")
    print(f"  Tr(P):             {np.trace(np.array(last.covariance_matrix)):.4f}")
    print(f"  confidence_score:  {last.confidence_score:.1f}%")


if __name__ == "__main__":
    _demo()
