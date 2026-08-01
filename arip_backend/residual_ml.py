"""Residual Correction Engine — hybrid physics + ML architecture.

    y_real ≈ y_physics + f_ML(x)

Loads a pre-trained ``RandomForestRegressor`` or ``GradientBoostingRegressor``
from ``arip_backend/models/`` via joblib. On cold-start (no model / no lab
history) returns zero discrepancy with ``is_pure_physics=True``.

Run demo
--------
    python -m arip_backend.residual_ml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import joblib
import numpy as np

from arip_backend.schemas.residual import (
    RefinedPrediction,
    ResidualCorrection,
    ScientificFeatures,
)

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BACKEND_ROOT / "models"

# Fixed feature / target column order for sklearn estimators
FEATURE_NAMES = [
    "conversion",
    "Q_rxn_W",
    "mass_transfer_ratio",
    "T_reactor_c",
    "agitator_rpm",
]
TARGET_NAMES = [
    "delta_T_exotherm_c",
    "delta_C_nitro",
    "delta_yield",
]

# Physics keys that receive residual corrections
PHYSICS_T_KEYS = ("T_reactor", "T_reactor_c", "T")
PHYSICS_CNITRO_KEYS = ("C_nitro", "C_NX", "nitro")
PHYSICS_YIELD_KEYS = ("yield", "xylidine_yield", "amine_yield", "Y")


class ResidualCorrectionEngine:
    """Hybrid residual learner: δ_ML = f_ML(scientific features).

    Parameters
    ----------
    models_dir:
        Directory containing ``*.joblib`` / ``*.pkl`` regressors.
    model_filename:
        Optional explicit model file. If omitted, loads the newest matching
        artifact in ``models_dir``.
    """

    def __init__(
        self,
        models_dir: str | Path | None = None,
        model_filename: str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model: Any | None = None
        self.model_path: Path | None = None
        self.model_name: str | None = None
        self.is_pure_physics: bool = True
        self._load_error: str | None = None

        self._try_load_model(model_filename)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------
    def _candidate_paths(self, model_filename: str | None) -> list[Path]:
        if model_filename:
            p = Path(model_filename)
            if not p.is_absolute():
                p = self.models_dir / p
            return [p]

        patterns = ("*.joblib", "*.pkl", "*.pickle")
        found: list[Path] = []
        for pat in patterns:
            found.extend(self.models_dir.glob(pat))
        # Prefer residual_* names, then newest mtime
        found.sort(
            key=lambda p: (
                0 if p.name.startswith("residual") else 1,
                -p.stat().st_mtime,
            )
        )
        return found

    def _try_load_model(self, model_filename: str | None = None) -> None:
        for path in self._candidate_paths(model_filename):
            if not path.exists() or not path.is_file():
                continue
            try:
                obj = joblib.load(path)
            except Exception as exc:  # noqa: BLE001 — cold-start on any load failure
                self._load_error = f"Failed to load {path.name}: {exc}"
                continue

            estimator = self._unwrap_estimator(obj)
            if estimator is None:
                self._load_error = (
                    f"{path.name} is not a supported regressor "
                    "(expected RandomForest / GradientBoosting or a dict wrapper)"
                )
                continue

            self.model = estimator
            self.model_path = path
            self.model_name = path.stem
            self.is_pure_physics = False
            self._load_error = None
            return

        # Cold-start fallback
        self.model = None
        self.model_path = None
        self.model_name = None
        self.is_pure_physics = True
        if self._load_error is None:
            self._load_error = (
                f"No trained residual model found in {self.models_dir} "
                "(cold-start: δ_ML = 0, is_pure_physics=True)"
            )

    @staticmethod
    def _unwrap_estimator(obj: Any) -> Any | None:
        """Accept raw sklearn regressor or ``{'model': est, ...}`` joblib dict."""
        if isinstance(obj, dict):
            est = obj.get("model") or obj.get("estimator") or obj.get("regressor")
        else:
            est = obj

        if est is None:
            return None

        # Duck-type: must implement predict
        if not hasattr(est, "predict"):
            return None

        name = type(est).__name__
        supported = (
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "MultiOutputRegressor",
            "Pipeline",
        )
        if name not in supported and not hasattr(est, "estimators_"):
            # Still allow any regressor with predict — hybrid engine is model-agnostic
            # but prefer the named ensemble families from the requirements.
            pass
        return est

    def reload(self, model_filename: str | None = None) -> bool:
        """Re-scan ``models_dir`` and load a model if available."""
        self._try_load_model(model_filename)
        return not self.is_pure_physics

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------
    @staticmethod
    def mass_transfer_ratio(
        r_rxn: float,
        kla: float,
        C_H2_star: float,
        *,
        stoich_h2: float = 3.0,
    ) -> float:
        """Compute 3·r / (k_L a · C_H2*). Safe for C*→0."""
        denom = max(float(kla) * max(float(C_H2_star), 1e-12), 1e-12)
        return float(stoich_h2 * float(r_rxn) / denom)

    @staticmethod
    def features_from_mapping(data: Mapping[str, float]) -> ScientificFeatures:
        return ScientificFeatures(
            conversion=float(data["conversion"]),
            Q_rxn_W=float(data["Q_rxn_W"]),
            mass_transfer_ratio=float(data["mass_transfer_ratio"]),
            T_reactor_c=float(data["T_reactor_c"]),
            agitator_rpm=float(data["agitator_rpm"]),
        )

    def _feature_row(
        self,
        features: Union[ScientificFeatures, Mapping[str, float], Sequence[float]],
    ) -> np.ndarray:
        if isinstance(features, ScientificFeatures):
            row = [
                features.conversion,
                features.Q_rxn_W,
                features.mass_transfer_ratio,
                features.T_reactor_c,
                features.agitator_rpm,
            ]
        elif isinstance(features, Mapping):
            row = [float(features[name]) for name in FEATURE_NAMES]
        else:
            seq = list(features)
            if len(seq) != len(FEATURE_NAMES):
                raise ValueError(
                    f"Expected {len(FEATURE_NAMES)} features {FEATURE_NAMES}, got {len(seq)}"
                )
            row = [float(v) for v in seq]
        return np.asarray(row, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_residual(
        self,
        features: Union[ScientificFeatures, Mapping[str, float], Sequence[float]],
    ) -> ResidualCorrection:
        """Predict δ_ML = [ΔT_exotherm, ΔC_nitro, ΔYield].

        Cold-start (no model): returns the zero vector.
        """
        if self.is_pure_physics or self.model is None:
            return ResidualCorrection(
                delta_T_exotherm_c=0.0,
                delta_C_nitro=0.0,
                delta_yield=0.0,
            )

        X = self._feature_row(features)
        y_hat = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        if y_hat.size < 3:
            # Single-target model — pad remaining residuals with 0
            padded = np.zeros(3, dtype=float)
            padded[: y_hat.size] = y_hat
            y_hat = padded
        return ResidualCorrection(
            delta_T_exotherm_c=float(y_hat[0]),
            delta_C_nitro=float(y_hat[1]),
            delta_yield=float(y_hat[2]),
        )

    def apply_correction(
        self,
        physics_baseline: Mapping[str, float],
        features: Union[ScientificFeatures, Mapping[str, float], Sequence[float]],
    ) -> RefinedPrediction:
        """Return y_refined = y_physics + δ_ML as ``RefinedPrediction``."""
        residual = self.predict_residual(features)
        refined = dict(physics_baseline)

        # Apply ΔT
        for key in PHYSICS_T_KEYS:
            if key in refined:
                refined[key] = float(refined[key]) + residual.delta_T_exotherm_c
                break
        else:
            refined["T_reactor_c"] = (
                float(physics_baseline.get("T_reactor_c", physics_baseline.get("T_reactor", 0.0)))
                + residual.delta_T_exotherm_c
            )

        # Apply ΔC_nitro
        for key in PHYSICS_CNITRO_KEYS:
            if key in refined:
                refined[key] = float(refined[key]) + residual.delta_C_nitro
                break
        else:
            refined["C_nitro"] = float(physics_baseline.get("C_nitro", 0.0)) + residual.delta_C_nitro

        # Apply ΔYield
        for key in PHYSICS_YIELD_KEYS:
            if key in refined:
                refined[key] = float(refined[key]) + residual.delta_yield
                break
        else:
            base_yield = float(
                physics_baseline.get(
                    "yield",
                    physics_baseline.get("nitro_conversion", physics_baseline.get("conversion", 0.0)),
                )
            )
            refined["yield"] = base_yield + residual.delta_yield

        # Clamp non-physical negatives on concentrations / yield band
        if "C_nitro" in refined:
            refined["C_nitro"] = max(float(refined["C_nitro"]), 0.0)
        if "yield" in refined:
            refined["yield"] = float(np.clip(refined["yield"], 0.0, 1.5))

        feat_dict: dict[str, float] | None
        if isinstance(features, ScientificFeatures):
            feat_dict = features.model_dump()
        elif isinstance(features, Mapping):
            feat_dict = {k: float(features[k]) for k in FEATURE_NAMES if k in features}
        else:
            feat_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, features)}

        return RefinedPrediction(
            physics_baseline={k: float(v) for k, v in physics_baseline.items()},
            residual=residual,
            refined_state={k: float(v) for k, v in refined.items()},
            is_pure_physics=self.is_pure_physics,
            model_name=self.model_name,
            features_used=feat_dict,
            metadata={
                "architecture": "y_real = y_physics + f_ML(x)",
                "load_message": self._load_error,
                "models_dir": str(self.models_dir),
            },
        )


def train_and_save_demo_model(
    models_dir: Path | None = None,
    *,
    kind: str = "gradient_boosting",
    n_samples: int = 400,
    seed: int = 7,
) -> Path:
    """Fit a small multi-output residual model on synthetic lab-like data.

    Intended for demos / CI when no real historical dataset is present.
    Saves to ``models/residual_correction_gb.joblib`` (or ``_rf``).
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Synthetic features spanning operating window
    conversion = rng.uniform(0.0, 1.0, n_samples)
    Q_rxn = rng.uniform(0.0, 2.5e6, n_samples)
    mt_ratio = rng.uniform(0.0, 5.0, n_samples)
    T = rng.uniform(70.0, 120.0, n_samples)
    rpm = rng.uniform(60.0, 250.0, n_samples)
    X = np.column_stack([conversion, Q_rxn, mt_ratio, T, rpm])

    # Synthetic discrepancy correlating with heat / MT limitation
    dT = 0.015 * (Q_rxn / 1e5) + 0.4 * mt_ratio + rng.normal(0.0, 0.15, n_samples)
    dC = -0.01 * conversion * mt_ratio + rng.normal(0.0, 0.005, n_samples)
    dY = 0.02 * conversion - 0.005 * (T - 90.0) / 10.0 + rng.normal(0.0, 0.01, n_samples)
    Y = np.column_stack([dT, dC, dY])

    if kind == "random_forest":
        base = RandomForestRegressor(
            n_estimators=80,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
        )
        stem = "residual_correction_rf"
    else:
        base = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            random_state=seed,
        )
        stem = "residual_correction_gb"

    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    out = models_dir / f"{stem}.joblib"
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "kind": kind,
        "n_samples": n_samples,
        "architecture": "y_real = y_physics + f_ML(x)",
    }
    joblib.dump(payload, out)

    meta = models_dir / f"{stem}_meta.json"
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_file": out.name,
                "feature_names": FEATURE_NAMES,
                "target_names": TARGET_NAMES,
                "kind": kind,
                "n_samples": n_samples,
            },
            fh,
            indent=2,
        )
    return out


def _demo() -> None:
    print("Residual Correction Engine — Hybrid Physics + ML")
    print("=" * 56)

    # 1) Cold-start (no model)
    cold = ResidualCorrectionEngine(models_dir=DEFAULT_MODELS_DIR / "_empty_cold_start")
    feats = ScientificFeatures(
        conversion=0.85,
        Q_rxn_W=1.2e6,
        mass_transfer_ratio=1.8,
        T_reactor_c=92.0,
        agitator_rpm=180.0,
    )
    physics = {
        "T_reactor_c": 92.0,
        "C_nitro": 0.42,
        "yield": 0.83,
        "P_headspace_bar": 10.0,
    }
    r0 = cold.apply_correction(physics, feats)
    print("\n[Cold-start]")
    print(f"  is_pure_physics = {r0.is_pure_physics}")
    print(f"  residual        = {r0.residual.model_dump()}")
    print(f"  refined_state   = {r0.refined_state}")

    # 2) Train demo model + apply correction
    model_path = train_and_save_demo_model(DEFAULT_MODELS_DIR, kind="gradient_boosting")
    print(f"\n[Trained demo model] {model_path}")

    warm = ResidualCorrectionEngine(models_dir=DEFAULT_MODELS_DIR)
    r1 = warm.apply_correction(physics, feats)
    print("\n[With ML residual model]")
    print(f"  model_name      = {r1.model_name}")
    print(f"  is_pure_physics = {r1.is_pure_physics}")
    print(f"  residual δ_ML   = {r1.residual.model_dump()}")
    print(f"  physics         = {r1.physics_baseline}")
    print(f"  refined y       = {r1.refined_state}")
    print(f"\n  architecture: y_real = y_physics + f_ML(x)")


if __name__ == "__main__":
    _demo()
