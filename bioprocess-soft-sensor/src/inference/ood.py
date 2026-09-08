"""Mahalanobis OOD layer and beta_trust (Prompt 9).

Spec: docs/08_OOD_SPEC.md, docs/00_MASTER_SPEC.md §14.

Fit ``mean_train`` and ``covariance_train`` on **training features only**.
Never use test batches (or test-split statistics) to fit OOD.

Distance
--------
D_M = sqrt( (phi - mean)^T Sigma^{-1} (phi - mean) )

Trust
-----
beta_trust = exp(-kappa * max(0, D_M - D_threshold))
clipped to [0, 1].

Hybrid (applied per sequential timestamp, no future required)
------------------------------------------------------------
X_hybrid = X_mechanistic + beta_trust * delta_X_pred
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.covariance import LedoitWolf

from src.config import load_yaml

ALLOWED_FIT_SPLITS = frozenset({"train"})
FORBIDDEN_FIT_SPLITS = frozenset({"test", "validation", "val", "holdout"})

OOD_IN_DISTRIBUTION = "in_distribution"
OOD_MODERATE = "moderate_ood"
OOD_SEVERE = "severe_ood"
OOD_INVALID = "invalid"

FALLBACK_NONE = "none"
FALLBACK_ATTENUATED = "attenuated"
FALLBACK_PHYSICS = "physics"


def load_ood_config() -> dict[str, Any]:
    """Load OOD parameters from ``configs/ood.yaml`` (merged defaults)."""
    raw = load_yaml("ood.yaml")
    return {
        "method": str(raw.get("method", "mahalanobis")),
        "kappa": raw.get("kappa"),
        "d_threshold": raw.get("d_threshold"),
        "threshold_quantile": float(raw.get("threshold_quantile", 0.95)),
        "ridge": float(raw.get("ridge", 1.0e-6)),
        "relative_ridge": float(raw.get("relative_ridge", 1.0e-4)),
        "shrinkage": str(raw.get("shrinkage", "ledoit_wolf")),
        "beta_trust_formula": str(
            raw.get("beta_trust_formula", "exp(-kappa * max(0, D_M - D_threshold))")
        ),
        "fallback": str(raw.get("fallback", "physics")),
        "severe_beta": float(raw.get("severe_beta", 1.0e-3)),
        "default_kappa": float(raw.get("default_kappa", 1.0)),
    }


def beta_trust(
    d_m: ArrayLike,
    d_threshold: float,
    kappa: float,
) -> float | np.ndarray:
    """beta_trust = exp(-kappa * max(0, D_M - D_threshold)), clipped to [0, 1].

    Positional signature matches stress/eval callers: ``(d_m, d_threshold, kappa)``.
    """
    if not np.isfinite(kappa) or not np.isfinite(d_threshold):
        raise ValueError("kappa and d_threshold must be finite")
    if kappa < 0.0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    d_m_arr = np.asarray(d_m, dtype=np.float64)
    excess = np.maximum(0.0, d_m_arr - float(d_threshold))
    out = np.exp(-float(kappa) * excess)
    out = np.clip(out, 0.0, 1.0)
    out = np.where(np.isfinite(d_m_arr), out, 0.0)
    if np.isscalar(d_m) or d_m_arr.shape == ():
        val = float(np.asarray(out).reshape(-1)[0])
        return 0.0 if not np.isfinite(val) else val
    return out


def beta_trust_from_distance(
    d_m: float,
    *,
    kappa: float,
    d_threshold: float,
) -> float:
    """Scalar wrapper around ``beta_trust`` (keyword kappa / threshold)."""
    return float(beta_trust(d_m, d_threshold, kappa))


def apply_hybrid(
    x_mechanistic: float,
    delta_x_pred: float,
    beta_trust: float,
) -> HybridCombine:
    """X_hybrid = X_mechanistic + beta_trust * delta_X_pred (spec equation)."""
    beta = float(np.clip(beta_trust, 0.0, 1.0)) if np.isfinite(beta_trust) else 0.0
    mech = float(x_mechanistic) if np.isfinite(x_mechanistic) else float("nan")
    delta = float(delta_x_pred) if np.isfinite(delta_x_pred) else 0.0
    if not np.isfinite(mech):
        delta_applied = 0.0
        x_hybrid = float("nan")
    else:
        delta_applied = beta * delta
        x_hybrid = mech + delta_applied
    return HybridCombine(
        X_mechanistic=mech,
        delta_X_raw=delta,
        beta_trust=beta,
        delta_X_applied=delta_applied,
        X_hybrid=x_hybrid,
        physics_contribution=mech,
        ml_correction=delta_applied,
    )


@dataclass(frozen=True)
class HybridCombine:
    """One-timestamp hybrid combine. Independent of future samples."""

    X_mechanistic: float
    delta_X_raw: float
    beta_trust: float
    delta_X_applied: float
    X_hybrid: float
    physics_contribution: float
    ml_correction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "X_mechanistic": self.X_mechanistic,
            "delta_X_raw": self.delta_X_raw,
            "beta_trust": self.beta_trust,
            "delta_X_applied": self.delta_X_applied,
            "X_hybrid": self.X_hybrid,
            "physics_contribution": self.physics_contribution,
            "ml_correction": self.ml_correction,
        }


@dataclass(frozen=True)
class OODScore:
    """Logged OOD assessment for one feature vector (one timestamp)."""

    d_m: float
    d_threshold: float
    kappa: float
    beta_trust: float
    ood_state: str
    fallback_state: str
    valid: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "d_m": self.d_m,
            "d_threshold": self.d_threshold,
            "kappa": self.kappa,
            "beta_trust": self.beta_trust,
            "ood_state": self.ood_state,
            "fallback_state": self.fallback_state,
            "valid": self.valid,
        }


def _as_2d(features: ArrayLike) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Features must be 1-D or 2-D, got ndim={arr.ndim}")
    return arr


def _finite_rows(matrix: np.ndarray) -> np.ndarray:
    return np.isfinite(matrix).all(axis=1)


def _classify_ood(beta: float, d_m: float, d_threshold: float, severe_beta: float) -> tuple[str, str]:
    if d_m <= d_threshold:
        return OOD_IN_DISTRIBUTION, FALLBACK_NONE
    if beta <= severe_beta:
        return OOD_SEVERE, FALLBACK_PHYSICS
    return OOD_MODERATE, FALLBACK_ATTENUATED


class MahalanobisOOD:
    """Training-only Mahalanobis detector with sequential per-timestamp scoring."""

    def __init__(
        self,
        *,
        kappa: float | None = None,
        d_threshold: float | None = None,
        threshold_quantile: float | None = None,
        ridge: float | None = None,
        relative_ridge: float | None = None,
        severe_beta: float | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = dict(config) if config is not None else load_ood_config()
        method = str(cfg.get("method", "mahalanobis"))
        if method != "mahalanobis":
            raise ValueError(f"Locked OOD method is mahalanobis, got {method!r}")
        default_kappa = float(cfg.get("default_kappa", 1.0))
        cfg_kappa = cfg.get("kappa")
        self.kappa = float(
            kappa if kappa is not None else (default_kappa if cfg_kappa is None else cfg_kappa)
        )
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa}")
        cfg_thr = cfg.get("d_threshold")
        self._configured_threshold = (
            d_threshold if d_threshold is not None else (None if cfg_thr is None else float(cfg_thr))
        )
        q = threshold_quantile if threshold_quantile is not None else cfg.get("threshold_quantile", 0.95)
        self.threshold_quantile = float(q)
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        self.ridge = float(ridge if ridge is not None else cfg.get("ridge", 1.0e-6))
        self.relative_ridge = float(
            relative_ridge if relative_ridge is not None else cfg.get("relative_ridge", 1.0e-4)
        )
        self.severe_beta = float(severe_beta if severe_beta is not None else cfg.get("severe_beta", 1.0e-3))
        self.method = "mahalanobis"

        self.mean_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.d_threshold: float | None = None
        self.n_features_: int | None = None
        self.n_train_rows_: int | None = None
        self.feature_names_: tuple[str, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.precision_ is not None and self.d_threshold is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("MahalanobisOOD must be fit on training features before scoring")

    def fit(
        self,
        train_features: ArrayLike,
        *,
        split: str = "train",
        batch_ids: Sequence[Any] | np.ndarray | None = None,
        train_batch_ids: Iterable[Any] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> MahalanobisOOD:
        """Fit mean/covariance (and D_threshold if unset) on training rows only.

        ``split`` must be ``\"train\"``. Passing test/validation raises.
        If ``batch_ids`` and ``train_batch_ids`` are given, rows whose batch is
        not in the train set are dropped (test data cannot enter the fit).
        """
        split_n = str(split).strip().lower()
        if split_n in FORBIDDEN_FIT_SPLITS or split_n not in ALLOWED_FIT_SPLITS:
            raise ValueError(
                "OOD statistics must be fit on training data only "
                f"(got split={split!r}). Test batches are forbidden."
            )
        x = _as_2d(train_features)
        if x.shape[0] < 2:
            raise ValueError("Need at least two training rows to fit OOD covariance")
        if x.shape[1] < 1:
            raise ValueError("Need at least one feature column")

        if batch_ids is not None:
            bids = np.asarray(list(batch_ids), dtype=object).reshape(-1)
            if bids.shape[0] != x.shape[0]:
                raise ValueError(
                    f"batch_ids length {bids.shape[0]} != n_rows {x.shape[0]}"
                )
            if train_batch_ids is None:
                raise ValueError("train_batch_ids is required when batch_ids is provided")
            train_set = {b for b in train_batch_ids}
            mask = np.array([b in train_set for b in bids], dtype=bool)
            x = x[mask]
            if x.shape[0] < 2:
                raise ValueError("Fewer than two training-batch rows after filtering")

        finite = _finite_rows(x)
        x = x[finite]
        if x.shape[0] < 2:
            raise ValueError("Fewer than two finite training rows after dropping NaN/Inf")

        mean, cov, precision = _stable_covariance(x, ridge=self.ridge, relative_ridge=self.relative_ridge)
        self.mean_ = mean
        self.covariance_ = cov
        self.precision_ = precision
        self.n_features_ = int(x.shape[1])
        self.n_train_rows_ = int(x.shape[0])
        if feature_names is not None:
            names = tuple(str(n) for n in feature_names)
            if len(names) != self.n_features_:
                raise ValueError(
                    f"feature_names length {len(names)} != n_features {self.n_features_}"
                )
            self.feature_names_ = names
        else:
            self.feature_names_ = None

        train_d = _mahalanobis_batch(x, mean, precision)
        if self._configured_threshold is not None:
            self.d_threshold = float(self._configured_threshold)
        else:
            self.d_threshold = float(np.quantile(train_d, self.threshold_quantile))
        if not np.isfinite(self.d_threshold) or self.d_threshold < 0.0:
            self.d_threshold = 0.0
        return self

    def vector_from_features(
        self,
        features: Mapping[str, Any],
        feature_order: Sequence[str] | None = None,
    ) -> np.ndarray:
        order = list(feature_order) if feature_order is not None else list(self.feature_names_ or ())
        if not order:
            raise ValueError("feature_order is required to map a feature dict to phi")
        vec = np.empty(len(order), dtype=np.float64)
        for i, name in enumerate(order):
            if name not in features:
                vec[i] = np.nan
                continue
            try:
                vec[i] = float(features[name])
            except (TypeError, ValueError):
                vec[i] = np.nan
        return vec

    def mahalanobis(self, phi: ArrayLike) -> np.ndarray:
        """D_M for one vector or a batch of rows. Does not update fit statistics."""
        self._require_fitted()
        x = _as_2d(phi)
        if x.shape[1] != self.n_features_:
            raise ValueError(f"Expected n_features={self.n_features_}, got {x.shape[1]}")
        return _mahalanobis_batch(x, self.mean_, self.precision_)

    def score(self, phi: ArrayLike) -> OODScore:
        """Score one timestamp feature vector (sequential / causal)."""
        self._require_fitted()
        vec = np.asarray(phi, dtype=np.float64).reshape(-1)
        if vec.size != self.n_features_:
            return OODScore(
                d_m=float("nan"),
                d_threshold=float(self.d_threshold),
                kappa=self.kappa,
                beta_trust=0.0,
                ood_state=OOD_INVALID,
                fallback_state=FALLBACK_PHYSICS,
                valid=False,
            )
        if not np.isfinite(vec).all():
            return OODScore(
                d_m=float("nan"),
                d_threshold=float(self.d_threshold),
                kappa=self.kappa,
                beta_trust=0.0,
                ood_state=OOD_INVALID,
                fallback_state=FALLBACK_PHYSICS,
                valid=False,
            )
        d_m = float(self.mahalanobis(vec)[0])
        if not np.isfinite(d_m):
            return OODScore(
                d_m=d_m,
                d_threshold=float(self.d_threshold),
                kappa=self.kappa,
                beta_trust=0.0,
                ood_state=OOD_INVALID,
                fallback_state=FALLBACK_PHYSICS,
                valid=False,
            )
        beta = beta_trust_from_distance(d_m, kappa=self.kappa, d_threshold=float(self.d_threshold))
        ood_state, fallback = _classify_ood(beta, d_m, float(self.d_threshold), self.severe_beta)
        return OODScore(
            d_m=d_m,
            d_threshold=float(self.d_threshold),
            kappa=self.kappa,
            beta_trust=beta,
            ood_state=ood_state,
            fallback_state=fallback,
            valid=True,
        )

    def hybrid_at(
        self,
        *,
        x_mechanistic: float,
        delta_x_pred: float,
        phi: ArrayLike,
    ) -> tuple[HybridCombine, OODScore]:
        """Sequential hybrid combine at one timestamp using this detector."""
        ood = self.score(phi)
        hybrid = apply_hybrid(x_mechanistic, delta_x_pred, ood.beta_trust)
        return hybrid, ood


def _stable_covariance(
    x: np.ndarray,
    *,
    ridge: float,
    relative_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ledoit-Wolf shrinkage + ridge + Hermitian pinv (singular-safe)."""
    n, d = x.shape
    mean = np.mean(x, axis=0)
    try:
        lw = LedoitWolf(assume_centered=False, store_precision=False)
        lw.fit(x)
        cov = np.asarray(lw.covariance_, dtype=np.float64)
        mean = np.asarray(lw.location_, dtype=np.float64)
    except Exception:
        centered = x - mean
        cov = centered.T @ centered / max(n - 1, 1)

    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (d, d):
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    scale = trace / d if d else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    eps = max(float(ridge), float(relative_ridge) * scale, 1.0e-12)
    cov = cov + eps * np.eye(d, dtype=np.float64)
    precision = np.linalg.pinv(cov, rcond=1e-12)
    precision = 0.5 * (precision + precision.T)
    return mean, cov, precision


def _mahalanobis_batch(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = x - mean
    # diag(delta @ precision @ delta.T) without forming the full product
    mid = delta @ precision
    d2 = np.einsum("ij,ij->i", mid, delta)
    d2 = np.maximum(d2, 0.0)
    out = np.sqrt(d2)
    return out
