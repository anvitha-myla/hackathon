"""Residual MLP: learns delta_X = X_reference - X_mechanistic.

Single network (not three phase NNs). Inputs may include observables, causal
derived features, mechanistic outputs, and soft phase indicators.

Training uses train batches for optimization and scaler fit; validation
batches for model selection. The test split is never used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROJECT_ROOT, load_yaml
from src.models.mlp import LOCKED_HIDDEN, SmallMLP, count_linear_layers
from src.models.training import (
    TEST_SPLIT_FORBIDDEN,
    assert_disjoint_splits,
    assert_no_test_ids,
    fit_feature_scaler,
    load_model_config,
    load_split_manifest,
    normalize_split,
    set_seed,
)

REFERENCE_COL = "X_reference"
MECHANISTIC_COL = "X_mechanistic"
BATCH_COL = "batch_id"
TARGET_NAME = "delta_X"

# Default residual inputs until the feature-engineering layer publishes a registry.
DEFAULT_FEATURE_ORDER: tuple[str, ...] = (
    "pH",
    "DO",
    "T_vessel",
    "T_jacket",
    "agitation",
    "feed_rate",
    "gas_flow",
    "outlet_CO2",
    "outlet_O2",
    "OUR",
    "CER",
    "RQ",
    "delta_T",
    "d_tau_dt",
    "dDO_dt",
    "cumulative_feed",
    "volume",
    "progress",
    "X_mechanistic",
    "mu_mechanistic",
    "product_rate_mechanistic",
    "phase_growth",
    "phase_production",
    "phase_autolysis",
)


def residual_target(
    x_reference: np.ndarray | Sequence[float],
    x_mechanistic: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Training label: delta_X = X_reference - X_mechanistic."""
    ref = np.asarray(x_reference, dtype=np.float64)
    mech = np.asarray(x_mechanistic, dtype=np.float64)
    if ref.shape != mech.shape:
        raise ValueError(f"X_reference shape {ref.shape} != X_mechanistic shape {mech.shape}")
    return ref - mech


def valid_residual_mask(
    x_reference: np.ndarray | Sequence[float],
    x_mechanistic: np.ndarray | Sequence[float],
) -> np.ndarray:
    ref = np.asarray(x_reference, dtype=np.float64)
    mech = np.asarray(x_mechanistic, dtype=np.float64)
    if ref.shape != mech.shape:
        raise ValueError(f"X_reference shape {ref.shape} != X_mechanistic shape {mech.shape}")
    return np.isfinite(ref) & np.isfinite(mech)


class ResidualMLP(nn.Module):
    """One residual MLP with a scalar delta_X head."""

    n_networks = 1
    phase_specific = False
    output_name = TARGET_NAME

    def __init__(self, n_in: int, hidden: Sequence[int] = LOCKED_HIDDEN) -> None:
        super().__init__()
        self.backbone = SmallMLP(n_in=n_in, n_out=1, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out.squeeze(-1)


@dataclass
class ResidualFitResult:
    model: ResidualMLP
    scaler: StandardScaler
    feature_order: list[str]
    config: dict[str, Any]
    validation_metrics: dict[str, float]
    history: dict[str, list[float]]
    artifact_dir: Path | None = None
    train_batch_ids: list[Any] = field(default_factory=list)
    validation_batch_ids: list[Any] = field(default_factory=list)


def _feature_order_from_config(config: Mapping[str, Any] | None) -> list[str]:
    if config and config.get("input_features"):
        return [str(name) for name in config["input_features"]]
    return list(DEFAULT_FEATURE_ORDER)


def _column_map(config: Mapping[str, Any] | None) -> dict[str, str]:
    cols = dict((config or {}).get("columns") or {})
    return {
        "batch_id": str(cols.get("batch_id", BATCH_COL)),
        "x_reference": str(cols.get("x_reference", REFERENCE_COL)),
        "x_mechanistic": str(cols.get("x_mechanistic", MECHANISTIC_COL)),
    }


def _select_rows(frame: pd.DataFrame, batch_col: str, ids: Sequence[Any]) -> pd.DataFrame:
    id_set = set(ids)
    return frame.loc[frame[batch_col].isin(id_set)].copy()


def build_residual_arrays(
    frame: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    x_reference_col: str = REFERENCE_COL,
    x_mechanistic_col: str = MECHANISTIC_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, delta_X, valid_mask) for rows in ``frame``."""
    missing = [c for c in feature_order if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing residual input columns: {missing}")
    if x_reference_col not in frame.columns:
        raise KeyError(f"Missing reference column {x_reference_col}")
    if x_mechanistic_col not in frame.columns:
        raise KeyError(f"Missing mechanistic column {x_mechanistic_col}")
    ref = frame[x_reference_col].to_numpy(dtype=np.float64)
    mech = frame[x_mechanistic_col].to_numpy(dtype=np.float64)
    mask = valid_residual_mask(ref, mech)
    features = frame.loc[:, list(feature_order)].to_numpy(dtype=np.float64)
    delta = residual_target(ref, mech)
    return features, delta, mask


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size == 0:
        return float("nan")
    return float(np.mean((pred - target) ** 2))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - target)))


@torch.no_grad()
def predict_delta(
    model: ResidualMLP,
    scaler: StandardScaler,
    features: np.ndarray,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cpu")
    model.eval()
    x = scaler.transform(np.asarray(features, dtype=np.float64))
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    pred = model(tensor).cpu().numpy()
    return np.asarray(pred, dtype=np.float64)


def _epoch_loss(
    model: ResidualMLP,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        if train_mode:
            loss.backward()
            optimizer.step()
        bs = int(yb.shape[0])
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def train(
    frame: pd.DataFrame,
    split: Mapping[str, Any],
    *,
    artifact_dir: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    max_epochs: int | None = None,
    device: str | torch.device | None = None,
) -> ResidualFitResult:
    """Fit the residual MLP on train batches; select on validation; ignore test.

    ``frame`` must include batch_id, X_reference, X_mechanistic, and feature columns.
    """
    cfg = dict(load_model_config() if config is None else config)
    split_n = normalize_split(split)
    assert_disjoint_splits(split_n)
    cols = _column_map(cfg)
    batch_col = cols["batch_id"]
    if batch_col not in frame.columns:
        raise KeyError(f"Missing batch column {batch_col}")

    present_ids = set(frame[batch_col].unique())
    test_in_frame = present_ids & set(split_n["test"])
    if test_in_frame:
        frame = frame.loc[~frame[batch_col].isin(split_n["test"])].copy()

    train_ids = [i for i in split_n["train"] if i in set(frame[batch_col].unique())]
    val_ids = [i for i in split_n["validation"] if i in set(frame[batch_col].unique())]
    if not train_ids:
        raise ValueError("No training batch ids present in the provided table")
    if not val_ids:
        raise ValueError("No validation batch ids present in the provided table")

    assert_no_test_ids(train_ids, split_n["test"], context="scaler/training batch ids")
    assert_no_test_ids(val_ids, split_n["test"], context="validation batch ids")

    feature_order = _feature_order_from_config(cfg)
    train_df = _select_rows(frame, batch_col, train_ids)
    val_df = _select_rows(frame, batch_col, val_ids)

    x_train, y_train, m_train = build_residual_arrays(
        train_df,
        feature_order,
        x_reference_col=cols["x_reference"],
        x_mechanistic_col=cols["x_mechanistic"],
    )
    x_val, y_val, m_val = build_residual_arrays(
        val_df,
        feature_order,
        x_reference_col=cols["x_reference"],
        x_mechanistic_col=cols["x_mechanistic"],
    )
    if not np.any(m_train):
        raise ValueError("No valid training samples (finite X_reference and X_mechanistic)")
    if not np.any(m_val):
        raise ValueError("No valid validation samples (finite X_reference and X_mechanistic)")

    x_train, y_train = x_train[m_train], y_train[m_train]
    x_val, y_val = x_val[m_val], y_val[m_val]

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    scaler = fit_feature_scaler(x_train)
    x_train_s = scaler.transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)

    device_t = torch.device(device or "cpu")
    model = ResidualMLP(n_in=len(feature_order), hidden=tuple(cfg.get("hidden", LOCKED_HIDDEN)))
    model.to(device_t)

    lr = float(cfg.get("learning_rate", 1e-3))
    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(max_epochs if max_epochs is not None else cfg.get("max_epochs", 100))
    patience = int(cfg.get("early_stopping_patience", 10))
    loss_name = str(cfg.get("loss", "mse")).lower()
    if loss_name == "huber":
        criterion: nn.Module = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train_s),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_s),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    for _ in range(max(epochs, 1)):
        train_loss = _epoch_loss(model, train_loader, criterion, optimizer=optimizer, device=device_t)
        val_loss = _epoch_loss(model, val_loader, criterion, optimizer=None, device=device_t)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device_t)
    val_pred = predict_delta(model, scaler, x_val, device=device_t)
    train_pred = predict_delta(model, scaler, x_train, device=device_t)
    validation_metrics = {
        "val_mse": _mse(val_pred, y_val),
        "val_mae": _mae(val_pred, y_val),
        "val_loss_best": float(best_val),
        "train_mse": _mse(train_pred, y_train),
        "n_train": float(len(y_train)),
        "n_val": float(len(y_val)),
        "n_features": float(len(feature_order)),
        "n_networks": 1.0,
        "epochs_ran": float(len(history["train_loss"])),
    }

    persist_cfg = {
        **cfg,
        "hidden": list(cfg.get("hidden", LOCKED_HIDDEN)),
        "activation": cfg.get("activation", "relu"),
        "n_networks": 1,
        "phase_specific": False,
        "residual_output": TARGET_NAME,
        "n_in": len(feature_order),
        "seed": seed,
        "linear_layers": count_linear_layers(model),
        "forbidden": ["lstm", "transformer", "xgboost", "three_phase_nns"],
        "test_split_used": False,
        "note": TEST_SPLIT_FORBIDDEN,
    }
    result = ResidualFitResult(
        model=model.cpu(),
        scaler=scaler,
        feature_order=list(feature_order),
        config=persist_cfg,
        validation_metrics=validation_metrics,
        history=history,
        train_batch_ids=list(train_ids),
        validation_batch_ids=list(val_ids),
    )
    if artifact_dir is not None:
        result.artifact_dir = save_residual_artifacts(result, artifact_dir)
    return result


def save_residual_artifacts(result: ResidualFitResult, artifact_dir: str | Path) -> Path:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), artifact_dir / "residual_weights.pt")
    joblib.dump(result.scaler, artifact_dir / "scaler.joblib")
    feature_payload = {
        "feature_order": result.feature_order,
        "feature_names": result.feature_order,
        "n_features": len(result.feature_order),
        "target": TARGET_NAME,
        "n_networks": 1,
    }
    (artifact_dir / "feature_order.json").write_text(
        json.dumps(feature_payload, indent=2), encoding="utf-8"
    )
    (artifact_dir / "config.json").write_text(json.dumps(result.config, indent=2, default=str), encoding="utf-8")
    (artifact_dir / "validation_metrics.json").write_text(
        json.dumps(result.validation_metrics, indent=2), encoding="utf-8"
    )
    (artifact_dir / "history.json").write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    meta = {
        "train_batch_ids": [str(x) for x in result.train_batch_ids],
        "validation_batch_ids": [str(x) for x in result.validation_batch_ids],
        "test_split_used": False,
        "output": TARGET_NAME,
        "architecture": "mlp_64_32_16_relu",
        "n_networks": 1,
    }
    (artifact_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return artifact_dir


def load_residual_artifacts(artifact_dir: str | Path, *, map_location: str = "cpu") -> ResidualFitResult:
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    feature_order = json.loads((artifact_dir / "feature_order.json").read_text(encoding="utf-8"))[
        "feature_order"
    ]
    scaler: StandardScaler = joblib.load(artifact_dir / "scaler.joblib")
    model = ResidualMLP(n_in=len(feature_order), hidden=tuple(config.get("hidden", LOCKED_HIDDEN)))
    weights_path = artifact_dir / "residual_weights.pt"
    try:
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    metrics = json.loads((artifact_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    history = json.loads((artifact_dir / "history.json").read_text(encoding="utf-8"))
    return ResidualFitResult(
        model=model,
        scaler=scaler,
        feature_order=list(feature_order),
        config=config,
        validation_metrics=metrics,
        history=history,
        artifact_dir=artifact_dir,
    )


def train_from_disk(
    *,
    project_root: str | Path | None = None,
    table_path: str | Path | None = None,
    split_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> ResidualFitResult:
    """Train from a processed table + split manifest. Does not download IndPenSim."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    split_path = Path(split_path) if split_path is not None else root / "data" / "splits" / "split.json"
    candidates = []
    if table_path is not None:
        candidates.append(Path(table_path))
    else:
        processed = root / "data" / "processed"
        candidates.extend(
            [
                processed / "residual_table.parquet",
                processed / "features.parquet",
                processed / "table.csv",
            ]
        )
    table_file = next((p for p in candidates if p.is_file()), None)
    if not split_path.is_file() or table_file is None:
        raise FileNotFoundError(
            "IndPenSim processed table or split manifest is not on disk. "
            "Refusing to download a large dataset. Call train(frame, split) with in-memory "
            "rows, or place a table under data/processed and a manifest at data/splits/split.json."
        )
    if table_file.suffix == ".csv":
        frame = pd.read_csv(table_file)
    else:
        frame = pd.read_parquet(table_file)
    split = load_split_manifest(split_path)
    artifacts = Path(artifact_dir) if artifact_dir is not None else root / "models" / "residual"
    return train(frame, split, artifact_dir=artifacts, config=load_yaml("model.yaml"))


def default_artifact_dir() -> Path:
    return PROJECT_ROOT / "models" / "residual"
