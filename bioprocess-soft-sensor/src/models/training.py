"""Training helpers: seeds, split isolation, scaler fit on train only."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, load_yaml

TEST_SPLIT_FORBIDDEN = (
    "The final test split must not be used to fit the scaler, OOD stats, or the residual MLP."
)


def set_seed(seed: int) -> None:
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_split(manifest: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Accept train/validation/test or train_batches/validation_batches/test_batches."""

    def _ids(keys: tuple[str, ...]) -> list[Any]:
        for key in keys:
            if key in manifest and manifest[key] is not None:
                return list(manifest[key])
        return []

    train = _ids(("train", "train_batches"))
    validation = _ids(("validation", "val", "validation_batches"))
    test = _ids(("test", "test_batches"))
    return {"train": train, "validation": validation, "test": test}


def load_split_manifest(path: str | Path | None = None) -> dict[str, list[Any]]:
    if path is None:
        default = PROJECT_ROOT / "data" / "splits" / "split.json"
        alt = PROJECT_ROOT / "data" / "splits" / "manifest.json"
        path = default if default.is_file() else alt
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Split manifest must be a mapping, got {type(raw)}")
    return normalize_split(raw)


def assert_disjoint_splits(split: Mapping[str, Sequence[Any]]) -> None:
    train = set(split.get("train", []))
    validation = set(split.get("validation", []))
    test = set(split.get("test", []))
    if train & validation:
        raise ValueError(f"Train and validation batch ids overlap: {train & validation}")
    if train & test:
        raise ValueError(f"Train and test batch ids overlap: {train & test}")
    if validation & test:
        raise ValueError(f"Validation and test batch ids overlap: {validation & test}")
    if not train:
        raise ValueError("Split must include at least one training batch id")


def assert_no_test_ids(batch_ids: Sequence[Any], test_ids: Sequence[Any], *, context: str) -> None:
    leaked = set(batch_ids) & set(test_ids)
    if leaked:
        raise ValueError(f"{TEST_SPLIT_FORBIDDEN} ({context}: {sorted(leaked, key=str)})")


def fit_feature_scaler(x_train: np.ndarray) -> StandardScaler:
    if x_train.ndim != 2:
        raise ValueError("Scaler input must be 2-D (n_samples, n_features)")
    if x_train.shape[0] < 1:
        raise ValueError("Cannot fit scaler on an empty training matrix")
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return load_yaml("model.yaml")
    return load_yaml(path)
