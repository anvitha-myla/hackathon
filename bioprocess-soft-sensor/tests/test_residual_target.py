"""Residual target math, single-network architecture, and train/val-only scaler fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.mlp import LOCKED_HIDDEN, SmallMLP, count_linear_layers
from src.models.residual_nn import (
    DEFAULT_FEATURE_ORDER,
    ResidualMLP,
    residual_target,
    train,
    valid_residual_mask,
)
from src.models.training import TEST_SPLIT_FORBIDDEN, fit_feature_scaler, normalize_split


def _tiny_frame(n_train: int = 8, n_val: int = 4, n_test: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    features = list(DEFAULT_FEATURE_ORDER)
    for batch_id, n, x_shift in (("train_a", n_train, 0.0), ("val_a", n_val, 0.5), ("test_a", n_test, 50.0)):
        for i in range(n):
            x_mech = 2.0 + 0.1 * i + x_shift
            x_ref = x_mech + 0.25 + 0.01 * i
            row = {name: float(rng.normal()) for name in features}
            row["X_mechanistic"] = x_mech
            row["X_reference"] = x_ref
            row["batch_id"] = batch_id
            rows.append(row)
    return pd.DataFrame(rows)


def test_delta_x_equals_reference_minus_mechanistic() -> None:
    x_ref = np.array([1.0, 2.5, 0.0, np.nan])
    x_mech = np.array([0.4, 1.0, 0.2, 1.0])
    delta = residual_target(x_ref, x_mech)
    np.testing.assert_allclose(delta[:3], np.array([0.6, 1.5, -0.2]))
    assert np.isnan(delta[3])
    mask = valid_residual_mask(x_ref, x_mech)
    np.testing.assert_array_equal(mask, np.array([True, True, True, False]))


def test_residual_target_matches_tensor_subtraction() -> None:
    ref = torch.tensor([3.0, 4.0, 5.0])
    mech = torch.tensor([1.5, 4.0, 0.5])
    delta = residual_target(ref.numpy(), mech.numpy())
    np.testing.assert_allclose(delta, (ref - mech).numpy())


def test_single_residual_network_not_three() -> None:
    model = ResidualMLP(n_in=len(DEFAULT_FEATURE_ORDER))
    assert ResidualMLP.n_networks == 1
    assert ResidualMLP.phase_specific is False
    assert ResidualMLP.output_name == "delta_X"
    assert not hasattr(model, "phase_models")
    linears = count_linear_layers(model)
    assert linears == 4  # 64, 32, 16, output
    last = [m for m in model.modules() if type(m) is torch.nn.Linear][-1]
    assert last.out_features == 1
    hidden = [
        m.out_features
        for m in model.backbone.net
        if isinstance(m, torch.nn.Linear)
    ][:-1]
    assert tuple(hidden) == LOCKED_HIDDEN


def test_forward_pass_scalar_delta() -> None:
    n_in = 6
    model = ResidualMLP(n_in=n_in)
    x = torch.randn(3, n_in)
    y = model(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()


def test_shared_backbone_is_64_32_16() -> None:
    mlp = SmallMLP(n_in=4, n_out=1)
    widths = [m.out_features for m in mlp.net if isinstance(m, torch.nn.Linear)]
    assert widths == [64, 32, 16, 1]


def test_scaler_not_fit_on_test_ids(tmp_path: Path) -> None:
    frame = _tiny_frame()
    split = {
        "train": ["train_a"],
        "validation": ["val_a"],
        "test": ["test_a"],
    }
    result = train(frame, split, artifact_dir=tmp_path, max_epochs=1)
    assert "test_a" not in result.train_batch_ids
    assert "test_a" not in result.validation_batch_ids
    assert result.config["test_split_used"] is False
    assert result.config["n_networks"] == 1

    train_rows = frame[frame["batch_id"] == "train_a"]
    x_train = train_rows[list(DEFAULT_FEATURE_ORDER)].to_numpy(dtype=np.float64)
    expected = fit_feature_scaler(x_train)
    np.testing.assert_allclose(result.scaler.mean_, expected.mean_)
    np.testing.assert_allclose(result.scaler.scale_, expected.scale_)

    all_but_val = frame[frame["batch_id"] != "val_a"]
    leaked = fit_feature_scaler(all_but_val[list(DEFAULT_FEATURE_ORDER)].to_numpy(dtype=np.float64))
    assert not np.allclose(result.scaler.mean_, leaked.mean_)

    for name in (
        "residual_weights.pt",
        "scaler.joblib",
        "feature_order.json",
        "config.json",
        "validation_metrics.json",
        "history.json",
    ):
        assert (tmp_path / name).is_file()
    assert result.feature_order == list(DEFAULT_FEATURE_ORDER)
    assert "val_mse" in result.validation_metrics
    assert len(result.history["train_loss"]) >= 1


def test_train_raises_if_splits_overlap() -> None:
    frame = _tiny_frame()
    split = normalize_split({"train": ["train_a"], "validation": ["train_a"], "test": []})
    try:
        train(frame, split, max_epochs=1)
        raise AssertionError("expected overlap error")
    except ValueError as exc:
        assert "overlap" in str(exc).lower()


def test_test_split_forbidden_message() -> None:
    assert "test split" in TEST_SPLIT_FORBIDDEN.lower()
