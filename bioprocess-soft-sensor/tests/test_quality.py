"""Unit tests for quality flags and physical plausibility."""

from __future__ import annotations

import numpy as np

from src.data.cleaning import load_cleaning_params
from src.data.quality import (
    QualityFlag,
    apply_physical_flags,
    bounds_for_column,
    check_physical_plausibility,
    combine_flags,
    flag_names,
    has_flag,
)


def test_flag_combine_and_names() -> None:
    q = combine_flags(QualityFlag.MISSING, QualityFlag.SPIKE)
    assert has_flag(q, QualityFlag.MISSING)
    assert has_flag(q, QualityFlag.SPIKE)
    assert not has_flag(q, QualityFlag.IMPLAUSIBLE)
    assert set(flag_names(q)) == {"missing", "spike"}
    assert flag_names(0) == ["ok"]
    assert has_flag(0, QualityFlag.OK)


def test_physical_plausibility_does_not_edit_values() -> None:
    x = np.array([7.0, 15.0, np.nan], dtype=float)
    original = x.copy()
    mask = check_physical_plausibility(x, 0.0, 14.0)
    np.testing.assert_array_equal(x, original)
    assert list(mask) == [False, True, False]


def test_bounds_from_yaml_aliases() -> None:
    bounds = load_cleaning_params().physical_bounds
    assert bounds_for_column("pH", bounds) == (0.0, 14.0)
    assert bounds_for_column("DO", bounds)[0] == 0.0
    assert bounds_for_column("unknown_signal", bounds) == (None, None)


def test_apply_physical_flags_keeps_inf_as_implausible_and_missing() -> None:
    x = np.array([1.0, np.inf, -1.0])
    q = np.zeros(3, dtype=np.uint32)
    out = apply_physical_flags(x, q, 0.0, 10.0)
    assert has_flag(int(out[1]), QualityFlag.IMPLAUSIBLE)
    assert has_flag(int(out[1]), QualityFlag.MISSING)
    assert has_flag(int(out[2]), QualityFlag.IMPLAUSIBLE)
    assert not has_flag(int(out[0]), QualityFlag.IMPLAUSIBLE)
