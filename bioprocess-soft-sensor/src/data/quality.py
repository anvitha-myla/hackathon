"""Quality flags and physical-plausibility checks (causal; no future access).

Impossible values are flagged. Callers must keep the raw observation; this
module never deletes source data.
"""

from __future__ import annotations

from enum import IntFlag
from typing import Any, Iterable, Mapping

import numpy as np
import numpy.typing as npt

ArrayF = npt.NDArray[np.floating]
ArrayB = npt.NDArray[np.bool_]
ArrayU = npt.NDArray[np.uint32]


class QualityFlag(IntFlag):
    """Bit flags; combine with bitwise OR. OK is the absence of all bits."""

    OK = 0
    MISSING = 1 << 0
    FORWARD_FILLED = 1 << 1
    FILL_LIMIT_EXCEEDED = 1 << 2
    SPIKE = 1 << 3
    IMPLAUSIBLE = 1 << 4


FLAG_NAMES: tuple[tuple[QualityFlag, str], ...] = (
    (QualityFlag.MISSING, "missing"),
    (QualityFlag.FORWARD_FILLED, "forward_filled"),
    (QualityFlag.FILL_LIMIT_EXCEEDED, "fill_limit_exceeded"),
    (QualityFlag.SPIKE, "spike"),
    (QualityFlag.IMPLAUSIBLE, "implausible"),
)


def combine_flags(*flags: QualityFlag | int) -> int:
    acc = 0
    for flag in flags:
        acc |= int(flag)
    return acc


def has_flag(quality: int | np.integer, flag: QualityFlag) -> bool:
    if flag is QualityFlag.OK:
        return int(quality) == 0
    return bool(int(quality) & int(flag))


def flag_names(quality: int | np.integer) -> list[str]:
    value = int(quality)
    if value == 0:
        return ["ok"]
    return [name for flag, name in FLAG_NAMES if value & int(flag)]


def flag_names_array(quality: npt.NDArray[np.integer]) -> list[list[str]]:
    return [flag_names(q) for q in np.asarray(quality).ravel()]


def check_physical_plausibility(
    values: npt.ArrayLike,
    min_value: float | None,
    max_value: float | None,
) -> ArrayB:
    """True where a finite value lies outside ``[min_value, max_value]``.

    Non-finite entries are not marked implausible here (they are missing).
    """
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros(x.shape, dtype=bool)
    if min_value is not None:
        out |= finite & (x < float(min_value))
    if max_value is not None:
        out |= finite & (x > float(max_value))
    return out


def non_finite_mask(values: npt.ArrayLike) -> ArrayB:
    x = np.asarray(values, dtype=float)
    return ~np.isfinite(x)


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def bounds_for_column(
    column: str,
    physical_bounds: Mapping[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Return (min, max) for ``column`` using exact or alias match. No data fit."""
    if not physical_bounds:
        return None, None
    target = _norm_name(column)
    for key, spec in physical_bounds.items():
        if not isinstance(spec, Mapping):
            continue
        names = [_norm_name(str(key))]
        aliases = spec.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        names.extend(_norm_name(str(a)) for a in aliases)
        if target in names:
            lo = spec.get("min", spec.get("low"))
            hi = spec.get("max", spec.get("high"))
            return (
                None if lo is None else float(lo),
                None if hi is None else float(hi),
            )
    return None, None


def apply_physical_flags(
    values: npt.ArrayLike,
    quality: npt.NDArray[np.integer],
    min_value: float | None,
    max_value: float | None,
) -> ArrayU:
    """OR ``IMPLAUSIBLE`` into ``quality`` where bounds fail. Does not edit values."""
    q = np.asarray(quality, dtype=np.uint32).copy()
    implausible = check_physical_plausibility(values, min_value, max_value)
    inf_or_nan = non_finite_mask(values)
    x = np.asarray(values, dtype=float)
    q[implausible] |= np.uint32(QualityFlag.IMPLAUSIBLE)
    q[np.isinf(x)] |= np.uint32(QualityFlag.IMPLAUSIBLE)
    q[inf_or_nan] |= np.uint32(QualityFlag.MISSING)
    return q


def audit_record(
    *,
    raw: npt.ArrayLike,
    cleaned: npt.ArrayLike,
    quality: npt.ArrayLike,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-signal audit payload: raw, cleaned, quality, transformation metadata."""
    record: dict[str, Any] = {
        "raw": np.asarray(raw, dtype=float),
        "cleaned": np.asarray(cleaned, dtype=float),
        "quality": np.asarray(quality, dtype=np.uint32),
        "quality_labels": flag_names_array(np.asarray(quality, dtype=np.uint32)),
    }
    if extra:
        record["transformation"] = dict(extra)
    return record


def any_named_flags(quality: Iterable[int], flag: QualityFlag) -> bool:
    return any(has_flag(q, flag) for q in quality)
