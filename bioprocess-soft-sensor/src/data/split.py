"""Reproducible batch-level train/validation/test split.

Locked rule (docs/00_MASTER_SPEC.md, docs/02_DATA_SPEC.md):
split by complete ``batch_id`` only. Never split timestamps of one batch
across train/validation/test. There is no calibration split.

Counts
------
- If ``n == 100``: exactly 60 train / 20 validation / 20 test.
- If ``n != 100``: ``n_train = n * 60 // 100``, ``n_val = n * 20 // 100``,
  ``n_test = n - n_train - n_val`` (remainder assigned to test). Every
  unique ``batch_id`` is assigned to exactly one split.

Reproducibility
---------------
The same seed and the same set of ``batch_id`` values always produce the
same membership, regardless of input order or duplicate ids. Seed is read
from ``configs/split.yaml`` (and ``configs/default.yaml``) and defaults to 42.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.config import PROJECT_ROOT, load_yaml

BATCH_ID_COL = "batch_id"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "manifest.json"
SPLIT_NAMES = ("train", "validation", "test")


def load_split_config() -> dict[str, Any]:
    """Load split parameters from ``configs/split.yaml``."""
    return load_yaml("split.yaml")


def default_seed() -> int:
    cfg = load_split_config()
    return int(cfg["seed"])


def split_counts(n: int) -> tuple[int, int, int]:
    """Return (n_train, n_validation, n_test) for ``n`` unique batches."""
    if n < 1:
        raise ValueError("Need at least one unique batch_id to split")
    if n == 100:
        return 60, 20, 20
    n_train = n * 60 // 100
    n_val = n * 20 // 100
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _normalize_batch_id(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"Invalid batch_id: {value!r}")
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int):
        return int(value)
    return value


def unique_sorted_batch_ids(batch_ids: Iterable[Any]) -> list[Any]:
    """Deduplicate and sort batch ids so shuffle input is order-independent."""
    seen: dict[Any, None] = {}
    for raw in batch_ids:
        seen[_normalize_batch_id(value=raw)] = None
    return sorted(seen, key=lambda x: (str(type(x).__name__), str(x)))


@dataclass(frozen=True)
class SplitManifest:
    """Batch-id membership for train / validation / test."""

    seed: int
    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    test: tuple[Any, ...]
    unit: str = "complete_batch"

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def as_sets(self) -> dict[str, set[Any]]:
        return {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }

    def all_ids(self) -> set[Any]:
        return set(self.train) | set(self.validation) | set(self.test)

    def split_of(self, batch_id: Any) -> str:
        bid = _normalize_batch_id(batch_id)
        for name in SPLIT_NAMES:
            if bid in getattr(self, name):
                return name
        raise KeyError(f"batch_id {batch_id!r} is not in the split manifest")

    def to_dict(self) -> dict[str, Any]:
        n = sum(self.counts.values())
        return {
            "seed": self.seed,
            "unit": self.unit,
            "n_batches": n,
            "counts": self.counts,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "n_not_100_rule": (
                "If n==100: exact 60/20/20. "
                "If n!=100: n_train=n*60//100, n_val=n*20//100, remainder to test."
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SplitManifest:
        return cls(
            seed=int(payload["seed"]),
            train=tuple(payload["train"]),
            validation=tuple(payload["validation"]),
            test=tuple(payload["test"]),
            unit=str(payload.get("unit", "complete_batch")),
        )


def assert_zero_batch_overlap(manifest: SplitManifest) -> None:
    """Raise if any batch_id appears in more than one split."""
    sets = manifest.as_sets()
    overlaps = {
        "train∩validation": sets["train"] & sets["validation"],
        "train∩test": sets["train"] & sets["test"],
        "validation∩test": sets["validation"] & sets["test"],
    }
    leaked = {name: sorted(ids, key=str) for name, ids in overlaps.items() if ids}
    if leaked:
        raise AssertionError(f"Batch leakage across splits: {leaked}")


def split_batch_ids(
    batch_ids: Iterable[Any],
    seed: int | None = None,
) -> SplitManifest:
    """Assign each unique batch_id to exactly one of train/validation/test."""
    if seed is None:
        seed = default_seed()
    ids = unique_sorted_batch_ids(batch_ids)
    n_train, n_val, n_test = split_counts(len(ids))
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    train = tuple(shuffled[:n_train])
    validation = tuple(shuffled[n_train : n_train + n_val])
    test = tuple(shuffled[n_train + n_val :])
    assert len(test) == n_test
    manifest = SplitManifest(
        seed=int(seed),
        train=train,
        validation=validation,
        test=test,
    )
    assert_zero_batch_overlap(manifest)
    if manifest.all_ids() != set(ids):
        raise AssertionError("Split union does not equal the unique batch_id set")
    return manifest


def save_manifest(
    manifest: SplitManifest,
    path: str | Path | None = None,
) -> Path:
    path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(path: str | Path | None = None) -> SplitManifest:
    path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = SplitManifest.from_dict(payload)
    assert_zero_batch_overlap(manifest)
    return manifest


def extract_batch_ids(frame: pd.DataFrame) -> list[Any]:
    if BATCH_ID_COL not in frame.columns:
        raise KeyError(f"DataFrame must contain a {BATCH_ID_COL!r} column")
    return unique_sorted_batch_ids(frame[BATCH_ID_COL].tolist())


def split_dataframe(
    frame: pd.DataFrame,
    seed: int | None = None,
    batch_id_col: str = BATCH_ID_COL,
) -> tuple[SplitManifest, dict[str, pd.DataFrame]]:
    """Split a row table by complete batch_id (all timestamps stay together)."""
    if batch_id_col not in frame.columns:
        raise KeyError(f"DataFrame must contain a {batch_id_col!r} column")
    manifest = split_batch_ids(frame[batch_id_col].tolist(), seed=seed)
    sets = manifest.as_sets()
    parts: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        mask = frame[batch_id_col].map(_normalize_batch_id).isin(sets[name])
        parts[name] = frame.loc[mask].copy()
    _assert_batch_level_only(frame, parts, batch_id_col=batch_id_col)
    return manifest, parts


def _assert_batch_level_only(
    frame: pd.DataFrame,
    parts: Mapping[str, pd.DataFrame],
    batch_id_col: str = BATCH_ID_COL,
) -> None:
    """Every original batch_id must map to exactly one split (no row-level split)."""
    assignment: dict[Any, str] = {}
    for name, part in parts.items():
        for bid in part[batch_id_col].map(_normalize_batch_id):
            prev = assignment.get(bid)
            if prev is not None and prev != name:
                raise AssertionError(
                    f"Timestamp-level leakage: batch_id {bid!r} in {prev!r} and {name!r}"
                )
            assignment[bid] = name
    original = set(unique_sorted_batch_ids(frame[batch_id_col].tolist()))
    if set(assignment) != original:
        raise AssertionError("Some batch_ids were dropped or invented during the split")


def assign_split_column(
    frame: pd.DataFrame,
    manifest: SplitManifest,
    batch_id_col: str = BATCH_ID_COL,
    split_col: str = "split",
) -> pd.DataFrame:
    """Attach a split label to each row from its batch_id (not from timestamp)."""
    out = frame.copy()
    out[split_col] = out[batch_id_col].map(lambda bid: manifest.split_of(bid))
    return out


def write_split_from_batch_ids(
    batch_ids: Sequence[Any],
    path: str | Path | None = None,
    seed: int | None = None,
) -> tuple[SplitManifest, Path]:
    manifest = split_batch_ids(batch_ids, seed=seed)
    written = save_manifest(manifest, path=path)
    return manifest, written
