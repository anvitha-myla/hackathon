"""Leakage, count, and reproducibility tests for batch-level 60/20/20 splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_default
from src.data.split import (
    BATCH_ID_COL,
    DEFAULT_MANIFEST_PATH,
    assert_zero_batch_overlap,
    assign_split_column,
    default_seed,
    load_manifest,
    load_split_config,
    save_manifest,
    split_batch_ids,
    split_counts,
    split_dataframe,
    unique_sorted_batch_ids,
)


def _ids_100() -> list[int]:
    return list(range(1, 101))


def test_split_config_seed_matches_default_yaml() -> None:
    split_cfg = load_split_config()
    default_cfg = load_default()
    assert split_cfg["seed"] == 42
    assert default_cfg["split"]["seed"] == 42
    assert split_cfg["train_batches"] == 60
    assert split_cfg["validation_batches"] == 20
    assert split_cfg["test_batches"] == 20
    assert split_cfg["unit"] == "complete_batch"
    assert default_seed() == 42
    assert split_cfg["manifest_path"] == "data/splits/manifest.json"
    assert Path(DEFAULT_MANIFEST_PATH).name == "manifest.json"


def test_exact_60_20_20_on_100_batch_fixture() -> None:
    manifest = split_batch_ids(_ids_100(), seed=42)
    assert manifest.counts == {"train": 60, "validation": 20, "test": 20}
    assert split_counts(100) == (60, 20, 20)
    assert len(manifest.train) + len(manifest.validation) + len(manifest.test) == 100


def test_zero_batch_overlap() -> None:
    manifest = split_batch_ids(_ids_100(), seed=42)
    sets = manifest.as_sets()
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert_zero_batch_overlap(manifest)


def test_union_equals_all_batch_ids() -> None:
    ids = _ids_100()
    manifest = split_batch_ids(ids, seed=42)
    assert manifest.all_ids() == set(ids)


def test_no_batch_in_more_than_one_split() -> None:
    manifest = split_batch_ids(_ids_100(), seed=7)
    seen: dict[int, str] = {}
    for name in ("train", "validation", "test"):
        for bid in getattr(manifest, name):
            assert bid not in seen
            seen[bid] = name
    assert len(seen) == 100


def test_reproducible_same_seed_same_id_set() -> None:
    a = split_batch_ids(_ids_100(), seed=42)
    b = split_batch_ids(_ids_100(), seed=42)
    assert a.train == b.train
    assert a.validation == b.validation
    assert a.test == b.test


def test_membership_independent_of_input_order_and_duplicates() -> None:
    ids = _ids_100()
    shuffled = list(reversed(ids)) + ids[:10]
    a = split_batch_ids(ids, seed=42)
    b = split_batch_ids(shuffled, seed=42)
    assert a.as_sets() == b.as_sets()


def test_different_seed_changes_membership() -> None:
    a = split_batch_ids(_ids_100(), seed=42)
    b = split_batch_ids(_ids_100(), seed=43)
    assert a.as_sets() != b.as_sets()


def test_n_not_100_uses_floor_ratio_remainder_to_test() -> None:
    ids = list(range(10))
    assert split_counts(10) == (6, 2, 2)
    manifest = split_batch_ids(ids, seed=1)
    assert manifest.counts == {"train": 6, "validation": 2, "test": 2}
    assert_zero_batch_overlap(manifest)
    assert manifest.all_ids() == set(ids)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    original = split_batch_ids(_ids_100(), seed=42)
    saved = save_manifest(original, path)
    assert saved == path
    loaded = load_manifest(path)
    assert loaded.as_sets() == original.as_sets()
    assert loaded.seed == 42
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) >= {"train", "validation", "test", "seed"}
    assert payload["counts"] == {"train": 60, "validation": 20, "test": 20}


def test_dataframe_split_is_batch_level_not_timestamp() -> None:
    rows = []
    for batch_id in _ids_100():
        for t in range(5):
            rows.append(
                {
                    BATCH_ID_COL: batch_id,
                    "timestamp": t,
                    "relative_time": float(t),
                    "value": batch_id * 10 + t,
                }
            )
    frame = pd.DataFrame(rows)
    # Row-level 60/20/20 would put timestamps of the same batch in different splits.
    assert len(frame) == 500
    manifest, parts = split_dataframe(frame, seed=42)
    assert manifest.counts == {"train": 60, "validation": 20, "test": 20}

    labeled = assign_split_column(frame, manifest)
    nunique = labeled.groupby(BATCH_ID_COL)["split"].nunique()
    assert (nunique == 1).all(), "timestamps from one batch leaked across splits"

    for name, part in parts.items():
        assert part[BATCH_ID_COL].nunique() == manifest.counts[name]
        assert set(part[BATCH_ID_COL]) == set(getattr(manifest, name))
        # All five timestamps stay with the batch.
        assert (part.groupby(BATCH_ID_COL).size() == 5).all()

    union_rows = sum(len(p) for p in parts.values())
    assert union_rows == len(frame)
    assert_zero_batch_overlap(manifest)


def test_unique_sorted_batch_ids_dedupes() -> None:
    assert unique_sorted_batch_ids([3, 1, 2, 1, 3]) == [1, 2, 3]


def test_empty_ids_raise() -> None:
    with pytest.raises(ValueError, match="at least one"):
        split_batch_ids([])
