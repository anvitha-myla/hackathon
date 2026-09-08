"""Prompt 1: IndPenSim ingestion (no training)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    DatasetNotFoundError,
    apply_published_v3_header_swap,
    identify_batch_id,
    load_dataset,
    load_fixture,
    load_indpensim,
    load_tables,
)
from src.data.schema import (
    IDENTITY_BATCH_ID,
    LIVE_FEATURE_ROLES,
    SOURCE_DATASET_FIXTURE,
    live_feature_columns,
    map_roles,
)
from src.data.validation import (
    SchemaValidationError,
    assert_biomass_not_required_for_live,
    validate_batch_boundaries,
    validate_required_headers,
    validate_required_roles,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "indpensim_synthetic_fixture.csv"
)


def test_load_fixture_identifies_batches_and_time() -> None:
    loaded = load_fixture(FIXTURE)
    assert loaded.source_kind == SOURCE_DATASET_FIXTURE
    assert loaded.dataset_found is False
    assert loaded.n_batches == 3
    assert set(loaded.batch_ids) == {"1", "2", "3"}
    assert loaded.frame["timestamp_h"].notna().all()
    assert loaded.column_map["time"] == "Time (h)"
    counts = loaded.frame.groupby(IDENTITY_BATCH_ID).size().to_dict()
    assert counts == {"1": 12, "2": 10, "3": 8}


def test_batch_boundaries_preserved() -> None:
    loaded = load_fixture(FIXTURE)
    validate_batch_boundaries(loaded.frame)
    order = list(loaded.frame[IDENTITY_BATCH_ID])
    compressed = [order[0]]
    for bid in order[1:]:
        if bid != compressed[-1]:
            compressed.append(bid)
    assert compressed == ["1", "2", "3"]
    assert len(set(compressed)) == len(compressed)


def test_schema_rejects_invented_requirements() -> None:
    loaded = load_fixture(FIXTURE)
    with pytest.raises(SchemaValidationError, match="invented"):
        validate_required_roles(["jacket_temperature_invented", "magic_sensor"])
    with pytest.raises(SchemaValidationError, match="not a documented IndPenSim column"):
        validate_required_headers(
            list(loaded.frame.columns),
            ["Unobtainium probe (g/L)", "Fake industrial PLC tag"],
        )


def test_reference_biomass_identified_but_not_required_for_live() -> None:
    loaded = load_fixture(FIXTURE)
    assert "biomass_reference" in loaded.column_map
    assert "biomass_reference" not in LIVE_FEATURE_ROLES
    live = live_feature_columns(loaded.column_map)
    assert "biomass_reference" not in live
    notes = assert_biomass_not_required_for_live(loaded.column_map, loaded.frame)
    assert any("not required for the live feature subset" in n for n in notes)
    live_cols = [loaded.column_map[r] for r in live]
    subset = loaded.frame[live_cols]
    assert not subset.empty
    bio = loaded.column_map["biomass_reference"]
    without_bio = loaded.frame.drop(columns=[bio])
    assert bio not in without_bio.columns
    assert all(c in without_bio.columns for c in live_cols)


def test_load_dataset_falls_back_when_raw_missing(tmp_path: Path) -> None:
    empty = tmp_path / "raw"
    empty.mkdir()
    loaded = load_dataset(empty, fallback_fixture=True)
    assert loaded.dataset_found is False
    assert loaded.n_batches == 3
    assert any("not found" in n.lower() for n in loaded.notes)


def test_load_indpensim_errors_if_missing(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        load_indpensim(tmp_path / "missing-raw", allow_missing=False)


def test_filename_batch_id(tmp_path: Path) -> None:
    src = pd.read_csv(FIXTURE, comment="#")
    src = src[src["Batch reference(Batch_ref:Batch ref)"] == 1].copy()
    src = src.drop(columns=["Batch reference(Batch_ref:Batch ref)", "Batch ID"])
    one = src[src["Time (h)"] <= 0.8].copy()
    path = tmp_path / "batch_07.csv"
    one.to_csv(path, index=False)
    loaded = load_tables(
        [path],
        source_dataset="IndPenSim",
        infer_from_time_resets=True,
        apply_v3_swap=False,
    )
    assert set(loaded.batch_ids) == {"7"}


def test_v3_header_swap_uses_pat_column_as_batch_ref() -> None:
    src = pd.read_csv(FIXTURE, comment="#")
    true_batch = src["Batch reference(Batch_ref:Batch ref)"].copy()
    src["2-PAT control(PAT_ref:PAT ref)"] = true_batch
    src["Batch reference(Batch_ref:Batch ref)"] = 0  # mislabeled PAT-like values
    swapped, applied = apply_published_v3_header_swap(src)
    assert applied is True
    assert swapped["Batch reference(Batch_ref:Batch ref)"].tolist() == true_batch.tolist()
    batches, method = identify_batch_id(swapped, infer_from_time_resets=True)
    assert method.startswith("column:")
    assert set(pd.Series(batches).astype(str)) == {"1", "2", "3"}


def test_time_reset_batch_inference_when_id_columns_are_junk() -> None:
    src = pd.read_csv(FIXTURE, comment="#")
    src["Batch reference(Batch_ref:Batch ref)"] = 0
    src["Batch ID"] = range(len(src))
    batches, method = identify_batch_id(src, infer_from_time_resets=True)
    assert method == "time_index_reset"
    assert pd.Series(batches).nunique() == 3


def test_map_roles_does_not_invent_jacket_temperature() -> None:
    loaded = load_fixture(FIXTURE)
    roles = map_roles(list(loaded.frame.columns))
    assert "jacket_temperature" not in roles
    assert "vessel_temperature" in roles
