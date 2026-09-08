"""Batch-aware IndPenSim loader. No training, no derived ML features."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_yaml
from src.data.schema import (
    IDENTITY_BATCH_ID,
    IDENTITY_RELATIVE_TIME,
    IDENTITY_SOURCE,
    IDENTITY_SOURCE_PATH,
    IDENTITY_TIMESTAMP,
    SOURCE_DATASET_FIXTURE,
    SOURCE_DATASET_INDPENSIM,
    V3_SWAPPED_PAIR,
    LoadedDataset,
    map_roles,
    match_role,
)
from src.data.validation import (
    assert_biomass_not_required_for_live,
    validate_batch_boundaries,
    validate_identity,
)

TABLE_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}
BATCH_NAME_RE = re.compile(
    r"(?:batch|run)[_\-\s]*0*(\d+)",
    re.IGNORECASE,
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the configured IndPenSim path has no loadable tables."""


def _data_cfg() -> dict[str, Any]:
    try:
        return load_yaml("data.yaml")
    except FileNotFoundError:
        return {}


def resolve_raw_dir(raw_path: str | Path | None = None) -> Path:
    cfg = _data_cfg()
    if raw_path is not None:
        path = Path(raw_path)
    else:
        path = Path(cfg.get("raw_path") or "data/raw")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_fixture_path(fixture_path: str | Path | None = None) -> Path:
    cfg = _data_cfg()
    if fixture_path is not None:
        path = Path(fixture_path)
    else:
        path = Path(
            cfg.get("fixture_path") or "tests/fixtures/indpensim_synthetic_fixture.csv"
        )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def discover_tables(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    found: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in TABLE_SUFFIXES:
            found.append(path)
    return found


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, comment="#")
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise DatasetNotFoundError(f"Unsupported table type: {path}")


def apply_published_v3_header_swap(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Apply Goldrick notebook rename when both swapped headers are present.

    Official note: batch reference of the published dump is stored under
    ``2-PAT control(PAT_ref:PAT ref)`` until the two names are exchanged.
    """
    a, b = V3_SWAPPED_PAIR
    if a in frame.columns and b in frame.columns:
        out = frame.rename(columns={a: b, b: a})
        return out, True
    return frame, False


def _run_lengths(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)
    lengths: list[int] = []
    run = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return np.asarray(lengths, dtype=int)


def _time_reset_labels(time: pd.Series) -> np.ndarray | None:
    t = pd.to_numeric(time, errors="coerce").to_numpy()
    if t.size == 0 or np.all(np.isnan(t)):
        return None
    resets = np.where(np.diff(t.astype(float)) < -1e-9)[0] + 1
    n_batches = int(len(resets) + 1)
    if n_batches < 2:
        return None
    labels = np.empty(len(t), dtype=object)
    bounds = np.r_[0, resets, len(t)]
    for i in range(len(bounds) - 1):
        labels[bounds[i] : bounds[i + 1]] = i + 1
    return labels


def _candidate_looks_like_batch_id(
    series: pd.Series,
    n_time_reset_batches: int | None,
) -> bool:
    values = series.to_numpy()
    n_rows = len(values)
    nunique = pd.Series(values).nunique(dropna=True)
    if nunique < 2 or n_rows == 0:
        return False
    if nunique > max(2, int(0.25 * n_rows)):
        return False
    lengths = _run_lengths(pd.Series(values).astype(str).fillna("__na__").to_numpy())
    if lengths.size == 0 or float(np.median(lengths)) < 3:
        return False
    if n_time_reset_batches is not None and n_time_reset_batches >= 3:
        if nunique < max(2, int(0.5 * n_time_reset_batches)):
            return False
        if nunique > n_time_reset_batches * 3:
            return False
    return True


def _batch_id_from_path(path: Path) -> str | None:
    for part in (path.stem, path.parent.name):
        m = BATCH_NAME_RE.search(part)
        if m:
            return str(int(m.group(1)))
    return None


def identify_batch_id(
    frame: pd.DataFrame,
    *,
    source_path: Path | None = None,
    infer_from_time_resets: bool = True,
) -> tuple[pd.Series, str]:
    """Return (batch_id series, method used)."""
    columns = list(frame.columns)
    time_col = match_role(columns, "time")
    n_reset = None
    reset_labels = None
    if time_col is not None:
        reset_labels = _time_reset_labels(frame[time_col])
        if reset_labels is not None:
            n_reset = int(pd.Series(reset_labels).nunique())

    # Goldrick: after swap, Batch reference is the batch index (1..100).
    ordered_roles = ("batch_ref", "pat_ref", "batch_id_column")
    for role in ordered_roles:
        col = match_role(columns, role)
        if col is None:
            continue
        series = frame[col]
        if _candidate_looks_like_batch_id(series, n_reset):
            return series.astype(str), f"column:{col}"

    if source_path is not None:
        from_name = _batch_id_from_path(source_path)
        if from_name is not None:
            return pd.Series([from_name] * len(frame), index=frame.index), (
                f"path:{source_path.name}"
            )

    if infer_from_time_resets and reset_labels is not None:
        return pd.Series(reset_labels, index=frame.index).astype(str), (
            "time_index_reset"
        )

    if len(frame) > 0:
        fallback = _batch_id_from_path(source_path) if source_path else None
        if fallback is None:
            fallback = "1"
        return pd.Series([str(fallback)] * len(frame), index=frame.index), (
            "single_file_fallback"
        )
    raise DatasetNotFoundError("Cannot identify batch_id in empty table")


def identify_time(frame: pd.DataFrame) -> pd.Series:
    col = match_role(list(frame.columns), "time")
    if col is None:
        raise DatasetNotFoundError(
            "No timestamp column found. Expected documented header 'Time (h)'."
        )
    return pd.to_numeric(frame[col], errors="coerce")


def _standardize_frame(
    frame: pd.DataFrame,
    *,
    source_path: Path,
    source_dataset: str,
    infer_from_time_resets: bool,
    apply_v3_swap: bool,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    notes: list[str] = []
    work = frame.copy()
    if apply_v3_swap:
        work, swapped = apply_published_v3_header_swap(work)
        if swapped:
            notes.append(
                "Applied published V3 header swap between "
                "'Batch reference(Batch_ref:Batch ref)' and "
                "'2-PAT control(PAT_ref:PAT ref)' (Goldrick notebook)."
            )
    column_map = map_roles(list(work.columns))
    batch_ids, method = identify_batch_id(
        work,
        source_path=source_path,
        infer_from_time_resets=infer_from_time_resets,
    )
    notes.append(f"batch_id identified via {method}")
    timestamps = identify_time(work)
    if timestamps.isna().any():
        raise DatasetNotFoundError(f"Non-numeric Time (h) values in {source_path}")

    out = work.copy()
    out[IDENTITY_BATCH_ID] = batch_ids.astype(str).to_numpy()
    out[IDENTITY_TIMESTAMP] = timestamps.to_numpy()
    out[IDENTITY_SOURCE] = source_dataset
    out[IDENTITY_SOURCE_PATH] = str(source_path)
    pieces = []
    for _, grp in out.groupby(IDENTITY_BATCH_ID, sort=False):
        pieces.append(grp[IDENTITY_TIMESTAMP] - grp[IDENTITY_TIMESTAMP].iloc[0])
    out[IDENTITY_RELATIVE_TIME] = pd.concat(pieces).reindex(out.index)

    id_front = [
        IDENTITY_BATCH_ID,
        IDENTITY_TIMESTAMP,
        IDENTITY_RELATIVE_TIME,
        IDENTITY_SOURCE,
        IDENTITY_SOURCE_PATH,
    ]
    rest = [c for c in out.columns if c not in id_front]
    out = out[id_front + rest]
    return out, column_map, notes


def _concat_preserving_batch_blocks(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def load_tables(
    paths: list[Path],
    *,
    source_dataset: str,
    infer_from_time_resets: bool = True,
    apply_v3_swap: bool = True,
) -> LoadedDataset:
    frames: list[pd.DataFrame] = []
    maps: list[dict[str, str]] = []
    notes: list[str] = []
    for path in paths:
        raw = _read_table(path)
        std, column_map, extra_notes = _standardize_frame(
            raw,
            source_path=path,
            source_dataset=source_dataset,
            infer_from_time_resets=infer_from_time_resets,
            apply_v3_swap=apply_v3_swap,
        )
        frames.append(std)
        maps.append(column_map)
        notes.extend([f"{path.name}: {n}" for n in extra_notes])

    if len(frames) > 1:
        used: set[str] = set()
        for i, std in enumerate(frames):
            bids = std[IDENTITY_BATCH_ID].astype(str)
            overlap = used.intersection(set(bids.unique()))
            if overlap:
                stem = paths[i].stem
                std[IDENTITY_BATCH_ID] = stem + ":" + bids
                notes.append(
                    f"{paths[i].name}: prefixed batch_id with {stem!r} to avoid colliding ids {sorted(overlap)}"
                )
            used.update(std[IDENTITY_BATCH_ID].astype(str).unique())
            frames[i] = std
    frame = _concat_preserving_batch_blocks(frames)
    column_map: dict[str, str] = {}
    for m in maps:
        column_map.update(m)
    validate_identity(frame)
    validate_batch_boundaries(frame)
    notes.extend(assert_biomass_not_required_for_live(column_map, frame))
    kind = (
        SOURCE_DATASET_FIXTURE
        if source_dataset == SOURCE_DATASET_FIXTURE
        else SOURCE_DATASET_INDPENSIM
    )
    return LoadedDataset(
        frame=frame,
        column_map=column_map,
        source_kind=kind,
        source_paths=list(paths),
        notes=notes,
        dataset_found=source_dataset == SOURCE_DATASET_INDPENSIM,
    )


def load_indpensim(
    raw_path: str | Path | None = None,
    *,
    allow_missing: bool = False,
) -> LoadedDataset:
    """Load IndPenSim tables from the configured raw directory.

    Does not invent columns. Does not train models.
    """
    cfg = _data_cfg()
    raw_dir = resolve_raw_dir(raw_path)
    tables = discover_tables(raw_dir)
    infer = bool(cfg.get("infer_batches_from_time_resets", True))
    swap = bool(cfg.get("apply_published_v3_batch_ref_pat_swap", True))
    if not tables:
        msg = (
            f"IndPenSim dataset not found under {raw_dir}. "
            "Place the published files (e.g. 100_Batches_IndPenSim_V3.csv from "
            "https://data.mendeley.com/datasets/pdnjz7zz5x/1 ) in data/raw/. "
            "The dump is a simulator reference, not physical industrial telemetry."
        )
        if allow_missing:
            return LoadedDataset(
                frame=pd.DataFrame(),
                column_map={},
                source_kind="not_found",
                source_paths=[],
                notes=[msg],
                dataset_found=False,
            )
        raise DatasetNotFoundError(msg)
    return load_tables(
        tables,
        source_dataset=SOURCE_DATASET_INDPENSIM,
        infer_from_time_resets=infer,
        apply_v3_swap=swap,
    )


def load_fixture(fixture_path: str | Path | None = None) -> LoadedDataset:
    path = resolve_fixture_path(fixture_path)
    if not path.is_file():
        raise DatasetNotFoundError(
            f"Synthetic fixture not found at {path}. "
            "Expected tests/fixtures/indpensim_synthetic_fixture.csv"
        )
    cfg = _data_cfg()
    infer = bool(cfg.get("infer_batches_from_time_resets", True))
    loaded = load_tables(
        [path],
        source_dataset=SOURCE_DATASET_FIXTURE,
        infer_from_time_resets=infer,
        apply_v3_swap=False,
    )
    loaded.notes.insert(
        0,
        "SYNTHETIC FIXTURE: rows are labeled fake data using published "
        "IndPenSim-like column names only. Not industrial measurements and not "
        "the official 100-batch simulator dump.",
    )
    loaded.dataset_found = False
    return loaded


def load_dataset(
    raw_path: str | Path | None = None,
    *,
    fallback_fixture: bool = True,
) -> LoadedDataset:
    """Prefer real files in data/raw/; optionally fall back to the labeled fixture."""
    try:
        return load_indpensim(raw_path, allow_missing=False)
    except DatasetNotFoundError as err:
        if not fallback_fixture:
            raise
        loaded = load_fixture()
        loaded.notes.insert(0, str(err))
        return loaded
