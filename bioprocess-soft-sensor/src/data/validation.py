"""Schema and batch-boundary validation for IndPenSim ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.schema import (
    IDENTITY_BATCH_ID,
    IDENTITY_RELATIVE_TIME,
    IDENTITY_SOURCE,
    IDENTITY_TIMESTAMP,
    KNOWN_ROLES,
    LIVE_FEATURE_ROLES,
    is_documented_header,
    live_feature_columns,
)


class SchemaValidationError(ValueError):
    """Raised when a required field is invented or identity is incomplete."""


class BatchBoundaryError(ValueError):
    """Raised when batch_id mixing would break complete-batch isolation."""


def validate_required_roles(required: Iterable[str]) -> None:
    """Reject invented role names that are not in the documented schema."""
    unknown = [r for r in required if r not in KNOWN_ROLES]
    if unknown:
        raise SchemaValidationError(
            "Unknown/invented required roles (not in IndPenSim schema): "
            + ", ".join(unknown)
        )


def validate_required_headers(
    columns: Sequence[str],
    required_headers: Iterable[str],
) -> None:
    """Reject required source headers that are neither documented nor present."""
    present = {str(c) for c in columns}
    bad: list[str] = []
    for header in required_headers:
        if header in present:
            continue
        if is_documented_header(header):
            bad.append(f"{header!r} (documented IndPenSim header missing from file)")
        else:
            bad.append(f"{header!r} (not a documented IndPenSim column)")
    if bad:
        raise SchemaValidationError(
            "Schema validation rejected required columns: " + "; ".join(bad)
        )


def validate_identity(frame: pd.DataFrame) -> None:
    missing = [
        c
        for c in (
            IDENTITY_BATCH_ID,
            IDENTITY_TIMESTAMP,
            IDENTITY_RELATIVE_TIME,
            IDENTITY_SOURCE,
        )
        if c not in frame.columns
    ]
    if missing:
        raise SchemaValidationError(f"Standardized frame missing identity columns: {missing}")
    if frame[IDENTITY_BATCH_ID].isna().any():
        raise BatchBoundaryError("batch_id contains nulls; batch boundaries are not preserved")
    if frame[IDENTITY_TIMESTAMP].isna().any():
        raise SchemaValidationError("timestamp_h contains nulls")


def validate_batch_boundaries(frame: pd.DataFrame) -> None:
    """Each batch_id is a single contiguous block; times increase within a batch."""
    if frame.empty:
        raise BatchBoundaryError("empty frame")
    validate_identity(frame)
    batch = frame[IDENTITY_BATCH_ID].to_numpy()
    time = pd.to_numeric(frame[IDENTITY_TIMESTAMP], errors="coerce").to_numpy()

    seen: dict[object, int] = {}
    prev = object()
    for i, bid in enumerate(batch):
        key = bid
        if key != prev:
            if key in seen:
                raise BatchBoundaryError(
                    f"batch_id {bid!r} is split across non-contiguous blocks "
                    f"(first at row {seen[key]}, again at row {i})"
                )
            seen[key] = i
            prev = key

    dup = frame.duplicated(subset=[IDENTITY_BATCH_ID, IDENTITY_TIMESTAMP], keep=False)
    if dup.any():
        n = int(dup.sum())
        raise BatchBoundaryError(
            f"{n} rows share a (batch_id, timestamp_h) pair; timestamps are not unique per batch"
        )

    for bid, grp in frame.groupby(IDENTITY_BATCH_ID, sort=False):
        t = pd.to_numeric(grp[IDENTITY_TIMESTAMP], errors="coerce")
        if t.isna().any():
            raise SchemaValidationError(f"non-numeric timestamps in batch {bid!r}")
        deltas = np.diff(t.to_numpy())
        if np.any(deltas <= 0):
            raise BatchBoundaryError(
                f"timestamps are not strictly increasing within batch_id={bid!r}"
            )
        rel = pd.to_numeric(grp[IDENTITY_RELATIVE_TIME], errors="coerce")
        expected = t - t.iloc[0]
        if not np.allclose(rel.to_numpy(), expected.to_numpy(), equal_nan=True):
            raise BatchBoundaryError(
                f"relative_time_h is inconsistent with timestamp_h in batch_id={bid!r}"
            )


def validate_live_subset_excludes_biomass(column_map: dict[str, str]) -> None:
    live = live_feature_columns(column_map)
    if "biomass_reference" in live:
        raise SchemaValidationError(
            "biomass_reference must not be part of the live feature subset"
        )
    overlap = set(live) & {"biomass_reference"}
    if overlap:
        raise SchemaValidationError(f"live subset leaked reference roles: {overlap}")
    if "biomass_reference" in LIVE_FEATURE_ROLES:
        raise SchemaValidationError("LIVE_FEATURE_ROLES must not include biomass_reference")


def assert_biomass_not_required_for_live(
    column_map: dict[str, str],
    frame: pd.DataFrame,
) -> list[str]:
    """Biomass may be identified for training/eval; live columns must still load without it."""
    validate_live_subset_excludes_biomass(column_map)
    live_cols = [column_map[r] for r in live_feature_columns(column_map)]
    missing_live = [c for c in live_cols if c not in frame.columns]
    if missing_live:
        raise SchemaValidationError(f"mapped live columns missing from frame: {missing_live}")
    notes: list[str] = []
    if "biomass_reference" in column_map:
        notes.append(
            f"Reference biomass identified as {column_map['biomass_reference']!r} "
            "but is not required for the live feature subset."
        )
    else:
        notes.append(
            "Reference biomass column was not found; live feature subset still valid."
        )
    return notes
