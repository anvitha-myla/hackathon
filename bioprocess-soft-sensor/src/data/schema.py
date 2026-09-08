"""IndPenSim column schema and standardized identity fields.

Source headers are taken from the published 100-batch dump
(``100_Batches_IndPenSim_V3.csv`` / Goldrick et al., Mendeley
``pdnjz7zz5x``). Aliases include documented typos (e.g. ``concentratio``).

This module does not invent sensors. Unknown extra columns in a file are
preserved but never required. Roles that are not present stay unmapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

SOURCE_DATASET_INDPENSIM = "IndPenSim"
SOURCE_DATASET_FIXTURE = "IndPenSim_synthetic_fixture"

# Identity columns added to the standardized frame (docs/02_DATA_SPEC.md).
IDENTITY_BATCH_ID = "batch_id"
IDENTITY_TIMESTAMP = "timestamp_h"
IDENTITY_RELATIVE_TIME = "relative_time_h"
IDENTITY_SOURCE = "source_dataset"
IDENTITY_SOURCE_PATH = "source_path"

IDENTITY_COLUMNS = (
    IDENTITY_BATCH_ID,
    IDENTITY_TIMESTAMP,
    IDENTITY_RELATIVE_TIME,
    IDENTITY_SOURCE,
    IDENTITY_SOURCE_PATH,
)

# Documented published headers (process variables; Raman wavelengths omitted).
# Exact strings from the V3 CSV / process-only extracts.
PUBLISHED_PROCESS_HEADERS: tuple[str, ...] = (
    "Time (h)",
    "Aeration rate(Fg:L/h)",
    "Agitator RPM(RPM:RPM)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Heating water flow rate(Fh:L/h)",
    "Water for injection/dilution(Fw:L/h)",
    "Air head pressure(pressure:bar)",
    "Dumped broth flow(Fremoved:L/h)",
    "Substrate concentration(S:g/L)",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "Penicillin concentration(P:g/L)",
    "Vessel Volume(V:L)",
    "Vessel Weight(Wt:Kg)",
    "pH(pH:pH)",
    "Temperature(T:K)",
    "Generated heat(Q:kJ)",
    "carbon dioxide percent in off-gas(CO2outgas:%)",
    "PAA flow(Fpaa:PAA flow (L/h))",
    "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
    "Oil flow(Foil:L/hr)",
    "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "Oxygen in percent in off-gas(O2:O2  (%))",
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
    "Offline Biomass concentration(X_offline:X(g L^{-1}))",
    "Carbon evolution rate(CER:g/h)",
    "Ammonia shots(NH3_shots:kgs)",
    "Viscosity(Viscosity_offline:centPoise)",
    "Fault reference(Fault_ref:Fault ref)",
    "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)",
    "1- No Raman spec",
    "1-Raman spec recorded",
    "Batch reference(Batch_ref:Batch ref)",
    "2-PAT control(PAT_ref:PAT ref)",
    "Batch ID",
    "Fault flag",
)

# Canonical roles -> documented source header aliases (first match wins).
ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("Time (h)", "Time(h)", "Time (t: h)"),
    "batch_ref": (
        "Batch reference(Batch_ref:Batch ref)",
        "Batch reference",
        "Batch_ref",
    ),
    "pat_ref": ("2-PAT control(PAT_ref:PAT ref)",),
    "batch_id_column": ("Batch ID", "BatchID", "batch id"),
    "aeration": ("Aeration rate(Fg:L/h)",),
    "agitation": ("Agitator RPM(RPM:RPM)",),
    "sugar_feed": ("Sugar feed rate(Fs:L/h)",),
    "acid_flow": ("Acid flow rate(Fa:L/h)",),
    "base_flow": ("Base flow rate(Fb:L/h)",),
    "heating_cooling_water": ("Heating/cooling water flow rate(Fc:L/h)",),
    "heating_water": ("Heating water flow rate(Fh:L/h)",),
    "dilution_water": ("Water for injection/dilution(Fw:L/h)",),
    "air_head_pressure": ("Air head pressure(pressure:bar)",),
    "dumped_broth": ("Dumped broth flow(Fremoved:L/h)",),
    "substrate": ("Substrate concentration(S:g/L)",),
    "dissolved_oxygen": ("Dissolved oxygen concentration(DO2:mg/L)",),
    "penicillin": ("Penicillin concentration(P:g/L)",),
    "vessel_volume": ("Vessel Volume(V:L)",),
    "vessel_weight": ("Vessel Weight(Wt:Kg)",),
    "ph": ("pH(pH:pH)",),
    "vessel_temperature": ("Temperature(T:K)",),
    "generated_heat": ("Generated heat(Q:kJ)",),
    "outlet_co2": ("carbon dioxide percent in off-gas(CO2outgas:%)",),
    "paa_flow": ("PAA flow(Fpaa:PAA flow (L/h))", "PAA flow(PAA:L/h)"),
    "paa_offline": ("PAA concentration offline(PAA_offline:PAA (g L^{-1}))",),
    "oil_flow": ("Oil flow(Foil:L/hr)", "Oil flow(Oil:L/h)"),
    "nh3_offline": ("NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",),
    "our": ("Oxygen Uptake Rate(OUR:(g min^{-1}))",),
    "outlet_o2": ("Oxygen in percent in off-gas(O2:O2  (%))",),
    "penicillin_offline": (
        "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
    ),
    "biomass_reference": (
        "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
        "Offline Biomass concentration(X_offline:X(g L^{-1}))",
    ),
    "cer": ("Carbon evolution rate(CER:g/h)",),
    "ammonia_shots": ("Ammonia shots(NH3_shots:kgs)",),
    "viscosity_offline": ("Viscosity(Viscosity_offline:centPoise)",),
    "fault_reference": ("Fault reference(Fault_ref:Fault ref)",),
    "control_reference": (
        "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)",
    ),
    "raman_flag": ("1- No Raman spec", "1-Raman spec recorded"),
    "fault_flag": ("Fault flag",),
}

# Observable / live channels from the master spec, only if present in the file.
# Reference biomass is intentionally absent.
LIVE_FEATURE_ROLES: frozenset[str] = frozenset(
    {
        "aeration",
        "agitation",
        "sugar_feed",
        "acid_flow",
        "base_flow",
        "heating_cooling_water",
        "heating_water",
        "dilution_water",
        "air_head_pressure",
        "dumped_broth",
        "dissolved_oxygen",
        "ph",
        "vessel_temperature",
        "generated_heat",
        "outlet_co2",
        "outlet_o2",
        "paa_flow",
        "oil_flow",
        "our",
        "cer",
        "vessel_volume",
        "vessel_weight",
        "ammonia_shots",
        "fault_flag",
        "fault_reference",
        "control_reference",
    }
)

REFERENCE_STATE_ROLES: frozenset[str] = frozenset(
    {
        "biomass_reference",
        "penicillin",
        "penicillin_offline",
        "substrate",
        "paa_offline",
        "nh3_offline",
        "viscosity_offline",
        "vessel_volume",
    }
)

KNOWN_ROLES: frozenset[str] = frozenset(ROLE_ALIASES) | frozenset(IDENTITY_COLUMNS)

# Goldrick notebook: published V3 swapped these two header labels.
V3_SWAPPED_PAIR = (
    "Batch reference(Batch_ref:Batch ref)",
    "2-PAT control(PAT_ref:PAT ref)",
)


def normalize_header(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("^{\\mathrm{-1}}", "^-1")
    text = text.replace("^{-1}", "^-1")
    text = text.replace("–", "-")
    return "".join(ch for ch in text if ch.isalnum())


_ALIAS_NORM: dict[str, dict[str, str]] = {
    role: {normalize_header(alias): alias for alias in aliases}
    for role, aliases in ROLE_ALIASES.items()
}

_PUBLISHED_NORM: dict[str, str] = {
    normalize_header(h): h for h in PUBLISHED_PROCESS_HEADERS
}


def is_documented_header(name: str) -> bool:
    return normalize_header(name) in _PUBLISHED_NORM


def match_role(columns: list[str], role: str) -> str | None:
    if role not in _ALIAS_NORM:
        return None
    wanted = _ALIAS_NORM[role]
    by_norm = {normalize_header(c): c for c in columns}
    for key, _canonical_alias in wanted.items():
        if key in by_norm:
            return by_norm[key]
    return None


def map_roles(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for role in ROLE_ALIASES:
        hit = match_role(columns, role)
        if hit is not None:
            mapping[role] = hit
    return mapping


def live_feature_columns(column_map: Mapping[str, str]) -> dict[str, str]:
    return {
        role: col
        for role, col in column_map.items()
        if role in LIVE_FEATURE_ROLES
    }


def reference_columns(column_map: Mapping[str, str]) -> dict[str, str]:
    return {
        role: col
        for role, col in column_map.items()
        if role in REFERENCE_STATE_ROLES
    }


@dataclass
class LoadedDataset:
    """Standardized internal representation of IndPenSim (or a labeled fixture)."""

    frame: pd.DataFrame
    column_map: dict[str, str]
    source_kind: str
    source_paths: list[Path]
    notes: list[str] = field(default_factory=list)
    dataset_found: bool = False

    @property
    def n_batches(self) -> int:
        if self.frame.empty or IDENTITY_BATCH_ID not in self.frame.columns:
            return 0
        return int(self.frame[IDENTITY_BATCH_ID].nunique(dropna=True))

    @property
    def batch_ids(self) -> list:
        if self.frame.empty or IDENTITY_BATCH_ID not in self.frame.columns:
            return []
        return list(pd.unique(self.frame[IDENTITY_BATCH_ID].dropna()))
