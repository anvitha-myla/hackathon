"""Equipment and reaction package loader for the ARIP Digital Twin.

Recursively scans ``equipment_packages/`` and ``reaction_packages/``, validates
every ``.json`` file against the Pydantic schemas in ``schemas/``, and exposes
an in-memory registry dictionary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arip_backend.schemas.equipment import (
    EquipmentModel,
    EquipmentType,
    parse_equipment_package,
)
from arip_backend.schemas.reaction import ReactionPackage

logger = logging.getLogger(__name__)

# Default roots relative to this file (arip_backend/)
_BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_EQUIPMENT_ROOT = _BACKEND_ROOT / "equipment_packages"
DEFAULT_REACTION_ROOT = _BACKEND_ROOT / "reaction_packages"

# In-memory registry populated by :func:`load_all_packages`.
# Structure:
#   {
#     "equipment": {equipment_id: validated_model, ...},
#     "reactions": {reaction_id: validated_model, ...},
#     "by_type": {equipment_type_value: [equipment_id, ...], ...},
#     "errors": [{"path": str, "error": str}, ...],
#     "sources": {id: relative_path_str, ...},
#   }
REGISTRY: dict[str, Any] = {
    "equipment": {},
    "reactions": {},
    "by_type": {t.value: [] for t in EquipmentType},
    "errors": [],
    "sources": {},
}


def _iter_json_files(root: Path) -> list[Path]:
    """Return all ``*.json`` files under *root*, sorted for deterministic load order."""
    if not root.exists():
        logger.warning("Package root does not exist: %s", root)
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object, got {type(data).__name__}")
    return data


def _relative_source(path: Path, backend_root: Path = _BACKEND_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(backend_root))
    except ValueError:
        return str(path)


def load_equipment_packages(
    root: Path | str | None = None,
    *,
    registry: dict[str, Any] | None = None,
) -> dict[str, EquipmentModel]:
    """Scan *root* for equipment JSON packages and validate them.

    Returns a dict keyed by ``equipment_id``. Validation failures are recorded
    in ``registry["errors"]`` when a registry is provided (or the module-level
    :data:`REGISTRY`).
    """
    root_path = Path(root) if root is not None else DEFAULT_EQUIPMENT_ROOT
    reg = registry if registry is not None else REGISTRY
    loaded: dict[str, EquipmentModel] = {}

    for path in _iter_json_files(root_path):
        source = _relative_source(path)
        try:
            raw = _load_json(path)
            package = parse_equipment_package(raw)
            eq_id = package.equipment_id
            if eq_id in loaded or eq_id in reg["equipment"]:
                raise ValueError(f"Duplicate equipment_id={eq_id!r} at {source}")
            loaded[eq_id] = package
            reg["equipment"][eq_id] = package
            reg["by_type"][package.equipment_type.value].append(eq_id)
            reg["sources"][eq_id] = source
            logger.info("Loaded equipment %s from %s", eq_id, source)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            reg["errors"].append({"path": source, "kind": "equipment", "error": message})
            logger.error("Failed to load equipment package %s: %s", source, message)

    return loaded


def load_reaction_packages(
    root: Path | str | None = None,
    *,
    registry: dict[str, Any] | None = None,
) -> dict[str, ReactionPackage]:
    """Scan *root* for reaction JSON packages and validate them.

    Returns a dict keyed by ``reaction_id``.
    """
    root_path = Path(root) if root is not None else DEFAULT_REACTION_ROOT
    reg = registry if registry is not None else REGISTRY
    loaded: dict[str, ReactionPackage] = {}

    for path in _iter_json_files(root_path):
        source = _relative_source(path)
        try:
            raw = _load_json(path)
            package = ReactionPackage.model_validate(raw)
            rxn_id = package.reaction_id
            if rxn_id in loaded or rxn_id in reg["reactions"]:
                raise ValueError(f"Duplicate reaction_id={rxn_id!r} at {source}")
            loaded[rxn_id] = package
            reg["reactions"][rxn_id] = package
            reg["sources"][rxn_id] = source
            logger.info("Loaded reaction %s from %s", rxn_id, source)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            reg["errors"].append({"path": source, "kind": "reaction", "error": message})
            logger.error("Failed to load reaction package %s: %s", source, message)

    return loaded


def clear_registry(registry: dict[str, Any] | None = None) -> None:
    """Reset the in-memory registry to an empty state."""
    reg = registry if registry is not None else REGISTRY
    reg["equipment"] = {}
    reg["reactions"] = {}
    reg["by_type"] = {t.value: [] for t in EquipmentType}
    reg["errors"] = []
    reg["sources"] = {}


def load_all_packages(
    equipment_root: Path | str | None = None,
    reaction_root: Path | str | None = None,
    *,
    clear: bool = True,
) -> dict[str, Any]:
    """Load all equipment and reaction packages into :data:`REGISTRY`.

    Parameters
    ----------
    equipment_root, reaction_root:
        Optional overrides for package directories.
    clear:
        If True (default), wipe the registry before loading.

    Returns
    -------
    dict
        The module-level :data:`REGISTRY` after loading.
    """
    if clear:
        clear_registry()

    load_equipment_packages(equipment_root, registry=REGISTRY)
    load_reaction_packages(reaction_root, registry=REGISTRY)

    n_eq = len(REGISTRY["equipment"])
    n_rxn = len(REGISTRY["reactions"])
    n_err = len(REGISTRY["errors"])
    logger.info(
        "Registry ready: %d equipment, %d reactions, %d errors",
        n_eq,
        n_rxn,
        n_err,
    )
    return REGISTRY


def get_equipment(equipment_id: str) -> EquipmentModel:
    """Fetch a validated equipment package from the registry."""
    try:
        return REGISTRY["equipment"][equipment_id]
    except KeyError as exc:
        raise KeyError(f"Equipment {equipment_id!r} not found in registry") from exc


def get_reaction(reaction_id: str) -> ReactionPackage:
    """Fetch a validated reaction package from the registry."""
    try:
        return REGISTRY["reactions"][reaction_id]
    except KeyError as exc:
        raise KeyError(f"Reaction {reaction_id!r} not found in registry") from exc


def registry_summary() -> dict[str, Any]:
    """Return a JSON-serialisable summary of the current registry."""
    return {
        "equipment_count": len(REGISTRY["equipment"]),
        "reaction_count": len(REGISTRY["reactions"]),
        "equipment_ids": sorted(REGISTRY["equipment"]),
        "reaction_ids": sorted(REGISTRY["reactions"]),
        "by_type": {k: list(v) for k, v in REGISTRY["by_type"].items()},
        "errors": list(REGISTRY["errors"]),
        "sources": dict(REGISTRY["sources"]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_all_packages()
    summary = registry_summary()
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)
