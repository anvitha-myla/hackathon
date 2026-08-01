"""Equipment and reaction package loader for the ARIP Digital Twin.

Recursively scans ``equipment_packages/`` (and optionally ``reaction_packages/``),
validates JSON against Pydantic schemas, and exposes an in-memory registry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from arip_backend.schemas.equipment import parse_equipment_package
from arip_backend.schemas.reaction import (
    ReactionMasterPackage,
    ReactionPackage,
    ThermodynamicPropertiesPackage,
    parse_reaction_document,
)

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_EQUIPMENT_ROOT = _BACKEND_ROOT / "equipment_packages"
DEFAULT_REACTION_ROOT = _BACKEND_ROOT / "reaction_packages"


class EquipmentRegistry:
    """In-memory registry of validated equipment package JSON specifications."""

    def __init__(
        self,
        packages_dir: str | Path | None = None,
        *,
        validate: bool = True,
        auto_load: bool = True,
    ) -> None:
        self.packages_dir = Path(packages_dir) if packages_dir is not None else DEFAULT_EQUIPMENT_ROOT
        # equipment_id -> {"path": str, "data": dict, "model": BaseModel | None}
        self.registry: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, str]] = []
        self.validate = validate
        if auto_load:
            self.load_all_packages()

    def load_all_packages(self) -> None:
        """Recursively scan the directory and load all equipment JSON files."""
        self.registry.clear()
        self.errors.clear()

        if not self.packages_dir.exists():
            print(f"Directory '{self.packages_dir}' not found.")
            logger.warning("Equipment packages directory not found: %s", self.packages_dir)
            return

        json_files = sorted(self.packages_dir.glob("**/*.json"))
        print(f"Found {len(json_files)} equipment JSON files. Loading...")
        logger.info("Found %d equipment JSON files under %s", len(json_files), self.packages_dir)

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"JSON root must be an object, got {type(data).__name__}")

                eq_id = data.get("equipment_id", file_path.stem)
                model: BaseModel | None = None
                if self.validate:
                    model = parse_equipment_package(data)

                if eq_id in self.registry:
                    raise ValueError(f"Duplicate equipment_id={eq_id!r}")

                self.registry[eq_id] = {
                    "path": str(file_path),
                    "data": data,
                    "model": model,
                }
            except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.errors.append({"path": str(file_path), "error": message})
                print(f"Failed to parse {file_path}: {exc}")
                logger.error("Failed to parse %s: %s", file_path, message)

        print(f"Successfully loaded {len(self.registry)} equipment packages into registry.\n")
        logger.info(
            "Loaded %d equipment packages (%d errors)",
            len(self.registry),
            len(self.errors),
        )

    def get_equipment(self, equipment_id: str) -> Optional[dict[str, Any]]:
        """Retrieve dynamic equipment configuration by ID."""
        item = self.registry.get(equipment_id)
        return item["data"] if item else None

    def get_model(self, equipment_id: str) -> Optional[BaseModel]:
        """Retrieve the validated Pydantic model for an equipment ID."""
        item = self.registry.get(equipment_id)
        return item["model"] if item else None

    def list_by_module(self, module_name: str) -> dict[str, dict[str, Any]]:
        """Retrieve all equipment matching a specific module substring."""
        needle = module_name.lower()
        return {
            eq_id: item["data"]
            for eq_id, item in self.registry.items()
            if needle in item["data"].get("module", "").lower()
        }

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable registry summary."""
        return {
            "equipment_count": len(self.registry),
            "equipment_ids": sorted(self.registry),
            "errors": list(self.errors),
            "modules": sorted(
                {item["data"].get("module", "") for item in self.registry.values() if item["data"].get("module")}
            ),
        }


# Module-level registry populated on import / explicit reload for digital-twin services.
REGISTRY: dict[str, Any] = {
    "equipment": {},
    "reactions": {},
    "reaction_masters": {},
    "reaction_steps": {},
    "thermodynamics": {},
    "errors": [],
    "sources": {},
    "equipment_registry": None,
    "reaction_registry": None,
}


class ReactionRegistry:
    """In-memory registry for master, step, and thermodynamic reaction packages."""

    def __init__(
        self,
        packages_dir: str | Path | None = None,
        *,
        validate: bool = True,
        auto_load: bool = True,
    ) -> None:
        self.packages_dir = Path(packages_dir) if packages_dir is not None else DEFAULT_REACTION_ROOT
        self.registry: dict[str, dict[str, Any]] = {}
        self.masters: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}
        self.thermodynamics: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, str]] = []
        self.validate = validate
        if auto_load:
            self.load_all_packages()

    def load_all_packages(self) -> None:
        """Recursively scan reaction_packages/ and load all JSON documents."""
        self.registry.clear()
        self.masters.clear()
        self.steps.clear()
        self.thermodynamics.clear()
        self.errors.clear()

        if not self.packages_dir.exists():
            print(f"Directory '{self.packages_dir}' not found.")
            logger.warning("Reaction packages directory not found: %s", self.packages_dir)
            return

        json_files = sorted(self.packages_dir.glob("**/*.json"))
        print(f"Found {len(json_files)} reaction JSON files. Loading...")
        logger.info("Found %d reaction JSON files under %s", len(json_files), self.packages_dir)

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"JSON root must be an object, got {type(data).__name__}")

                model = parse_reaction_document(data, source_name=str(file_path)) if self.validate else None

                if isinstance(model, ReactionMasterPackage) or data.get("package_type") == "reaction_master":
                    key = data.get("reaction_id", file_path.stem)
                    bucket = self.masters
                    kind = "master"
                elif isinstance(model, ThermodynamicPropertiesPackage) or data.get("package_type") == "thermodynamic_properties":
                    key = data.get("package_id", file_path.stem)
                    bucket = self.thermodynamics
                    kind = "thermodynamics"
                else:
                    key = data.get("reaction_id", file_path.stem)
                    bucket = self.steps
                    kind = "step"

                if key in self.registry:
                    raise ValueError(f"Duplicate reaction document id={key!r}")

                entry = {"path": str(file_path), "data": data, "model": model, "kind": kind}
                self.registry[key] = entry
                bucket[key] = entry
            except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.errors.append({"path": str(file_path), "error": message})
                print(f"Failed to parse {file_path}: {exc}")
                logger.error("Failed to parse %s: %s", file_path, message)

        print(
            f"Successfully loaded {len(self.registry)} reaction documents "
            f"({len(self.masters)} masters, {len(self.steps)} steps, "
            f"{len(self.thermodynamics)} thermo packages).\n"
        )

    def get_reaction(self, reaction_id: str) -> Optional[dict[str, Any]]:
        """Retrieve reaction JSON by ID (master or step)."""
        item = self.registry.get(reaction_id)
        return item["data"] if item else None

    def list_steps(self, parent_reaction_id: str) -> dict[str, dict[str, Any]]:
        """Return elementary steps belonging to a master reaction."""
        return {
            rid: item["data"]
            for rid, item in self.steps.items()
            if item["data"].get("parent_reaction_id") == parent_reaction_id
        }

    def summary(self) -> dict[str, Any]:
        return {
            "document_count": len(self.registry),
            "master_ids": sorted(self.masters),
            "step_ids": sorted(self.steps),
            "thermo_ids": sorted(self.thermodynamics),
            "errors": list(self.errors),
        }


def load_reaction_packages(root: Path | str | None = None) -> dict[str, Any]:
    """Scan reaction_packages/ into REGISTRY via ReactionRegistry."""
    rxn_db = ReactionRegistry(root, validate=True, auto_load=True)
    REGISTRY["reaction_registry"] = rxn_db

    for key, item in rxn_db.masters.items():
        REGISTRY["reaction_masters"][key] = item["model"] or item["data"]
        REGISTRY["reactions"][key] = item["model"] or item["data"]
        REGISTRY["sources"][key] = item["path"]
        logger.info("Loaded reaction master %s from %s", key, item["path"])

    for key, item in rxn_db.steps.items():
        REGISTRY["reaction_steps"][key] = item["model"] or item["data"]
        REGISTRY["reactions"][key] = item["model"] or item["data"]
        REGISTRY["sources"][key] = item["path"]
        logger.info("Loaded reaction step %s from %s", key, item["path"])

    for key, item in rxn_db.thermodynamics.items():
        REGISTRY["thermodynamics"][key] = item["model"] or item["data"]
        REGISTRY["sources"][key] = item["path"]
        logger.info("Loaded thermo package %s from %s", key, item["path"])

    for err in rxn_db.errors:
        REGISTRY["errors"].append({"path": err["path"], "kind": "reaction", "error": err["error"]})

    return REGISTRY["reactions"]


def load_all_packages(
    equipment_root: Path | str | None = None,
    reaction_root: Path | str | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load equipment + reaction packages into :data:`REGISTRY`."""
    REGISTRY["equipment"] = {}
    REGISTRY["reactions"] = {}
    REGISTRY["reaction_masters"] = {}
    REGISTRY["reaction_steps"] = {}
    REGISTRY["thermodynamics"] = {}
    REGISTRY["errors"] = []
    REGISTRY["sources"] = {}

    eq_db = EquipmentRegistry(equipment_root, validate=validate, auto_load=True)
    REGISTRY["equipment_registry"] = eq_db
    for eq_id, item in eq_db.registry.items():
        REGISTRY["equipment"][eq_id] = item["data"]
        REGISTRY["sources"][eq_id] = item["path"]
    for err in eq_db.errors:
        REGISTRY["errors"].append({"path": err["path"], "kind": "equipment", "error": err["error"]})

    load_reaction_packages(reaction_root)
    return REGISTRY


def get_equipment(equipment_id: str) -> dict[str, Any]:
    """Fetch equipment JSON data from the module-level registry."""
    if equipment_id not in REGISTRY["equipment"]:
        raise KeyError(f"Equipment {equipment_id!r} not found in registry")
    return REGISTRY["equipment"][equipment_id]


def get_reaction(reaction_id: str) -> Any:
    """Fetch a validated reaction master or step from the registry."""
    if reaction_id not in REGISTRY["reactions"]:
        raise KeyError(f"Reaction {reaction_id!r} not found in registry")
    return REGISTRY["reactions"][reaction_id]


def registry_summary() -> dict[str, Any]:
    """Return a JSON-serialisable summary of the current module registry."""
    eq_db: EquipmentRegistry | None = REGISTRY.get("equipment_registry")
    rxn_db: ReactionRegistry | None = REGISTRY.get("reaction_registry")
    modules = eq_db.summary()["modules"] if eq_db else []
    return {
        "equipment_count": len(REGISTRY["equipment"]),
        "reaction_count": len(REGISTRY["reactions"]),
        "reaction_master_count": len(REGISTRY["reaction_masters"]),
        "reaction_step_count": len(REGISTRY["reaction_steps"]),
        "thermo_count": len(REGISTRY["thermodynamics"]),
        "equipment_ids": sorted(REGISTRY["equipment"]),
        "reaction_ids": sorted(REGISTRY["reactions"]),
        "reaction_master_ids": sorted(REGISTRY["reaction_masters"]),
        "reaction_step_ids": sorted(REGISTRY["reaction_steps"]),
        "thermo_ids": sorted(REGISTRY["thermodynamics"]),
        "modules": modules,
        "errors": list(REGISTRY["errors"]),
        "sources": dict(REGISTRY["sources"]),
        "reaction_registry_summary": rxn_db.summary() if rxn_db else {},
    }


# --- QUICK TEST / DEMONSTRATION ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Initialize Registry (default: arip_backend/equipment_packages)
    eq_db = EquipmentRegistry(DEFAULT_EQUIPMENT_ROOT)

    # Fetch a specific reactor package
    reactor_data = eq_db.get_equipment("EQ-PBR-100")
    if reactor_data:
        print("--- Loaded Specific Equipment ---")
        print(f"ID: {reactor_data['equipment_id']}")
        print(f"Name: {reactor_data['name']}")
        print(f"Max Operating Pressure: {reactor_data['limits']['max_operating_pressure_bar']} bar")

    # Query all Calorimetry Module tools
    print("\n--- Module 6 (Calorimetry) Inventory ---")
    calorimetry_suite = eq_db.list_by_module("Module 6")
    for eq_id, spec in calorimetry_suite.items():
        print(f" • [{eq_id}] {spec['name']}")

    # Load reactions
    rxn_db = ReactionRegistry(DEFAULT_REACTION_ROOT)
    master = rxn_db.get_reaction("RXN-NITROXYLENE-H2-MASTER")
    if master:
        print("\n--- Nitroxylene Hydrogenation Master ---")
        print(f"ID: {master['reaction_id']}")
        print(f"Name: {master['name']}")
        print(f"Steps: {len(master.get('steps', []))}")
        print(f"Overall ΔH_rxn: {master['overall_kinetics']['delta_H_rxn_J_mol']} J/mol")

    print("\n--- Reaction Steps ---")
    for rid, spec in rxn_db.list_steps("RXN-NITROXYLENE-H2-MASTER").items():
        print(f" • [{rid}] {spec['name']} | Ea={spec['kinetics']['Ea_J_mol']} J/mol")

    # Full registry summary
    load_all_packages()
    print("\n--- Registry Summary ---")
    print(json.dumps(registry_summary(), indent=2))
    if REGISTRY["errors"]:
        raise SystemExit(1)
