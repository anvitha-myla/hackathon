"""Digital-twin simulation pipeline orchestration.

Reaction Package + Equipment Package
        │
        ▼
1. Batch Package (Initial Conditions & Feed Recipe)
        │
        ▼
2. Process Control Logic (Thermal TCU & Pressure Loops)
        │
        ▼
3. Numerical ODE Solver Engine + Event State Machine
        │
        ▼
   Time-Series Simulation Output Data
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from arip_backend.equipment_loader import (
    DEFAULT_EQUIPMENT_ROOT,
    DEFAULT_REACTION_ROOT,
    EquipmentRegistry,
    ReactionRegistry,
)
from arip_backend.schemas.batch import BatchPackage
from arip_backend.schemas.control import ControlPackage
from arip_backend.simulation.ode_engine import ODEEngine, ODEEngineConfig

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH_ROOT = _BACKEND_ROOT / "batch_packages"
DEFAULT_CONTROL_ROOT = _BACKEND_ROOT / "control_packages"
DEFAULT_OUTPUT_ROOT = _BACKEND_ROOT / "simulation_output"


def load_batch_package(path: Path | str) -> BatchPackage:
    with open(path, "r", encoding="utf-8") as fh:
        return BatchPackage.model_validate(json.load(fh))


def load_control_package(path: Path | str) -> ControlPackage:
    with open(path, "r", encoding="utf-8") as fh:
        return ControlPackage.model_validate(json.load(fh))


def resolve_control_package(
    control_id: str,
    control_root: Path | str | None = None,
) -> ControlPackage:
    root = Path(control_root) if control_root is not None else DEFAULT_CONTROL_ROOT
    for path in sorted(root.glob("**/*.json")):
        ctrl = load_control_package(path)
        if ctrl.control_id == control_id:
            return ctrl
    raise FileNotFoundError(f"Control package {control_id!r} not found under {root}")


def run_pipeline(
    batch_path: Path | str | None = None,
    *,
    equipment_root: Path | str | None = None,
    reaction_root: Path | str | None = None,
    control_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    write_output: bool = True,
) -> dict[str, Any]:
    """Execute the full digital-twin pipeline and return results."""
    batch_file = (
        Path(batch_path)
        if batch_path is not None
        else DEFAULT_BATCH_ROOT / "batch_nx_h2_001.json"
    )
    batch = load_batch_package(batch_file)
    control = resolve_control_package(batch.control_package_id, control_root)

    eq_db = EquipmentRegistry(equipment_root or DEFAULT_EQUIPMENT_ROOT)
    rxn_db = ReactionRegistry(reaction_root or DEFAULT_REACTION_ROOT)

    reactor = eq_db.get_equipment(batch.reactor_equipment_id)
    tcu = eq_db.get_equipment(batch.tcu_equipment_id)
    if reactor is None:
        raise KeyError(f"Reactor equipment {batch.reactor_equipment_id!r} not found")
    if tcu is None:
        raise KeyError(f"TCU equipment {batch.tcu_equipment_id!r} not found")

    master = rxn_db.get_master_model(batch.reaction_id)
    if master is None:
        raise KeyError(f"Reaction master {batch.reaction_id!r} not found")
    steps = rxn_db.get_step_models(batch.reaction_id)
    thermo = rxn_db.get_thermo_model()
    if thermo is None:
        raise KeyError("Thermodynamic properties package not found")

    logger.info(
        "Pipeline: batch=%s reaction=%s reactor=%s tcu=%s control=%s",
        batch.batch_id,
        batch.reaction_id,
        batch.reactor_equipment_id,
        batch.tcu_equipment_id,
        control.control_id,
    )

    engine = ODEEngine(
        batch=batch,
        control=control,
        master=master,
        steps=steps,
        thermo=thermo,
        config=ODEEngineConfig(method="BDF"),
    )
    result = engine.run()

    payload = {
        "pipeline": [
            "reaction_package+equipment_package",
            "batch_package",
            "process_control_logic",
            "ode_solver+event_state_machine",
            "timeseries_output",
        ],
        "batch_id": batch.batch_id,
        "reaction_id": batch.reaction_id,
        "reactor_equipment_id": batch.reactor_equipment_id,
        "tcu_equipment_id": batch.tcu_equipment_id,
        "control_id": control.control_id,
        "equipment_snapshot": {
            "reactor_name": reactor.get("name"),
            "tcu_name": tcu.get("name"),
        },
        "success": result["success"],
        "solver": result["solver"],
        "message": result["message"],
        "t_final_s": result["t_final_s"],
        "final_phase": result["final_phase"],
        "summary": result["summary"],
        "events": result["events"],
        "n_timeseries_points": len(result["timeseries_records"]),
        "timeseries_preview": result["timeseries_records"][:5]
        + (result["timeseries_records"][-2:] if len(result["timeseries_records"]) > 5 else []),
    }

    if write_output:
        out_root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT
        out_root.mkdir(parents=True, exist_ok=True)
        csv_path = out_root / f"{batch.batch_id}_timeseries.csv"
        json_path = out_root / f"{batch.batch_id}_result.json"
        result["timeseries"].to_csv(csv_path, index=False)
        serializable = {
            **payload,
            "timeseries_records": result["timeseries_records"],
        }
        # drop preview duplication in full dump
        serializable.pop("timeseries_preview", None)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
        payload["output_csv"] = str(csv_path)
        payload["output_json"] = str(json_path)
        logger.info("Wrote %s and %s", csv_path, json_path)

    payload["timeseries"] = result["timeseries"]
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_pipeline()
    summary = {
        k: out[k]
        for k in (
            "pipeline",
            "batch_id",
            "reaction_id",
            "success",
            "solver",
            "t_final_s",
            "final_phase",
            "summary",
            "events",
            "n_timeseries_points",
            "output_csv",
            "output_json",
        )
        if k in out
    }
    print(json.dumps(summary, indent=2))
