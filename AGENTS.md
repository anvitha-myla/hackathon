# ARIP Digital Twin — Agent Notes

## Cursor Cloud specific instructions

This repository is a **Python-only** chemical-process digital twin (no frontend, database, or Docker). End-to-end development means running CLI simulation modules against bundled JSON packages.

### Prerequisites

- **Python 3.12+** with the `python3.12-venv` system package (Ubuntu: `sudo apt-get install -y python3.12-venv`).
- A virtual environment at `/workspace/.venv` (created automatically by the VM update script).

### Environment

Always run commands from the repo root (`/workspace`) with:

```bash
export PYTHONPATH=/workspace
```

Use the project venv for Python and pip:

```bash
.venv/bin/python ...
.venv/bin/pip ...
```

### Install / refresh dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -r arip_backend/requirements.txt
```

### Lint and validation

There is no project linter or test suite configured yet. Use these checks:

```bash
export PYTHONPATH=/workspace
.venv/bin/python -m compileall -q arip_backend   # syntax check
.venv/bin/pip check                               # dependency integrity
.venv/bin/python -m arip_backend.equipment_loader # validate all JSON packages
```

### Run simulations (primary workflows)

**Main pipeline** (batch `BATCH-NX-H2-001`, writes CSV/JSON under `arip_backend/simulation_output/`):

```bash
export PYTHONPATH=/workspace
.venv/bin/python -m arip_backend.simulation.pipeline
```

**Legacy closed-loop engine** (batch `batch_run_001`, STBR-5000L reactor):

```bash
export PYTHONPATH=/workspace
.venv/bin/python -m arip_backend.simulation_engine
```

### Services

| Component | Required? | Notes |
|-----------|-----------|-------|
| Python venv + pip deps | Yes | SciPy, NumPy, Pandas, Pydantic |
| Local JSON packages | Yes | Under `arip_backend/*_packages/` |
| FastAPI / Uvicorn | No | Listed in requirements but not implemented |
| Database / Docker | No | Not used |

### Gotchas

- Imports use the `arip_backend` package namespace; **`PYTHONPATH` must include the repo root** (not `arip_backend/` itself).
- `simulation_output/` is gitignored; simulations always write fresh output files there.
- `batch_nx_h2_001.json` is for `simulation.pipeline`; `batch_run_001.json` is for `simulation_engine` only.
