# Bioprocess Hybrid Soft Sensor

Prototype hybrid **mechanistic + residual MLP** soft sensor for estimating **hidden biomass** on **IndPenSim** simulated fed-batch trajectories.

IndPenSim is a **simulator / reference dataset**, not physical industrial telemetry. Live inference must hide reference biomass.

## Read this first

Cursor and other agents **must read [`docs/00_MASTER_SPEC.md`](docs/00_MASTER_SPEC.md) before changing code.** Numbered files `01`–`13` own each layer. Architecture changes go in [`docs/14_CHANGELOG.md`](docs/14_CHANGELOG.md).

Agent rules: [`AGENTS.md`](AGENTS.md) and [`.cursor/rules/00-master-spec.mdc`](.cursor/rules/00-master-spec.mdc).

## Locked architecture (Prompt 0)

| Decision | Value |
| --- | --- |
| Target | Biomass \(X\) only |
| Data | 100 IndPenSim batches |
| Split | 60 / 20 / 20 **by complete batch** |
| Models | A mechanistic-only, B NN-only MLP, C hybrid (reduced mech + residual MLP) |
| Residual MLP | 64-ReLU → 32-ReLU → 16-ReLU → output |
| Hybrid | \(X_\mathrm{hybrid} = X_\mathrm{mech} + \beta_\mathrm{trust}\,\Delta X\) |
| OOD | Mahalanobis + \(\beta_\mathrm{trust}\); physics fallback |
| UI | Four Streamlit tabs |
| Processing | Causal (no future leakage) |

Prompt 0: packages, configs, specs, and import smoke tests.

Prompt 2: reproducible **60/20/20 batch_id split** (`src/data/split.py`, `configs/split.yaml`, leakage tests). Seed `42`. Manifest path `data/splits/manifest.json`. No model training.

Residual training, NN-only training, hybrid inference, OOD, evaluation, stress tests, and Streamlit tab logic are **not** implemented yet.

## Layout

```
bioprocess-soft-sensor/
├── docs/                 # locked specification (source of truth)
├── src/                  # packages (stubs in Prompt 0)
│   ├── data/
│   ├── features/
│   ├── mechanistic/
│   ├── models/
│   ├── inference/
│   ├── evaluation/
│   └── monitoring/
├── app/                  # Streamlit package placeholder
├── tests/
├── configs/              # YAML; start with configs/default.yaml
├── data/{raw,processed,splits}/
├── models/
├── results/
├── notebooks/
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest
```

## Config

[`configs/default.yaml`](configs/default.yaml) encodes the locked split, biomass target, MLP widths, OOD placeholders, and four-tab UI. Load with `src.config.load_default()`.
