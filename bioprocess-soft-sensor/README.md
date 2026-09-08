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

Prompt 9: Mahalanobis OOD in `src/inference/ood.py` (`beta_trust`, train-only fit). Sequential hybrid combine is `X_hybrid = X_mechanistic + beta_trust * delta_X_pred` (`src/inference/engine.py` `hybrid_step`). No production training; OOD is not fit on test batches.

Residual NN-only training, full hybrid telemetry pipeline, evaluation, stress tests, and Streamlit tab logic may still be later prompts.

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

## IndPenSim data (Prompt 1)

The 100-batch dump is **not** in git (~2.5 GB with Raman). It is a **simulator**, not physical industrial telemetry.

1. Download from [Mendeley `pdnjz7zz5x`](https://data.mendeley.com/datasets/pdnjz7zz5x/1) (CC BY 4.0).
2. Put files in [`data/raw/`](data/raw/) (for example `100_Batches_IndPenSim_V3.csv`).
3. Path is configured in [`configs/data.yaml`](configs/data.yaml) (`raw_path`).

Without those files, the loader falls back to a **labeled synthetic fixture** (`tests/fixtures/indpensim_synthetic_fixture.csv`) that uses published IndPenSim column names only.

```bash
PYTHONPATH=. python -m src.data.inspect
```

writes `reports/indpensim_ingestion_report.md`.
