# Architecture Changelog

## Initial locked architecture

- Target: biomass X
- Dataset: IndPenSim
- Split: 60/20/20 batches
- Three comparison models:
  - Mechanistic-only
  - NN-only
  - Hybrid
- Hybrid ML: one residual MLP
- Reduced-order mechanistic baseline
- Causal preprocessing
- Causal feature engineering
- Soft phase information
- Mahalanobis OOD detection
- ML attenuation under OOD
- Minimal four-tab Streamlit application

## Important rejected designs

### Three separate phase NNs
Rejected for POC due unnecessary complexity and data fragmentation.

### XGBoost residual model
Not primary model because smooth neural residual modeling is more compatible with continuous biological trajectories.

### Full IndPenSim equations as physics baseline
Rejected because of circularity.

### Full-batch DTW
Rejected because it introduces lookahead leakage.

### Two-sided Savitzky-Golay
Rejected for live inference because it uses future samples.

### Static conformal prediction as final uncertainty solution
Not locked as final because temporal dependence violates simple exchangeability assumptions.

## Prompt 2 — batch split implementation

- Implemented reproducible **60/20/20 by complete `batch_id`** only (no calibration split, no timestamp-row split).
- Seed lives in `configs/split.yaml` and `configs/default.yaml` (`seed: 42`).
- Manifest written to `data/splits/manifest.json`.
- If `n != 100`: `n_train = n*60//100`, `n_val = n*20//100`, remainder to test. Exact 60/20/20 is required and tested when `n == 100`.

## Rule

Do not remove or change a locked decision without documenting why.
