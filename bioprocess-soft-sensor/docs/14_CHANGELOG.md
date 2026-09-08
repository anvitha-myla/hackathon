# Architecture Changelog

## Prompt 4 — causal feature engine

- Source of truth: `src/features/feature_engine.py`, `registry.py`, `derivatives.py`, `stoichiometry.py`. `engineering.py` is a wrap.
- Feature **definitions** are fixed in the registry; **values** update each timestamp.
- Implemented from IndPenSim-available columns: OUR/CER (off-gas inert balance, dataset OUR/CER fallback), RQ with near-zero OUR guard, dDO/dt, cumulative sugar/oil feed, measured + integrated reactor volume (no evaporation column), causal progress (cumulative feed, not DTW), lags 1/5/10, backward rolling mean/std/slope, soft phase indicators for the **single** residual MLP.
- Not implemented (missing IndPenSim sensors, not invented): ΔT (no jacket temperature), dτ/dt (no shaft torque). Mechanistic outputs wait for Prompt 5.
- Live feature tables exclude biomass, penicillin, substrate, and Raman bins. Leakage tests mutate a future raw sample and require features at t to be unchanged.

## Prompt 11 — robustness / stress-testing environment

- Stress testing is a **separate** environment from demo inference (`src/evaluation/stress.py`, `configs/stress.yaml`, `tests/test_stress.py`).
- Controlled perturbations (in-memory copies only): random noise, spikes, missing values, impossible values, sensor drift, distribution shift, feed interruption, temperature disturbance, combined faults, plus a severe-OOD shift.
- Clean final-test files are never overwritten; records/plots/tables go under `results/stress`.
- Each row records prediction, reference, error, OOD distance, `beta_trust`, ML contribution, and physics contribution.
- Expected fallback (tested, including with hybrid/OOD stubs): normal → ML correction active; severe OOD → `beta_trust` → 0 and `X_hybrid` → `X_mechanistic`; corrupted sensors must not crash.

## Prompt 7 — residual MLP implemented

- Residual training target remains `delta_X = X_reference - X_mechanistic`.
- One PyTorch MLP: Dense 64-ReLU → 32-ReLU → 16-ReLU → scalar `delta_X`.
- Scaler, feature order, and weights are fit/selected using train/validation batches only.
- Still rejected: three phase-specific NNs, XGBoost residual, LSTM/Transformer.

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

## Prompt 5 — reduced mechanistic baseline

Implemented reduced-order Monod / Luedeking-Piret sequential model (`solve_ivp` BDF with Radau fallback). Primary live output is biomass `X_mechanistic`. Substrate and optional product are internal states. Dissolved oxygen is an exogenous measurement (no full IndPenSim kLa/OUR oxygen balance). Live `step()` rejects reference-biomass keys and does not initialize from future reference X.

## Prompt 3 — Causal cleaning (implemented)

- Cleaning runs strictly causally: at time t only `x(t), x(t-1), …` are used.
- Components: bounded forward-fill, missingness indicators, backward Hampel (~5 min conceptual window), causal EMA / one-sided MA, physical-plausibility flags.
- Raw values, cleaned values, and quality flags are all retained. Impossible readings are flagged, not silently deleted.
- Constants live in `configs/cleaning.yaml` and are not estimated from test batches.
- Two-sided Savitzky–Golay, full-batch smoothing, and future-aware interpolation are not used.

## Prompt 9 — Mahalanobis OOD

- Spec source of truth: `src/inference/ood.py`.
- Fit `mean_train` / `covariance_train` on **training features only** (Ledoit–Wolf + ridge + `pinv`). Test/validation `split=` is rejected; non-train `batch_id` rows are dropped.
- `D_M = sqrt((phi - mean)^T Sigma^{-1} (phi - mean))`.
- `beta_trust = exp(-kappa * max(0, D_M - D_threshold))`, clipped to `[0, 1]`. `D_threshold` defaults to the train D_M 0.95 quantile when unset.
- Hybrid (one timestamp, sequential): `X_hybrid = X_mechanistic + beta_trust * delta_X_pred`. Injected into Prompt 8 via `MahalanobisTrustHook` without changing `SequentialHybridEngine.step(observation)`. `apply_hybrid` is the equation helper. `src/monitoring/ood.py` re-exports this module (including `beta_trust`).
- `src/monitoring/ood.py` re-exports the inference module (shim).
- Invalid/NaN features: `beta_trust = 0` (physics fallback). No production model training.

## Prompt 8 — sequential hybrid inference

- Added `src/inference/pipeline.py`, `state.py`, `result.py` (plus `trust.py` hook and `components.py` adapters).
- Live equation: `X_hybrid = X_mechanistic + beta_trust * delta_X_pred` with `beta_trust = 1.0` until a fitted OOD hook is injected.
- Does not reimplement Mahalanobis. `trust_hook` / `MahalanobisTrustHook` wraps Prompt 9 `MahalanobisOOD` when fitted.
- Reference biomass is stripped before cleaning/features/mechanistic/residual; evaluation may attach it after `step()`.
- Sequential API: `step(observation)` one timestamp; `run_sequential` only iterates `step` (no prefetch).
- Wires to causal `clean_series`, `FeatureEngine` (history through t only), `ReducedMechanisticModel`, and residual artifacts when present; otherwise hold/zero stubs.

## Prompt 12 — computational monitoring

- Implemented measured CPU/RAM (`psutil`) and stage latencies (`time.perf_counter`) in `src/monitoring/system.py` and `src/monitoring/timing.py`.
- Sequential inference loop records cleaning, feature-engineering, mechanistic solver, NN, OOD, and total inference latency, plus throughput, serialized model size, and `nn.Module` parameter count.
- Sub-150 ms remains a **benchmark target to measure against**, not a hardcoded or claimed latency.

## Prompt 10 — evaluation framework

- Implemented A vs B vs C metrics (RMSE, MAE, R², phase-wise, early/late, trajectory stability, OOD-stratified error, latency) in `src/evaluation/`.
- `run_final_test()` refuses if test `batch_id`s appear in train/scaler/OOD artifacts (`test_split_used=true` also fails closed).
- Final 20-batch IndPenSim scoring is **not** claimed when data or trained models are missing. Hybrid is not assumed to win.
- Plots label the simulator series **IndPenSim Reference** (not physical ground truth).
- NN-only: thin `src/models/nn_only.py` interface only; Prompt 6 training was not executed.

## Prompt 1 — ingestion (no architecture change)

- Added batch-aware IndPenSim loader (`src/data/loader.py`), documented-header schema (`src/data/schema.py`), and validation (`src/data/validation.py`).
- Reference biomass is identified when present and excluded from the live feature subset.
- Official 100-batch dump is not shipped; `configs/data.yaml` `raw_path` points at `data/raw/`.
- No model training.

## Rule

Do not remove or change a locked decision without documenting why.
