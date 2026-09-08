# Bioprocess Hybrid Soft Sensor — Master Specification

## 1. Project Purpose

Build a computer-science-heavy prototype of a hybrid mechanistic + machine-learning soft sensor for estimating hidden biomass concentration in a fed-batch bioprocess.

The prototype uses IndPenSim simulated batches as the controlled benchmark/reference dataset.

IMPORTANT:
- IndPenSim is a simulator, not physical ground-truth industrial data.
- Do not claim the public dataset is real industrial telemetry.
- The simulator historically relates to real industrial fermentation data, but this prototype evaluates against simulated reference trajectories.
- The system must be designed so the reference biomass is hidden during live inference.

## 2. Primary Research Question

Can a residual neural network combined with a deliberately reduced mechanistic model estimate hidden biomass more accurately and robustly than either:
1. a standalone mechanistic model, or
2. a standalone neural network?

## 3. Three Models Being Compared

### Model A — Mechanistic-only
Reduced-order biological/kinetic model.

### Model B — NN-only
Standalone MLP predicting biomass directly from observable/derived process features.

### Model C — Hybrid
Reduced mechanistic model + residual MLP.

The residual MLP learns:

delta_X = X_reference - X_mechanistic

During training.

Final hybrid estimate:

X_hybrid = X_mechanistic + beta_trust * delta_X_predicted

## 4. Target

Primary target:
- Biomass concentration X

Future extension:
- Substrate S
- Product P

Do NOT implement S/P prediction as primary functionality unless explicitly requested.

## 5. Dataset Strategy

Use 100 IndPenSim batches.

Split by complete batch:

- 60 training batches
- 20 validation batches
- 20 final test batches

NEVER split individual timestamps from the same batch across train/validation/test.

The final 20 test batches must remain untouched during model development.

## 6. Critical Anti-Leakage Rule

The entire pipeline must be causal.

At timestamp t, the live system may only use:
- current measurements
- previous measurements
- previously computed state
- previously accumulated quantities

It must NEVER use future timestamps.

Do not use:
- two-sided filters
- full-batch DTW
- future-aware interpolation
- future-derived normalization
- test-batch statistics
- reference biomass during live inference

## 7. Architecture

Raw process telemetry
        ↓
Batch-aware ingestion
        ↓
Causal data cleaning
        ↓
Feature engineering
        ↓
Reduced mechanistic model
        ↓
Phase information
        ↓
Residual MLP
        ↓
OOD trust factor
        ↓
Hybrid biomass estimate

The NN is ONE residual MLP.

Do NOT create three separate phase-specific NNs.

Do NOT use XGBoost as the primary residual model.

## 8. Phase Handling

Phase information may be provided to the single residual MLP.

Initial conceptual phases:
- Growth / trophophase
- Production / idiophase
- Autolysis

Phase detection must not cause hard model switching.

Use continuous/soft phase indicators rather than routing between separate neural networks.

The feature definitions remain fixed across the batch, while their numerical values change over time.

## 9. Observable Inputs

Initial observable process channels include:

- pH
- dissolved oxygen (DO)
- vessel temperature
- jacket temperature
- agitation / torque
- substrate/feed rate
- gas flow
- outlet CO2
- outlet O2

Use only variables actually available in the IndPenSim dataset.

Do not invent unavailable sensors.

## 10. Derived Features

Candidate derived features:

- OUR
- CER
- RQ
- causal progress coordinate
- ΔT = vessel temperature - jacket temperature
- dτ/dt
- dDO/dt
- cumulative substrate/feed influx
- reactor broth volume
- mechanistic growth-rate output
- mechanistic product-rate output
- causal lag features
- causal rolling statistics

Feature definitions remain mathematically fixed.
Feature values vary with time and process phase.

## 11. Mechanistic Model

Do NOT reproduce the full IndPenSim mathematical model as the baseline.

This would create circularity because IndPenSim generates the benchmark data.

Use a deliberately reduced-order model based on:
- Monod-style growth
- substrate limitation
- oxygen limitation
- decay
- substrate consumption
- product formation where appropriate
- volume/feed effects

The purpose is to create a physically meaningful but imperfect baseline.

The residual NN should learn the systematic gap between the reduced physics and the IndPenSim reference.

## 12. Neural Network

Use a small MLP.

Initial architecture:

Input
↓
Dense 64
↓
ReLU
↓
Dense 32
↓
ReLU
↓
Dense 16
↓
ReLU
↓
Output

For NN-only:
Output = predicted biomass X

For Hybrid:
Output = predicted residual delta_X

Architecture/hyperparameters may be tuned using the validation set.

Do not unnecessarily introduce:
- LSTM
- Transformer
- Neural ODE
- large deep networks

unless explicitly requested as later experiments.

## 13. Residual Training

For each training timestamp with valid reference biomass:

delta_X = X_reference - X_mechanistic

The residual MLP learns delta_X.

The model is NOT learning biomass directly in the hybrid branch.

The NN-only baseline separately learns biomass directly.

## 14. OOD Protection

Use feature-space OOD detection.

Initial method:
Mahalanobis distance.

Compare incoming feature vector against the training distribution.

Calculate:

D_M = Mahalanobis distance

Then:

beta_trust =
exp(-kappa * max(0, D_M - D_threshold))

Normal data:
beta ≈ 1

Severe OOD:
beta → 0

Final hybrid:

X_hybrid =
X_mechanistic + beta_trust * delta_X_predicted

This causes the system to fall back toward physics when ML confidence should be reduced.

## 15. Uncertainty

Static conformal prediction is NOT the final preferred architecture because fermentation telemetry is temporally dependent.

The long-term design may use adaptive/conformal PID-style uncertainty.

For the initial POC:
- implement uncertainty in a modular way
- do not falsely claim formal 95% coverage unless it has actually been demonstrated
- make conformal/adaptive uncertainty a replaceable module

## 16. Physical Constraints

Avoid blindly applying post-hoc clipping to claim statistical guarantees.

Where physical constraints are required, prefer:
- constrained training
- safe loss penalties
- physically valid mechanistic states
- controlled fallback

At minimum:
- biomass cannot be negative
- mechanistic states must remain numerically stable

## 17. Evaluation

Compare:

1. Mechanistic-only
2. NN-only
3. Hybrid

Metrics:
- RMSE
- MAE
- R²
- phase-wise RMSE/MAE
- early-stage error
- late-stage error
- trajectory stability
- OOD performance
- inference latency
- CPU usage
- RAM usage

The main research claim must be based on measured results, not assumptions.

The hybrid model is NOT assumed to win.

## 18. Stress Testing

Separate from normal demo inference.

Test:
- random noise
- sensor spikes
- missing values
- impossible values
- sensor drift
- distribution shift
- feed interruptions
- temperature disturbances
- extreme process conditions
- multiple simultaneous faults

Record:
- prediction
- reference
- error
- OOD score
- beta_trust
- phase indicators
- mechanistic contribution
- ML contribution

## 19. Live Inference

During live/test inference:

The model sees:
- current/past process signals
- derived causal features
- mechanistic state/output

It does NOT see:
- reference biomass
- future timestamps
- test-batch statistics

Reference biomass is used only after prediction to calculate evaluation error and display benchmark plots.

## 20. Application

Minimal four-tab application.

### Tab 1 — Data Cleaning
Show:
- selected batch
- raw/clean signal preview
- cleaning status
- basic quality statistics

### Tab 2 — Feature Engineering
Show:
- feature names
- source
- formula/definition
- model destination

Do not overload the application with biological explanations.

### Tab 3 — Live Inference
Show:
- mechanistic biomass
- NN-only biomass
- hybrid biomass
- IndPenSim reference benchmark
- error
- OOD status
- beta trust
- ML contribution
- physics contribution

Main visual:
trajectory comparison graph.

### Tab 4 — Computational Monitor
Show:
- CPU
- RAM
- inference latency
- throughput
- cleaning latency
- feature-engineering latency
- mechanistic-model latency
- NN latency
- total pipeline latency
- model size

## 21. Software Principles

Prioritize:
- reproducibility
- modularity
- batch isolation
- causal processing
- deterministic behavior
- testability
- logging
- configuration-driven parameters
- clear separation of training and inference
- no hidden data leakage

Every important transformation should be independently testable.

## 22. Cursor Rules

Before changing code:
1. Read this file.
2. Read the relevant specification file.
3. Inspect existing implementation.
4. Do not overwrite working functionality unnecessarily.
5. Run tests after implementation.
6. Fix failures before moving to the next stage.
7. Update docs/14_CHANGELOG.md for architectural changes.
8. Never silently change a research decision.
9. If a requested change conflicts with this specification, explain the conflict before implementing it.

## 23. Definition of Done

The project is complete only when:

- dataset loads correctly
- batches are isolated
- 60/20/20 split is reproducible
- preprocessing is causal
- features are reproducible
- mechanistic baseline runs
- NN-only baseline runs
- residual NN runs
- hybrid inference runs sequentially
- OOD layer works
- evaluation produces metrics
- stress testing works separately
- computational monitoring works
- UI displays all three model trajectories
- tests pass
- results are reproducible
