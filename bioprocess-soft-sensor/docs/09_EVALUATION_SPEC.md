# Evaluation Specification

## Models

Compare:

A. Mechanistic-only
B. NN-only
C. Hybrid

## Primary metrics

RMSE
MAE
R²

## Additional metrics

Phase-wise RMSE
Phase-wise MAE
Early-stage error
Late-stage error
Trajectory stability
OOD error
Inference latency
CPU
RAM

## Main visualization

For selected test batches:

Reference biomass
Mechanistic biomass
NN-only biomass
Hybrid biomass

on the same trajectory plot.

## Aggregate visualization

Show model performance across all 20 test batches.

## Leakage audit

Verify:
- no test scaling
- no future features
- no cross-batch contamination
- no reference biomass in live inference
- no future-aware filtering

## Research claim

Do not assume hybrid wins.

Report actual results.

Possible conclusions:
- hybrid improves accuracy
- hybrid improves robustness
- hybrid improves OOD behavior
- hybrid does not improve enough to justify complexity

All are valid research outcomes.

## Final test

The final 20 batches are untouched until this stage.
