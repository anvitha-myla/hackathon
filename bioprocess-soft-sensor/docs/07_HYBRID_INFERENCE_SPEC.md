# Hybrid Inference Specification

## Pipeline

At every timestamp:

1. Receive current telemetry.
2. Apply causal cleaning.
3. Calculate causal features.
4. Update mechanistic model.
5. Calculate phase indicators.
6. Calculate residual NN prediction.
7. Calculate OOD distance.
8. Calculate trust factor.
9. Combine physics and ML.
10. Log result.

## Residual

delta_X_pred =
ResidualNN(features)

## OOD trust

beta_trust ∈ [0,1]

## Final prediction

X_hybrid =
X_mechanistic +
beta_trust * delta_X_pred

## Contributions

Define:

physics_contribution = X_mechanistic

ml_correction =
beta_trust * delta_X_pred

## Reference

Reference biomass may be accessed ONLY by the evaluation layer.

It must never enter:
- cleaning
- features
- mechanistic inference
- NN inference
- OOD calculation

## Sequential requirement

The inference engine must support:

step(data_t)
step(data_t+1)
step(data_t+2)

without requiring the future batch trajectory.

## Output object

Each prediction should record:

batch_id
timestamp
X_mechanistic
delta_X_raw
beta_trust
delta_X_applied
X_hybrid
phase indicators
OOD distance
latency
solver status
