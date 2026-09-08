# Software Testing Specification

## Unit tests

Test:
- data loading
- batch splitting
- causal filtering
- missing-value handling
- spike detection
- derivatives
- OUR
- CER
- RQ
- cumulative feed
- mechanistic equations
- solver
- NN forward pass
- residual calculation
- OOD distance
- beta trust
- hybrid equation

## Leakage tests

Explicitly test that:
- future timestamps cannot enter feature calculations
- test batches cannot affect scalers
- test batches cannot affect OOD statistics
- reference biomass cannot enter inference features

## Integration tests

Test:

raw data
→ cleaning
→ features
→ mechanistic
→ residual NN
→ OOD
→ hybrid

## Failure tests

Test:
- NaN
- missing columns
- empty batch
- corrupted input
- solver failure
- invalid sensor values

## Reproducibility

Same:
- dataset
- seed
- configuration

should produce reproducible results within documented numerical tolerance.
