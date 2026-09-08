# OOD and Safety Specification

## Purpose

Detect process conditions unlike those seen during training.

## Initial method

Mahalanobis distance.

Fit training feature distribution:

mean_train
covariance_train

Use only training data.

## Distance

D_M =
sqrt(
(phi - mean)^T
Sigma^-1
(phi - mean)
)

Use numerically stable covariance handling.

## Trust factor

beta_trust =
exp(
-kappa * max(0, D_M - D_threshold)
)

## Behavior

In-distribution:

beta ≈ 1

Moderately OOD:

beta decreases

Severely OOD:

beta approaches 0

## Fallback

When OOD becomes severe:

X_hybrid approaches X_mechanistic.

Do not allow the ML correction to grow uncontrollably.

## Logging

Record:
- Mahalanobis distance
- threshold
- beta
- OOD state
- fallback state

## Future extension

Adaptive uncertainty / conformal PID may be added later.

Do not claim formal coverage until experimentally validated.
