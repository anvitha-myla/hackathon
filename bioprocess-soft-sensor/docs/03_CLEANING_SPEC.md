# Causal Data Cleaning Specification

## Objective

Clean process telemetry without using future information.

## Sampling

Target grid:
1-minute intervals where supported by the dataset.

## Required principles

At time t:
clean(t) may depend only on:
x(t), x(t-1), x(t-2), ...

Never x(t+1) or later.

## Cleaning components

### 1. Missing values

Use causal methods only.

Possible:
- forward fill with bounded duration
- causal interpolation where explicitly safe
- missingness indicators

Never interpolate using future data during live inference.

### 2. Spike detection

Use a backward-looking rolling Hampel-style detector where appropriate.

Initial conceptual window:
5 minutes.

Use robust statistics.

### 3. Causal smoothing

Use:
- causal EMA
- causal one-sided filter

Do not use standard two-sided Savitzky-Golay.

### 4. Physical plausibility

Flag impossible sensor values.

Do not silently erase observations.

Maintain:
- cleaned value
- original value
- quality flag

### 5. Cleaning audit

For every signal retain:
- raw
- cleaned
- quality status
- transformation metadata

## No global leakage

Any fitted preprocessing parameters must be learned only from training batches.

Test data must never influence them.
