# Feature Engineering Specification

## Principle

Feature definitions are fixed.

Feature values are dynamic.

The same feature formula is evaluated at every timestamp.

## Base process variables

Use variables actually present in IndPenSim.

Candidate channels:

pH
DO
vessel temperature
jacket temperature
agitation/torque
feed/substrate rate
gas flow
outlet O2
outlet CO2

## Derived features

### OUR
Oxygen uptake rate from off-gas balance.

### CER
Carbon evolution rate from off-gas balance.

### RQ

RQ = CER / OUR

Handle near-zero OUR safely.

### Delta temperature

Delta_T = T_vessel - T_jacket

### Torque derivative

Backward derivative of torque.

### DO derivative

Backward derivative of DO.

### Cumulative substrate/feed

Causal cumulative integral.

### Reactor volume

Causal volume calculation using feed and evaporation terms where supported.

### Progress coordinate

Prefer a causal cumulative process-progress variable rather than full-batch DTW.

### Mechanistic features

Possible:
- mechanistic biomass
- mechanistic substrate
- mechanistic product
- mechanistic growth rate
- mechanistic product rate

Clearly label these as model-derived features.

## History features

Use only past/current values.

Candidate:
- lag 1
- lag 5
- lag 10
- rolling mean
- rolling standard deviation
- rolling slope

All windows must be backward-looking.

## Phase features

Provide soft phase indicators.

Initial conceptual phases:
- growth
- production
- autolysis

Do not create separate NNs.

## Feature registry

Every feature must have:

name
source
unit
formula
causal=True/False
model_usage
missing-data behavior
