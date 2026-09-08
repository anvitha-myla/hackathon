# Reduced Mechanistic Model Specification

## Purpose

Provide a physically meaningful baseline that is intentionally imperfect.

## Important

Do NOT implement the complete IndPenSim model.

Use a reduced-order model.

## States

Primary:
X = biomass

Optional internal states:
S = substrate
P = product

## Model family

Reduced Monod / Luedeking-Piret style model.

Include, where supported:
- growth
- substrate limitation
- oxygen limitation
- biomass decay
- substrate consumption
- product formation
- feed dilution effects

## Growth

Use a Monod-style formulation.

Potential form:

mu(S, DO) =
mu_max *
S/(K_s + S) *
DO/(K_DO + DO)

Optional substrate inhibition can be implemented if supported by the dataset and validated.

## Biomass

dX/dt =
mu X - k_d X - (F/V)X

## Substrate

dS/dt =
feed contribution
- consumption
- maintenance
- dilution

## Product

dP/dt =
q_p X - (F/V)P

Only implement P if required.

## Solver

Use scipy.integrate.solve_ivp.

Start with:
- BDF
or
- Radau

Profile runtime.

## Initial conditions

Initial state values must come from:
- known batch conditions
- documented assumptions
- or a short observable-data initialization procedure

Never initialize the live model using future reference biomass.

## Outputs

At minimum:

X_mechanistic

Also expose:
- growth rate
- substrate state if implemented
- product state if implemented
- solver status
- solver latency

## Stability

Detect:
- solver failure
- divergence
- NaN
- negative physical states
- excessive runtime

Never hide solver failure.
