# Project Context

## Problem

Bioprocesses contain hidden biological states that are difficult to measure continuously.

The soft sensor estimates hidden biomass from observable process telemetry.

## Core idea

Physics provides a stable baseline.

Machine learning learns what the simplified physics fails to capture.

Therefore:

Hybrid = Physics + Learned Residual

## Why not use full IndPenSim physics?

Because IndPenSim generates the benchmark data.

Using the complete simulator equations as the baseline would create circularity and leave little meaningful residual for ML to learn.

Therefore the baseline must intentionally be reduced.

## Research positioning

This is a proof-of-concept / benchmark study.

Do not claim:
- industrial deployment
- real plant validation
- real-time pharmaceutical control
- formal uncertainty guarantees

unless experimentally demonstrated.

## Main scientific comparison

Mechanistic-only vs NN-only vs Hybrid.

The goal is to determine whether the hybrid approach provides a meaningful improvement.
