# Computational Monitoring Specification

Measure actual application performance.

## System metrics

CPU usage
RAM usage

## Pipeline metrics

Data cleaning latency
Feature engineering latency
Mechanistic solver latency
NN inference latency
OOD latency
Total prediction latency

## Throughput

Samples processed per second.

## Model metrics

Model file size
Parameter count

## UI

Display current:
- CPU
- RAM
- latency
- throughput

Also provide historical values for the current inference session.

## Runtime requirement

The system should comfortably process the intended sampling interval.

The original architecture proposed a sub-150 ms execution target for the mechanistic loop; treat this as a benchmark target to measure, not as an already-achieved claim.
