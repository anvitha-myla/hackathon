# Application UI Specification

## Design principle

Minimal, professional, visual.

Do not put the entire scientific explanation into the UI.

Technical explanations belong in documentation/PPT.

## Tab 1 — Data Cleaning

Components:
- batch selector
- raw signal preview
- cleaned signal preview
- quality status
- missing values
- outlier count

## Tab 2 — Feature Engineering

Show a feature map.

Columns:

Feature
Source
Formula
Unit
Model usage

Example:

OUR
Off-gas O2
gas balance
rate
NN + analysis

Mechanistic biomass
Mechanistic model
ODE output
concentration
Hybrid NN

## Tab 3 — Live Inference

Main graph:

Reference
Mechanistic
NN-only
Hybrid

Show:
- current biomass
- prediction error
- phase
- OOD status
- beta trust
- ML correction
- physics estimate

The reference is labeled clearly as:
"IndPenSim Reference"

Do not label it "real-world ground truth."

## Tab 4 — Computational Monitor

Show:

CPU
RAM
Latency
Throughput

Pipeline breakdown:

Cleaning
Features
Physics
NN
OOD
Total

## UI technology

Streamlit for the initial prototype.

Keep business logic outside the UI.
