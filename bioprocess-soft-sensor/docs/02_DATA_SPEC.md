# Data Specification

## Dataset

IndPenSim simulated penicillin fermentation dataset.

Use 100 batches.

## Split

60 train
20 validation
20 final test

Split by batch_id.

## Rule

No timestamps from one batch may appear in multiple splits.

## Training data

Used for:
- fitting scalers
- fitting preprocessing parameters
- fitting mechanistic parameters where appropriate
- training NN
- fitting OOD distribution

## Validation data

Used for:
- hyperparameter tuning
- threshold selection
- early stopping
- model selection

## Test data

Used exactly once for final evaluation.

Do not fit anything on test data.

## Reference

IndPenSim reference biomass is available for:
- supervised training
- validation
- final evaluation

It must not be supplied to the live inference feature vector.

## Batch identity

Every row must retain:
- batch_id
- timestamp
- relative time
- source dataset identifier

Batch boundaries must never be lost.
