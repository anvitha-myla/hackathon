# Neural Network Specification

## Two ML baselines

### NN-only

Inputs:
observable and derived causal features.

Target:
reference biomass X.

### Residual NN

Inputs:
observable/derived causal features plus mechanistic outputs and phase information where appropriate.

Target:

delta_X =
X_reference - X_mechanistic

## Architecture

Initial MLP:

Input
↓
Linear/Dense 64
↓
ReLU
↓
Linear/Dense 32
↓
ReLU
↓
Linear/Dense 16
↓
ReLU
↓
Output

Keep the network small.

## Framework

PyTorch.

## Scaling

Fit feature scaling only on training batches.

Persist:
- scaler
- feature ordering
- feature names

## Training

Use:
- training batches for optimization
- validation batches for model selection

Use:
- early stopping
- reproducible random seeds

## Loss

Initial:
MSE or Huber loss.

Residual target:

delta_X = X_reference - X_mechanistic

The final hybrid prediction is not the training target of the residual network.

## Output

Residual model:
delta_X_pred

NN-only:
X_NN_pred

## Persistence

Save:
- model weights
- architecture configuration
- feature list
- scaler
- training metadata
- random seed
- validation metrics

## Important

Do not create:
- three phase-specific networks
- large deep networks
- LSTM
- Transformer

unless explicitly added as a future experiment.
