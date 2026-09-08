# Stress Testing Specification

Stress testing is separate from normal demo inference.

## Perturbations

### Sensor noise
Add controlled random noise.

### Spikes
Inject abrupt sensor spikes.

### Missing values
Remove sections of telemetry.

### Impossible values
Inject physically impossible measurements.

### Sensor drift
Gradually bias a sensor.

### Distribution shift
Move operating conditions outside training distribution.

### Feed interruption
Simulate feed pump interruption.

### Temperature disturbance
Simulate temperature ramps/disturbances.

### Combined faults
Apply multiple disturbances simultaneously.

## Record

For every test:

batch_id
timestamp
fault_type
fault_magnitude
reference
mechanistic_prediction
nn_prediction
hybrid_prediction
error
OOD_distance
beta_trust
ML_contribution
physics_contribution

## Expected behavior

Normal:
ML correction active.

OOD:
ML correction attenuated.

Severe OOD:
prediction approaches mechanistic baseline.

System must not crash because of corrupted sensor input.
