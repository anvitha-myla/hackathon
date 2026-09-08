# Agent instructions

**Read `docs/00_MASTER_SPEC.md` before any code change.** Then read the numbered spec that owns the layer you are touching (`01`–`13`). Then inspect existing code. Do not overwrite working code. Do not silently change research decisions.

Locked decisions (do not reverse without an entry in `docs/14_CHANGELOG.md`):

- Primary target is **biomass only**.
- Dataset is **IndPenSim** simulated batches (not claimed as physical industrial telemetry).
- Split is **60/20/20 by complete batch**, never by timestamp. Final 20 test batches stay untouched during development.
- Pipeline is **causal**.
- Physics is **reduced-order Monod-style**, not full IndPenSim equations.
- One residual MLP: **64-ReLU-32-ReLU-16-ReLU-Output**. No LSTM / Transformer / Neural ODE / XGBoost primary residual / three phase NNs.
- Hybrid: `X_hybrid = X_mechanistic + beta_trust * delta_X_predicted`.
- OOD: Mahalanobis distance with `beta_trust`; fallback to physics.
- UI: four Streamlit tabs (Data Cleaning, Feature Engineering, Live Inference, Computational Monitor). Live inference hides reference biomass.

This repository is at Prompt 0 (initialize). Do not implement later prompts unless the user explicitly starts them.
