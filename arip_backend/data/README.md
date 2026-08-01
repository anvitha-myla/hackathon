# Lab historical datasets for residual ML

Place measured batch histories here to enable the hybrid residual model:

- `*.csv` / `*.parquet` / `*.jsonl` with columns matching residual features
  (`conversion`, `Q_rxn_W`, `mass_transfer_ratio`, `T_reactor_c`, `agitator_rpm`)
  and targets (`delta_T_exotherm_c`, `delta_C_nitro`, `delta_yield`)

**Strict pure-physics policy:** if this directory has no lab dataset files,
`ResidualCorrectionEngine` forces $\delta_{\text{ML}} = 0$ and
`is_pure_physics = True` even when a joblib model artifact exists under
`models/`. The ODE twin then runs raw physics only.
