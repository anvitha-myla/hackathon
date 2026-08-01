# Cursor Composer Prompt — ARIP Backend

Paste into Cursor Composer when extending or debugging `arip_backend/`.

---

## Role

You are working in the ARIP nitroxylene-hydrogenation digital twin backend (`arip_backend/`). Preserve industrial correctness: stiff ODE physics first, optional residual ML only when lab history exists, then EKF → PSM decisions → AI explainer.

## Hard invariants (do not regress)

1. **Strict pure-physics fallback**
   - Scan `arip_backend/data` and `/data` for lab history (`*.csv`, `*.parquet`, `*.jsonl`, `lab_*.json`, `*_history.json`).
   - README alone does **not** count as a dataset.
   - If no dataset: force `is_pure_physics=True`, `δ_ML = 0.0` for all residual targets — even if `models/*.joblib` exists.
   - Endpoint: `GET /api/v1/twin/physics-mode` must report gate status + zero sample residual.

2. **Dual execution engine**
   - **Mode A (live):** `WS /ws/v1/twin/stream` at speed `1 | 5 | 10`. Physics step = `dt_s * speed_x`.
   - **Mode B (jump):** `POST /api/v1/twin/jump` with `{ "t_min": 32.5 }` or `{ "t_s": ... }`. Integrate from `t=0` to target; return final frame + sparse keyframes. Do not make the client wait in real time.
   - WebSocket **must not auto-start** on connect. Emit `hello` with `running: false`. Client sends `{ "action": "start" }` / `pause` / `reset` / `configure`. Ack with `start_ack` / `pause_ack` / `reset_ack`.
   - Idle heartbeats ≤ ~1 Hz (never flood at telemetry Hz).

3. **Pipeline order** (per step): SciPy ODE → residual refine → EKF fuse → decision engine → optional AI advisory.

4. **Telemetry labels** must stay explicit engineering names in `UnifiedTwinFrame.metrics[].display_name`:
   - Reactor Temperature (`T_reactor`)
   - Headspace Pressure (`P_headspace`)
   - Jacket Inlet Flow Rate
   - Intermediate Hydroxylamine Concentration (or Intermediate Concentration)
   - Never ship cryptic-only tags as the only UI-facing text.

## Module map

| Module | Path | Responsibility |
|---|---|---|
| ODE physics | `simulation_engine.py` | 6-state BDF; no stateful PID inside RHS |
| Residual ML | `residual_ml.py` | `y = y_physics + f_ML(x)` with lab-dataset gate |
| EKF | `ekf_estimator.py` | Fuse `[T, P, MFC_H2_rate]` → state + confidence |
| Decisions | `decision_engine.py` | Tier-1 interlocks + Tier-2 advice |
| AI | `ai_explainer.py` | Ollama `qwen2.5:7b` + template fallback |
| Orchestrator | `twin_runtime.py` | `step()`, `jump_to()`, metric assembly |
| Gateway | `main.py` | REST + WS; serve `arip_frontend/dist` |

## Schemas

- `schemas/telemetry.py` — `TwinStepRequest`, `StreamControlMessage` (`action`, `speed_x`), `TwinJumpRequest`, `UnifiedTwinFrame`
- Residual deltas: `delta_T_exotherm_c`, `delta_C_nitro`, `delta_yield`

## Acceptance checks

```bash
curl -s localhost:8000/api/v1/twin/physics-mode   # is_pure_physics true, sample ~0
curl -s -X POST localhost:8000/api/v1/twin/jump -H 'Content-Type: application/json' -d '{"t_min":32.5}'
# WS: connect → hello (no frames) → action:start → start_ack + frames → action:reset → reset_ack idle
```

## Task template

> Extend \<module\> for \<feature\>. Keep pure-physics gate, Mode A/B APIs, and explicit metric `display_name`s. Add/adjust tests or a smoke curl. Do not auto-start the WebSocket stream.
