# Cursor Composer Prompt — ARIP Frontend

Paste into Cursor Composer when extending or debugging `arip_frontend/`.

---

## Role

You are working on the ARIP engineering workbench (React + TypeScript). This is a **desktop-class process engineering UI**, not a generic SaaS dashboard. Prefer Aspen / LabPlot / SCADA density and clarity over marketing layouts.

## Hard invariants (do not regress)

1. **Three primary screen modes** (top view-switcher)
   - **View 1 — Process:** PFD / reactor visuals (Focus Mode) + side telemetry
   - **View 2 — Analytics:** Full-screen LabPlot plot grid
   - **View 3 — Safety:** Interlocks + AI Operator Assistant
   - Keep the **bottom Logs / Plots / Hide** toggle across views.

2. **Dual execution controls**
   - Mode A: Start / Pause with speed `1× | 5× | 10×` over WebSocket.
   - Mode B: Time-jump input (default `32.5` min) → `POST /api/v1/twin/jump` → hydrate vessel levels, stage, temperatures, and rebuild plots from keyframes **without** waiting for live stream.

3. **Glitch-free Start/Reset state machine**
   - Phases: `idle | starting | running | pausing | paused | resetting | jumping`
   - WebSocket connect ≠ Start. Client may `configure` on open; never send `action: start` until the user presses Start.
   - Reset: pause → `POST /api/v1/twin/reset` → clear history → hydrate IC frame → WS `action: reset` → land in **`idle`** (never auto-restart).
   - Disable buttons while `starting|pausing|resetting|jumping` to avoid races.
   - Use ack handlers (`start_ack`, `pause_ack`, `reset_ack`) to advance phase; do not assume stream messages imply control success.

4. **Unambiguous engineering labels**
   - Render `metric.display_name` everywhere (gauges, plots, console).
   - Prefer: Reactor Temperature (`T_reactor`), Headspace Pressure (`P_headspace`), Jacket Inlet Flow Rate, Intermediate Concentration — never cryptic-only abbreviations as the sole label.

5. **Pure-physics badge**
   - Surface `is_pure_physics` from hello / physics-mode / frame residual (`PURE PHYSICS` vs `HYBRID ML`).

## Module map

| Area | Path |
|---|---|
| Shell / views / lifecycle | `src/App.tsx` |
| WS + jump/reset APIs | `src/api/twinClient.ts` |
| Zustand store | `src/store/appStore.ts` |
| Toolbar + view switcher | `src/components/shell/ToolBar.tsx` |
| PFD | `src/components/pfd/` |
| Plots / telemetry / safety / AI / console | `src/components/panels/Panels.tsx` |
| Aspen Input Expert | `src/components/expert/InputExpertDialog.tsx` |

## API contracts

- WS URL: `/ws/v1/twin/stream` — control payload uses **`action`** (not `type`): `start|pause|reset|configure`
- Jump: `postTimeJump(tMin)` → `{ frame, keyframes, t_min, is_pure_physics }`
- Reset: hydrate from `res.frame` via `hydrateTwin` (do not double-append history like live `setTwin`)

## Design notes

- Follow existing industrial chrome (dark engineering palette already in `index.css`).
- Do not turn the hero/workbench into a card-heavy marketing dashboard.
- Physics traces dashed; EKF fused traces solid in LabPlot mode.

## Acceptance checks

- Connect → status idle, no live frames until Start
- Start @ 10× → `start_ack` then rising `t_min`
- Jump `32.5` → stage/T/levels update immediately; plots rebuild
- Reset → `idle`, history cleared, no auto-run
- Switch Process / Analytics / Safety; bottom Logs↔Plots still works
- Readouts show full `display_name` strings

## Task template

> Update \<component\> for \<feature\>. Keep the 3-view switcher, bottom Logs/Plots toggle, explicit engineering labels, and the Start/Reset state machine decoupled from WebSocket connect. Verify Mode A speeds and Mode B jump still work.
