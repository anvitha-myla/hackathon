# ARIP Digital Twin (Hackathon)

Hybrid physics + ML digital twin for **semi-batch nitroxylene hydrogenation**.

## Backend (`arip_backend/`)

SciPy ODE · Residual ML · EKF · PSM decision engine · Local AI explainer · FastAPI + WebSocket

```bash
pip install -r arip_backend/requirements.txt
uvicorn arip_backend.main:app --host 0.0.0.0 --port 8000
```

## Frontend (`arip_frontend/`) — desktop-class engineering workbench

React + TypeScript + **Dockview** + **React Flow** + Aspen-style **Input Expert** dialogs.

```bash
cd arip_frontend && npm install && npm run dev
# http://localhost:5173  (proxies API/WS to :8000)
```

Production UI (after `npm run build`) is also served by FastAPI at `http://localhost:8000/`.

### Workbench features

- **Strict pure-physics fallback** — no lab history in `arip_backend/data` (or `/data`) ⇒ residual ML off (`δ_ML = 0`); raw ODE only
- **Dual execution** — Mode A live WebSocket at 1×/5×/10×; Mode B instant time jump (`POST /api/v1/twin/jump`, e.g. `t = 32.5` min)
- **3 primary views** — Process Diagram · LabPlot Analytics · Safety & AI Assistant (bottom Logs/Plots toggle)
- Full engineering labels (Reactor Temperature, Headspace Pressure, Jacket Inlet Flow Rate, …)
- Glitch-free Start/Reset state machine — WS connects idle; Start begins Mode A; Reset returns to idle (no auto-run)
- Top menu / ribbon · asset tree · interactive PFD (STBR/TCU/MFC/ANF/WFE) · Aspen Input Expert dialogs
