# ARIP Digital Twin — Desktop-Class Engineering Workbench

React + TypeScript application with:

- **Dockview** dockable multi-window workspace (PFD / plots / console)
- **React Flow** interactive PFD canvas (STBR, TCU, MFC, ANF, WFE + animated streams)
- **Aspen-style Input Expert** multi-page modal dialogs (Connections → Heat → Kinetics → IC → Apply)
- Top **menu bar + ribbon toolbar** (File / Equipment / Flowsheet / Simulation / Safety / …)
- Live twin telemetry via FastAPI WebSocket proxy

## Develop

```bash
# terminal 1 — API
uvicorn arip_backend.main:app --host 0.0.0.0 --port 8000

# terminal 2 — Vite (proxies /api and /ws to :8000)
cd arip_frontend
npm install
npm run dev
# open http://localhost:5173
```

## Production build (served by FastAPI at `/`)

```bash
cd arip_frontend && npm run build
uvicorn arip_backend.main:app --port 8000
# open http://localhost:8000/
```
