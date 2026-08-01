"""ARIP Digital Twin — FastAPI REST + WebSocket SCADA gateway.

Endpoints
---------
POST /api/v1/twin/step
    One pipeline cycle: SciPy ODE → Residual ML → EKF → Decision → AI Explainer.
    Returns a unified industrial telemetry JSON frame.

WS /ws/v1/twin/stream
    Continuous ~10 Hz WebSocket feed of batch execution telemetry
    (Aspen / LabPlot / SCADA aesthetic metrics with engineering units).

Run
---
    uvicorn arip_backend.main:app --host 0.0.0.0 --port 8000
    python -m arip_backend.main
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from arip_backend import __version__
from arip_backend.schemas.telemetry import (
    StreamControlMessage,
    TwinJumpRequest,
    TwinStepRequest,
    UnifiedTwinFrame,
)
from arip_backend.twin_runtime import TwinOrchestrator
from arip_backend.residual_ml import has_lab_historical_dataset

STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "arip_frontend" / "dist"

# Shared process twin (single-batch demo runtime)
_runtime: Optional[TwinOrchestrator] = None
_stream_clients: set[WebSocket] = set()


def get_runtime() -> TwinOrchestrator:
    global _runtime
    if _runtime is None:
        _runtime = TwinOrchestrator()
    return _runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_runtime()
    yield


app = FastAPI(
    title="ARIP Digital Twin Gateway",
    description=(
        "Industrial digital-twin API for nitroxylene hydrogenation — "
        "ODE physics, residual ML, EKF fusion, PSM decisions, and local AI advisories "
        "formatted for Aspen Plus / LabPlot / SCADA dashboards."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def dashboard_root():
    """Prefer the desktop-class React workbench build; fall back to legacy HTML."""
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    legacy = STATIC_DIR / "dashboard.html"
    if legacy.exists():
        return FileResponse(legacy)
    return JSONResponse({"message": "ARIP twin API", "docs": "/docs", "frontend": "build arip_frontend"})


# ---------------------------------------------------------------------------
# Health / catalog
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, Any]:
    rt = get_runtime()
    return {
        "status": "ok",
        "service": "arip-digital-twin",
        "version": __version__,
        "batch_id": rt.batch_id,
        "t_s": rt.t_s,
        "stage_index": rt.stage_index,
        "dashboard": "/",
    }


@app.post("/api/v1/twin/setpoints")
async def set_setpoints(body: dict[str, Any]) -> dict[str, Any]:
    """Update operator setpoints from the control panel."""
    rt = get_runtime()
    if "agitator_rpm" in body and body["agitator_rpm"] is not None:
        rt.agitator_rpm = float(body["agitator_rpm"])
    if "T_jacket_c" in body and body["T_jacket_c"] is not None:
        rt.T_jacket_c = float(body["T_jacket_c"])
    if "T_sp_c" in body and body["T_sp_c"] is not None:
        rt.T_sp_c = float(body["T_sp_c"])
    if "P_sp_bar" in body and body["P_sp_bar"] is not None:
        rt.P_sp_bar = float(body["P_sp_bar"])
    return {
        "ok": True,
        "agitator_rpm": rt.agitator_rpm,
        "T_jacket_c": rt.T_jacket_c,
        "T_sp_c": rt.T_sp_c,
        "P_sp_bar": rt.P_sp_bar,
    }


@app.get("/api/v1/twin/units")
async def units_catalog() -> dict[str, Any]:
    """Engineering unit dictionary for frontend axis / table headers."""
    return {
        "schema_version": "arip.twin.v1",
        "units": {
            "RX.T": {"unit": "°C", "display_name": "Reactor Temperature", "category": "thermal"},
            "RX.P": {"unit": "bar", "display_name": "Headspace Pressure", "category": "pressure"},
            "RX.C_NITRO": {"unit": "mol/L", "display_name": "Nitroxylene Conc.", "category": "composition"},
            "RX.C_XYL": {"unit": "mol/L", "display_name": "Xylidine Conc.", "category": "composition"},
            "RX.C_H2": {"unit": "mol/L", "display_name": "Dissolved H₂", "category": "composition"},
            "RX.C_OH": {"unit": "mol/L", "display_name": "Hydroxylamine Conc.", "category": "composition"},
            "RX.CONV": {"unit": "%", "display_name": "Nitro Conversion", "category": "quality"},
            "RX.YIELD": {"unit": "%", "display_name": "Xylidine Yield", "category": "quality"},
            "RX.Q_RXN": {"unit": "kW", "display_name": "Heat Release Rate", "category": "energy"},
            "RX.R": {"unit": "mol/L·s", "display_name": "Reaction Rate", "category": "kinetics"},
            "RX.MT_RATIO": {"unit": "—", "display_name": "Mass Transfer Ratio", "category": "kinetics"},
            "RX.U": {"unit": "W/m²K", "display_name": "Effective HTC", "category": "thermal"},
            "AG.RPM": {"unit": "RPM", "display_name": "Agitator Speed", "category": "mechanical"},
            "H2.MFC": {"unit": "kg/min", "display_name": "H₂ MFC Rate", "category": "mechanical"},
            "EKF.CONF": {"unit": "%", "display_name": "EKF Confidence", "category": "quality"},
            "ML.dT": {"unit": "°C", "display_name": "Residual ΔT_exotherm", "category": "quality"},
            "BATCH.TIME": {"unit": "s", "display_name": "Batch Time", "category": "status"},
        },
        "safety_badges": ["RUN", "WARN", "TRIP", "HOLD", "IDLE"],
        "safety_status": ["NOMINAL", "WARNING", "CRITICAL"],
    }


@app.post("/api/v1/twin/advisory")
async def twin_advisory() -> dict[str, Any]:
    """Generate an operator advisory from the current twin state (no time advance)."""
    from arip_backend.ekf_estimator import I_HYDROXYL, I_NITRO, I_P, I_T, I_XYLIDINE
    from arip_backend.twin_runtime import STAGE_NAMES

    rt = get_runtime()
    x = rt.ekf.state_vector()
    last = rt.decision.history[-1] if rt.decision.history else None
    advisory = await rt.explainer.generate_operator_advisory(
        stage=STAGE_NAMES.get(rt.stage_index, f"STAGE_{rt.stage_index}"),
        fused_state={
            "T_reactor_c": float(x[I_T]),
            "P_headspace_bar": float(x[I_P]),
            "C_nitro": float(x[I_NITRO]),
            "C_xylidine": float(x[I_XYLIDINE]),
        },
        ekf_confidence=float(rt.ekf.confidence_score()),
        active_interlocks=last.active_interlocks if last else [],
        optimization_advice=last.optimization_recommendations if last else [],
        safety_status=last.safety_status if last else "NOMINAL",
    )
    return {
        "ai_advisory": advisory.model_dump(),
        "t_s": rt.t_s,
        "stage_index": rt.stage_index,
        "ekf_confidence": rt.ekf.confidence_score(),
        "C_hydroxyl": float(x[I_HYDROXYL]),
    }


@app.get("/api/v1/twin/physics-mode")
async def physics_mode() -> dict[str, Any]:
    """Report strict pure-physics gate status (δ_ML = 0 when no lab dataset)."""
    rt = get_runtime()
    # Zero-feature sample proves residual contribution is identically off
    sample = rt.residual.predict_residual(
        {
            "conversion": 0.0,
            "Q_rxn_W": 0.0,
            "mass_transfer_ratio": 1.0,
            "T_reactor_c": 80.0,
            "agitator_rpm": 200.0,
        }
    )
    return {
        "is_pure_physics": bool(rt.residual.is_pure_physics),
        "lab_dataset_present": bool(getattr(rt.residual, "lab_dataset_present", False)),
        "has_lab_dataset_scan": has_lab_historical_dataset(),
        "model_name": rt.residual.model_name,
        "message": rt.residual._load_error,
        "sample": {
            "dT_K": float(sample.delta_T_exotherm_c),
            "delta_T_exotherm_c": float(sample.delta_T_exotherm_c),
            "delta_C_nitro": float(sample.delta_C_nitro),
            "delta_yield": float(sample.delta_yield),
        },
    }


@app.post("/api/v1/twin/jump")
async def twin_jump(body: TwinJumpRequest) -> dict[str, Any]:
    """Mode B: instant time scrub — evaluate exact state at t without live wait."""
    rt = get_runtime()
    try:
        target = body.target_seconds()
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    frame, keyframes = await rt.jump_to(target, keyframe_every_s=body.keyframe_every_s)
    return {
        "ok": True,
        "mode": "time_jump",
        "t_s": frame.t_s,
        "t_min": frame.t_min,
        "frame": frame.model_dump(),
        "keyframes": keyframes,
        "n_keyframes": len(keyframes),
        "is_pure_physics": frame.residual.get("is_pure_physics"),
    }


@app.post("/api/v1/twin/reset")
async def reset_twin() -> dict[str, Any]:
    """Reset batch to t=0 and leave the twin idle (does not auto-start Mode A)."""
    rt = get_runtime()
    rt.reset()
    # Snapshot IC frame for UI hydration without advancing the batch clock
    frame = await rt.step(
        TwinStepRequest(dt_s=1e-6, include_ai_advisory=False, sensor_noise=False)
    )
    rt.t_s = 0.0
    return {
        "ok": True,
        "batch_id": rt.batch_id,
        "t_s": rt.t_s,
        "stage": rt.stage_index,
        "is_pure_physics": rt.residual.is_pure_physics,
        "status": {"running": False, "t_s": rt.t_s, "stage": rt.stage_index},
        "frame": frame.model_dump(),
    }


# ---------------------------------------------------------------------------
# POST /api/v1/twin/step
# ---------------------------------------------------------------------------
@app.post("/api/v1/twin/step", response_model=UnifiedTwinFrame)
async def twin_step(body: TwinStepRequest) -> UnifiedTwinFrame:
    """Advance the twin one step through the full hybrid pipeline."""
    rt = get_runtime()
    frame = await rt.step(body)
    return frame


# ---------------------------------------------------------------------------
# WS /ws/v1/twin/stream  — ~10 Hz telemetry
# ---------------------------------------------------------------------------
@app.websocket("/ws/v1/twin/stream")
async def twin_stream(websocket: WebSocket) -> None:
    """Mode A: live real-time stream at 1× / 5× / 10× speed.

    Does **not** auto-start. Client must send ``{"action":"start"}`` so UI
    trigger events stay decoupled from the WebSocket lifecycle.
    """
    await websocket.accept()
    _stream_clients.add(websocket)
    rt = get_runtime()

    hz = 10.0
    dt_s = 0.1  # sim seconds per tick at 1×
    speed_x = 1
    include_ai = False
    running = False  # idle until explicit start
    stop = asyncio.Event()

    await websocket.send_json(
        {
            "type": "hello",
            "schema_version": "arip.twin.v1",
            "message": "ARIP twin stream connected (paused — send start)",
            "default_hz": hz,
            "speed_x": speed_x,
            "running": False,
            "batch_id": rt.batch_id,
            "is_pure_physics": rt.residual.is_pure_physics,
            "units_endpoint": "/api/v1/twin/units",
        }
    )

    async def _recv_loop() -> None:
        nonlocal hz, dt_s, include_ai, running, speed_x
        try:
            while not stop.is_set():
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                    msg = StreamControlMessage.model_validate(data)
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

                if msg.action == "pause":
                    running = False
                    await websocket.send_json(
                        {"type": "pause_ack", "t_s": rt.t_s, "running": False}
                    )
                elif msg.action in ("start", "configure"):
                    hz = float(msg.hz)
                    dt_s = float(msg.dt_s)
                    speed_x = int(msg.speed_x)
                    include_ai = bool(msg.include_ai_advisory)
                    if msg.agitator_rpm is not None:
                        rt.agitator_rpm = float(msg.agitator_rpm)
                    if msg.T_jacket_c is not None:
                        rt.T_jacket_c = float(msg.T_jacket_c)
                    if msg.P_sp_bar is not None:
                        rt.P_sp_bar = float(msg.P_sp_bar)
                    if msg.T_sp_c is not None:
                        rt.T_sp_c = float(msg.T_sp_c)
                    if msg.action == "start":
                        running = True
                        await websocket.send_json(
                            {
                                "type": "start_ack",
                                "t_s": rt.t_s,
                                "running": True,
                                "speed_x": speed_x,
                                "dt_s": dt_s * speed_x,
                            }
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "configure_ack",
                                "speed_x": speed_x,
                                "hz": hz,
                                "dt_s": dt_s,
                                "running": running,
                            }
                        )
                elif msg.action == "reset":
                    running = False
                    rt.reset()
                    await websocket.send_json(
                        {"type": "reset_ack", "t_s": rt.t_s, "running": False}
                    )
        except WebSocketDisconnect:
            stop.set()
        except Exception:  # noqa: BLE001
            stop.set()

    recv_task = asyncio.create_task(_recv_loop())
    last_heartbeat = 0.0

    try:
        while not stop.is_set():
            loop_start = asyncio.get_event_loop().time()
            try:
                if running:
                    req = TwinStepRequest(
                        dt_s=float(dt_s) * float(speed_x),
                        include_ai_advisory=include_ai,
                        sensor_noise=True,
                        agitator_rpm=rt.agitator_rpm,
                        T_jacket_c=rt.T_jacket_c,
                        P_sp_bar=rt.P_sp_bar,
                    )
                    frame = await rt.step(req)
                    await websocket.send_json(
                        {
                            "type": "frame",
                            "payload": frame.model_dump(),
                            "speed_x": speed_x,
                        }
                    )
                else:
                    # Idle keepalive at ~1 Hz — do not flood at telemetry hz
                    now = asyncio.get_event_loop().time()
                    if now - last_heartbeat >= 1.0:
                        last_heartbeat = now
                        await websocket.send_json(
                            {
                                "type": "heartbeat",
                                "t_s": rt.t_s,
                                "running": False,
                                "scada_badge": "IDLE",
                                "speed_x": speed_x,
                            }
                        )
            except WebSocketDisconnect:
                break

            elapsed = asyncio.get_event_loop().time() - loop_start
            tick = (1.0 / hz) if running else 0.2
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(0.0, tick - elapsed))
                break
            except asyncio.TimeoutError:
                continue
    finally:
        stop.set()
        recv_task.cancel()
        _stream_clients.discard(websocket)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    uvicorn.run(
        "arip_backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
