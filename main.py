import os
import sys
from pathlib import Path
import io
import json
import asyncio
import traceback
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

# Add project root and backend directory to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR if (CURRENT_DIR / "index.html").exists() else CURRENT_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"
for d in [str(ROOT_DIR), str(BACKEND_DIR), str(CURRENT_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from cv_solver import solve_cv, read_csv_text

# NOTE: Render runs app.py, not this module (see render.yaml). This streaming
# variant is kept for local development only; app.js talks to app.py's plain
# JSON endpoint and cannot parse the NDJSON body this file returns.

DEFAULTS = {
    "film_thickness": (float, 1e-4), "v_min": (float, -1.0), "v_max": (float, 1.0),
    "skip_factor": (int, 2), "num_peaks": (int, 20), "peak_sharpness": (float, 38.92),
    "dos_smoothness": (float, 0.01), "max_iter": (int, 1000),
    "tol_ftol": (float, 1e-9), "tol_gtol": (float, 1e-8),
    "num_terms": (int, 20), "loss_weight_const": (float, 1.0),
}


def build_config(raw):
    cfg = {k: cast(raw.get(k, dflt)) for k, (cast, dflt) in DEFAULTS.items()}
    # The retired 'background'/'use_tafel' settings selected exponential edge tails
    # and are ignored. Transport is the fast/slow split instead.
    mode = str(raw.get("transport", "")).lower()
    cfg["transport"] = mode if mode in ("two_site", "single") else "two_site"
    cfg["smooth_width_V"] = float(raw.get("smooth_width_V", 0.35))
    return cfg


def build_scans(data, raw_config):
    """Accept the batch body, falling back to the old single-file shape."""
    files = data.get("files")
    if not files:
        content = data.get("file_content", "")
        if not content:
            return []
        files = [{"name": data.get("file_name", "scan"), "content": content,
                  "scan_rate": raw_config.get("scan_rate", 0.010)}]
    scans = []
    for f in files:
        if f.get("content", "").strip():
            scans.append({"name": f.get("name", ""),
                          "df": read_csv_text(f["content"]),
                          "scan_rate": float(f.get("scan_rate", 0.010))})
    return scans


def compute_solve_cv(scans, config, pot_col, cur_col, queue, loop):
    solve_cv(scans, config, pot_col, cur_col, queue, loop)

app = FastAPI(title="CV Curve Fitting Pro - JAX Engine")

# Enable Cross-Origin Resource Sharing (CORS) for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.get("/api/health")
def health_check():
    import jax
    devices = [str(d) for d in jax.devices()]
    is_gpu = any("gpu" in d.lower() or "cuda" in d.lower() for d in devices)
    return {
        "status": "ok",
        "engine": "JAX Hardware Accelerated (GPU/A100)" if is_gpu else "JAX Hardware Accelerated (CPU/XLA)",
        "hardware": "GPU" if is_gpu else "CPU / Local",
        "cost": "100% Free",
        "devices": devices,
        "features": ["Automatic Differentiation", "JIT Parallelized Scan", "L-BFGS-B Multi-stage"]
    }

@app.post("/api/solve")
@app.post("/solve")
async def api_solve_stream(request: Request):
    data = await request.json()
    raw_config = data.get("config", {})
    config = build_config(raw_config)
    pot_col = int(raw_config.get("pot_col", 8))
    cur_col = int(raw_config.get("cur_col", 9))
    scans = build_scans(data, raw_config)

    if not scans:
        return {"type": "error", "message": "No CSV data received. Please upload at least one CV file."}

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def run_solver():
        try:
            compute_solve_cv(scans, config, pot_col, cur_col, queue, loop)
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait, {
                    "type": "error",
                    "message": str(e),
                    "trace": traceback.format_exc()
                }
            )
            
    asyncio.create_task(asyncio.to_thread(run_solver))
    
    async def event_generator():
        while True:
            msg = await queue.get()
            yield json.dumps(msg) + "\n"
            if msg.get("type") in ("done", "error"):
                break
                
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

async def handle_solver_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        raw_config = data.get("config", {})
        config = build_config(raw_config)
        pot_col = int(raw_config.get("pot_col", 8))
        cur_col = int(raw_config.get("cur_col", 9))
        scans = build_scans(data, raw_config)

        if not scans:
            await websocket.send_json({"type": "error", "message": "No CSV data received. Please upload at least one CV file."})
            return

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        def run_solver():
            try:
                compute_solve_cv(scans, config, pot_col, cur_col, queue, loop)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {
                        "type": "error",
                        "message": str(e),
                        "trace": traceback.format_exc()
                    }
                )
                
        asyncio.create_task(asyncio.to_thread(run_solver))
        
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") in ("done", "error"):
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

@app.websocket("/ws/solve")
async def websocket_solve(websocket: WebSocket):
    await handle_solver_websocket(websocket)

@app.websocket("/ws")
async def websocket_root(websocket: WebSocket):
    await handle_solver_websocket(websocket)

# Mount static frontend files for direct web serving
frontend_static_dir = ROOT_DIR / "frontend"
if not frontend_static_dir.exists() or not (frontend_static_dir / "index.html").exists():
    frontend_static_dir = ROOT_DIR

if (frontend_static_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_static_dir), html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
