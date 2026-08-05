import os
import sys
from pathlib import Path
import io
import asyncio
import traceback
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root and backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from backend.cv_solver import solve_cv
except ImportError:
    from cv_solver import solve_cv

# ZeroGPU Support: Use Hugging Face ZeroGPU @spaces.GPU decorator if running on ZeroGPU
try:
    import spaces
    has_spaces = True
except Exception:
    has_spaces = False

if has_spaces:
    @spaces.GPU(duration=120)
    def compute_solve_cv(df, config, pot_col, cur_col, queue, loop):
        solve_cv(df, config, pot_col, cur_col, queue, loop)
else:
    def compute_solve_cv(df, config, pot_col, cur_col, queue, loop):
        solve_cv(df, config, pot_col, cur_col, queue, loop)

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
        "engine": "JAX Hardware Accelerated (ZeroGPU / A100)" if (has_spaces or is_gpu) else "JAX Hardware Accelerated (CPU/XLA)",
        "hardware": "Hugging Face ZeroGPU (NVIDIA A100/H100)" if has_spaces else ("GPU" if is_gpu else "CPU / Local"),
        "cost": "100% Free",
        "devices": devices,
        "features": ["ZeroGPU Dynamic Allocation", "Automatic Differentiation", "JIT Parallelized Scan", "L-BFGS-B Multi-stage"]
    }

async def handle_solver_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        
        # Parse configuration
        raw_config = data.get("config", {})
        file_content = data.get("file_content", "")
        
        if not file_content:
            await websocket.send_json({"type": "error", "message": "No CSV file content received. Please select and upload a CV file."})
            return
            
        config = {
            "scan_rate_v_s": float(raw_config.get("scan_rate", 0.010)),
            "film_thickness": float(raw_config.get("film_thickness", 1e-4)),
            "v_min": float(raw_config.get("v_min", -1.0)),
            "v_max": float(raw_config.get("v_max", 1.0)),
            "skip_factor": int(raw_config.get("skip_factor", 5)),
            "num_peaks": int(raw_config.get("num_peaks", 50)),
            "max_iter": int(raw_config.get("max_iter", 100)),
            "tol_ftol": float(raw_config.get("tol_ftol", 1e-8)),
            "tol_gtol": float(raw_config.get("tol_gtol", 1e-7)),
            "num_terms": int(raw_config.get("num_terms", 50)),
            "loss_weight_const": float(raw_config.get("loss_weight_const", 1.0))
        }
        
        pot_col = int(raw_config.get("pot_col", 8))
        cur_col = int(raw_config.get("cur_col", 9))
        
        # Read file into pandas DataFrame
        df = pd.read_csv(io.StringIO(file_content), sep=None, engine='python')
        
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # Run JAX solver (with ZeroGPU acceleration if available) in background thread
        def run_solver():
            try:
                compute_solve_cv(df, config, pot_col, cur_col, queue, loop)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {
                        "type": "error",
                        "message": str(e),
                        "trace": traceback.format_exc()
                    }
                )
                
        asyncio.create_task(asyncio.to_thread(run_solver))
        
        # Stream live progress updates to WebSocket client
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
