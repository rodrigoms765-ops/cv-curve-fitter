from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import uvicorn
import asyncio
from cv_solver import solve_cv
import traceback
import os

app = FastAPI(title="CV Curve Fitting Pro - JAX Engine")

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
    return {
        "status": "ok",
        "engine": "JAX Hardware Accelerated (XLA)",
        "features": ["Automatic Differentiation", "JIT Parallelized Scan", "L-BFGS-B Multi-stage"]
    }

async def handle_solver_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        
        # Parse configuration
        raw_config = data.get("config", {})
        file_content = data.get("file_content", "")
        
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
        
        # Read the file content into pandas
        df = pd.read_csv(io.StringIO(file_content), sep=None, engine='python')
        
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # Start optimization in a background thread
        def run_solver():
            try:
                solve_cv(df, config, pot_col, cur_col, queue, loop)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {
                        "type": "error",
                        "message": str(e),
                        "trace": traceback.format_exc()
                    }
                )
                
        # Run in thread pool
        task = asyncio.create_task(asyncio.to_thread(run_solver))
        
        # Stream updates from queue to websocket
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg["type"] in ("done", "error"):
                break
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

@app.websocket("/ws/solve")
async def websocket_solve(websocket: WebSocket):
    await handle_solver_websocket(websocket)

@app.websocket("/ws")
async def websocket_root(websocket: WebSocket):
    await handle_solver_websocket(websocket)

# Mount frontend files if serving locally
frontend_path = os.path.join(os.path.dirname(__file__), '..')
if os.path.exists(os.path.join(frontend_path, 'index.html')):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
