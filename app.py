import os
import sys
from pathlib import Path
import json
import traceback
import pandas as pd
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.requests import Request

# Configure paths
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
for d in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Import solver
from cv_solver import solve_cv, read_csv_text

def _background_mode(raw_config):
    """Background tail mode, falling back to the old use_tafel boolean."""
    mode = str(raw_config.get("background", "")).lower()
    if mode in ("per_scan", "shared_k", "off"):
        return mode
    return "per_scan" if str(raw_config.get("use_tafel", "true")).lower() == "true" else "off"


def solve_cv_api(files, config_json: str):
    """Fit every uploaded scan at once against one shared D(V) and DOS.

    files: [{"name": str, "content": str, "scan_rate": float in V/s}, ...]
    """
    try:
        if not files:
            return json.dumps({"type": "error", "message": "No CSV data received. Please upload at least one cyclic voltammetry file."})

        raw_config = json.loads(config_json) if isinstance(config_json, str) else (config_json or {})
        config = {
            "film_thickness": float(raw_config.get("film_thickness", 1e-4)),
            "v_min": float(raw_config.get("v_min", -1.0)),
            "v_max": float(raw_config.get("v_max", 1.0)),
            "skip_factor": int(raw_config.get("skip_factor", 2)),
            "num_peaks": int(raw_config.get("num_peaks", 20)),
            "peak_sharpness": float(raw_config.get("peak_sharpness", 38.92)),
            "dos_smoothness": float(raw_config.get("dos_smoothness", 0.01)),
            "max_iter": int(raw_config.get("max_iter", 300)),
            "tol_ftol": float(raw_config.get("tol_ftol", 1e-12)),
            "tol_gtol": float(raw_config.get("tol_gtol", 1e-10)),
            "num_terms": int(raw_config.get("num_terms", 50)),
            "loss_weight_const": float(raw_config.get("loss_weight_const", 1.0)),
            "smooth_width_V": float(raw_config.get("smooth_width_V", 0.35)),
            "background": _background_mode(raw_config)
        }
        pot_col = int(raw_config.get("pot_col", 0))
        cur_col = int(raw_config.get("cur_col", 1))

        scans = []
        for f in files:
            content = f.get("content", "")
            if not content or not content.strip():
                continue
            scans.append({
                "name": f.get("name", ""),
                "df": read_csv_text(content),
                "scan_rate": float(f.get("scan_rate", 0.010)),
            })
        if not scans:
            return json.dumps({"type": "error", "message": "Uploaded files contained no readable CSV data."})

        result = solve_cv(scans, config, pot_col, cur_col, queue=None, loop=None)
        shared = result["shared"]

        return json.dumps({
            "type": "done",
            "params": {
                "D0": shared["diffusivity"],
                "Vc": shared["v_center"],
                "beta_L": shared["beta_left"],
                "beta_R": shared["beta_right"],
                "sharpness": shared["sharpness"],
                "dos_fwhm": shared["dos_fwhm"],
                "num_scans": shared["num_scans"],
                "final_loss": shared["final_loss"],
                "background": shared["background"],
                "dos_charge": shared["dos_charge"]
            },
            "notes": result["notes"],
            "scans": result["scans"],
            "plots": {
                "v_plot": result["plots"]["v_plot"],
                "d_of_v": result["plots"]["d_of_v"],
                "dos_total": result["plots"]["dos_total"],
                "dos_peaks": result["plots"]["dos_matrix"]
            }
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        })

def get_app_assets():
    """Load index.html, style.css, and app.js."""
    index_file = ROOT_DIR / "index.html"
    css_file = ROOT_DIR / "style.css"
    js_file = ROOT_DIR / "app.js"

    html_content = index_file.read_text(encoding="utf-8") if index_file.exists() else "<h2>Dashboard Loading...</h2>"
    css_content = css_file.read_text(encoding="utf-8") if css_file.exists() else ""
    js_content = js_file.read_text(encoding="utf-8") if js_file.exists() else ""

    # Inline CSS & JS for 100% self-contained execution
    inlined = html_content
    inlined = inlined.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css_content}\n</style>")
    inlined = inlined.replace('<script src="app.js"></script>', f"<script>\n{js_content}\n</script>")
    return inlined, css_content, js_content

inlined_html, app_css, app_js = get_app_assets()

# ---------------------------------------------------------
# RENDER DEPLOYMENT: Pure FastAPI Setup
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the static HTML frontend on the root path for Render deployments."""
    return inlined_html

@app.get("/style.css")
async def serve_css():
    return FileResponse(ROOT_DIR / "style.css")

@app.get("/app.js")
async def serve_js():
    return FileResponse(ROOT_DIR / "app.js")

@app.get("/health")
@app.get("/api/health")
def health_info_route():
    import jax
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception:
        devices = []
    return {
        "status": "ok",
        "engine": "JAX Auto-Diff Hardware Accelerated Engine",
        "devices": devices
    }

@app.post("/solve")
@app.post("/api/solve")
async def api_solve_handler(request: Request):
    """Direct HTTP API for pure FastAPI (Render) deployments."""
    data = await request.json()
    config = data.get("config", {})
    files = data.get("files")
    if not files:
        # Single-file callers from before the multi-scan solver.
        content = data.get("file_content", "")
        if content:
            files = [{"name": data.get("file_name", "scan"),
                      "content": content,
                      "scan_rate": config.get("scan_rate", 0.010)}]
    res_str = solve_cv_api(files or [], json.dumps(config))
    return JSONResponse(content=json.loads(res_str))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    import uvicorn
    print(f"Starting Pure FastAPI Server (Render mode) on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
