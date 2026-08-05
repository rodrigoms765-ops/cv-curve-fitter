# Hugging Face ZeroGPU: import spaces at the top of the file
try:
    import spaces
    GPU = spaces.GPU
except Exception:
    def GPU(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

import os
import sys
from pathlib import Path
import json
import traceback
import pandas as pd
import io

# Configure paths
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
for d in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Import solver and backend handlers
try:
    from backend.cv_solver import solve_cv
except ImportError:
    try:
        from cv_solver import solve_cv
    except ImportError:
        from FD_solver import solve_cv

# Top-level @spaces.GPU function registered with Gradio for ZeroGPU A100 allocation
@GPU(duration=120)
def gradio_solve_cv(file_content: str, config_json: str):
    """ZeroGPU registered execution point for JAX optimization."""
    try:
        if not file_content or not file_content.strip():
            return json.dumps({"type": "error", "message": "No CSV data file content provided. Please upload a cyclic voltammetry data file."})

        raw_config = json.loads(config_json) if isinstance(config_json, str) else (config_json or {})
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
        pot_col = int(raw_config.get("pot_col", 0))
        cur_col = int(raw_config.get("cur_col", 1))

        df = pd.read_csv(io.StringIO(file_content), sep=None, engine='python')
        result_dict = solve_cv(df, config, pot_col, cur_col, queue=None, loop=None)

        return json.dumps({
            "type": "done",
            "params": {
                "D0": result_dict["parameters"]["diffusivity"],
                "Vc": result_dict["parameters"]["v_center"],
                "beta_L": result_dict["parameters"]["beta_left"],
                "beta_R": result_dict["parameters"]["beta_right"],
                "I_offset": result_dict["parameters"]["baseline_offset"]
            },
            "plots": {
                "v_plot": result_dict["plots"]["v_plot"],
                "d_of_v": result_dict["plots"]["d_of_v"],
                "dos_total": result_dict["plots"]["dos_total"],
                "dos_peaks": result_dict["plots"]["dos_matrix"],
                "exp_potential": result_dict["plots"]["exp_potential"],
                "exp_current": result_dict["plots"]["exp_current"],
                "sim_current": result_dict["plots"]["sim_current"]
            },
            "total_iterations": 100
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

# Health check helper
def health_info():
    import jax
    devices = [str(d) for d in jax.devices()]
    return {
        "status": "ok",
        "engine": "JAX Auto-Diff Hardware Accelerated Engine",
        "devices": devices
    }

# Gradio integration for Hugging Face ZeroGPU
has_gradio = False
try:
    import gradio as gr
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.requests import Request
    from fastapi.responses import JSONResponse

    head_html = f"""
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    {app_css}
    footer {{visibility: hidden !important; display: none !important;}}
    .gradio-container {{max-width: 100% !important; padding: 0 !important; margin: 0 !important; background: #0b1120 !important;}}
    .prose {{max-width: 100% !important;}}
    .hidden-bridge {{position: absolute !important; opacity: 0 !important; pointer-events: none !important; height: 0 !important; width: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; border: none !important;}}
    </style>
    """

    with gr.Blocks(title="Cyclic Voltammetry Parameter Extraction & Physical Model Fitting", head=head_html, js=app_js) as demo:
        # Native Gradio components for ZeroGPU hardware event dispatching
        gr_input_file = gr.Textbox(value="", elem_id="gr_input_file", elem_classes=["hidden-bridge"])
        gr_input_config = gr.Textbox(value="{}", elem_id="gr_input_config", elem_classes=["hidden-bridge"])
        gr_output_json = gr.Textbox(value="", elem_id="gr_output_json", elem_classes=["hidden-bridge"])
        gr_trigger_btn = gr.Button("Execute ZeroGPU", elem_id="gr_trigger_btn", elem_classes=["hidden-bridge"])

        gr.HTML(inlined_html)

        gr_trigger_btn.click(
            fn=gradio_solve_cv,
            inputs=[gr_input_file, gr_input_config],
            outputs=[gr_output_json]
        )

    # Attach FastAPI routes directly to demo.app
    demo.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def api_solve_handler(request: Request):
        data = await request.json()
        file_content = data.get("file_content", "")
        config = data.get("config", {})
        res_str = gradio_solve_cv(file_content, json.dumps(config))
        return JSONResponse(content=json.loads(res_str))

    demo.app.add_api_route("/health", health_info, methods=["GET"])
    demo.app.add_api_route("/api/health", health_info, methods=["GET"])
    demo.app.add_api_route("/api/solve", api_solve_handler, methods=["POST"])
    demo.app.add_api_route("/solve", api_solve_handler, methods=["POST"])

    has_gradio = True
except ImportError:
    has_gradio = False

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    if has_gradio:
        print(f"Starting Gradio server on port {port}...")
        demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
    else:
        import uvicorn
        from fastapi import FastAPI
        fastapi_app = FastAPI()
        fastapi_app.add_api_route("/health", health_info, methods=["GET"])
        print(f"Starting FastAPI server on port {port}...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
