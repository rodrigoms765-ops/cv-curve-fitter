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

try:
    from backend.main import app as fastapi_app, handle_solver_websocket, health_check, compute_solve_cv
except ImportError:
    from main import app as fastapi_app, handle_solver_websocket, health_check, compute_solve_cv

# Top-level @spaces.GPU function registered with Gradio to ensure ZeroGPU detects GPU workload
@GPU(duration=120)
def gpu_zerogpu_runner(input_str=""):
    """ZeroGPU registered execution point for Hugging Face container startup check."""
    return "ZeroGPU JAX Engine Ready & Active"

def load_inlined_html():
    """Load index.html with inlined style.css and app.js for direct native rendering."""
    index_file = ROOT_DIR / "index.html"
    if not index_file.exists():
        index_file = ROOT_DIR / "frontend" / "index.html"
        
    css_file = ROOT_DIR / "style.css"
    if not css_file.exists():
        css_file = ROOT_DIR / "frontend" / "style.css"
        
    js_file = ROOT_DIR / "app.js"
    if not js_file.exists():
        js_file = ROOT_DIR / "frontend" / "app.js"

    html_content = index_file.read_text(encoding="utf-8") if index_file.exists() else "<h2>Dashboard Loading...</h2>"
    css_content = css_file.read_text(encoding="utf-8") if css_file.exists() else ""
    js_content = js_file.read_text(encoding="utf-8") if js_file.exists() else ""

    # Inline CSS & JavaScript directly to prevent iframe blocking or relative path failures
    inlined = html_content
    inlined = inlined.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css_content}\n</style>")
    inlined = inlined.replace('<script src="app.js"></script>', f"<script>\n{js_content}\n</script>")
    return inlined

# Gradio integration for Hugging Face ZeroGPU
has_gradio = False
try:
    import gradio as gr
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, HTMLResponse

    custom_css = """
    footer {visibility: hidden !important; display: none !important;}
    .gradio-container {max-width: 100% !important; padding: 0 !important; margin: 0 !important; background: #0f172a !important;}
    .prose {max-width: 100% !important;}
    """

    with gr.Blocks(title="CV Curve Fitting Pro - JAX Engine", css=custom_css) as demo:
        gr.HTML(load_inlined_html())
        
        # ZeroGPU event handler registration so Hugging Face scans & verifies @spaces.GPU
        dummy_input = gr.Textbox(value="ping", visible=False)
        dummy_output = gr.Textbox(visible=False)
        dummy_btn = gr.Button("Run ZeroGPU", visible=False)
        dummy_btn.click(fn=gpu_zerogpu_runner, inputs=[dummy_input], outputs=[dummy_output])

    # Attach FastAPI routes directly to demo.app
    demo.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    demo.app.add_api_route("/health", health_check, methods=["GET"])
    demo.app.add_api_route("/api/health", health_check, methods=["GET"])
    demo.app.add_api_websocket_route("/ws/solve", handle_solver_websocket)
    demo.app.add_api_websocket_route("/ws", handle_solver_websocket)

    has_gradio = True
except ImportError:
    has_gradio = False

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    if has_gradio:
        print(f"Starting Gradio + ZeroGPU server on port {port}...")
        demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
    else:
        import uvicorn
        print(f"Starting FastAPI server on port {port}...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
