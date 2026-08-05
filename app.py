import os
import sys
from pathlib import Path
import gradio as gr

# Add project root and backend folder to Python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.main import app as fastapi_app

# Create Gradio interface for Hugging Face Spaces
with gr.Blocks(title="CV Curve Fitting Pro - JAX Engine") as demo:
    gr.Markdown("# ⚡ CV Curve Fitting Pro - JAX Engine")
    gr.Markdown("""
    ### Hardware-Accelerated Electrochemical CV Solver (Pure Python JAX + SciPy)
    - **Engine**: JAX Reverse-Mode Automatic Differentiation & L-BFGS-B Optimization
    - **WebSocket API**: `/ws/solve`
    - **Status Check**: `/health`
    """)
    gr.HTML("""
    <div style="margin: 20px 0; padding: 24px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-family: sans-serif;">
        <h2 style="color: #60a5fa; margin-top: 0;">🚀 Live Web Application</h2>
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6;">
            The backend solver engine is active and ready to process simulations.
            You can use the full interactive visual application on GitHub Pages:
        </p>
        <div style="margin-top: 16px;">
            <a href="https://rodrigoms765-ops.github.io/cv-curve-fitter/" target="_blank" style="display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; border-radius: 8px; font-weight: bold; text-decoration: none; box-shadow: 0 2px 6px rgba(59,130,246,0.4);">
                Open Interactive CV Solver Web App &rarr;
            </a>
        </div>
    </div>
    """)

# Mount Gradio onto the FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
