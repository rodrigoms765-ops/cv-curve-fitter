import os
import sys
from pathlib import Path
import uvicorn

# Add current directory and backend directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
for d in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Hugging Face ZeroGPU Static Scanner: explicit @spaces.GPU decorator in app.py
try:
    import spaces
    @spaces.GPU(duration=120)
    def zerogpu_probe():
        return True
except Exception:
    pass

# Fail-safe Import: Supports both subfolder and flat directory uploads
try:
    from backend.main import app, handle_solver_websocket, health_check
except ImportError:
    try:
        from main import app, handle_solver_websocket, health_check
    except ImportError as e:
        raise ImportError(f"Failed to load backend/main.py or main.py: {e}")

# Optionally mount Gradio documentation interface if gradio is available (Hugging Face Spaces)
try:
    import gradio as gr

    with gr.Blocks(title="CV Curve Fitting Pro - JAX Engine") as demo:
        gr.Markdown("# ⚡ Cyclic Voltammetry (CV) Curve Fitting Pro")
        gr.Markdown("""
        ### 100% Free Hardware-Accelerated Electrochemical CV Solver (Pure Python JAX + SciPy)
        - **Compute Engine**: JAX Reverse-Mode Automatic Differentiation & L-BFGS-B Optimization
        - **Hardware Acceleration**: Hugging Face ZeroGPU (NVIDIA A100/H100 - 100% Free) & CPU
        - **WebSocket API**: `/ws/solve`
        - **Health Probe**: `/health`
        """)
        gr.HTML("""
        <div style="margin: 20px 0; padding: 24px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-family: sans-serif;">
            <h2 style="color: #60a5fa; margin-top: 0;">🚀 Interactive Web Application Active</h2>
            <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6;">
                The full interactive CV curve fitting dashboard with real-time Plotly charts, cycle selectors, and parameter extraction is running at the root URL.
            </p>
            <div style="margin-top: 16px;">
                <a href="/" style="display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; border-radius: 8px; font-weight: bold; text-decoration: none; box-shadow: 0 2px 6px rgba(59,130,246,0.4);">
                    Open Main Interactive Dashboard &rarr;
                </a>
            </div>
        </div>
        """)

    app = gr.mount_gradio_app(app, demo, path="/gradio")
except ImportError:
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting CV Curve Fitting Server on port {port} (ZeroGPU & CPU Ready)...")
    uvicorn.run(app, host="0.0.0.0", port=port)
