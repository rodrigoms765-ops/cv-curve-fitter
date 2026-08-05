---
title: CV Curve Fitting Pro - JAX Diffusion Engine
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# ⚡ Cyclic Voltammetry (CV) Curve Fitting & Diffusion Engine

A high-performance, physics-based simulation and optimization web application to determine the potential-dependent **diffusion coefficient $D(V)$** and **electronic density of states $DOS(V)$** from Cyclic Voltammetry experimental data.

**100% Free CPU Compatible**: Optimized for Hugging Face Spaces' Free 2 vCPU tier ($0.00 cost) using JAX automatic differentiation and JIT-compiled Fourier diffusion scan.

---

## 🌟 Features
- **100% Free Cloud & Local Execution**: Runs smoothly on free CPU hardware without requiring paid GPUs.
- **Physics-Based PDE Modeling**: Fast Fourier spectral decomposition of 1D ion transport and diffusion within thin-film electrodes.
- **Hardware-Accelerated Auto-Diff**: JAX reverse-mode gradients enable rapid multi-stage L-BFGS-B parameter convergence in seconds.
- **Interactive Visualizations (Plotly)**:
  - Real-time live updating $I(V)$ curve fit overlay.
  - Potential-dependent diffusion coefficient $D(V) = D_0 \exp(\beta (V - V_0)^2)$.
  - Electronic Density of States ($DOS$) Gaussian deconvolution.
- **User-Friendly Chemistry Workflow**:
  - **Drag-and-Drop File Upload**: Direct upload for user `.csv` / `.txt` potentiostat files.
  - **Potentiostat Preset Selector**: Quick buttons for Cycle 1, 2, 3, 4 (Raw vs. Adjusted) from Biologic, Gamry, or CH Instruments data.
  - **1-Click Data Export**: Export fitted curve coordinates (`.csv`) and extracted physical parameters (`.json`).

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** or **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## ☁️ Hugging Face Spaces Free Deployment

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Space SDK**: `Gradio` or `Docker`.
3. Choose **Hardware**: **CPU basic &bull; 2 vCPU &bull; 16 GB &bull; Free ($0/mo)**.
4. Clone your Space repository and push these files:
```bash
git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
git push hf main
```
5. Your web app will automatically build and go live at `https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>`!

---

## 🔬 Mathematical Method Summary

The solver fits experimental CV data through a 4-stage optimization procedure using L-BFGS-B minimization:
1. **Stage 1 (Baseline Extraction)**: Identifies constant and non-faradaic background current offsets.
2. **Stage 1.5 (Exponential Charging Tails)**: Fits Tafel-like charging behaviors near electrochemical potential window boundaries.
3. **Stage 2 (Peak Anchoring)**: Optimizes discrete potential state distributions under baseline diffusivity $D_0$.
4. **Stage 3 (Non-Linear Global Polish)**: Jointly refines potential-dependent diffusivity $D(V) = D_0 \exp(\beta (V - V_0)^2)$, state densities, and charging offsets.
