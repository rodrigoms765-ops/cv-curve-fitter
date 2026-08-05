# Cyclic Voltammetry Diffusion Model Fitting (Python JAX + SciPy)

A high-performance physics-based simulation and optimization engine for Cyclic Voltammetry (CV) data analysis.

## Core Features
- **100% Python Architecture**: Pure Python mathematical engine matching `FD_solver.ipynb` with exact numerical precision.
- **Hardware-Accelerated PDE Solving**: JAX JIT-compiled Fourier diffusion scan with automatic differentiation.
- **Multi-Stage Projected L-BFGS-B Optimization**: Staged parameter discovery across baseline, background exponential tails, peak anchoring, and full non-linear polish.
- **Dual Interfaces**:
  1. **Interactive Web Dashboard**: Real-time optimization progress, Plotly interactive curves for $I(V)$, $D(V)$, and $DOS(V)$.
  2. **Standalone Python CLI**: Run directly from terminal with `python FD_solver.py` for automated data processing and publication-quality figure generation.

---

## Quick Start

### 1. Run the Interactive Web Application
Double-click `start_backend.bat` or run:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 2. Run Direct Python Solver Script
```bash
python FD_solver.py
```

---

## Method Summary

The model fits experimental CV data through a 4-stage optimization procedure using L-BFGS-B minimization:
1. **Stage 1 (Baseline Extraction)**: Identifies constant and background current offsets.
2. **Stage 1.5 (Background Exponential Tails)**: Fits exponential Tafel charging behaviors near potential limits.
3. **Stage 2 (Peak Anchoring)**: Optimizes discrete potential state distributions under baseline diffusivity.
4. **Stage 3 (Non-Linear Global Polish)**: Simultaneously refines potential-dependent diffusivity $D(V) = D_0 \exp(\beta (V - V_0)^2)$ and state densities.

