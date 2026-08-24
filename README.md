# Cyclic Voltammetry (CV) Curve Fitting & Diffusion Simulation Engine

## Overview

This repository provides a high-performance, physics-based simulation and optimization framework for analyzing Cyclic Voltammetry (CV) experimental data. The primary objective of this software is to extract the potential-dependent diffusion coefficient, $D(V)$, and the electronic density of states, $DOS(V)$, from raw potentiostat datasets.

The engine leverages JAX for hardware-accelerated automatic differentiation and JIT compilation, paired with SciPy's L-BFGS-B optimization algorithm, to solve the underlying partial differential equations (PDEs) governing 1D ion transport and diffusion within thin-film electrodes.

## Key Features

* **Physics-Based PDE Modeling:** Employs fast Fourier spectral decomposition to solve 1D ion transport and diffusion equations.
* **Hardware-Accelerated Automatic Differentiation:** Utilizes JAX for reverse-mode gradients, enabling rapid and stable multi-stage parameter convergence.
* **Multi-Stage Optimization:**
  * *Stage 1 (Baseline Extraction):* Identifies constant and non-faradaic background current offsets.
  * *Stage 1.5 (Exponential Charging Tails):* Fits Tafel-like charging behaviors near electrochemical potential window boundaries.
  * *Stage 2 (Peak Anchoring):* Optimizes discrete potential state distributions under a constant baseline diffusivity, $D_0$.
  * *Stage 3 (Non-Linear Global Polish):* Jointly refines the potential-dependent diffusivity $D(V) = D_0 \exp(eta (V - V_0)^2)$, state densities, and charging offsets.
* **Interactive Diagnostic Visualizations:** Provides real-time, interactive visualizations of the $I(V)$ curve fit overlay, extracted $D(V)$ profile, and Gaussian-deconvoluted $DOS(V)$.
* **Web-Based Interface:** Includes a responsive FastAPI-driven web interface for uploading .csv or .txt potentiostat files, configuring hyperparameters, and exporting fitted curves and physical parameters.

## Installation and Execution

### Requirements
The software requires Python 3.10 or higher. 

### Local Setup
1. Clone the repository and navigate to the project root.
2. Install the required dependencies:
   `ash
   pip install -r requirements.txt
   `

### Running the Server
Execute the following command to start the local FastAPI web server:
`ash
python app.py
`
Alternatively, you can start the uvicorn server directly:
`ash
uvicorn app:app --host 0.0.0.0 --port 7860
`
Navigate to http://localhost:7860 in your web browser to access the graphical interface.

## Deployment

The application is configured for deployment on Render. The included 
ender.yaml specifies the build and start commands required to host the FastAPI application as a web service.

## License

This project is licensed under the MIT License.
