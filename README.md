# Cyclic Voltammetry (CV) Curve Fitting & Diffusion Simulation Engine

## Overview

This repository provides a high-performance, physics-based simulation and optimization for analyzing Cyclic Voltammetry (CV) experimental data. The objective of this software is to extract the potential-dependent diffusion coefficient, $D(V)$, and the density of states, $DOS(V)$, from raw potentiostat datasets.

The engine utilizes JAX for automatic differentiation and JIT compilation, paired with SciPy's L-BFGS-B optimization algorithm, to solve the PDEs governing 1D ion transport and diffusion within thin-film electrodes.

## Live Web Application

**Access the live web application here:** [https://cv-curve-fitter.onrender.com/](https://cv-curve-fitter.onrender.com/)

The application provides a responsive, easy-to-use interface for uploading `.csv` or `.txt` potentiostat files, configuring parameters, and exporting fitted curves and physical parameters without needing any local setup.

## Key Features

* **Physics-Based PDE Modeling:** Employs Spectral Decomposition to solve 1D ion transport and diffusion equations.
* **Hardware-Accelerated Automatic Differentiation:** Utilizes JAX for reverse-mode gradients, allowing for rapid multi-stage parameter convergence.
* **Multi-Stage Optimization:**
  * *Stage 1 (Baseline Extraction):* Identifies constant and non-faradaic background current offsets.
  * *Stage 1.5 (Exponential Charging Tails):* Fits Tafel-like charging behaviors near electrochemical potential window boundaries.
  * *Stage 2 (Peak Anchoring):* (Optional) Optimizes discrete potential state distributions under a constant baseline diffusivity, $D_0$.
  * *Stage 3 (Non-Linear Global Polish):* Refines the potential-dependent diffusivity $D(V) = D_0 \exp(\beta (V - V_0)^2)$, state densities, and charging offsets.
* **Interactive Diagnostic Visualizations:** Provides real-time, visualizations of the $I(V)$ curve fit overlay, extracted $D(V)$ profile, and $DOS(V)$.

## License

This project is licensed under the MIT License.
