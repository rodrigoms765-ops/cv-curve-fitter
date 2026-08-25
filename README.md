# Cyclic Voltammetry (CV) Curve Fitting & Diffusion Simulation Engine

## Overview

This repository provides a high-performance, physics-based simulation and optimization framework for analyzing Cyclic Voltammetry (CV) experimental data. The primary objective of this software is to extract the potential-dependent diffusion coefficient, $D(V)$, and the electronic density of states, $DOS(V)$, from raw potentiostat datasets.

The engine leverages JAX for hardware-accelerated automatic differentiation and JIT compilation, paired with SciPy's L-BFGS-B optimization algorithm, to solve the underlying partial differential equations (PDEs) governing 1D ion transport and diffusion within thin-film electrodes.

## Live Web Application

**Access the live web application here:** [https://cv-curve-fitter.onrender.com/](https://cv-curve-fitter.onrender.com/)

The application provides a responsive, easy-to-use interface for uploading `.csv` or `.txt` potentiostat files, configuring hyperparameters, and exporting fitted curves and physical parameters without needing any local setup.

## Key Features

* **Physics-Based PDE Modeling:** Employs fast Fourier spectral decomposition to solve 1D ion transport and diffusion equations.
* **Hardware-Accelerated Automatic Differentiation:** Utilizes JAX for reverse-mode gradients, enabling rapid and stable multi-stage parameter convergence.
* **Multi-Stage Optimization:**
  * *Stage 1 (Baseline Extraction):* Identifies constant and non-faradaic background current offsets.
  * *Stage 1.5 (Exponential Charging Tails):* Fits Tafel-like charging behaviors near electrochemical potential window boundaries.
  * *Stage 2 (Peak Anchoring):* Optimizes discrete potential state distributions under a constant baseline diffusivity, $D_0$.
  * *Stage 3 (Non-Linear Global Polish):* Jointly refines the potential-dependent diffusivity $D(V) = D_0 \exp(\beta (V - V_0)^2)$, state densities, and charging offsets.
* **Interactive Diagnostic Visualizations:** Provides real-time, interactive visualizations of the $I(V)$ curve fit overlay, extracted $D(V)$ profile, and Gaussian-deconvoluted $DOS(V)$.

## License

This project is licensed under the MIT License.
