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
* **Joint Multi-Scan Fitting:** All uploaded scan rates are fit simultaneously against a
  single shared $D(V)$ and $DOS(V)$; only the non-faradaic offset and background tails are
  allowed to differ between scans. This is what makes $D_0$ identifiable at all — the DOS
  contributes current linearly in the scan rate, while diffusion enters through the ratio of
  sweep time to $L^2/D$, so a single scan cannot separate them.
* **Fixed-Basis Density of States:** The DOS is expanded on a uniform grid of sigmoids with
  one shared width, and only the heights are fitted. Floating peak positions and widths
  independently makes the recovered DOS non-unique.
* **Multi-Stage Optimization:**
  * *Baseline:* constant non-faradaic offset per scan, window edges masked.
  * *Tails:* Tafel-like charging behaviour near the potential window boundaries.
  * *Peaks:* DOS heights and shared width, against a frozen background.
  * *Diffusion:* $D(V) = D_0 \exp(\beta (V - V_c)^2)$, held U-shaped with $\beta \geq 0$.
  * *Polish and restart:* joint refinement of every parameter.
* **Interactive Diagnostic Visualizations:** Provides real-time, visualizations of the $I(V)$ curve fit overlay, extracted $D(V)$ profile, and $DOS(V)$.

## License

This project is licensed under the MIT License.
