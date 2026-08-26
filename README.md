# Cyclic Voltammetry Curve Fitting and Diffusion Simulation Engine

## Overview

This repository implements a physically based simulation and optimisation framework for the quantitative analysis of cyclic voltammetry (CV) measurements. Its purpose is the recovery of the potential-dependent chemical diffusion coefficient, $D(V)$, and the density of electrochemically accessible states, $DOS(V)$, from raw potentiostat data.

Ion transport within the thin-film electrode is described by a one-dimensional diffusion equation subject to local equilibrium at the electrolyte interface and a no-flux condition at the current collector. The governing equation is solved by spectral decomposition, and model parameters are recovered by gradient-based non-linear least squares. Gradients are obtained through reverse-mode automatic differentiation (JAX), and minimisation is performed with the L-BFGS-B algorithm (SciPy).

## Live Web Application

**Access the live web application here:** [https://cv-curve-fitter.onrender.com/](https://cv-curve-fitter.onrender.com/)

The interface accepts `.csv` or `.txt` potentiostat files, exposes the model and solver parameters, and permits export of the fitted curves and extracted quantities without local installation.

## Methodology

### Forward model

The occupancy of the film is expressed as a sum of sigmoidal sub-bands in potential, and its evolution is propagated by an exponential integrator over the measured potential program. The initial condition is taken as the periodic steady state of the cycle, so the result is independent of an assumed starting concentration profile.

### Joint multi-scan fitting

All supplied scan rates are fitted simultaneously against a single shared $D(V)$ and $DOS(V)$. Only the non-faradaic contributions — a constant baseline offset and two exponential background terms — are permitted to differ between scans, consistent with the interpretation of $D(V)$ and $DOS(V)$ as intrinsic properties of the film.

### Density-of-states representation

The density of states is expanded over a fixed uniform grid of sigmoidal sub-bands of common width, with only the amplitudes treated as free parameters. Permitting the position and width of each sub-band to vary independently renders the recovered distribution non-unique. A second-difference (Tikhonov) penalty is imposed on the amplitudes to suppress the noise amplification characteristic of this class of linear inverse problem.

### Diffusivity model

The potential dependence of the diffusion coefficient is parameterised as

$$D(V) = D_0 \exp\left[\beta (V - V_c)^2\right], \qquad \beta \geq 0,$$

with distinct exponents either side of $V_c$. The non-negativity constraint restricts the profile to a minimum at $V_c$, and $V_c$ is confined to the interior of the measured potential window.

### Staged optimisation

Parameters are released in stages to limit coupling between the faradaic and non-faradaic contributions:

1. *Baseline* — the constant non-faradaic offset of each scan, with the potential window edges masked.
2. *Background tails* — Tafel-like charging behaviour near the window boundaries.
3. *Sub-band amplitudes* — the density of states and its shared width, against a frozen background.
4. *Diffusivity* — $D_0$, the two exponents, and $V_c$.
5. *Joint refinement* — simultaneous relaxation of all parameters, followed by a restart of the limited-memory Hessian.

### Numerical considerations

Downsampling is applied per scan subject to a lower bound on the retained point count, since the simulator integrates on the grid supplied by the measurement and the fastest scans are the most sparsely sampled. Parameters that converge onto a constraint boundary are reported as such, as their fitted values then reflect the imposed bound rather than the data.

## Diagnostic Output

The application reports the shared physical parameters, the per-scan residual relative to the measured current range, and the extracted $D(V)$ and $DOS(V)$ profiles, together with an overlay of the measured and simulated voltammograms.

## License

This project is licensed under the MIT License.
