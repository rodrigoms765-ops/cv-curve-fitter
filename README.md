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

The density of states is expanded over a fixed uniform grid of sigmoidal sub-bands of common width, with only the amplitudes treated as free parameters. Permitting the position and width of each sub-band to vary independently renders the recovered distribution non-unique.

Because the simulated current is linear in these amplitudes, their recovery constitutes a linear inverse problem and inherits its characteristic noise amplification: unregularised amplitudes oscillate between adjacent sub-bands. A second-difference (Tikhonov) penalty of weight $\lambda$ is therefore imposed,

$$\mathcal{L} = \mathcal{L}_{\mathrm{misfit}} + \lambda \sum_i \left( h_{i+2} - 2h_{i+1} + h_i \right)^2 \big/ \langle h \rangle^2 ,$$

where $h_i$ denotes the amplitude of sub-band $i$. The penalty is normalised by the mean amplitude so that $\lambda$ is independent of the overall current scale. Larger values yield a smoother, lower-resolution distribution; setting $\lambda = 0$ disables the regularisation. The default was selected from the corner of the corresponding L-curve, at which the oscillation is suppressed for a negligible increase in misfit.

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

## Model and Solver Parameters

The scan rate of each uploaded file is inferred from its filename where possible and may be corrected individually. The remaining parameters are common to the fit.

### Physical

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Film thickness | $L$ | $10^{-4}$ cm | Diffusion length of the film. The voltammogram constrains $D/L^2$, so the absolute diffusion coefficient scales with $L^2$. |
| Potential window | $V_{\min}$, $V_{\max}$ | $-1.0$, $1.0$ V | Limits of the swept potential range; also sets the extent of the sub-band grid. |
| Tafel edge fitting | — | enabled | Admits exponential background terms near the window boundaries. |

### Density of states

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Sub-bands | $N$ | 20 | Number of fixed sigmoidal basis functions spanning the potential window. |
| Sub-band width | $s$ | 38.92 V⁻¹ | Shared inverse width. The default is $F/RT$ at 298 K, the ideal one-electron Nernstian limit. Constrained so that adjacent sub-bands remain overlapping. |
| DOS smoothing | $\lambda$ | 2.0 | Weight of the second-difference penalty on the sub-band amplitudes. Increase for a smoother distribution; set to zero to disable regularisation. |

### Numerical

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Spectral terms | — | 50 | Number of eigenmodes retained in the spectral solution. |
| Downsample factor | — | 2 | Upper bound on the per-scan sampling stride, subject to a lower bound on the retained point count. |
| Iterations | — | 300 | Maximum L-BFGS-B iterations per stage. |
| Tolerance | $f_{\mathrm{tol}}$ | $10^{-12}$ | Relative convergence tolerance on the objective. |
| Weight constant | — | 1.0 | Uniform term added to the curvature- and magnitude-based residual weighting. |

## Diagnostic Output

The application reports the shared physical parameters, the per-scan residual relative to the measured current range, and the extracted $D(V)$ and $DOS(V)$ profiles, together with an overlay of the measured and simulated voltammograms.

## License

This project is licensed under the MIT License.
