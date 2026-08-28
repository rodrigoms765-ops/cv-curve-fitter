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

All supplied scan rates are fitted simultaneously against a single shared $D(V)$ and $DOS(V)$. A constant baseline offset is the only quantity permitted to differ between scans, consistent with the interpretation of $D(V)$ and $DOS(V)$ as intrinsic properties of the film.

Earlier revisions additionally admitted exponential edge terms. These have been removed: the fitted decay constant converged onto whichever bound was imposed rather than onto a value determined by the data, indicating that the terms were acting as a general-purpose smoother. More seriously, they carried per-scan freedom and thereby masked a genuine disagreement between scan rates — precisely the discrepancy that a joint multi-scan fit exists to expose.

### Density-of-states representation

The density of states is expanded over a fixed uniform grid of sigmoidal sub-bands of common width, with only the amplitudes treated as free parameters. Permitting the position and width of each sub-band to vary independently renders the recovered distribution non-unique.

Because the simulated current is linear in these amplitudes, their recovery constitutes a linear inverse problem and inherits its characteristic noise amplification: unregularised amplitudes oscillate between adjacent sub-bands. A second-difference (Tikhonov) penalty of weight $\lambda$ is therefore imposed,

$$\mathcal{L} = \mathcal{L}_{\mathrm{misfit}} + \lambda \, (\Delta V)^{-3} \sum_i \left( h_{i+2} - 2h_{i+1} + h_i \right)^2 \big/ \langle h \rangle^2 ,$$

where $h_i$ denotes the amplitude of sub-band $i$, $\Delta V$ their spacing, and $\langle h \rangle$ their mean. The spacing factor renders the penalty an approximation to $\int (\mathrm{d}^2 DOS/\mathrm{d}V^2)^2\,\mathrm{d}V$, so that $\lambda$ carries the same meaning at any number of sub-bands. Larger values yield a smoother, lower-resolution distribution; setting $\lambda = 0$ disables the regularisation. The default was selected from the corner of the corresponding L-curve, at which the oscillation is suppressed for a negligible increase in misfit.

Division by $\langle h \rangle^2$ makes the penalty invariant under $h \mapsto ch$, and hence $\lambda$ invariant to the internal current normalisation. The amplitudes are defined relative to a scale fixed from one supplied scan and from the initial guess, and therefore depend on the assumed film thickness and on the number of sub-bands; without this division a given $\lambda$ would impose different smoothing as those settings varied. The misfit term is already scale-free, being expressed relative to the range of each measured scan, so the two contributions now share a common normalisation.

### Diffusivity model

The potential dependence of the diffusion coefficient is parameterised as

$$D(V) = D_0 \exp\left[\beta (V - V_c)^2\right], \qquad \beta \geq 0,$$

with distinct exponents either side of $V_c$. The non-negativity constraint restricts the profile to a minimum at $V_c$, and $V_c$ is confined to the interior of the measured potential window. Where the fit returns $\beta_L = \beta_R = 0$ the profile is flat, $V_c$ is then unconstrained, and both quantities are withheld from the reported results rather than presented as fitted values.

### Transport environments

A single diffusivity cannot in general reconcile scans acquired at different sweep rates. Fitted individually, each scan is well described but demands its own $D$, which rises approximately as $v^{1/2}$; the joint residual accordingly grows with the ratio of sweep rates rather than with any individual scan. This is the expected signature of a distribution of transport timescales, of which the coarsest representation is two environments,

$$I(t) = f\,I\!\left(D\right) + (1-f)\,I\!\left(r D\right), \qquad 0 < r < 1,$$

in which a fraction $f$ of the sites reside where transport is rapid and the remainder where it is slower by a factor $r$. Both populations share the same $DOS(V)$ and the same $D(V)$ shape. The construction introduces two global parameters and no per-scan freedom, and must therefore hold at every sweep rate simultaneously. Physically it corresponds to the ordered and disordered regions of a semicrystalline film, which admit solvated ions at markedly different rates.

The split is refused below three scan rates, since the evidence separating the two environments resides almost entirely in the sweep-rate dependence; the solver then falls back to a single diffusivity and reports having done so.

Note that $f$ and $r$ are independent of the assumed film thickness, whereas the absolute diffusivities scale with $L^2$.

### Staged optimisation

Parameters are released in stages to limit coupling between the faradaic and non-faradaic contributions:

1. *Baseline* — the constant non-faradaic offset of each scan, with the potential window edges masked.
2. *Sub-band amplitudes* — the density of states and its shared width, against a frozen baseline.
3. *Diffusivity* — $D_0$, the two exponents, $V_c$, and where applicable the environment fraction and ratio.
4. *Joint refinement* — simultaneous relaxation of all parameters.

### Numerical considerations

Downsampling is applied per scan subject to a lower bound on the retained point count, since the simulator integrates on the grid supplied by the measurement and the fastest scans are the most sparsely sampled. Parameters that converge onto a constraint boundary are reported as such, as their fitted values then reflect the imposed bound rather than the data.

## Model and Solver Parameters

The scan rate of each uploaded file is inferred from its filename where possible and may be corrected individually. The remaining parameters are common to the fit.

### Physical

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Film thickness | $L$ | $10^{-4}$ cm | Diffusion length of the film. The voltammogram constrains $D/L^2$, so the absolute diffusion coefficient scales with $L^2$. |
| Potential window | $V_{\min}$, $V_{\max}$ | $-1.0$, $1.0$ V | Limits of the swept potential range; also sets the extent of the sub-band grid. |
| Transport model | — | two environments | Fast/slow split sharing one $DOS(V)$, or a single diffusivity. Requires at least three scan rates; below that the solver falls back to a single diffusivity. |

### Density of states

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Sub-bands | $N$ | 16 | Number of fixed sigmoidal basis functions spanning the potential window. |
| Sub-band width | $s$ | 38.92 V⁻¹ | Shared inverse width. The default is $F/RT$ at 298 K, the ideal one-electron Nernstian limit. Constrained so that adjacent sub-bands remain overlapping. |
| DOS smoothing | $\lambda$ | 0.01 | Weight of the second-difference penalty on the sub-band amplitudes. Increase for a smoother distribution; set to zero to disable regularisation. |

### Numerical

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Spectral terms | — | 20 | Number of eigenmodes retained in the spectral solution. Truncation at 20 alters the simulated current by roughly 1%, but the error is smooth and is absorbed by the sub-band amplitudes: relative to 60 terms the fitted $D$ shifts by under 1% and the per-scan residuals are unchanged. The cost is linear in this number. |
| Downsample factor | — | 4 | Upper bound on the per-scan sampling stride, subject to a lower bound on the retained point count. |
| Iterations | — | 500 | Maximum L-BFGS-B iterations per stage. Raising this to 1000 lowers the objective by a further 0.5% and shifts the fitted fast fraction by roughly 0.06; the free hosting tier is approximately thirty times slower than local hardware, so the additional iterations carry a substantial wall-clock cost. |
| Tolerance | $f_{\mathrm{tol}}$ | $10^{-9}$ | Relative convergence tolerance on the objective. A tolerance of $10^{-12}$ required approximately half again as many iterations while displacing the fitted diffusivity by one part in $10^5$. |
| Weight constant | — | 1.0 | Uniform term added to the curvature- and magnitude-based residual weighting. |

## Diagnostic Output

The application reports the shared physical parameters, the per-scan residual relative to the measured current range, and the extracted $D(V)$ and $DOS(V)$ profiles, together with an overlay of the measured and simulated voltammograms.

## License

This project is licensed under the MIT License.
