# Cyclic Voltammetry Curve Fitting and Diffusion Simulation Engine

## Overview

This repository fits cyclic voltammetry data to a diffusion model in order to recover two things from raw potentiostat files: the potential-dependent diffusion coefficient $D(V)$, and the density of electrochemically accessible states.

We treat the film as a one-dimensional slab in which ions diffuse, held at local equilibrium with the electrolyte at one face and with no flux through the current collector at the other. The diffusion equation is solved by the spectral method, and the model parameters are recovered by non-linear least squares. Gradients come from automatic differentiation in JAX, and the minimisation itself is done with L-BFGS-B from SciPy.

The full derivation of the forward solution is in [`docs/mathematical_methods.tex`](docs/mathematical_methods.tex); what follows summarises the model as the code now implements it.

## Live Web Application

**Access the live web application here:** [https://cv-curve-fitter.onrender.com/](https://cv-curve-fitter.onrender.com/)

The interface accepts `.csv` or `.txt` potentiostat files, exposes the model and solver parameters, and lets you export the fitted curves and the extracted quantities without installing anything.

Example data is provided for demonstration. *Load example scans* stages four voltammograms of the same film (40, 80, 160 and 320 mV/s, third cycle of each file, 2 cm² electrode) together with the settings they were fitted with. *Show example result* displays the stored joint fit of those scans immediately, which is useful because a live fit on the hosted instance takes several minutes. The files, settings and stored result all live in `examples/`. The stored result is only valid for the solver that produced it, so regenerate it with `python examples/build_example.py` after any change to `cv_solver.py`.

### Running it locally

A fit runs far faster on a laptop than on the hosted instance. With Python 3.10 or later:

```
pip install -r requirements.txt
python -m uvicorn app:app --port 8000
```

and open `http://127.0.0.1:8000`. On Windows, `start_backend.bat` does the same. The four-scan example takes about a minute and a half. ECP-Magenta with the series resistance switched on takes two to four minutes.

## The Model

### The diffusion problem

We begin by writing the problem down explicitly, noting that the diffusion coefficient $D(t) = D(V(t))$ depends on time because the potential is being swept:

$$u_t = D(t) u_{xx}, \qquad u(0,t) = f(t), \qquad u_x(L,t) = 0, \qquad u(x,0) = u(x, k\tilde{T})$$

Here $f(t)$ is the equilibrium occupancy at the electrolyte face, set by the applied potential, and the last condition says that we want the periodic steady state of the cycle rather than the response to some assumed starting profile.

### Spectral solution

Writing $u = v + f(t)$ moves the inhomogeneity off the boundary and into the equation, leaving $v$ with homogeneous boundary conditions. We then expand

$$v(x,t) = \sum_{n=1}^{\infty} T_n(t) \sin(\lambda_n x), \qquad \lambda_n = \frac{(2n-1)\pi}{2L}$$

which satisfies both boundary conditions term by term. Projecting onto $\sin(\lambda_m x)$ and using orthogonality decouples the modes,

$$T_m'(t) = -D(t) \lambda_m^2 T_m(t) - f'(t) c_m$$

and integrating over one time step, with $D$ taken as constant across the step, gives the recursion the code actually runs:

$$T_m(t_i) = T_m(t_{i-1}) e^{-D(t_i)\lambda_m^2 \Delta t_i} + c_m\left[f(t_{i-1}) - f(t_i)\right]$$

Iterating this over a full cycle and imposing periodicity gives $T_m(0)$ in closed form, so we get the periodic initial condition directly rather than by cycling the simulation until it settles. Both the accumulation and the stepping are done with `jax.lax.scan`.

The number of particles in the film is the integral of $u$ over the slab, and the faradaic current is its time derivative, evaluated as a finite difference. No other term is added, and nothing is fitted per scan. An earlier version carried a constant current offset for each scan; removing it changed the fits by less than 0.05 percentage points.

Earlier revisions also carried exponential edge terms meant to imitate Tafel kinetics at the ends of the sweep. These have been removed. The fitted decay constant converged onto whichever bound it was given rather than onto a value fixed by the data, which is the mark of a general-purpose smoother; worse, the terms carried per-scan freedom and so hid a genuine disagreement between scan rates, which is exactly what a joint fit exists to expose.

### The boundary condition and the density of states

The boundary value $f(V)$ is the equilibrium occupancy, which is the density of states integrated up to the present potential. Integrating a Gaussian exactly gives an error function, which is expensive to evaluate at every step, so we approximate the integrated peaks with logistic functions instead:

$$f(V) = \sum_{p=1}^{P} \frac{w_p}{1 + \exp\left[-\gamma(V - V_p)\right]}$$

Unlike the original formulation, the centres $V_p$ sit on a fixed uniform grid across the potential window and the width $\gamma$ is shared by every sub-band, so only the weights $w_p$ are free. Letting the positions and widths float as well makes the recovered distribution non-unique, since neighbouring sub-bands simply collapse onto one another. We hold the width at $F/RT$ at 298 K, the sharpest a one-electron site can physically be; fitting it instead drove it to the narrowest the grid allowed at every sub-band count we tried.

One consequence of fixing the centres and the width is that the simulated current is exactly linear in the weights, which is what makes recovering them well behaved.

### The diffusion coefficient

We model the potential dependence of the diffusion coefficient phenomenologically, as a curve with its minimum at $V_c$ and separate exponents on either side:

$$D(V) = D_0 \exp\left[\beta_{L,R}(V - V_c)^2\right], \qquad \beta_{L,R} \geq 0$$

Keeping the exponents non-negative forces the minimum to sit at $V_c$, and $V_c$ itself is confined to the interior of the measured window. If the fit returns both exponents at zero then $D(V)$ is flat, $V_c$ means nothing, and we withhold both from the reported results rather than presenting them as measurements.

### Two transport environments

A single diffusivity cannot reconcile scans taken at different sweep rates. Fitted one at a time, each scan is described well but demands its own $D$, rising roughly as the square root of the sweep rate, so the joint residual grows with the ratio of sweep rates rather than with any individual scan. This is what a distribution of transport timescales looks like, and the coarsest way to represent one is with two environments:

$$I = f I(D) + (1-f) I(rD)$$

A fraction $f$ of the sites sit where transport is fast, and the rest where it is slower by a factor $r$ between 0 and 1. Both populations share the same density of states and the same shape of $D(V)$. The split costs two global parameters and no per-scan freedom, so it still has to hold at every sweep rate at once. Physically it corresponds to the ordered and disordered regions of a semicrystalline film, which take up solvated ions at very different rates.

Below three scan rates we refuse the split and fall back to a single diffusivity, reporting that we have done so, because almost all of the evidence separating the two environments lives in the sweep-rate dependence.

Note that $f$ and $r$ do not depend on the assumed film thickness, whereas the absolute diffusivities scale with $L^2$.

### Series resistance

Some films show a peak separation that grows with the current, together with sharp cut-offs on one side of a peak. ECP-Magenta is the clearest case. This is the signature of an uncompensated resistance $R$ in the cell: the film does not see the applied potential but

$$V_{\text{film}}(t) = V(t) - R I(t)$$

where $I$ is the film's own current. Using the measured current here is ill-conditioned, since wherever the measured current changes quickly the film potential jumps and the simulated current spikes, so we solve the condition self-consistently. Over one time step the number of ions in the film is affine in the new edge occupancy, $N = a f(V_{\text{film}}) + b$ with $a$ between 0 and $L$, so $V_{\text{film}} - V + R I(V_{\text{film}})$ increases with $V_{\text{film}}$ and has exactly one root, which a few Newton steps find. The periodic state comes from the same closed form as above, applied to the previous pass over the cycle. Steps are refined to at most 10 mV where the data points are further apart than that.

With $R = 0$ the model is the one above, unchanged. When $R$ is fitted, it is released last, starting from the fit without it, since released from the start it can run to its bound. Once $R$ accounts for the peak separation the film may turn out to be at equilibrium at every sweep rate, in which case $D$ runs to its upper bound and only a lower bound on it is reported. On ECP-Magenta (20 to 320 mV/s, window from the data, 26 sub-bands) we find $R$ of about 89 Ω and a residual of 2.0%, against 5.4% without it. The resistance is well determined. The shape of $D(V)$ is not: both exponents and $V_c$ end on their bounds.

## Fitting

### The objective

We minimise the mean squared error between the simulated and the measured current, each scan divided by its own measured range so that no scan dominates simply by carrying more current. Every supplied scan rate is fitted at the same time against one shared $D(V)$ and one shared density of states, which is what it means to treat them as properties of the film.

That is the whole objective. Earlier versions weighted each point by the curvature of the smoothed current and added a second-difference smoothing penalty on the sub-band weights. We tested both on three films (the 0.1 g/L film, ProDOT and ECP-Magenta). Neither was needed. Without the weighting the plain residual came out slightly lower on all three. Without the penalty the error on a held-out scan was the same or lower. The density of states was just as reproducible from fit to fit with one scan left out each time. The penalty had also flattened the narrow peaks of ECP-Magenta.

The potential window should match the data. Sub-bands centred well outside the data are full or empty throughout the sweep and carry no current, so the data cannot set their weights. A window reaching more than 50 mV past the data is clipped to it.

### Optimisation

The fit starts from a density of states read off the slowest scan. Near equilibrium $I/v$ is the density of states, so half the gap between the two sweeps, divided by the sub-band spacing, gives the starting weights. A flat start does not work: on ECP-Magenta it stalls at a poor minimum.

From that start every parameter is fitted at once, with $D$ as $\log_{10} D$, since a film at equilibrium sends it up by decades. Earlier versions opened the parameters up in stages (baseline, weights, diffusivity, then everything). On ProDOT and ECP-Magenta those warm-up stages led to a worse minimum, so they are gone.

When the series resistance is switched on, a second stage refits everything, including $R$, with the film driven at $V - RI$.

### Numerical notes

Each scan is thinned to about one point every 10 mV of potential rather than by a fixed stride in index. The two are very different: one cycle at 10 mV/s carries around 1360 points per volt and one at 640 mV/s only about 18. 10 mV is a ninth of the 90 mV sub-band width, and sampling finer changes the fit by nothing measurable.

Parameters that converge onto a bound are reported as such, since their fitted values then reflect the bound rather than the data.

## Model and Solver Parameters

The scan rate of each uploaded file is read from its filename where possible and can be corrected individually. The rest of the parameters are shared by the whole fit.

### Physical

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Film thickness | $L$ | $10^{-4}$ cm | Diffusion length of the film. The voltammogram constrains $D/L^2$, so the absolute diffusion coefficient scales with $L^2$. |
| Electrode area | $A$ | 1.0 cm² | Used only to express the density of states per unit volume. It does not enter the fit, and the shape of the recovered distribution does not depend on it. |
| Potential window | $V_{\min}$, $V_{\max}$ | from the data | Limits of the swept range, and the extent of the sub-band grid. Set from the data when files are loaded; a window more than 50 mV past the data is clipped to it. |
| Transport model | — | two environments | Fast/slow split sharing one density of states, or a single diffusivity. Needs at least three scan rates; below that the solver falls back to a single diffusivity. |
| Series resistance | $R$ | 0 Ω | Uncompensated cell resistance. The starting value when fitted, the value used when held. |
| Fit series resistance | — | No | Fit $R$ as a final stage. Switch it on for films whose peak separation grows with the current, such as ECP-Magenta. |

### Density of states

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Sub-bands | $P$ | 50 | Number of fixed logistic sub-bands spanning the potential window. About one every 40 mV is enough: 50 for a 2 V window, 25 for a 1 V window. |
| Sub-band width | $\gamma$ | 38.92 V⁻¹ | Shared inverse width, held fixed rather than fitted. The default is $F/RT$ at 298 K, the ideal one-electron Nernstian limit. |

### Numerical

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| Spectral terms | — | 20 | Modes retained in the spectral solution. Truncating at 20 changes the simulated current by roughly 1%, but the error is smooth and the sub-band weights absorb it: against 60 terms the fitted $D$ moves by under 1% and the per-scan residuals are unchanged. The cost is linear in this number. |
| Iterations per stage | — | 10000 | Maximum L-BFGS-B iterations in each stage. A fit normally stops on the tolerance first, after a few thousand iterations. Capping it at 500 takes the example from about 80 s to 10 s, and moves its RMSE by 0.02 percentage points and the fast fraction by 0.007. |
| Tolerance | $f_{\mathrm{tol}}$ | $10^{-9}$ | Relative convergence tolerance on the objective. Asking for $10^{-12}$ needed about half again as many iterations and moved the fitted diffusivity by one part in $10^5$. |

## Output

The application reports the shared physical parameters, the residual of each scan relative to its measured current range, and the extracted $D(V)$ and density of states, together with an overlay of the measured and simulated voltammograms.

## License

This project is licensed under the MIT License.
