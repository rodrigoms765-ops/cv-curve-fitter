# Cyclic Voltammetry Diffusion Model Fitting

A client-side scientific tool for parameter estimation and physics-based diffusion fitting of Cyclic Voltammetry (CV) data. Computations run locally in the browser using WebAssembly (Pyodide, SciPy, and NumPy).

## Live Application

The tool is accessible at:
**[https://rodrigoms765-ops.github.io/cv-curve-fitter/](https://rodrigoms765-ops.github.io/cv-curve-fitter/)**

---

## Method Summary

The model fits experimental CV data through a 4-stage optimization procedure using L-BFGS-B minimization:
1. **Baseline Extraction**: Identifies constant and background current offsets.
2. **Background Exponential Tails**: Fits exponential charging behaviors near potential limits.
3. **Peak Anchoring**: Optimizes discrete potential state distributions under baseline diffusivity.
4. **Non-Linear Global Polish**: Simultaneously refines potential-dependent diffusivity $D(E) = D_0 \exp(\beta (E - E_0)^2)$ and state densities.

---

## Local Execution

To run locally without internet deployment:

```bash
# Start a local web server in the repository directory
python -m http.server 8000
```

Open `http://localhost:8000` in any modern web browser.

---

## Enabling GitHub Pages

If the live URL returns a 404 error:
1. Open the repository on GitHub: **[github.com/rodrigoms765-ops/cv-curve-fitter](https://github.com/rodrigoms765-ops/cv-curve-fitter)**
2. Navigate to **Settings** &rarr; **Pages** (in the left-hand menu).
3. Under **Build and deployment**:
   - Set **Source** to **Deploy from a branch**.
   - Set **Branch** to `main` and **Folder** to `/ (root)`.
   - Click **Save**.
4. Allow 1–2 minutes for GitHub Pages to complete initial provisioning.
