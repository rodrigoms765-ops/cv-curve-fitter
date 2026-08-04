# CV Curve Fitting Pro

A professional, high-performance web application for fitting Cyclic Voltammetry (CV) curves using a physics-based Fourier diffusion model.

Powered by **WebAssembly (Pyodide, SciPy, NumPy)** running **100% in-browser on the client's local CPU** — zero server costs, zero cold starts, and zero CPU throttling.

---

## 🌐 Live Website (GitHub Pages)

The application is deployed directly on GitHub Pages:

👉 **[https://rodrigoms765-ops.github.io/cv-curve-fitter/](https://rodrigoms765-ops.github.io/cv-curve-fitter/)**

---

## ✨ Key Features

- **⚡ Client-Side WebAssembly (WASM)**: Solves CV curves directly on your machine's CPU via Pyodide and SciPy L-BFGS-B optimization.
- **🔄 4-Stage Physics Solver**:
  1. *Stage 1*: Flat baseline calibration
  2. *Stage 1.5*: Background exponential edge tails
  3. *Stage 2*: Peak anchoring with constant diffusivity
  4. *Stage 3*: Non-linear global polish
- **📊 Real-Time Interactive Visualizations**: Live Plotly streaming of the fitted CV curve, Density of States (DOS), and potential-dependent Diffusivity $D(V)$.
- **📥 Instant One-Click Demo**: Includes synthetic CV dataset for instant in-browser demonstration.
- **🚀 Zero Backend Dependencies**: Hosted statically on GitHub Pages with automatic GitHub Actions deployment.

---

## 🛠️ How to Enable GitHub Pages on Your Repo

1. Open your repository on GitHub: **[github.com/rodrigoms765-ops/cv-curve-fitter](https://github.com/rodrigoms765-ops/cv-curve-fitter)**
2. Go to **Settings** > **Pages** (in the left sidebar).
3. Under **Build and deployment**:
   - **Source**: Select **GitHub Actions** (or select *Deploy from a branch* -> `main` -> `/ (root)`).
4. On your next push to `main`, GitHub Actions will automatically deploy your live site!

---

## 💻 Running Locally

Because the application is 100% static, you can run it locally with any simple HTTP server:

### Option 1: Python HTTP Server (Built-in)
```bash
# In the project root directory
python -m http.server 8000
```
Open your browser at **http://localhost:8000**.

### Option 2: VS Code Live Server
Right-click `index.html` and select **"Open with Live Server"**.

---

## 📈 Usage Guide

1. **Input Data**: Upload your experimental `.csv` CV file or click **⚡ Load Sample CV Data**.
2. **Column Mapping**: Ensure `Potential Col #` and `Current Col #` match your CSV structure (0-indexed).
3. **Set Parameters**: Adjust scan rate, film thickness, potential range, and peak modes if needed.
4. **Execute**: Click **Run Multi-Stage Optimization**.
5. **Analyze & Export**: Inspect extracted physical parameters ($D_0$, $\beta_{left}$, $\beta_{right}$, $V_0$) and download the result JSON report.

---

## 📄 License
MIT License
