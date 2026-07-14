# CV Curve Fitting Pro

A professional, high-performance web application for fitting Cyclic Voltammetry (CV) curves using a physics-based diffusion model. Built with Python (JAX, SciPy, FastAPI) and Vanilla JS/HTML/CSS.

## Cloud Access

You do not need to install this software to use it. A live cloud version is available here:
**[Insert your Render URL here]**

If you wish to run it locally for maximum performance (which utilizes your local CPU's multiple cores and AVX instructions via JAX), follow the instructions below.

## Local Installation

### Prerequisites
- Python 3.8 or higher installed on your machine.
- Git (optional, for cloning).

### Setup Instructions

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/rodrigoms765-ops/cv-curve-fitter.git
   cd cv-curve-fitter
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Navigate to the `backend` folder and install the required Python packages.
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Running the Application

Once the dependencies are installed, you can start the local server:

```bash
# Make sure you are in the backend directory
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open your web browser and navigate to: **http://localhost:8000**
The UI will load automatically.

## Usage
1. Upload your `.csv` CV data file.
2. Ensure the "Potential Column Index" and "Current Column Index" correctly match your CSV file's structure (Note: 0-indexed).
3. Adjust physical parameters and advanced settings (like Data Downsample Factor or Number of Fourier Modes) to balance speed and accuracy.
4. Click **Run Optimization** and watch the real-time fit evolve.
