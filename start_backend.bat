@echo off
cd /d "%~dp0"
title CV Curve Fitting Pro - High-Speed JAX Backend
echo ======================================================================
echo   CV Curve Fitting Pro - High-Speed JAX Optimization Backend
echo ======================================================================
echo.
echo Checking Python environment...
python -c "import jax; print('JAX Version:', jax.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] JAX not detected or error importing JAX.
    echo Installing required packages...
    pip install fastapi uvicorn websockets pandas scipy numpy jax
)
echo.
echo Starting FastAPI + JAX Backend on http://127.0.0.1:8000 ...
echo Opening web interface in your browser...
start http://127.0.0.1:8000
echo.
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
