// CV Curve Fitting Pro - Web Worker Solver powered by Pyodide (WASM)
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null;
let isReady = false;

const pythonSolverCode = `
import numpy as np
import pandas as pd
import io
import json
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import js

def load_and_preprocess_cv_data(df, pot_col, cur_col, scan_rate_v_s, skip_factor):
    raw_potential = df.iloc[:, pot_col].dropna().values.astype(np.float64)
    raw_current = df.iloc[:, cur_col].dropna().values.astype(np.float64)
    
    min_len = min(len(raw_potential), len(raw_current))
    raw_potential = raw_potential[:min_len]
    raw_current = raw_current[:min_len]
    
    voltage_steps = np.abs(np.diff(raw_potential, prepend=raw_potential[0]))
    raw_time = np.cumsum(voltage_steps) / scan_rate_v_s

    exp_potential = raw_potential[::skip_factor]
    exp_current = raw_current[::skip_factor]
    exp_time = raw_time[::skip_factor]

    return exp_time, exp_potential, exp_current

def extract_physics_priors(potential, turn_idx, num_peaks, v_min, v_max):
    global_params = [2.0, 1.0, 1.0, float(np.mean(potential)), 0.0, 0.1, 1.0, 0.1, 1.0]
    peaks_matrix = np.zeros((num_peaks, 3), dtype=np.float64)
    v_crits = np.linspace(v_min + 0.1, v_max - 0.1, num_peaks)
    for i in range(num_peaks):
        peaks_matrix[i] = [1.0, v_crits[i], 15.0]
    return np.concatenate([global_params, peaks_matrix.flatten()])

def get_parameter_bounds(idx, val, num_globals, v_min, v_max):
    var = np.abs(val) * 0.5
    if idx == 0: return (max(1e-8, val - 5.0), val + 5.0)
    if idx in (1, 2): return (val - 1.0, val + 1.0)
    if idx == 3: return (val - 0.6, val + 0.6)
    if idx == 4:
        var = var if val != 0 else 10.0
        return (val - var, val + var)
    if idx in (5, 7): return (max(1e-8, val - (var + 0.1)), val + var + 0.1)
    if idx in (6, 8): return (max(0.1, val - (var + 0.5)), val + var + 0.5)
    offset = (idx - num_globals) % 3
    if offset == 0: return (max(1e-4, val - (var + 1e-4)), val + 5.0)
    if offset == 1: return (max(v_min, val - (var + 1e-4)), min(v_max, val + var + 1e-4))
    return (max(0.1, val - (var + 1e-4)), val + 20.0)

def create_staged_bounds(target_params, active_indices, v_min, v_max, num_globals):
    bounds = []
    for i, val in enumerate(target_params):
        if i not in active_indices:
            bounds.append((val - 1e-9, val + 1e-9))
        else:
            bounds.append(get_parameter_bounds(i, val, num_globals, v_min, v_max))
    return bounds

def run_fourier_simulation_numpy(time_array, potential_array, diffusivity, beta_left, beta_right, v_center, peaks_matrix, num_terms, thickness):
    n_arr = np.arange(1.0, num_terms + 1.0, dtype=np.float64)
    wavenumbers = (2.0 * n_arr - 1.0) * np.pi / (2.0 * thickness)
    fourier_coeffs = 4.0 / ((2.0 * n_arr - 1.0) * np.pi)
    sin_integrals = 1.0 / wavenumbers
    
    dt_arr = np.diff(time_array)
    dt_arr = np.where(dt_arr <= 0.0, 1e-6, dt_arr)
    
    weights = peaks_matrix[:, 0, np.newaxis] 
    v_crits = peaks_matrix[:, 1, np.newaxis]
    sharpnesses = peaks_matrix[:, 2, np.newaxis]
    
    occ_matrix = weights / (1.0 + np.exp(-sharpnesses * (potential_array[np.newaxis, :] - v_crits)))
    occ_eq_arr = np.sum(occ_matrix, axis=0)
    
    occ_eq_old_arr = occ_eq_arr[:-1]
    occ_eq_new_arr = occ_eq_arr[1:]
    
    beta_arr = np.where(potential_array[1:] < v_center, beta_left, beta_right)
    d_val_arr = diffusivity * np.exp(beta_arr * (potential_array[1:] - v_center)**2)
    
    k_dt_matrix = np.outer(d_val_arr * dt_arr, wavenumbers**2)
    decay_matrix = np.exp(-k_dt_matrix)
    
    forcing_factor = np.where(
        k_dt_matrix < 1e-8,
        1.0 - k_dt_matrix / 2.0,
        (1.0 - decay_matrix) / (k_dt_matrix + 1e-15)
    )
    
    base_forcing = np.outer(occ_eq_old_arr - occ_eq_new_arr, fourier_coeffs)
    forcing_matrix = base_forcing * forcing_factor
    
    N_steps = len(dt_arr)
    cum_dec = np.ones(num_terms, dtype=np.float64)
    acc_forc = np.zeros(num_terms, dtype=np.float64)
    
    for i in range(N_steps):
        dec = decay_matrix[i]
        forc = forcing_matrix[i]
        cum_dec = cum_dec * dec
        acc_forc = acc_forc * dec + forc
        
    T_m_0 = acc_forc / (1.0 - cum_dec + 1e-15)
    
    fourier_history = np.empty((N_steps, num_terms), dtype=np.float64)
    fourier_modes = T_m_0.copy()
    for i in range(N_steps):
        fourier_modes = fourier_modes * decay_matrix[i] + forcing_matrix[i]
        fourier_history[i] = fourier_modes
        
    sum_fourier = np.dot(fourier_history, sin_integrals)
    total_ions_all = thickness * occ_eq_new_arr + sum_fourier
    total_ions_old_init = thickness * occ_eq_arr[0] + np.sum(T_m_0 * sin_integrals)
    total_ions_shifted = np.concatenate([[total_ions_old_init], total_ions_all])
    simulated_currents = np.diff(total_ions_shifted) / dt_arr
    
    return simulated_currents

def solve_cv_pyodide(file_content, config_json_str):
    config = json.loads(config_json_str)
    
    scan_rate_v_s = float(config.get("scan_rate", 0.010))
    film_thickness = float(config.get("film_thickness", 1e-4))
    v_min = float(config.get("v_min", -1.0))
    v_max = float(config.get("v_max", 1.0))
    skip_factor = int(config.get("skip_factor", 10))
    num_peaks = int(config.get("num_peaks", 30))
    max_iter = int(config.get("max_iter", 100))
    tol_ftol = float(config.get("tol_ftol", 1e-8))
    tol_gtol = float(config.get("tol_gtol", 1e-7))
    num_terms = int(config.get("num_terms", 50))
    loss_weight_const = float(config.get("loss_weight_const", 1.0))
    pot_col = int(config.get("pot_col", 8))
    cur_col = int(config.get("cur_col", 9))
    
    OPTIMIZER_CONFIG = {
        "max_iter": max_iter, 
        "tol_ftol": tol_ftol,
        "tol_gtol": tol_gtol,
        "num_globals": 9,
        "mult_diff": (film_thickness**2) / 10.0,
        "mult_beta": 1.0,
        "mult_offset": 1e-4,
        "mult_bg_a": 1e-4, 
        "mult_bg_k": 10.0
    }
    
    # Read CSV
    df = pd.read_csv(io.StringIO(file_content), sep=None, engine='python')
    
    exp_time, exp_potential, exp_current = load_and_preprocess_cv_data(
        df, pot_col, cur_col, scan_rate_v_s, skip_factor
    )
    
    global_target_current = exp_current[1:].ravel()

    turn_idx = np.argmax(np.abs(exp_potential - exp_potential[0]))
    if turn_idx < len(exp_potential) * 0.1:
        turn_idx = len(exp_potential) // 2

    window_len = min(51, len(exp_current) if len(exp_current) % 2 != 0 else len(exp_current) - 1)
    if window_len < 5:
        smoothed_current = exp_current
    else:
        smoothed_current = savgol_filter(exp_current, window_length=window_len, polyorder=3)
    
    d2I_raw = np.abs(np.diff(smoothed_current, n=2))
    d2I = np.pad(d2I_raw, (1, 1), mode='edge')
    max_d2I = np.max(d2I) if np.max(d2I) > 0 else 1.0
    loss_weights = (d2I / max_d2I) + loss_weight_const

    edge_threshold = (v_max - v_min) * 0.05
    left_mask = exp_potential[1:] < (v_min + edge_threshold)
    right_mask = exp_potential[1:] > (v_max - edge_threshold)

    global_weights = loss_weights[1:].ravel().copy()
    global_weights_masked = global_weights.copy()
    global_weights_masked[left_mask] = 0.0
    global_weights_masked[right_mask] = 0.0

    data_driven_initial_guess = extract_physics_priors(
        exp_potential, turn_idx, num_peaks, v_min, v_max
    )

    initial_peaks_matrix = data_driven_initial_guess[OPTIMIZER_CONFIG["num_globals"]:].reshape((-1, 3))
    baseline_diffusivity = data_driven_initial_guess[0] * OPTIMIZER_CONFIG["mult_diff"]

    calibration_sim = run_fourier_simulation_numpy(
        exp_time, exp_potential,
        baseline_diffusivity, 0.0, 0.0,
        data_driven_initial_guess[3],
        initial_peaks_matrix, num_terms, film_thickness
    )

    pure_faradaic_target = exp_current[1:]
    v_range = v_max - v_min
    safe_min = v_min + (v_range * 0.15)
    safe_max = v_max - (v_range * 0.15)
    safe_mask = (exp_potential[1:] > safe_min) & (exp_potential[1:] < safe_max)

    if np.any(safe_mask):
        real_faradaic_ptp = np.ptp(pure_faradaic_target[safe_mask])
        sim_ptp = np.ptp(calibration_sim[safe_mask])
    else:
        real_faradaic_ptp = np.ptp(pure_faradaic_target)
        sim_ptp = np.ptp(calibration_sim)
        
    if sim_ptp < 1e-12: sim_ptp = 1e-6 

    calibrated_scale = real_faradaic_ptp / sim_ptp
    total_simulated_baseline = (calibration_sim * calibrated_scale)
    real_mean = np.mean(exp_current[1:])
    sim_mean = np.mean(total_simulated_baseline)
    calibrated_offset = (real_mean - sim_mean) / OPTIMIZER_CONFIG["mult_offset"]
    data_driven_initial_guess[4] = calibrated_offset

    # Notify UI of initialization
    js.postMessage(json.dumps({
        "type": "init",
        "exp_potential": exp_potential[1:].tolist(),
        "exp_current": exp_current[1:].tolist()
    }))

    def compute_forward(scaled_params, weights):
        diffusivity = scaled_params[0] * OPTIMIZER_CONFIG["mult_diff"]
        beta_left = scaled_params[1] * OPTIMIZER_CONFIG["mult_beta"]
        beta_right = scaled_params[2] * OPTIMIZER_CONFIG["mult_beta"]
        v_center = scaled_params[3]
        baseline_offset = scaled_params[4] * OPTIMIZER_CONFIG["mult_offset"]
        a_right = scaled_params[5] * OPTIMIZER_CONFIG["mult_bg_a"]
        k_right = scaled_params[6] * OPTIMIZER_CONFIG["mult_bg_k"]
        a_left  = scaled_params[7] * OPTIMIZER_CONFIG["mult_bg_a"]
        k_left  = scaled_params[8] * OPTIMIZER_CONFIG["mult_bg_k"]
        peaks_matrix = np.reshape(scaled_params[OPTIMIZER_CONFIG["num_globals"]:], (-1, 3))
            
        simulated_currents = run_fourier_simulation_numpy(
            exp_time, exp_potential,
            diffusivity, beta_left, beta_right, v_center, peaks_matrix, num_terms, film_thickness
        )
        bg_current = a_right * np.exp(k_right * (exp_potential[1:] - v_max)) \
                   - a_left * np.exp(-k_left * (exp_potential[1:] - v_min))
        final_sim = (simulated_currents * calibrated_scale) + baseline_offset + bg_current
        squared_errors = (final_sim - global_target_current)**2
        weighted_mse = np.average(squared_errors, weights=weights)
        loss = np.sqrt(weighted_mse) * 1e6 
        return loss, final_sim

    def objective_func(params, weights):
        loss, _ = compute_forward(params, weights)
        return loss

    class PyOptimizationTracker:
        def __init__(self):
            self.iter_count = 0
            self.stage_label = ""
            self.active_weights = None

        def set_stage(self, label, weights):
            self.stage_label = label
            self.active_weights = weights
            self.iter_count = 0

        def __call__(self, xk):
            self.iter_count += 1
            if self.iter_count % 5 == 0 or self.iter_count == 1:
                loss, final_sim = compute_forward(xk, self.active_weights)
                js.postMessage(json.dumps({
                    "type": "update",
                    "stage": self.stage_label,
                    "iter": self.iter_count,
                    "loss": float(loss),
                    "sim_current": final_sim.tolist()
                }))

    tracker = PyOptimizationTracker()

    all_indices = list(range(len(data_driven_initial_guess)))
    idx_baseline = [4]
    idx_bg = [5, 6, 7, 8]
    idx_diffusion_base = [0, 3]
    idx_beta = [1, 2]
    idx_peaks = list(range(OPTIMIZER_CONFIG["num_globals"], len(data_driven_initial_guess)))

    optim_options = {
        'maxiter': OPTIMIZER_CONFIG["max_iter"],
        'ftol': OPTIMIZER_CONFIG["tol_ftol"],
        'gtol': OPTIMIZER_CONFIG["tol_gtol"],
        'disp': False
    }

    stages = [
        ("Stage 1: Pure Flat Baseline", idx_baseline, global_weights_masked),
        ("Stage 1.5: Background Tails", idx_bg, global_weights),
        ("Stage 2: Anchor Peaks (Constant D)", idx_baseline + idx_bg + idx_diffusion_base + idx_peaks, global_weights),
        ("Stage 3: Full Non-Linear Polish", all_indices, global_weights)
    ]

    current_x = data_driven_initial_guess.copy()

    for label, active_idx, weights in stages:
        tracker.set_stage(label, weights)
        bounds = create_staged_bounds(current_x, active_idx, v_min, v_max, OPTIMIZER_CONFIG["num_globals"])
        res = minimize(
            objective_func, 
            current_x, 
            args=(weights,), 
            bounds=bounds, 
            method='L-BFGS-B', 
            callback=tracker,
            options=optim_options 
        )
        current_x = res.x

    final_result = res
    final_peaks = final_result.x[OPTIMIZER_CONFIG["num_globals"]:].reshape((-1, 3))
    
    diffusivity = final_result.x[0] * OPTIMIZER_CONFIG["mult_diff"]
    beta_left   = final_result.x[1] * OPTIMIZER_CONFIG["mult_beta"]
    beta_right  = final_result.x[2] * OPTIMIZER_CONFIG["mult_beta"]
    v_center    = final_result.x[3]
    baseline_offset = final_result.x[4] * OPTIMIZER_CONFIG["mult_offset"]

    v_plot = np.linspace(v_min, v_max, 500)
    beta_plot = np.where(v_plot < v_center, beta_left, beta_right)
    d_of_v = diffusivity * np.exp(beta_plot * (v_plot - v_center)**2)

    weights_p = final_peaks[:, 0, np.newaxis]
    v_crits = final_peaks[:, 1, np.newaxis]
    sharpnesses = final_peaks[:, 2, np.newaxis]

    exp_terms = np.exp(-sharpnesses * (v_plot - v_crits))
    dos_matrix = weights_p * sharpnesses * exp_terms / (1.0 + exp_terms)**2
    dos_total = np.sum(dos_matrix, axis=0)
    
    _, final_sim = compute_forward(final_result.x, global_weights)

    result_data = {
        "parameters": {
            "diffusivity": float(diffusivity),
            "beta_left": float(beta_left),
            "beta_right": float(beta_right),
            "baseline_offset": float(baseline_offset),
            "v_center": float(v_center)
        },
        "plots": {
            "v_plot": v_plot.tolist(),
            "d_of_v": d_of_v.tolist(),
            "dos_total": dos_total.tolist(),
            "dos_matrix": dos_matrix.tolist(),
            "exp_potential": exp_potential[1:].tolist(),
            "exp_current": exp_current[1:].tolist(),
            "sim_current": final_sim.tolist()
        }
    }
    
    js.postMessage(json.dumps({
        "type": "done",
        "data": result_data
    }))
`;

async function init() {
    try {
        postMessage(JSON.stringify({ type: 'status', message: 'Initializing Pyodide WebAssembly Engine...' }));
        pyodide = await loadPyodide();
        postMessage(JSON.stringify({ type: 'status', message: 'Loading Scientific Packages (NumPy, SciPy, Pandas)...' }));
        await pyodide.loadPackage(['numpy', 'scipy', 'pandas']);
        
        postMessage(JSON.stringify({ type: 'status', message: 'Compiling Physics Solver Model...' }));
        await pyodide.runPythonAsync(pythonSolverCode);
        
        isReady = true;
        postMessage(JSON.stringify({ type: 'ready', message: 'WASM Physics Engine Ready' }));
    } catch (err) {
        postMessage(JSON.stringify({ type: 'error', message: 'Failed to initialize Pyodide engine: ' + err.message }));
    }
}

init();

self.onmessage = async (e) => {
    if (!isReady) {
        postMessage(JSON.stringify({ type: 'error', message: 'Optimization engine is still initializing. Please wait a few seconds...' }));
        return;
    }
    
    const { action, file_content, config } = e.data;
    if (action === 'solve') {
        try {
            pyodide.globals.set("temp_file_content", file_content);
            pyodide.globals.set("temp_config_json", JSON.stringify(config));
            
            await pyodide.runPythonAsync(`
                try:
                    solve_cv_pyodide(temp_file_content, temp_config_json)
                except Exception as e:
                    import traceback
                    import js, json
                    js.postMessage(json.dumps({
                        "type": "error",
                        "message": str(e),
                        "trace": traceback.format_exc()
                    }))
            `);
        } catch (err) {
            postMessage(JSON.stringify({ type: 'error', message: err.message }));
        }
    }
};
