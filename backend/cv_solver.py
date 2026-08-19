import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import scan
import asyncio

def load_and_preprocess_cv_data(df, pot_col, cur_col, scan_rate_v_s, skip_factor):
    if isinstance(df, str):
        df = pd.read_csv(df, sep=None, engine='python')
    if pot_col >= df.shape[1] or cur_col >= df.shape[1]:
        raise ValueError(f"Selected column index (Potential: {pot_col}, Current: {cur_col}) exceeds total available columns ({df.shape[1]}).")
        
    s_pot = pd.to_numeric(df.iloc[:, pot_col], errors='coerce').dropna()
    s_cur = pd.to_numeric(df.iloc[:, cur_col], errors='coerce').dropna()
    
    common_idx = s_pot.index.intersection(s_cur.index)
    raw_potential = s_pot.loc[common_idx].values.astype(np.float64)
    raw_current = s_cur.loc[common_idx].values.astype(np.float64)
    
    if len(raw_potential) == 0:
        raise ValueError(f"No valid numeric data found in Column {pot_col} (Potential) and Column {cur_col} (Current).")
    
    voltage_steps = np.abs(np.diff(raw_potential, prepend=raw_potential[0]))
    raw_time = np.cumsum(voltage_steps) / scan_rate_v_s

    exp_potential = raw_potential[::skip_factor]
    exp_current = raw_current[::skip_factor]
    exp_time = raw_time[::skip_factor]

    return exp_time, exp_potential, exp_current

def extract_physics_priors(potential, turn_idx, num_peaks, v_min, v_max):
    global_params = [2.0, 1.0, 1.0, np.mean(potential), 0.0, 0.1, 1.0, 0.1, 1.0]
    peaks_matrix = np.zeros((num_peaks, 3))
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

@partial(jax.jit, static_argnames=['num_terms'])
def run_fourier_simulation_with_data(time_array, potential_array, diffusivity, beta_left, beta_right, v_center, peaks_matrix, num_terms, thickness):
    n_arr = jnp.arange(1.0, num_terms + 1.0, dtype=jnp.float64)
    wavenumbers = (2.0 * n_arr - 1.0) * jnp.pi / (2.0 * thickness)
    fourier_coeffs = 4.0 / ((2.0 * n_arr - 1.0) * jnp.pi)
    sin_integrals = 1.0 / wavenumbers
    
    dt_arr = jnp.diff(time_array)
    dt_arr = jnp.where(dt_arr <= 0.0, 1e-6, dt_arr)
    
    weights = peaks_matrix[:, 0, jnp.newaxis] 
    v_crits = peaks_matrix[:, 1, jnp.newaxis]
    sharpnesses = peaks_matrix[:, 2, jnp.newaxis]
    
    occ_matrix = weights / (1.0 + jnp.exp(-sharpnesses * (potential_array[jnp.newaxis, :] - v_crits)))
    occ_eq_arr = jnp.sum(occ_matrix, axis=0)
    
    occ_eq_old_arr = occ_eq_arr[:-1]
    occ_eq_new_arr = occ_eq_arr[1:]
    
    beta_arr = jnp.where(potential_array[1:] < v_center, beta_left, beta_right)
    d_val_arr = diffusivity * jnp.exp(beta_arr * (potential_array[1:] - v_center)**2)
    
    k_dt_matrix = jnp.outer(d_val_arr * dt_arr, wavenumbers**2)
    decay_matrix = jnp.exp(-k_dt_matrix)
    
    forcing_factor = jnp.where(
        k_dt_matrix < 1e-8,
        1.0 - k_dt_matrix / 2.0,
        (1.0 - decay_matrix) / k_dt_matrix
    )
    
    base_forcing = jnp.outer(occ_eq_old_arr - occ_eq_new_arr, fourier_coeffs)
    forcing_matrix = base_forcing * forcing_factor
    
    def init_step(carry, xs):
        cum_dec, acc_forc = carry
        dec, forc = xs
        return (cum_dec * dec, acc_forc * dec + forc), None
    
    init_carry = (jnp.ones(num_terms, dtype=jnp.float64), jnp.zeros(num_terms, dtype=jnp.float64))
    (final_cum_dec, final_acc_forc), _ = scan(init_step, init_carry, (decay_matrix, forcing_matrix))
    
    T_m_0 = final_acc_forc / (1.0 - final_cum_dec + 1e-15)
    
    def history_step(fourier_modes, xs):
        dec, forc = xs
        fourier_modes = fourier_modes * dec + forc
        return fourier_modes, fourier_modes
        
    _, fourier_history = scan(history_step, T_m_0, (decay_matrix, forcing_matrix))
    sum_fourier = jnp.dot(fourier_history, sin_integrals)
    total_ions_all = thickness * occ_eq_new_arr + sum_fourier
    total_ions_old_init = thickness * occ_eq_arr[0] + jnp.sum(T_m_0 * sin_integrals)
    total_ions_shifted = jnp.concatenate([jnp.array([total_ions_old_init]), total_ions_all])
    simulated_currents = jnp.diff(total_ions_shifted) / dt_arr
    
    return simulated_currents

def solve_cv(df, config, pot_col, cur_col, queue, loop):
    USER_CONFIG = config
    OPTIMIZER_CONFIG = {
        "max_iter": int(config.get("max_iter", 100)), 
        "tol_ftol": float(config.get("tol_ftol", 1e-8)),
        "tol_gtol": float(config.get("tol_gtol", 1e-7)),
        "num_globals": 9,
        "mult_diff": (USER_CONFIG["film_thickness"]**2) / 10.0,
        "mult_beta": 1.0,
        "mult_offset": 1e-4,
        "mult_bg_a": 1e-4, 
        "mult_bg_k": 10
    }
    
    num_terms = int(config.get("num_terms", 50))
    loss_weight_const = float(config.get("loss_weight_const", 1.0))
    
    exp_time, exp_potential, exp_current = load_and_preprocess_cv_data(
        df, pot_col, cur_col,
        USER_CONFIG["scan_rate_v_s"], USER_CONFIG["skip_factor"]
    )

    exp_time_jax = jnp.array(exp_time)
    exp_potential_jax = jnp.array(exp_potential)
    global_target_current_jax = jnp.array(exp_current[1:].ravel())

    if loop and queue:
        loop.call_soon_threadsafe(
            queue.put_nowait, {
                "type": "init",
                "exp_potential": exp_potential[1:].tolist(),
                "exp_current": exp_current[1:].tolist()
            }
        )

    turn_idx = np.argmax(np.abs(exp_potential - exp_potential[0]))
    if turn_idx < len(exp_potential) * 0.1:
        turn_idx = len(exp_potential) // 2

    smoothed_current = savgol_filter(exp_current, window_length=51, polyorder=3)
    d2I_raw = np.abs(np.diff(smoothed_current, n=2))
    d2I = np.pad(d2I_raw, (1, 1), mode='edge')
    loss_weights = (d2I / np.max(d2I)) + loss_weight_const

    edge_threshold = (USER_CONFIG["v_max"] - USER_CONFIG["v_min"]) * 0.05
    left_mask = exp_potential[1:] < (USER_CONFIG["v_min"] + edge_threshold)
    right_mask = exp_potential[1:] > (USER_CONFIG["v_max"] - edge_threshold)

    global_weights = loss_weights[1:].ravel().copy()
    global_weights_masked = global_weights.copy()
    global_weights_masked[left_mask] = 0.0
    global_weights_masked[right_mask] = 0.0

    dt_array = np.diff(exp_time)
    dt_array[dt_array <= 0] = 1e-6

    data_driven_initial_guess = extract_physics_priors(
        exp_potential, turn_idx, USER_CONFIG["num_peaks"], USER_CONFIG["v_min"], USER_CONFIG["v_max"]
    )

    initial_peaks_matrix = data_driven_initial_guess[OPTIMIZER_CONFIG["num_globals"]:].reshape((-1, 3))
    baseline_diffusivity = data_driven_initial_guess[0] * OPTIMIZER_CONFIG["mult_diff"]

    calibration_sim = np.array(run_fourier_simulation_with_data(
        exp_time_jax, exp_potential_jax,
        baseline_diffusivity, 0.0, 0.0,
        data_driven_initial_guess[3],
        jnp.array(initial_peaks_matrix), num_terms, USER_CONFIG["film_thickness"]
    ))

    pure_faradaic_target = exp_current[1:]
    v_range = USER_CONFIG["v_max"] - USER_CONFIG["v_min"]
    safe_min = USER_CONFIG["v_min"] + (v_range * 0.15)
    safe_max = USER_CONFIG["v_max"] - (v_range * 0.15)
    safe_mask = (exp_potential[1:] > safe_min) & (exp_potential[1:] < safe_max)

    real_faradaic_ptp = np.ptp(pure_faradaic_target[safe_mask]) 
    sim_ptp = np.ptp(calibration_sim[safe_mask])
    if sim_ptp < 1e-12: sim_ptp = 1e-6 

    calibrated_scale = real_faradaic_ptp / sim_ptp
    total_simulated_baseline = (calibration_sim * calibrated_scale)
    real_mean = np.mean(exp_current[1:])
    sim_mean = np.mean(total_simulated_baseline)
    calibrated_offset = (real_mean - sim_mean) / OPTIMIZER_CONFIG["mult_offset"]
    data_driven_initial_guess[4] = calibrated_offset
    calibrated_scale_jax = jnp.array(calibrated_scale)

    @jax.jit
    def compute_forward(scaled_params, weights):
        use_tafel = float(USER_CONFIG.get("use_tafel", True))
        diffusivity = scaled_params[0] * OPTIMIZER_CONFIG["mult_diff"]
        beta_left = scaled_params[1] * OPTIMIZER_CONFIG["mult_beta"]
        beta_right = scaled_params[2] * OPTIMIZER_CONFIG["mult_beta"]
        v_center = scaled_params[3]
        baseline_offset = scaled_params[4] * OPTIMIZER_CONFIG["mult_offset"]
        a_right = scaled_params[5] * OPTIMIZER_CONFIG["mult_bg_a"] * use_tafel
        k_right = scaled_params[6] * OPTIMIZER_CONFIG["mult_bg_k"]
        a_left  = scaled_params[7] * OPTIMIZER_CONFIG["mult_bg_a"] * use_tafel
        k_left  = scaled_params[8] * OPTIMIZER_CONFIG["mult_bg_k"]
        peaks_matrix = jnp.reshape(scaled_params[OPTIMIZER_CONFIG["num_globals"]:], (-1, 3))
            
        simulated_currents = run_fourier_simulation_with_data(
            exp_time_jax, exp_potential_jax,
            diffusivity, beta_left, beta_right, v_center, peaks_matrix, num_terms, USER_CONFIG["film_thickness"]
        )
        bg_current = a_right * jnp.exp(k_right * (exp_potential_jax[1:] - USER_CONFIG["v_max"])) \
                   - a_left * jnp.exp(-k_left * (exp_potential_jax[1:] - USER_CONFIG["v_min"]))
        final_sim = (simulated_currents * calibrated_scale_jax) + baseline_offset + bg_current
        squared_errors = (final_sim - global_target_current_jax)**2
        weighted_mse = jnp.average(squared_errors, weights=weights)
        loss = jnp.sqrt(weighted_mse) * 1e6 
        return loss, final_sim

    @jax.jit
    def objective_function_jax(scaled_params, weights):
        loss, _ = compute_forward(scaled_params, weights)
        return loss

    loss_and_grad_jax = jax.jit(jax.value_and_grad(objective_function_jax, argnums=0))

    def scipy_objective(x, weights):
        loss, grad = loss_and_grad_jax(jnp.array(x), jnp.array(weights))
        return np.array(loss, dtype=np.float64), np.array(grad, dtype=np.float64)

    class OptimizationTracker:
        def __init__(self):
            self.iter_count = 0
            self.stage_label = ""
            self.active_weights = None

        def set_stage(self, label, weights):
            self.stage_label = label
            self.active_weights = weights
            self.iter_count = 0

        def __call__(self, xk):
            if self.iter_count % 10 == 0:
                loss, final_sim = compute_forward(jnp.array(xk), jnp.array(self.active_weights))
                # Send update to queue via loop
                if loop and queue:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {
                            "type": "update",
                            "stage": self.stage_label,
                            "iter": self.iter_count,
                            "loss": float(loss),
                            "sim_current": np.array(final_sim).tolist()
                        }
                    )
            self.iter_count += 1

    tracker = OptimizationTracker()

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

    current_x = data_driven_initial_guess
    
    # Send initial data arrays to client so it can setup the base plot
    if loop and queue:
        loop.call_soon_threadsafe(
            queue.put_nowait, {
                "type": "init",
                "exp_potential": exp_potential[1:].tolist(),
                "exp_current": exp_current[1:].tolist()
            }
        )

    for label, active_idx, weights in stages:
        tracker.set_stage(label, weights)
        bounds = create_staged_bounds(current_x, active_idx, USER_CONFIG["v_min"], USER_CONFIG["v_max"], OPTIMIZER_CONFIG["num_globals"])
        res = minimize(
            scipy_objective, 
            current_x, 
            args=(weights,), 
            bounds=bounds, 
            jac=True, 
            method='L-BFGS-B', 
            callback=tracker,
            options=optim_options 
        )
        current_x = res.x

    final_result = res
    final_peaks = final_result.x[OPTIMIZER_CONFIG["num_globals"]:].reshape((-1, 3))
    
    # Generate plot data
    diffusivity = final_result.x[0] * OPTIMIZER_CONFIG["mult_diff"]
    beta_left   = final_result.x[1] * OPTIMIZER_CONFIG["mult_beta"]
    beta_right  = final_result.x[2] * OPTIMIZER_CONFIG["mult_beta"]
    v_center    = final_result.x[3]
    baseline_offset = final_result.x[4] * OPTIMIZER_CONFIG["mult_offset"]

    v_plot = np.linspace(USER_CONFIG["v_min"], USER_CONFIG["v_max"], 500)
    beta_plot = np.where(v_plot < v_center, beta_left, beta_right)
    d_of_v = diffusivity * np.exp(beta_plot * (v_plot - v_center)**2)

    weights_p = final_peaks[:, 0, np.newaxis]
    v_crits = final_peaks[:, 1, np.newaxis]
    sharpnesses = final_peaks[:, 2, np.newaxis]

    exp_terms = np.exp(-sharpnesses * (v_plot - v_crits))
    dos_matrix = weights_p * sharpnesses * exp_terms / (1.0 + exp_terms)**2
    dos_total = np.sum(dos_matrix, axis=0)
    
    _, final_sim = compute_forward(jnp.array(final_result.x), jnp.array(global_weights))

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
            "dos_matrix": dos_matrix.T.tolist(),
            "exp_potential": exp_potential[1:].tolist(),
            "exp_current": exp_current[1:].tolist(),
            "sim_current": np.array(final_sim).tolist()
        }
    }
    
    if loop and queue:
        loop.call_soon_threadsafe(
            queue.put_nowait, {
                "type": "done",
                "data": result_data
            }
        )
    return result_data
