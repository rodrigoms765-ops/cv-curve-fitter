import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import jax
jax.config.update("jax_enable_x64", True) #minimizes flop errors during scans
import jax.numpy as jnp
from jax.lax import scan

#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_0p1gL_2cm2\1p0k 30s 0p1g 2cm2_-1V1V_10mVs.csv"
#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_1p0gL_2cm2\1p0k 30s 1p0g 2cm2_-1V1V_10mVs.csv"
#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_1p0gL_4cm2\1p0k 30s 1p0g 4cm2_-1V1V_10mVs.csv"
#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_10p0gL_2cm2\1p0k 30s 10p0g 2cm2_-1V1V_10mVs.csv"
#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\1000rpm_10p0gL_3cm2\1p0k 30s 10p0g 3cm2_-1V1V_10mVs.csv"
#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\2000rpm_1p0gL_2cm2\2p0k 30s 1p0g 2cm2_-1V1V_10mVs.csv"
#"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\2000rpm_1p0gL_3cm2\2p0k 30s 1p0g 3cm2_-1V1V_10mVs.csv"

USER_CONFIG = {
    "filepath": r"C:\Users\rodri\OneDrive\Documents\Chem Project\Rodrigo CV\2000rpm_1p0gL_3cm2\2p0k 30s 1p0g 3cm2_-1V1V_10mVs.csv",
    "scan_rate_v_s": 0.010, #from file
    "skip_factor": 10, #skipping data points
    "film_thickness": 1e-4, #L = 1e-7 experimentally
    "v_min": -1.0, #scan region min
    "v_max": 1.0, #scan region max
    "num_peaks": 30 # num of gaussians forming DOS
}

OPTIMIZER_CONFIG = {
    "max_iter": 100, 
    "tol_ftol": 1e-8, #function tolerance
    "tol_gtol": 1e-7, #gradient tolerance
    "num_globals": 9,  # D0, beta_L, beta_R, V_center, offset, a_R, k_R, a_L, k_L
    "mult_diff": (USER_CONFIG["film_thickness"]**2) / 10.0,
    "mult_beta": 1.0, #from D(V)
    "mult_offset": 1e-4, #experimental apram
    "mult_bg_a": 1e-4, 
    "mult_bg_k": 10
}

@jax.jit(static_argnames=['num_terms']) #just-in-time compilation, hard codes array shapes w/ num_terms
def run_fourier_simulation_with_data(time_array, potential_array, diffusivity, beta_left, beta_right, v_center, peaks_matrix, num_terms, thickness):
    n_arr = jnp.arange(1.0, num_terms + 1.0, dtype=jnp.float64)
    wavenumbers = (2.0 * n_arr - 1.0) * jnp.pi / (2.0 * thickness)
    fourier_coeffs = 4.0 / ((2.0 * n_arr - 1.0) * jnp.pi)
    sin_integrals = 1.0 / wavenumbers
    
    dt_arr = jnp.diff(time_array) #time interval in data
    dt_arr = jnp.where(dt_arr <= 0.0, 1e-6, dt_arr) #replace bad vals w 1e-6, o/w keep dt
    
    # 3 params / gaussian
    weights = peaks_matrix[:, 0, jnp.newaxis] 
    v_crits = peaks_matrix[:, 1, jnp.newaxis]
    sharpnesses = peaks_matrix[:, 2, jnp.newaxis]
    
    # matrix of fermi-diracs (approximating erf)
    occ_matrix = weights / (1.0 + jnp.exp(-sharpnesses * (potential_array[jnp.newaxis, :] - v_crits)))
    occ_eq_arr = jnp.sum(occ_matrix, axis=0)
    
    occ_eq_old_arr = occ_eq_arr[:-1]
    occ_eq_new_arr = occ_eq_arr[1:]
    
    #splits into two diffusion coefficients (left vs right)
    beta_arr = jnp.where(potential_array[1:] < v_center, beta_left, beta_right)
    #diffusion coeff as gaussian from paper (D(V))
    d_val_arr = diffusivity * jnp.exp(beta_arr * (potential_array[1:] - v_center)**2)
    

    # math for T(t) ODE
    k_dt_matrix = jnp.outer(d_val_arr * dt_arr, wavenumbers**2)
    decay_matrix = jnp.exp(-k_dt_matrix)
    
    forcing_factor = jnp.where(
        k_dt_matrix < 1e-8,
        1.0 - k_dt_matrix / 2.0,
        (1.0 - decay_matrix) / k_dt_matrix
    )
    
    base_forcing = jnp.outer(occ_eq_old_arr - occ_eq_new_arr, fourier_coeffs)
    forcing_matrix = base_forcing * forcing_factor
    
    #weighted sum of past forcing terms (X_new = X_old * decay + forcing)
    def init_step(carry, xs):
        cum_dec, acc_forc = carry
        dec, forc = xs
        return (cum_dec * dec, acc_forc * dec + forc), None
    
    init_carry = (jnp.ones(num_terms, dtype=jnp.float64), jnp.zeros(num_terms, dtype=jnp.float64))
    (final_cum_dec, final_acc_forc), _ = scan(init_step, init_carry, (decay_matrix, forcing_matrix))
    
    T_m_0 = final_acc_forc / (1.0 - final_cum_dec + 1e-15)
    
    #similar to init_step, but for t != t_0
    def history_step(fourier_modes, xs):
        dec, forc = xs
        fourier_modes = fourier_modes * dec + forc
        return fourier_modes, fourier_modes
        
    _, fourier_history = scan(history_step, T_m_0, (decay_matrix, forcing_matrix))
    
    sum_fourier = jnp.dot(fourier_history, sin_integrals)
    
    #calculate N = integral of solution dx
    total_ions_all = thickness * occ_eq_new_arr + sum_fourier
    total_ions_old_init = thickness * occ_eq_arr[0] + jnp.sum(T_m_0 * sin_integrals)
    
    total_ions_shifted = jnp.concatenate([jnp.array([total_ions_old_init]), total_ions_all])
    # I = dN/dt
    simulated_currents = jnp.diff(total_ions_shifted) / dt_arr
    
    return simulated_currents

def load_and_preprocess_cv_data(filepath, scan_rate_v_s, skip_factor):
    data = pd.read_csv(filepath, sep=None, engine='python')
    
    #8 and 9 to read 3rd cycle (data file specific)
    raw_potential = data.iloc[:, 8].dropna().values
    raw_current = data.iloc[:, 9].dropna().values
    
    min_len = min(len(raw_potential), len(raw_current))
    raw_potential = raw_potential[:min_len]
    raw_current = raw_current[:min_len]
    
    #determine time array from triangular wave
    voltage_steps = np.abs(np.diff(raw_potential, prepend=raw_potential[0]))
    raw_time = np.cumsum(voltage_steps) / scan_rate_v_s

    exp_potential = raw_potential[::skip_factor]
    exp_current = raw_current[::skip_factor]
    exp_time = raw_time[::skip_factor]

    return exp_time, exp_potential, exp_current

def extract_physics_priors(potential, turn_idx, num_peaks, v_min, v_max):
    # D0, beta_left, beta_right, v_center, baseline_offset, a_right, k_right, a_left, k_left
    #used an an edcucated initial guess
    global_params = [2.0, 1.0, 1.0, np.mean(potential), 0.0, 0.1, 1.0, 0.1, 1.0]
    
    #even spaced peaks
    peaks_matrix = np.zeros((num_peaks, 3))
    v_crits = np.linspace(v_min + 0.1, v_max - 0.1, num_peaks)
    
    for i in range(num_peaks):
        peaks_matrix[i] = [1.0, v_crits[i], 15.0]
        
    return np.concatenate([global_params, peaks_matrix.flatten()])

def get_parameter_bounds(idx, val, num_globals, v_min, v_max):
    var = np.abs(val) * 0.5
    
    #chanigng bounds in steps so as to follow correct gradients

    if idx == 0:
        return (max(1e-8, val - 5.0), val + 5.0)
    if idx in (1, 2):
        return (val - 1.0, val + 1.0)
    if idx == 3:
        return (val - 0.6, val + 0.6)
    if idx == 4:
        var = var if val != 0 else 10.0
        return (val - var, val + var)
    if idx in (5, 7):
        return (max(1e-8, val - (var + 0.1)), val + var + 0.1)
    if idx in (6, 8):
        return (max(0.1, val - (var + 0.5)), val + var + 0.5)
        
    offset = (idx - num_globals) % 3
    if offset == 0:
        return (max(1e-4, val - (var + 1e-4)), val + 5.0)
    if offset == 1:
        return (max(v_min, val - (var + 1e-4)), min(v_max, val + var + 1e-4))
        
    return (max(0.1, val - (var + 1e-4)), val + 20.0)

def create_staged_bounds(target_params, active_indices, v_min, v_max):
    num_globals = OPTIMIZER_CONFIG["num_globals"]
    bounds = []
    
    # allow "frozen" params to move slightly so gradient is defined
    for i, val in enumerate(target_params):
        if i not in active_indices:
            bounds.append((val - 1e-9, val + 1e-9))
        else:
            bounds.append(get_parameter_bounds(i, val, num_globals, v_min, v_max))
            
    return bounds


exp_time, exp_potential, exp_current = load_and_preprocess_cv_data(
    filepath = USER_CONFIG["filepath"], 
    scan_rate_v_s = USER_CONFIG["scan_rate_v_s"],
    skip_factor = USER_CONFIG["skip_factor"]
)

exp_time_jax = jnp.array(exp_time)
exp_potential_jax = jnp.array(exp_potential)
global_target_current_jax = jnp.array(exp_current[1:].ravel())


turn_idx = np.argmax(np.abs(exp_potential - exp_potential[0]))
if turn_idx < len(exp_potential) * 0.1:
    turn_idx = len(exp_potential) // 2

# smooth current so that d2I is well defined
smoothed_current = savgol_filter(exp_current, window_length=51, polyorder=3)

#used for emphasizing areas with high change in current
d2I_raw = np.abs(np.diff(smoothed_current, n=2))
d2I = np.pad(d2I_raw, (1, 1), mode='edge')

loss_weights = (d2I / np.max(d2I)) + 1 #can change constant to place +/- emphasis on tails

edge_threshold = (USER_CONFIG["v_max"] - USER_CONFIG["v_min"]) * 0.05
left_mask = exp_potential[1:] < (USER_CONFIG["v_min"] + edge_threshold)
right_mask = exp_potential[1:] > (USER_CONFIG["v_max"] - edge_threshold)

global_weights = loss_weights[1:].ravel().copy()
global_weights_masked = global_weights.copy()
global_weights_masked[left_mask] = 0.0
global_weights_masked[right_mask] = 0.0

fig, ax = plt.subplots(figsize=(10, 6))
line_exp, = ax.plot(exp_potential[1:], exp_current[1:], marker='o', markersize=3, label='Raw Data', color='black', alpha=0.5)
line_sim, = ax.plot(exp_potential[1:], np.zeros_like(exp_potential[1:]), label='Simulated', color='red', linestyle='-', linewidth=2)
ax.set_xlabel('Potential (V)')
ax.set_ylabel('Current (A)')
ax.legend()
ax.grid(True)
title_text = ax.set_title('Initializing...')
dh = display(fig, display_id=True)
plt.close(fig)

dt_array = np.diff(exp_time)
dt_array[dt_array <= 0] = 1e-6

data_driven_initial_guess = extract_physics_priors(
    exp_potential, 
    turn_idx, 
    num_peaks=USER_CONFIG["num_peaks"],
    v_min=USER_CONFIG["v_min"],
    v_max=USER_CONFIG["v_max"]
)


initial_peaks_matrix = data_driven_initial_guess[OPTIMIZER_CONFIG["num_globals"]:].reshape((-1, 3))
baseline_diffusivity = data_driven_initial_guess[0] * OPTIMIZER_CONFIG["mult_diff"] #mul by const so optimizer works in O(1)

#gather initial simulation for calibration (better initial guess)
calibration_sim = np.array(run_fourier_simulation_with_data(
    exp_time_jax, exp_potential_jax,
    baseline_diffusivity, 
    0.0, 0.0,
    data_driven_initial_guess[3],
    jnp.array(initial_peaks_matrix), 
    50, 
    USER_CONFIG["film_thickness"]
))

#make sure current magnitude and off-set are close enough for optimizer
pure_faradaic_target = exp_current[1:]

#ignore the high jumps at large magnitude voltages
v_range = USER_CONFIG["v_max"] - USER_CONFIG["v_min"]
safe_min = USER_CONFIG["v_min"] + (v_range * 0.15)
safe_max = USER_CONFIG["v_max"] - (v_range * 0.15)
safe_mask = (exp_potential[1:] > safe_min) & (exp_potential[1:] < safe_max)


real_faradaic_ptp = np.ptp(pure_faradaic_target[safe_mask]) 
sim_ptp = np.ptp(calibration_sim[safe_mask])
if sim_ptp < 1e-12:
    sim_ptp = 1e-6 

# aligning magnitude
calibrated_scale = real_faradaic_ptp / sim_ptp

total_simulated_baseline = (calibration_sim * calibrated_scale)
real_mean = np.mean(exp_current[1:])
sim_mean = np.mean(total_simulated_baseline)
#aligning offset
calibrated_offset = (real_mean - sim_mean) / OPTIMIZER_CONFIG["mult_offset"]
data_driven_initial_guess[4] = calibrated_offset

calibrated_scale_jax = jnp.array(calibrated_scale)


@jax.jit
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
    
    peaks_matrix = jnp.reshape(scaled_params[OPTIMIZER_CONFIG["num_globals"]:], (-1, 3))
        
    simulated_currents = run_fourier_simulation_with_data(
        exp_time_jax, exp_potential_jax,
        diffusivity, beta_left, beta_right, v_center, peaks_matrix, 50, USER_CONFIG["film_thickness"]
    )
    
    bg_current = a_right * jnp.exp(k_right * (exp_potential_jax[1:] - USER_CONFIG["v_max"])) \
               - a_left * jnp.exp(-k_left * (exp_potential_jax[1:] - USER_CONFIG["v_min"]))
    
    # I_tot = I_diffusion + I_offset + background tails from Tafel equation
    final_sim = (simulated_currents * calibrated_scale_jax) + baseline_offset + bg_current
    #MSD
    squared_errors = (final_sim - global_target_current_jax)**2
    weighted_mse = jnp.average(squared_errors, weights=weights)
    loss = jnp.sqrt(weighted_mse) * 1e6 #scaled loss function
    
    return loss, final_sim


#simply gets rid of final sim to work with just loss instead (faster and cleaner)
@jax.jit
def objective_function_jax(scaled_params, weights):
    loss, _ = compute_forward(scaled_params, weights)
    return loss

#automatic differentiation (main use of JAX)
loss_and_grad_jax = jax.jit(jax.value_and_grad(objective_function_jax, argnums=0))

#splits output of automatic differentiation into loss function and its gradient
def scipy_objective(x, weights):
    loss, grad = loss_and_grad_jax(jnp.array(x), jnp.array(weights))
    return np.array(loss, dtype=np.float64), np.array(grad, dtype=np.float64)


#sets up display with updates in an efficient manner
class OptimizationTracker:
    def __init__(self, fig, ax, line_sim, title_text, display_handle):
        self.fig = fig
        self.ax = ax
        self.line_sim = line_sim
        self.title_text = title_text
        self.dh = display_handle
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
            self.line_sim.set_ydata(np.array(final_sim))
            self.ax.relim()
            self.ax.autoscale_view()
            self.title_text.set_text(
                f'{self.stage_label} | iter: {self.iter_count} | '
                f'Loss: {loss:.4f}\nDiffusivity: {xk[0]:.2f}x'
            )
            self.dh.update(self.fig)
        self.iter_count += 1

tracker = OptimizationTracker(fig, ax, line_sim, title_text, dh)

#naming each index after its parameter
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

v_min_cfg = USER_CONFIG["v_min"]
v_max_cfg = USER_CONFIG["v_max"]

#stages for optimization
stages = [
    ("Stage 1: Pure Flat Baseline", idx_baseline, global_weights_masked),
    ("Stage 1.5: Background Tails", idx_bg, global_weights),
    ("Stage 2: Anchor Peaks (Constant D)", idx_baseline + idx_bg + idx_diffusion_base + idx_peaks, global_weights),
    ("Stage 3: Full Non-Linear Polish", all_indices, global_weights)
]

#set starting point
current_x = data_driven_initial_guess

for label, active_idx, weights in stages:
    tracker.set_stage(label, weights)
    #looks only at parameters that are allowed to change
    bounds = create_staged_bounds(current_x, active_idx, v_min_cfg, v_max_cfg)
    res = minimize(
        scipy_objective, 
        current_x, 
        args=(weights,), 
        bounds=bounds, 
        jac=True, 
        method='L-BFGS-B', #smart gradient descent with memory and second deriv info
        callback=tracker, 
        options=optim_options #custom optimizations for optimizer set previously
    )
    #make sure starting point is same as previous' end point
    current_x = res.x

final_result = res

print("\n--- Optimization Complete ---")
print(f"Final Diffusivity: {final_result.x[0] * OPTIMIZER_CONFIG['mult_diff']:.4e} cm²/s")
print(f"Final Beta (Left): {final_result.x[1] * OPTIMIZER_CONFIG['mult_beta']:.4e}")
print(f"Final Beta (Right): {final_result.x[2] * OPTIMIZER_CONFIG['mult_beta']:.4e}")
print(f"Final Baseline Offset: {final_result.x[4] * OPTIMIZER_CONFIG['mult_offset']:.4e} A")

final_peaks = final_result.x[OPTIMIZER_CONFIG["num_globals"]:].reshape((-1, 3))
print("\nFinal Peaks Matrix [Weight, V_crit(V), Sharpness(k)]:")
print(np.round(final_peaks, 4))


def plot_extracted_physics(res_array, peaks_matrix):
    diffusivity = res_array[0] * OPTIMIZER_CONFIG["mult_diff"]
    beta_left   = res_array[1] * OPTIMIZER_CONFIG["mult_beta"]
    beta_right  = res_array[2] * OPTIMIZER_CONFIG["mult_beta"]
    v_center    = res_array[3]

    v_plot = np.linspace(-1, 1, 500)
    beta_plot = np.where(v_plot < v_center, beta_left, beta_right)
    d_of_v = diffusivity * np.exp(beta_plot * (v_plot - v_center)**2)

    weights = peaks_matrix[:, 0, np.newaxis]
    v_crits = peaks_matrix[:, 1, np.newaxis]
    sharpnesses = peaks_matrix[:, 2, np.newaxis]

    exp_terms = np.exp(-sharpnesses * (v_plot - v_crits))
    dos_matrix = weights * sharpnesses * exp_terms / (1.0 + exp_terms)**2
    dos_total = np.sum(dos_matrix, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax2.plot(v_plot, dos_matrix.T, color='gray', alpha=0.3, linestyle='--')

    ax1.plot(v_plot, d_of_v, color='red', linewidth=2)
    ax1.set_title('Extracted Diffusivity $D(V)$')
    ax1.set_xlabel('Potential (V)')
    ax1.set_ylabel('D (cm²/s)')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.plot(v_plot, dos_total, label='Total DOS', color='black', linewidth=2)
    ax2.set_title('Extracted Density of States')
    ax2.set_xlabel('Potential (V)')
    ax2.set_ylabel('Density of States (a.u.)')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_extracted_physics(final_result.x, final_peaks)

