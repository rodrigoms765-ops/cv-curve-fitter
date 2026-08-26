"""Multi-scan CV fitting engine.

One film has one diffusivity and one density of states, so every scan rate is fit
simultaneously against a single shared D(V) and DOS. Only the non-faradaic terms
(a DC offset and two background tails) are allowed to differ between scans.

This matters: the DOS contributes current linearly in the scan rate while diffusion
enters through the ratio of sweep time to L^2/D. Fitting one scan alone cannot
separate them, so a single-scan D0 is not identifiable.
"""

import io
import pathlib
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import scan

NERNST_SHARPNESS = 38.92   # F/RT at 298 K, the ideal one-electron limit


# ---------------------------------------------------------------- data loading

MIN_POINTS_PER_SCAN = 120
MAX_POINTS_PER_SCAN = 800


def sniff_delimiter(sample):
    """Pick the delimiter from the first non-empty line.

    pandas can do this itself with sep=None, but only through the Python engine,
    which is ~5x slower on the larger files. These are the same three delimiters
    the browser-side preview understands.
    """
    line = next((l for l in sample.splitlines() if l.strip()), "")
    counts = {d: line.count(d) for d in (',', '\t', ';')}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ','


def read_csv_text(content):
    """Parse uploaded CSV/TSV text with the fast C engine."""
    return pd.read_csv(io.StringIO(content), sep=sniff_delimiter(content[:8192]))


def resolve_stride(n_points, skip_factor):
    """Thin long scans, but never below MIN_POINTS_PER_SCAN.

    A blind stride hurts the fast scans most: they already have the fewest points,
    and the simulator integrates on whatever grid the data supplies. A factor of 5
    leaves a 320 mV/s cycle with ~28 points, which is far too coarse to resolve DOS
    features. So the requested factor is treated as an upper bound, not a mandate.
    """
    cap = -(-n_points // MAX_POINTS_PER_SCAN)          # ceil division
    stride = max(int(skip_factor), cap, 1)
    return max(1, min(stride, max(1, n_points // MIN_POINTS_PER_SCAN)))


def load_and_preprocess_cv_data(df, pot_col, cur_col, scan_rate_v_s, skip_factor):
    if isinstance(df, str):
        df = read_csv_text(pathlib.Path(df).read_text(encoding='utf-8'))
    if pot_col >= df.shape[1] or cur_col >= df.shape[1]:
        raise ValueError(
            f"Selected column index (Potential: {pot_col}, Current: {cur_col}) "
            f"exceeds total available columns ({df.shape[1]})."
        )

    s_pot = pd.to_numeric(df.iloc[:, pot_col], errors='coerce').dropna()
    s_cur = pd.to_numeric(df.iloc[:, cur_col], errors='coerce').dropna()

    common_idx = s_pot.index.intersection(s_cur.index)
    raw_potential = s_pot.loc[common_idx].values.astype(np.float64)
    raw_current = s_cur.loc[common_idx].values.astype(np.float64)

    if len(raw_potential) < 10:
        raise ValueError(
            f"No usable numeric data in column {pot_col} (potential) and {cur_col} (current)."
        )

    # The potentiostat samples at fixed time steps, so |dV| per sample is proportional
    # to the scan rate. Rebuilding time this way stays self-consistent with the data.
    voltage_steps = np.abs(np.diff(raw_potential, prepend=raw_potential[0]))
    raw_time = np.cumsum(voltage_steps) / scan_rate_v_s

    step = resolve_stride(len(raw_potential), skip_factor)
    return raw_time[::step], raw_potential[::step], raw_current[::step]


# ----------------------------------------------------------- forward simulation

@partial(jax.jit, static_argnames=['num_terms'])
def run_fourier_simulation_with_data(time_array, potential_array, diffusivity,
                                     beta_left, beta_right, v_center,
                                     peaks_matrix, num_terms, thickness):
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

    # D(V) = D0 * exp(beta * (V - Vc)^2), U-shaped with its minimum at Vc.
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

    # First pass finds the periodic steady state, so the fit does not depend on an
    # arbitrary starting concentration profile.
    def init_step(carry, xs):
        cum_dec, acc_forc = carry
        dec, forc = xs
        return (cum_dec * dec, acc_forc * dec + forc), None

    init_carry = (jnp.ones(num_terms, dtype=jnp.float64),
                  jnp.zeros(num_terms, dtype=jnp.float64))
    (final_cum_dec, final_acc_forc), _ = scan(init_step, init_carry,
                                              (decay_matrix, forcing_matrix))
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
    return jnp.diff(total_ions_shifted) / dt_arr


# ------------------------------------------------------------------ main solver

def solve_cv(scans, config, pot_col, cur_col, queue=None, loop=None):
    """Fit every scan at once against a shared D(V) and DOS.

    scans  : list of {"name": str, "df": DataFrame, "scan_rate": float in V/s}
    config : the UI configuration dict
    """
    if not scans:
        raise ValueError("No scans supplied.")

    thickness = float(config["film_thickness"])
    v_min = float(config["v_min"])
    v_max = float(config["v_max"])
    num_peaks = int(config["num_peaks"])
    num_terms = int(config.get("num_terms", 50))
    use_tafel = 1.0 if config.get("use_tafel", True) else 0.0

    NUM_GLOBALS = 5    # D0, beta_left, beta_right, V_center, sharpness
    NUM_INDEP = 5      # per scan: offset, a_right, k_right, a_left, k_left
    MULT = {
        "diff": thickness**2 / 10.0,
        "beta": 1.0,
        "offset": 1e-4,
        "bg_a": 1e-4,
        "bg_k": 10.0,
        "sharp": NERNST_SHARPNESS,
    }

    # --- DOS basis: fixed centres on a uniform grid, one shared width ----------
    # Only the heights are free. Letting each peak float in position and width as
    # well makes the recovered DOS non-unique - peaks collapse onto each other.
    peak_vcrits = np.linspace(v_min, v_max, num_peaks)
    peak_spacing = peak_vcrits[1] - peak_vcrits[0] if num_peaks > 1 else (v_max - v_min)
    peak_vcrits_jax = jnp.array(peak_vcrits)

    # Keep the nails overlapping; past this the basis becomes a comb with gaps.
    sharp_max = 3.5255 / (1.5 * peak_spacing)
    sharp_min = 5.0
    sharp_init = float(np.clip(float(config.get("peak_sharpness", NERNST_SHARPNESS)),
                               sharp_min, sharp_max))

    def build_peaks_matrix(peak_heights, sharpness):
        return jnp.stack([peak_heights,
                          peak_vcrits_jax,
                          sharpness * jnp.ones_like(peak_vcrits_jax)], axis=1)

    # --- load every scan ------------------------------------------------------
    loaded = []
    for s in scans:
        t, v, i = load_and_preprocess_cv_data(
            s["df"], pot_col, cur_col, float(s["scan_rate"]), int(config["skip_factor"])
        )
        loaded.append({"name": s.get("name", ""), "scan_rate": float(s["scan_rate"]),
                       "time": t, "potential": v, "current": i})
    loaded.sort(key=lambda d: d["scan_rate"])
    num_scans = len(loaded)

    time_jax = [jnp.array(d["time"]) for d in loaded]
    pot_jax = [jnp.array(d["potential"]) for d in loaded]
    target_jax = [jnp.array(d["current"][1:]) for d in loaded]

    # Emphasise regions of high curvature and large current, and mask the window
    # edges during the baseline stage so the tails do not drag the offset around.
    weights_full, weights_masked = [], []
    edge = (v_max - v_min) * 0.05
    for d in loaded:
        cur = d["current"]
        window = min(51, len(cur))
        if window % 2 == 0:
            window -= 1
        smoothed = savgol_filter(cur, window, 3) if window > 3 else cur
        d2 = np.pad(np.abs(np.diff(smoothed, n=2)), (1, 1), mode="edge")
        w = (d2 / (np.max(d2) + 1e-12)
             + np.abs(cur) / (np.max(np.abs(cur)) + 1e-12)
             + float(config.get("loss_weight_const", 1.0)))
        w = w[1:].ravel().copy()
        wm = w.copy()
        wm[d["potential"][1:] < v_min + edge] = 0.0
        wm[d["potential"][1:] > v_max - edge] = 0.0
        weights_full.append(jnp.array(w))
        weights_masked.append(jnp.array(wm))

    # --- initial guess --------------------------------------------------------
    mean_pot = float(np.mean([np.mean(d["potential"]) for d in loaded]))
    x0 = np.concatenate([
        [1.0, 0.0, 0.0, mean_pot, sharp_init / MULT["sharp"]],
        np.tile([0.0, 0.1, 1.0, 0.1, 1.0], num_scans),
        np.ones(num_peaks),
    ])

    # Fix the overall current scale from the slowest scan, then set each scan's
    # starting offset so simulated and measured means agree.
    init_peaks = jnp.array(np.column_stack(
        [np.ones(num_peaks), peak_vcrits, np.full(num_peaks, sharp_init)]))
    calib = [np.array(run_fourier_simulation_with_data(
        time_jax[i], pot_jax[i], x0[0] * MULT["diff"], 0.0, 0.0, mean_pot,
        init_peaks, num_terms, thickness)) for i in range(num_scans)]

    sim_ptp = max(np.max(np.abs(calib[0])), 1e-12)
    calibrated_scale = float(np.max(np.abs(loaded[0]["current"])) / sim_ptp)
    calibrated_scale_jax = jnp.array(calibrated_scale)

    for i in range(num_scans):
        gap = np.mean(loaded[i]["current"][1:]) - np.mean(calib[i] * calibrated_scale)
        x0[NUM_GLOBALS + i * NUM_INDEP] = gap / MULT["offset"]

    # --- forward model over all scans ----------------------------------------
    @jax.jit
    def compute_forward_multi(p, w_list):
        diffusivity = p[0] * MULT["diff"]
        beta_left = p[1] * MULT["beta"]
        beta_right = p[2] * MULT["beta"]
        v_center = p[3]
        sharpness = p[4] * MULT["sharp"]

        peak_heights = p[NUM_GLOBALS + NUM_INDEP * num_scans:]
        peaks_matrix = build_peaks_matrix(peak_heights, sharpness)

        total_loss = 0.0
        sims = []
        for i in range(num_scans):
            b = NUM_GLOBALS + i * NUM_INDEP
            offset = p[b] * MULT["offset"]
            a_right = p[b + 1] * MULT["bg_a"] * use_tafel
            k_right = p[b + 2] * MULT["bg_k"]
            a_left = p[b + 3] * MULT["bg_a"] * use_tafel
            k_left = p[b + 4] * MULT["bg_k"]

            v = pot_jax[i][1:]
            sim = run_fourier_simulation_with_data(
                time_jax[i], pot_jax[i], diffusivity, beta_left, beta_right,
                v_center, peaks_matrix, num_terms, thickness)
            bg = (a_right * jnp.exp(k_right * (v - v_max))
                  - a_left * jnp.exp(-k_left * (v - v_min)))
            final = sim * calibrated_scale_jax + offset + bg
            sims.append(final)

            wmse = jnp.average((final - target_jax[i])**2, weights=w_list[i])
            ptp = jnp.maximum(jnp.ptp(target_jax[i]), 1e-12)
            total_loss += jnp.sqrt(wmse) / ptp * 100.0

        # Without a smoothness penalty the heights ring from nail to nail; the
        # heights are a linear inverse problem and inherit its noise amplification.
        roughness = (jnp.sum(jnp.diff(peak_heights, n=2)**2)
                     / (jnp.mean(peak_heights)**2 + 1e-12))
        total_loss = total_loss / num_scans + float(config.get("dos_smoothness", 2.0)) * roughness
        return total_loss, sims

    @jax.jit
    def objective(p, w_list):
        return compute_forward_multi(p, w_list)[0]

    loss_and_grad = jax.jit(jax.value_and_grad(objective, argnums=0))

    def scipy_objective(x, w_list):
        loss, grad = loss_and_grad(jnp.array(x), w_list)
        return np.array(loss, dtype=np.float64), np.array(grad, dtype=np.float64)

    # --- bounds ---------------------------------------------------------------
    def bounds_for(idx, val):
        var = abs(val) * 2.0
        if idx == 0:
            return (max(1e-8, val - 5.0), val + 5.0)
        # Non-negative betas keep D(V) U-shaped. Opposite signs splice the two
        # halves into a monotonic curve with a kink at V_center.
        if idx in (1, 2):
            return (0.0, 20.0)
        # Absolute, not relative to the current value - V_center drifting outside
        # the window silently turns one of the betas into a dead parameter.
        if idx == 3:
            return (v_min + 0.2, v_max - 0.2)
        if idx == 4:
            return (sharp_min / MULT["sharp"], sharp_max / MULT["sharp"])
        if idx < NUM_GLOBALS + NUM_INDEP * num_scans:
            off = (idx - NUM_GLOBALS) % NUM_INDEP
            if off == 0:
                span = var if val != 0 else 10.0
                return (val - span, val + span)
            if off in (1, 3):
                return (max(1e-8, val - (var + 0.1)), val + var + 0.1)
            return (max(0.1, val - (var + 0.5)), val + var + 0.5)
        return (0.0, None)   # nail heights; a density of states cannot go negative

    def staged_bounds(target, active):
        return [bounds_for(i, v) if i in active else (v - 1e-9, v + 1e-9)
                for i, v in enumerate(target)]

    # --- staged optimisation --------------------------------------------------
    all_idx = list(range(len(x0)))
    idx_offset, idx_bg = [], []
    for i in range(num_scans):
        b = NUM_GLOBALS + i * NUM_INDEP
        idx_offset.append(b)
        idx_bg.extend([b + 1, b + 2, b + 3, b + 4])
    idx_diffusion = [0, 1, 2, 3]
    idx_peaks = list(range(NUM_GLOBALS + NUM_INDEP * num_scans, len(x0)))

    stages = [
        ("baseline", idx_offset, weights_masked),
        ("tails", idx_bg, weights_full),
        # Offset and background stay frozen so the DOS forms against a settled
        # background instead of absorbing it.
        ("peaks", idx_peaks + [4], weights_full),
        ("diffusion", idx_diffusion, weights_full),
        ("polish", all_idx, weights_full),
        # L-BFGS-B's limited-memory Hessian is stale by the end of the polish; a
        # restart from that optimum still buys a real reduction.
        ("restart", all_idx, weights_full),
    ]

    opts = {
        "maxiter": int(config.get("max_iter", 300)),
        "ftol": float(config.get("tol_ftol", 1e-12)),
        "gtol": float(config.get("tol_gtol", 1e-10)),
    }

    current_x = x0
    res = None
    for label, active, w_list in stages:
        res = minimize(scipy_objective, current_x, args=(w_list,),
                       bounds=staged_bounds(current_x, active), jac=True,
                       method="L-BFGS-B", options=opts)
        current_x = res.x
        if loop and queue:
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "update", "stage": label, "loss": float(res.fun)})

    # --- results --------------------------------------------------------------
    x = res.x
    D0 = float(x[0] * MULT["diff"])
    beta_left = float(x[1] * MULT["beta"])
    beta_right = float(x[2] * MULT["beta"])
    v_center = float(x[3])
    sharpness = float(x[4] * MULT["sharp"])
    fwhm = 3.5255 / sharpness

    _, final_sims = compute_forward_multi(jnp.array(x), weights_full)

    v_plot = np.linspace(v_min, v_max, 500)
    d_of_v = D0 * np.exp(np.where(v_plot < v_center, beta_left, beta_right)
                         * (v_plot - v_center)**2)

    heights = np.asarray(x[NUM_GLOBALS + NUM_INDEP * num_scans:]) * calibrated_scale
    e = np.exp(-sharpness * (v_plot[np.newaxis, :] - peak_vcrits[:, np.newaxis]))
    dos_matrix = heights[:, np.newaxis] * sharpness * e / (1.0 + e)**2
    dos_total = dos_matrix.sum(axis=0)

    # A parameter sitting on its bound is reporting the constraint, not the data.
    notes = []
    if num_scans < 2:
        notes.append("Only one scan rate was fit. D0 is not identifiable from a "
                     "single scan - the DOS and diffusion terms cannot be separated.")
    if min(abs(v_center - (v_min + 0.2)), abs(v_center - (v_max - 0.2))) < 0.01:
        notes.append("V_center is on its bound, so the data is not locating the D(V) minimum.")
    if beta_left < 1e-6 or beta_right < 1e-6:
        notes.append("A beta is zero, so D(V) is flat on that side of V_center.")
    if abs(sharpness - sharp_max) < 0.01 * sharp_max or abs(sharpness - sharp_min) < 0.01 * sharp_min:
        notes.append("DOS width is on its bound. It is a smoothing width degenerate "
                     "with the height profile - read the DOS curve, not this number.")

    scan_results = []
    for i, d in enumerate(loaded):
        sim = np.array(final_sims[i])
        tgt = d["current"][1:]
        rmse = float(100.0 * np.sqrt(np.mean((sim - tgt)**2)) / np.ptp(tgt))
        scan_results.append({
            "name": d["name"],
            "scan_rate": d["scan_rate"],
            "rmse_pct": rmse,
            "baseline_offset": float(x[NUM_GLOBALS + i * NUM_INDEP] * MULT["offset"]),
            "exp_potential": d["potential"][1:].tolist(),
            "exp_current": tgt.tolist(),
            "sim_current": sim.tolist(),
        })

    result_data = {
        "shared": {
            "diffusivity": D0,
            "beta_left": beta_left,
            "beta_right": beta_right,
            "v_center": v_center,
            "sharpness": sharpness,
            "dos_fwhm": float(fwhm),
            "dos_overlap": float(fwhm / peak_spacing),
            "num_scans": num_scans,
            "final_loss": float(res.fun),
        },
        "notes": notes,
        "scans": scan_results,
        "plots": {
            "v_plot": v_plot.tolist(),
            "d_of_v": d_of_v.tolist(),
            "dos_total": dos_total.tolist(),
            "dos_matrix": dos_matrix.T.tolist(),
        },
    }

    if loop and queue:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "data": result_data})
    return result_data
