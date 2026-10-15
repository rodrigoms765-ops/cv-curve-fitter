"""Multi-scan CV fitting engine.

One film has one density of states, so every scan rate is fit simultaneously
against a single shared DOS. Nothing is fitted per scan.

Transport is either one diffusivity or two environments - a fast fraction and a
slow remainder sharing that same DOS. Two environments exist because one
diffusivity cannot reconcile scans taken at different sweep rates: fitted alone,
each scan demands its own D, rising roughly as the square root of the scan rate.
The split costs two global parameters and no per-scan freedom, so it still has to
hold at every rate at once. Physically it is the ordered and disordered regions of
a semicrystalline film, which admit hydrated ions at very different rates.

Fitting one scan alone cannot separate the DOS from diffusion, so a single-scan D
is not identifiable, and two environments need more scan rates still.

An uncompensated series resistance can be switched on. The film is then driven
at V - R*I, with I the film's own current, solved self-consistently at every
time step. It is fitted last, starting from the fit without it.
"""

import io
import pathlib
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import scan

NERNST_SHARPNESS = 38.92   # F/RT at 298 K, the ideal one-electron limit
LOSS_SCALE = 1e4           # carries the misfit as (RMSE percent of range) squared
MIN_SCANS_TWO_SITE = 3     # below this the fast/slow split is not identifiable
ELEMENTARY_CHARGE = 1.602176634e-19   # C, to turn fitted charge into a state count


# ---------------------------------------------------------------- data loading

# A scan is thinned to about one point every 10 mV of potential, not by a fixed
# stride in index: one cycle at 10 mV/s carries ~1360 points per volt and one at
# 640 mV/s only ~18. 10 mV is a ninth of the 90 mV sub-band width; on the test
# films it gives the same fit as the basis-tied rule it replaced.
POINT_SPACING_V = 0.01


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


def resolve_stride(potential):
    """Stride leaving about one point every POINT_SPACING_V volts.

    Fast scans already carry fewer points than that, so the rule returns 1 for them.
    """
    if len(potential) < 2:
        return 1
    gap = float(np.median(np.abs(np.diff(potential))))
    if gap <= 0.0:
        return 1
    return max(1, int(POINT_SPACING_V // gap))


def load_and_preprocess_cv_data(df, pot_col, cur_col, scan_rate_v_s):
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

    step = resolve_stride(raw_potential)
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
    # (1 - exp(-k dt)) / (k dt), accurate down to small k dt
    forcing_factor = -jnp.expm1(-k_dt_matrix) / k_dt_matrix

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


# ------------------------------------------------------- series resistance stage

def _fit_with_resistance(x, r_start, fit_resistance, loaded, thickness, num_terms,
                         peak_vcrits_jax, scale, two_site, MULT, NUM_GLOBALS,
                         i_max, target_jax, bounds_for, opts, solver_step):
    """Refit everything with the film driven at V - R*I_film.

    Per time step the film's ion content is affine in the new edge occupancy,
    N = slope * occ + intercept with 0 < slope <= L, so V_film - V + R * I(V_film)
    is increasing in V_film and has one root, found by Newton. Only the last
    Newton step is differentiated. The periodic state comes from the closed form
    applied to the previous pass, starting from V - R * I_measured.
    """
    num_scans = len(loaded)
    n_x = len(x)
    n = np.arange(1, num_terms + 1)
    wavenumbers = jnp.array((2 * n - 1) * np.pi / (2 * thickness))
    fourier_coeffs = jnp.array(4 / ((2 * n - 1) * np.pi))
    sin_integrals = 1.0 / wavenumbers

    # Each data interval, plus the one closing the cycle, split into steps of at
    # most solver_step volts.
    grids = []
    for d in loaded:
        t, v, c = d["time"], d["potential"], d["current"]
        dt = np.diff(t)
        dt[dt <= 0] = 1e-6
        rate = np.sum(np.abs(np.diff(v))) / max(t[-1] - t[0], 1e-12)
        dt = np.append(dt, max(abs(v[0] - v[-1]) / rate, 1e-6))
        v_end = np.append(v[1:], v[0])
        i_end = np.append(c[1:], c[0])
        k = int(max(1, np.ceil(np.median(np.abs(np.diff(v))) / solver_step)))
        frac = np.arange(1, k + 1) / k
        grids.append({
            "k": k,
            "dt": jnp.array(dt),
            "v": jnp.array((v[:, None] + (v_end - v)[:, None] * frac).ravel()),
            "i": jnp.array((c[:, None] + (i_end - c)[:, None] * frac).ravel()),
            "fine_dt": jnp.array(np.repeat(dt / k, k)),
        })

    def bands(v, heights, sharpness):
        fermi = 1.0 / (1.0 + jnp.exp(-sharpness * (v[..., None] - peak_vcrits_jax)))
        occupancy = jnp.sum(heights * fermi, axis=-1)
        capacitance = jnp.sum(heights * sharpness * fermi * (1.0 - fermi), axis=-1)
        return occupancy, capacitance

    def mode_factors(v_path, dt, diffusivities, beta_left, beta_right, v_center):
        beta = jnp.where(v_path < v_center, beta_left, beta_right)
        stretch = jnp.exp(beta * (v_path - v_center) ** 2)
        k_dt = (stretch * dt)[:, None, None] * diffusivities[None, :, None] * wavenumbers ** 2
        return jnp.exp(-k_dt), -jnp.expm1(-k_dt) / k_dt

    def periodic_start(decays, pushes):
        def step(carry, xs):
            total_decay, summed = carry
            decay, push = xs
            return (total_decay * decay, summed * decay + push), None
        first = (jnp.ones_like(decays[0]), jnp.zeros_like(decays[0]))
        (total_decay, summed), _ = scan(step, first, (decays, pushes))
        return summed / (1.0 - total_decay + 1e-15)

    def simulate(grid, diffusivities, fractions, beta_left, beta_right, v_center,
                 sharpness, heights, r_ohm):
        weight = fractions[:, None] * sin_integrals
        v_applied = grid["v"]
        fine_dt = grid["fine_dt"]
        v_change = v_applied - jnp.roll(v_applied, 1)

        def run_cycle(state, v_path):
            decays, forcings = mode_factors(v_path, fine_dt, diffusivities,
                                            beta_left, beta_right, v_center)

            def step(state, xs):
                modes, occ_old, ions_old, v_film = state
                v_now, change, dt, decay, forcing = xs
                decayed = modes * decay
                pull = jnp.sum(weight * fourier_coeffs * forcing)
                slope = thickness - pull
                intercept = jnp.sum(weight * decayed) + occ_old * pull

                def error(v):
                    occ, cap = bands(v, heights, sharpness)
                    current = scale * (slope * occ + intercept - ions_old) / dt
                    gradient = 1.0 + r_ohm * scale * slope * cap / dt
                    return v - v_now + r_ohm * current, gradient

                v = v_film + change
                for _ in range(4):
                    e, g = error(v)
                    v = v - jnp.clip(e / g, -0.1, 0.1)
                v = jax.lax.stop_gradient(v)
                e, g = error(v)
                v = v - e / g
                occ, _ = bands(v, heights, sharpness)
                push = (occ_old - occ) * fourier_coeffs * forcing
                ions = slope * occ + intercept
                return (decayed + push, occ, ions, v), (ions, v, push)

            inputs = (v_applied, v_change, fine_dt, decays, forcings)
            end, (ions_end, new_path, pushes) = scan(step, state, inputs)
            return end, ions_end, new_path, decays, pushes

        def start_state(modes, v):
            occ, _ = bands(v, heights, sharpness)
            return (modes, occ, thickness * occ + jnp.sum(weight * modes), v)

        v_path = v_applied - r_ohm * grid["i"]
        occ, _ = bands(v_path, heights, sharpness)
        decays, forcings = mode_factors(v_path, fine_dt, diffusivities,
                                        beta_left, beta_right, v_center)
        pushes = (jnp.roll(occ, 1) - occ)[:, None, None] * fourier_coeffs * forcings
        state = start_state(periodic_start(decays, pushes), v_path[-1])
        for _ in range(2):
            start = state
            end, ions_end, v_path, decays, pushes = run_cycle(start, v_path)
            state = start_state(periodic_start(decays, pushes), end[3])
        k = grid["k"]
        ions_end = ions_end[k - 1::k]
        ions_before = jnp.concatenate([start[2][None], ions_end[:-1]])
        current = scale * (ions_end - ions_before) / grid["dt"]
        return current[:-1]

    @jax.jit
    def forward(q):
        diffusivity = MULT["diff"] * 10.0 ** q[0]
        beta_left = q[1] * MULT["beta"]
        beta_right = q[2] * MULT["beta"]
        v_center = q[3]
        sharpness = q[4] * MULT["sharp"]
        heights = q[NUM_GLOBALS:n_x]
        r_ohm = q[n_x] / i_max
        if two_site:
            diffusivities = jnp.stack([diffusivity, diffusivity * q[6]])
            fractions = jnp.stack([q[5], 1.0 - q[5]])
        else:
            diffusivities = jnp.stack([diffusivity])
            fractions = jnp.ones(1)
        total = 0.0
        sims = []
        for i in range(num_scans):
            sim = simulate(grids[i], diffusivities, fractions, beta_left, beta_right,
                           v_center, sharpness, heights, r_ohm)
            sims.append(sim)
            mse = jnp.mean((sim - target_jax[i]) ** 2)
            ptp = jnp.maximum(jnp.ptp(target_jax[i]), 1e-12)
            total += mse / ptp ** 2 * LOSS_SCALE
        return total / num_scans, sims

    loss_and_grad = jax.jit(jax.value_and_grad(lambda q: forward(q)[0]))

    def scipy_objective(q):
        loss, grad = loss_and_grad(jnp.array(q))
        return float(loss), np.array(grad, dtype=np.float64)

    q0 = np.append(np.array(x, dtype=np.float64), r_start * i_max)
    bounds = [bounds_for(i, value) for i, value in enumerate(q0[:n_x])]
    bounds.append((0.0, 1.0) if fit_resistance else (q0[n_x], q0[n_x]))
    res = minimize(scipy_objective, q0, bounds=bounds, jac=True,
                   method="L-BFGS-B", options=opts)
    q = res.x
    final_loss, sims = forward(jnp.array(q))
    return float(q[n_x] / i_max), q[:n_x].copy(), float(final_loss), sims


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

    # Sub-bands centred outside the data stay full or empty for the whole scan, so
    # they carry no current and the data cannot set their heights. A window
    # reaching well past the data is therefore clipped to it.
    data_lo, data_hi = np.inf, -np.inf
    for s in scans:
        if pot_col < s["df"].shape[1]:
            pot = pd.to_numeric(s["df"].iloc[:, pot_col], errors='coerce').dropna()
            if len(pot):
                data_lo = min(data_lo, float(pot.min()))
                data_hi = max(data_hi, float(pot.max()))
    window_clipped = False
    if np.isfinite(data_lo) and (v_min < data_lo - 0.05 or v_max > data_hi + 0.05):
        requested = (v_min, v_max)
        v_min = max(v_min, round(data_lo, 3))
        v_max = min(v_max, round(data_hi, 3))
        window_clipped = True
    # Electrode area, needed only to express the DOS per unit volume. It does
    # not enter the fit: the misfit is scale-free and the calibration constant
    # absorbs the overall current magnitude.
    electrode_area = float(config.get("electrode_area", 1.0))
    # Truncating the Fourier solution at 20 terms costs about 1% of the simulated
    # current, but almost none of that reaches the fitted parameters: the error is
    # smooth and the DOS amplitudes absorb it. Against 40 terms the fit moves D_fast
    # by 0.04% and D_slow by 0.7%, leaves every per-scan RMSE unchanged, and runs
    # about 30% faster. Cost is linear in this number.
    num_terms = int(config.get("num_terms", 20))
    transport_requested = str(config.get("transport", "two_site")).lower()
    if transport_requested not in ("two_site", "single"):
        transport_requested = "two_site"

    # log10(D / MULT["diff"]), two betas, V_center, sharpness, frac_fast, d_ratio,
    # then the sub-band heights. D is fitted on a log scale: a film at equilibrium
    # sends it up by decades.
    NUM_GLOBALS = 7
    MULT = {
        "diff": thickness**2 / 10.0,
        "beta": 1.0,
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
    sharp_want = float(config.get("peak_sharpness", NERNST_SHARPNESS))
    sharp_init = float(np.clip(sharp_want, sharp_min, sharp_max))

    # The width is held fixed rather than fitted. Fitted, it ran to the narrowest
    # the grid allowed at every sub-band count tried (10, 25, 50), and where the
    # grid was fine enough to let it choose freely it settled 2.1x sharper than a
    # one-electron site can physically be. That is not site broadening being
    # measured, it is a knob trading against the height profile - which is why the
    # results already warned against reading it. Holding it at the Nernstian width
    # costs nothing measurable (2.833 vs 2.868 percent at 50 sub-bands, 2.861 vs
    # 2.821 at 100, both inside run-to-run scatter) and buys an unambiguous
    # reading: each sub-band is then a population of ideal one-electron sites, so
    # DOS(V) is the distribution of formal potentials rather than that distribution
    # convolved with an arbitrary fitted blur.
    #
    # Set fit_dos_width to re-enable it, which is right only if the film is
    # expected to be genuinely non-ideal. Note the direction: attractive site-site
    # interactions sharpen a redox response and repulsive ones broaden it, so a
    # charged film taking in ions should come out broader than Nernstian, not
    # sharper, which is what fitting it actually produced.
    fit_dos_width = bool(config.get("fit_dos_width", False))

    # Sub-bands needed before the grid can carry the requested width at all.
    peaks_for_width = int(np.ceil((v_max - v_min) * 1.5 * sharp_want / 3.5255)) + 1

    def build_peaks_matrix(peak_heights, sharpness):
        return jnp.stack([peak_heights,
                          peak_vcrits_jax,
                          sharpness * jnp.ones_like(peak_vcrits_jax)], axis=1)

    # --- load every scan ------------------------------------------------------
    loaded = []
    for s in scans:
        t, v, i = load_and_preprocess_cv_data(s["df"], pot_col, cur_col,
                                              float(s["scan_rate"]))
        loaded.append({"name": s.get("name", ""), "scan_rate": float(s["scan_rate"]),
                       "time": t, "potential": v, "current": i})
    loaded.sort(key=lambda d: d["scan_rate"])
    num_scans = len(loaded)

    # The two-environment split carries three transport parameters, and nearly all
    # the evidence separating them lives in how the response changes with sweep
    # rate. Below three rates that evidence is not there and the optimiser will
    # still return confident-looking numbers, so refuse rather than oblige.
    two_site = transport_requested == "two_site" and num_scans >= MIN_SCANS_TWO_SITE
    transport = "two_site" if two_site else "single"

    time_jax = [jnp.array(d["time"]) for d in loaded]
    pot_jax = [jnp.array(d["potential"]) for d in loaded]
    target_jax = [jnp.array(d["current"][1:]) for d in loaded]

    r_start = max(float(config.get("series_resistance", 0.0)), 0.0)
    fit_resistance = bool(config.get("fit_resistance", False))
    use_resistance = fit_resistance or r_start > 0.0

    # --- initial guess --------------------------------------------------------
    mean_pot = float(np.mean([np.mean(d["potential"]) for d in loaded]))
    x0 = np.concatenate([
        [0.0, 0.0, 0.0, mean_pot, sharp_init / MULT["sharp"],
         0.5,      # frac_fast: share of the sites in the fast environment
         0.1],     # d_ratio: D_slow / D_fast
        np.ones(num_peaks),
    ])

    # The resistance is fitted as the ohmic drop at the largest measured current,
    # R * i_max in volts, so it reaches L-BFGS-B at O(1) like everything else.
    i_max = max(float(np.max(np.abs(d["current"]))) for d in loaded)

    # Fix the overall current scale from the slowest scan.
    init_peaks = jnp.array(np.column_stack(
        [np.ones(num_peaks), peak_vcrits, np.full(num_peaks, sharp_init)]))
    calib = np.array(run_fourier_simulation_with_data(
        time_jax[0], pot_jax[0], MULT["diff"] * 10.0 ** x0[0], 0.0, 0.0, mean_pot,
        init_peaks, num_terms, thickness))
    sim_ptp = max(np.max(np.abs(calib)), 1e-12)
    calibrated_scale = float(np.max(np.abs(loaded[0]["current"])) / sim_ptp)
    calibrated_scale_jax = jnp.array(calibrated_scale)

    # Starting DOS read off the slowest scan: near equilibrium I/v is the DOS, so
    # take half the gap between the two sweeps, on the V - R*I axis, spread over
    # the sub-band spacing. A flat start can leave the fit stuck far from the data.
    slow = loaded[0]
    v_corrected = slow["potential"] - r_start * slow["current"]
    charge = slow["current"] / slow["scan_rate"]
    forward_sweep = np.diff(slow["potential"], prepend=slow["potential"][0]) > 0
    branches = []
    for sweep in (forward_sweep, ~forward_sweep):
        order = np.argsort(v_corrected[sweep])
        branches.append(np.interp(peak_vcrits, v_corrected[sweep][order],
                                  charge[sweep][order]))
    start_dos = np.clip((branches[0] - branches[1]) / 2.0, 0.0, None)
    start_heights = start_dos * peak_spacing / (calibrated_scale * thickness)
    if start_heights.max() > 0.0:
        x0[NUM_GLOBALS:] = np.maximum(start_heights, 1e-3 * start_heights.max())

    # --- forward model over all scans ----------------------------------------
    @jax.jit
    def compute_forward_multi(p):
        diffusivity = MULT["diff"] * 10.0 ** p[0]
        beta_left = p[1] * MULT["beta"]
        beta_right = p[2] * MULT["beta"]
        v_center = p[3]
        sharpness = p[4] * MULT["sharp"]
        frac_fast = p[5]
        d_ratio = p[6]

        peak_heights = p[NUM_GLOBALS:]
        peaks_matrix = build_peaks_matrix(peak_heights, sharpness)

        total_loss = 0.0
        sims = []
        for i in range(num_scans):
            sim = run_fourier_simulation_with_data(
                time_jax[i], pot_jax[i], diffusivity, beta_left, beta_right,
                v_center, peaks_matrix, num_terms, thickness)
            if two_site:
                # Same DOS and the same D(V) shape, slower by d_ratio. The current
                # is linear in the heights, so blending the two runs is the same as
                # splitting every nail between a fast and a slow environment.
                sim_slow = run_fourier_simulation_with_data(
                    time_jax[i], pot_jax[i], diffusivity * d_ratio, beta_left,
                    beta_right, v_center, peaks_matrix, num_terms, thickness)
                sim = frac_fast * sim + (1.0 - frac_fast) * sim_slow
            final = sim * calibrated_scale_jax
            sims.append(final)

            mse = jnp.mean((final - target_jax[i])**2)
            ptp = jnp.maximum(jnp.ptp(target_jax[i]), 1e-12)
            # Squared rather than root, and carried in units of (RMSE as a percent
            # of range) squared so the loss stays of order one and the optimiser
            # tolerances keep their meaning.
            total_loss += mse / ptp**2 * LOSS_SCALE
        return total_loss / num_scans, sims

    @jax.jit
    def objective(p):
        return compute_forward_multi(p)[0]

    loss_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(x):
        loss, grad = loss_and_grad(jnp.array(x))
        return np.array(loss, dtype=np.float64), np.array(grad, dtype=np.float64)

    # --- bounds ---------------------------------------------------------------
    def bounds_for(idx, val):
        if idx == 0:
            # log10(D / (L^2/10)): L^2/D from 1e5 s down to 1 ms.
            return (-4.0, 4.0)
        # Non-negative betas keep D(V) U-shaped. Opposite signs splice the two
        # halves into a monotonic curve with a kink at V_center.
        if idx in (1, 2):
            return (0.0, 20.0)
        # Absolute, not relative to the current value - V_center drifting outside
        # the window silently turns one of the betas into a dead parameter.
        if idx == 3:
            return (v_min + 0.2, v_max - 0.2)
        if idx == 4:
            # Collapsed to a point unless explicitly re-enabled, so the width stays
            # wherever sharp_init put it.
            if not fit_dos_width:
                return (val, val)
            return (sharp_min / MULT["sharp"], sharp_max / MULT["sharp"])
        # Held off the endpoints: at d_ratio = 1 the two environments merge and the
        # fraction stops meaning anything, and at a fraction of 0 or 1 the ratio does.
        if idx == 5:
            return (0.02, 0.98) if two_site else (val, val)
        if idx == 6:
            return (1e-3, 0.9) if two_site else (val, val)
        return (0.0, None)   # nail heights; a density of states cannot go negative

    # --- optimisation ---------------------------------------------------------
    # One stage over everything, from the start above. Warm-up stages that fitted
    # the transport first led some films into a worse minimum.
    # 1e-12 asked L-BFGS-B for twelve-digit convergence and bought nothing: half
    # again as many iterations to move D by one part in 1e5, four orders below
    # the fit's own uncertainty.
    opts = {
        "maxiter": int(config.get("max_iter", 10000)),
        "ftol": float(config.get("tol_ftol", 1e-9)),
        "gtol": float(config.get("tol_gtol", 1e-8)),
    }
    res = minimize(scipy_objective, x0, jac=True, method="L-BFGS-B",
                   bounds=[bounds_for(i, v) for i, v in enumerate(x0)], options=opts)
    if loop and queue:
        loop.call_soon_threadsafe(queue.put_nowait, {
            "type": "update", "stage": "fit", "loss": float(res.fun)})

    x = res.x
    final_loss = float(res.fun)
    _, final_sims = compute_forward_multi(jnp.array(x))

    # --- series resistance ----------------------------------------------------
    # The film sees V - R*I with I its own current. Released from the start, R can
    # run to its bound, so it is fitted last, from the fit above.
    resistance = 0.0
    if use_resistance:
        resistance, x, final_loss, final_sims = _fit_with_resistance(
            x, r_start, fit_resistance, loaded, thickness, num_terms, peak_vcrits_jax,
            calibrated_scale_jax, two_site, MULT, NUM_GLOBALS, i_max, target_jax,
            bounds_for, opts, max(float(config.get("solver_step", 0.01)), 1e-4))
        if loop and queue:
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "update", "stage": "resistance", "loss": final_loss})

    # --- results --------------------------------------------------------------
    d_lower_bound_only = bool(x[0] > 3.9)
    D0 = float(MULT["diff"] * 10.0 ** x[0])
    beta_left = float(x[1] * MULT["beta"])
    beta_right = float(x[2] * MULT["beta"])
    v_center = float(x[3])
    sharpness = float(x[4] * MULT["sharp"])
    frac_fast = float(x[5])
    d_ratio = float(x[6])
    d_slow = D0 * d_ratio
    fwhm = 3.5255 / sharpness

    # With both betas at zero D(V) is a flat line and V_center has nothing to sit
    # on, so it drifts anywhere inside its bounds. Reporting it then would dress a
    # meaningless number as a measurement; the UI hides these when this is False.
    d_of_v_determined = not (beta_left < 1e-6 and beta_right < 1e-6)
    environments_merged = two_site and abs(d_ratio - 0.9) < 1e-3

    v_plot = np.linspace(v_min, v_max, 500)
    d_of_v = D0 * np.exp(np.where(v_plot < v_center, beta_left, beta_right)
                         * (v_plot - v_center)**2)
    d_of_v_slow = (d_of_v * d_ratio) if two_site else None

    heights = np.asarray(x[NUM_GLOBALS:]) * calibrated_scale
    e = np.exp(-sharpness * (v_plot[np.newaxis, :] - peak_vcrits[:, np.newaxis]))
    dos_matrix = heights[:, np.newaxis] * sharpness * e / (1.0 + e)**2
    dos_total = dos_matrix.sum(axis=0)

    # Put the DOS in absolute units. As built, dos_matrix is a charge per unit
    # length per volt (C cm^-1 V^-1): once the calibration constant is applied the
    # heights carry C/cm, and the sigmoid derivative contributes the V^-1. Dividing
    # by the electrode area turns per-length into per-volume, and by the elementary
    # charge turns charge into a count of states. For a one-electron transfer a
    # volt of potential is an electronvolt of energy, so V^-1 becomes eV^-1 without
    # a numerical factor. The result is states cm^-3 eV^-1.
    #
    # The SHAPE of this curve is what the data determines. Its absolute height is
    # only as good as the area and the thickness, neither of which is measured here.
    dos_unit = max(electrode_area, 1e-30) * ELEMENTARY_CHARGE
    dos_matrix = dos_matrix / dos_unit
    dos_total = dos_total / dos_unit

    # A parameter sitting on its bound is reporting the constraint, not the data.
    notes = []
    if window_clipped:
        notes.append(
            f"The window {requested[0]:g} to {requested[1]:g} V reached past the data, "
            f"so it was clipped to {v_min:g} to {v_max:g} V. Sub-bands outside the data "
            f"carry no current, so the data cannot set their heights.")
    if use_resistance:
        notes.append(
            f"Series resistance {'fitted' if fit_resistance else 'held'} at "
            f"{resistance:.1f} ohm, an ohmic drop of {1000 * resistance * i_max:.0f} mV "
            f"at the largest measured current.")
    if d_lower_bound_only:
        notes.append(
            "D is at its upper bound: the film is at equilibrium over every sweep, so "
            "only a lower bound on D can be given.")
    if transport_requested == "two_site" and not two_site:
        notes.append(
            f"Fell back to a single diffusivity: the fast/slow split needs at least "
            f"{MIN_SCANS_TWO_SITE} scan rates and only {num_scans} were supplied. "
            f"Nearly all the evidence separating the two environments is in how the "
            f"response changes with sweep rate, so fitting them here would return "
            f"numbers the data cannot support.")
    if num_scans < 2:
        notes.append("Only one scan rate was fit. D is not identifiable from a "
                     "single scan - the DOS and diffusion terms cannot be separated.")
    if environments_merged:
        notes.append("The slow/fast ratio is on its upper bound, so the two "
                     "environments have merged and the data is not resolving a split.")
    elif two_site and d_ratio < 1.01e-3:
        notes.append("The slow/fast ratio is on its lower bound: the slow environment "
                     "is as slow as the model permits, so it contributes almost nothing "
                     "over a sweep and its diffusivity is an upper limit rather than a "
                     "measurement.")
    if two_site and min(abs(frac_fast - 0.02), abs(frac_fast - 0.98)) < 1e-3:
        notes.append("The fast fraction is on a bound, so one environment carries "
                     "essentially everything and the second is not earning its place.")
    if not d_of_v_determined:
        notes.append(
            "D(V) came out flat, so its shape parameters (beta and V_center) are not "
            "determined and are not shown."
            + (" Under the two-environment model that is the expected outcome - the"
               " spread of transport rates is carried by the two environments rather"
               " than by curvature in D(V)." if two_site else ""))
    elif min(abs(v_center - (v_min + 0.2)), abs(v_center - (v_max - 0.2))) < 0.01:
        notes.append("V_center is on its bound, so the data is not locating the D(V) minimum.")
    if fit_dos_width and (abs(sharpness - sharp_max) < 0.01 * sharp_max
                          or abs(sharpness - sharp_min) < 0.01 * sharp_min):
        notes.append("DOS width is on its bound. It is a smoothing width degenerate "
                     "with the height profile - read the DOS curve, not this number.")
    if not fit_dos_width:
        notes.append(
            f"Sub-band width was held fixed at {sharpness:.2f} 1/V (FWHM "
            f"{1000 * fwhm:.0f} mV) rather than fitted, so each sub-band is one ideal "
            f"one-electron site population and the DOS reads as a distribution of "
            f"formal potentials. Fitted, this width is degenerate with the height "
            f"profile and runs to whichever bound the grid imposes.")
        if sharp_init < sharp_want - 1e-9:
            notes.append(
                f"The sub-band grid is too coarse for the requested width: "
                f"{num_peaks} sub-bands cap it at {sharp_max:.1f} 1/V against the "
                f"{sharp_want:.1f} 1/V asked for, so the sub-bands are broader than "
                f"intended and the DOS is correspondingly over-smoothed. About "
                f"{peaks_for_width} sub-bands would be needed to carry it.")
    notes.append(
        f"DOS is reported in states cm^-3 eV^-1 using the film thickness "
        f"({thickness:.3g} cm) and electrode area ({electrode_area:.3g} cm^2) as "
        f"supplied. Both are inputs, not measurements, and the absolute height of "
        f"the curve scales inversely with their product - the shape does not. The "
        f"integrated site density is an extrapolation beyond the measured window "
        f"and should not be quoted as a measured density.")
    notes.append(
        "D scales with the square of the film thickness, which is an input here, not "
        f"a measurement: at L = {thickness:.3g} cm, a thickness wrong by 2x moves every "
        "D by 4x. The fast fraction and the slow/fast ratio do not depend on it.")

    dos_charge = float(thickness * np.sum(heights))
    # Sites per unit volume implied by that charge. Reported for completeness,
    # but it is an extrapolation: the model says most sites barely participate
    # over the measured window, so this runs well above the anodic charge that
    # was actually measured. Not a measurement of site density.
    dos_site_density = float(dos_charge / ELEMENTARY_CHARGE
                             / max(electrode_area * thickness, 1e-30))

    scan_results = []
    for i, d in enumerate(loaded):
        sim = np.array(final_sims[i])
        tgt = d["current"][1:]
        rmse = float(100.0 * np.sqrt(np.mean((sim - tgt)**2)) / np.ptp(tgt))

        dt = np.diff(d["time"])
        fwd = np.diff(d["potential"]) > 0
        q_meas = float(np.sum(tgt[fwd] * dt[fwd]))

        scan_results.append({
            "name": d["name"],
            "scan_rate": d["scan_rate"],
            "rmse_pct": rmse,
            "anodic_charge": q_meas,
            "exp_potential": d["potential"][1:].tolist(),
            "exp_current": tgt.tolist(),
            "sim_current": sim.tolist(),
        })

    result_data = {
        "shared": {
            "transport": transport,
            "transport_requested": transport_requested,
            # Kept as the headline diffusivity for older callers; under two_site it
            # is the fast environment.
            "diffusivity": D0,
            "d_fast": D0,
            "d_slow": d_slow if two_site else None,
            "frac_fast": frac_fast if two_site else None,
            "d_ratio": d_ratio if two_site else None,
            "d_of_v_determined": d_of_v_determined,
            "beta_left": beta_left if d_of_v_determined else None,
            "beta_right": beta_right if d_of_v_determined else None,
            "v_center": v_center if d_of_v_determined else None,
            "sharpness": sharpness,
            "dos_fwhm": float(fwhm),
            "dos_overlap": float(fwhm / peak_spacing),
            "num_scans": num_scans,
            "final_loss": final_loss,
            "series_resistance": resistance,
            "resistance_used": use_resistance,
            "resistance_fitted": fit_resistance,
            "ir_drop_max": resistance * i_max,
            "d_lower_bound_only": d_lower_bound_only,
            "environments_merged": environments_merged,
            "v_min": v_min,
            "v_max": v_max,
            "film_thickness": thickness,
            "dos_charge": dos_charge,
            "electrode_area": electrode_area,
            "dos_site_density": dos_site_density,
            "dos_units": "cm^-3 eV^-1",
        },
        "notes": notes,
        "scans": scan_results,
        "plots": {
            "v_plot": v_plot.tolist(),
            "d_of_v": d_of_v.tolist(),
            "d_of_v_slow": d_of_v_slow.tolist() if d_of_v_slow is not None else None,
            "dos_total": dos_total.tolist(),
            "dos_matrix": dos_matrix.T.tolist(),
        },
    }

    if loop and queue:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "data": result_data})
    return result_data
