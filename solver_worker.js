// CV Curve Fitting Pro - High-Performance Native JavaScript Optimization Engine
// Pure Float64Array JIT-compiled simulation core - Blazing fast in-browser execution

self.onmessage = function(e) {
    const { action, file_content, config } = e.data;
    if (action === 'solve') {
        try {
            solveCVSimulation(file_content, config);
        } catch (err) {
            self.postMessage(JSON.stringify({
                type: 'error',
                message: err.message || String(err)
            }));
        }
    }
};

// Main solver entry point
function solveCVSimulation(fileContent, rawConfig) {
    self.postMessage(JSON.stringify({ type: 'status', message: 'Parsing and preprocessing data...' }));
    
    // Parse configuration
    const scan_rate_v_s = parseFloat(rawConfig.scan_rate || 0.010);
    const film_thickness = parseFloat(rawConfig.film_thickness || 1e-4);
    const v_min = parseFloat(rawConfig.v_min || -1.0);
    const v_max = parseFloat(rawConfig.v_max || 1.0);
    const skip_factor = parseInt(rawConfig.skip_factor || 10, 10);
    const num_peaks = parseInt(rawConfig.num_peaks || 30, 10);
    const max_iter = parseInt(rawConfig.max_iter || 50, 10);
    const num_terms = parseInt(rawConfig.num_terms || 25, 10);
    const loss_weight_const = parseFloat(rawConfig.loss_weight_const || 1.0);
    const pot_col = parseInt(rawConfig.pot_col !== undefined ? rawConfig.pot_col : 8, 10);
    const cur_col = parseInt(rawConfig.cur_col !== undefined ? rawConfig.cur_col : 9, 10);

    const mult_diff = (film_thickness * film_thickness) / 10.0;
    const mult_beta = 1.0;
    const mult_offset = 1e-4;
    const mult_bg_a = 1e-4;
    const mult_bg_k = 10.0;
    const NUM_GLOBALS = 9;

    // Parse CSV data
    const { time, potential, current } = parseAndPreprocessCSV(
        fileContent, pot_col, cur_col, scan_rate_v_s, skip_factor
    );

    if (potential.length < 10) {
        throw new Error("Dataset is too small or specified column indices are invalid.");
    }

    const exp_potential = potential.subarray(1);
    const exp_current = current.subarray(1);
    const exp_time = time;
    const M = exp_potential.length;

    // Notify UI with initial raw dataset for plotting
    self.postMessage(JSON.stringify({
        type: 'init',
        exp_potential: Array.from(exp_potential),
        exp_current: Array.from(exp_current)
    }));

    // Calculate weights via Savitzky-Golay smoothed d2I
    const loss_weights = computeLossWeights(current, loss_weight_const);
    const global_weights = loss_weights.subarray(1);

    // Edge masking for stage 1
    const edge_threshold = (v_max - v_min) * 0.05;
    const global_weights_masked = new Float64Array(M);
    for (let i = 0; i < M; i++) {
        const v = exp_potential[i];
        if (v < v_min + edge_threshold || v > v_max - edge_threshold) {
            global_weights_masked[i] = 0.0;
        } else {
            global_weights_masked[i] = global_weights[i];
        }
    }

    // Initial Physics Prior Extraction
    let mean_pot = 0;
    for (let i = 0; i < potential.length; i++) mean_pot += potential[i];
    mean_pot /= potential.length;

    const totalParamsCount = NUM_GLOBALS + num_peaks * 3;
    const current_x = new Float64Array(totalParamsCount);
    current_x[0] = 2.0;       // D0
    current_x[1] = 1.0;       // beta_L
    current_x[2] = 1.0;       // beta_R
    current_x[3] = mean_pot;  // V_center
    current_x[4] = 0.0;       // baseline_offset
    current_x[5] = 0.1;       // a_R
    current_x[6] = 1.0;       // k_R
    current_x[7] = 0.1;       // a_L
    current_x[8] = 1.0;       // k_L

    // Evenly spaced initial peaks
    const v_step = (v_max - v_min - 0.2) / (num_peaks > 1 ? num_peaks - 1 : 1);
    for (let i = 0; i < num_peaks; i++) {
        const base = NUM_GLOBALS + i * 3;
        current_x[base] = 1.0;                  // weight
        current_x[base + 1] = v_min + 0.1 + i * v_step; // v_crit
        current_x[base + 2] = 15.0;             // sharpness
    }

    // Calibration simulation to scale initial guess
    const calib_sim = runFourierSimulation(
        exp_time, potential,
        current_x[0] * mult_diff, 0.0, 0.0, current_x[3],
        current_x.subarray(NUM_GLOBALS), num_peaks, num_terms, film_thickness
    );

    // Scale and offset alignment
    const v_range = v_max - v_min;
    const safe_min = v_min + (v_range * 0.15);
    const safe_max = v_max - (v_range * 0.15);

    let min_faradaic = Infinity, max_faradaic = -Infinity;
    let min_sim = Infinity, max_sim = -Infinity;
    let sum_real = 0, sum_sim = 0;

    for (let i = 0; i < M; i++) {
        const v = exp_potential[i];
        sum_real += exp_current[i];
        sum_sim += calib_sim[i];
        if (v > safe_min && v < safe_max) {
            if (exp_current[i] < min_faradaic) min_faradaic = exp_current[i];
            if (exp_current[i] > max_faradaic) max_faradaic = exp_current[i];
            if (calib_sim[i] < min_sim) min_sim = calib_sim[i];
            if (calib_sim[i] > max_sim) max_sim = calib_sim[i];
        }
    }

    const real_ptp = (max_faradaic > min_faradaic) ? (max_faradaic - min_faradaic) : 1e-4;
    let sim_ptp = (max_sim > min_sim) ? (max_sim - min_sim) : 1e-6;
    if (sim_ptp < 1e-12) sim_ptp = 1e-6;

    const calibrated_scale = real_ptp / sim_ptp;
    const real_mean = sum_real / M;
    const sim_mean = (sum_sim / M) * calibrated_scale;
    current_x[4] = (real_mean - sim_mean) / mult_offset;

    // Fast Forward Model Evaluation Function
    const sim_buffer = new Float64Array(M);
    function computeForward(params, weights, outSim) {
        const diff = params[0] * mult_diff;
        const beta_l = params[1] * mult_beta;
        const beta_r = params[2] * mult_beta;
        const vc = params[3];
        const off = params[4] * mult_offset;
        const ar = params[5] * mult_bg_a;
        const kr = params[6] * mult_bg_k;
        const al = params[7] * mult_bg_a;
        const kl = params[8] * mult_bg_k;

        const raw_sim = runFourierSimulation(
            exp_time, potential,
            diff, beta_l, beta_r, vc,
            params.subarray(NUM_GLOBALS), num_peaks, num_terms, film_thickness
        );

        let weighted_sq_err = 0.0;
        let total_weight = 0.0;

        for (let i = 0; i < M; i++) {
            const v = exp_potential[i];
            const bg = ar * Math.exp(kr * (v - v_max)) - al * Math.exp(-kl * (v - v_min));
            const sim_val = (raw_sim[i] * calibrated_scale) + off + bg;
            if (outSim) outSim[i] = sim_val;

            const err = sim_val - exp_current[i];
            const w = weights[i];
            weighted_sq_err += w * err * err;
            total_weight += w;
        }

        const weighted_mse = total_weight > 0 ? (weighted_sq_err / total_weight) : 0;
        return Math.sqrt(weighted_mse) * 1e6;
    }

    // Optimization Stages Setup
    const idx_baseline = [4];
    const idx_bg = [5, 6, 7, 8];
    const idx_diffusion_base = [0, 3];
    const idx_peaks = [];
    for (let i = NUM_GLOBALS; i < totalParamsCount; i++) idx_peaks.push(i);
    const all_indices = [];
    for (let i = 0; i < totalParamsCount; i++) all_indices.push(i);

    const stages = [
        { label: "Stage 1: Pure Flat Baseline", active: idx_baseline, weights: global_weights_masked, iters: Math.min(30, max_iter) },
        { label: "Stage 1.5: Background Tails", active: idx_bg, weights: global_weights, iters: Math.min(30, max_iter) },
        { label: "Stage 2: Anchor Peaks (Constant D)", active: idx_baseline.concat(idx_bg, idx_diffusion_base, idx_peaks), weights: global_weights, iters: Math.min(60, max_iter) },
        { label: "Stage 3: Full Non-Linear Polish", active: all_indices, weights: global_weights, iters: max_iter }
    ];

    // Parameter Bounds Generator
    function getBounds(idx, val) {
        const v = Math.abs(val) * 0.5;
        if (idx === 0) return [Math.max(1e-8, val - 5.0), val + 5.0];
        if (idx === 1 || idx === 2) return [val - 1.0, val + 1.0];
        if (idx === 3) return [val - 0.6, val + 0.6];
        if (idx === 4) {
            const range = val !== 0 ? v : 10.0;
            return [val - range, val + range];
        }
        if (idx === 5 || idx === 7) return [Math.max(1e-8, val - (v + 0.1)), val + v + 0.1];
        if (idx === 6 || idx === 8) return [Math.max(0.1, val - (v + 0.5)), val + v + 0.5];
        
        const offset = (idx - NUM_GLOBALS) % 3;
        if (offset === 0) return [Math.max(1e-4, val - (v + 1e-4)), val + 5.0];
        if (offset === 1) return [Math.max(v_min, val - (v + 1e-4)), Math.min(v_max, val + v + 1e-4)];
        return [Math.max(0.1, val - (v + 1e-4)), val + 20.0];
    }

    // Execute Multi-Stage Coordinate/L-BFGS Gradient Optimization
    for (let s = 0; s < stages.length; s++) {
        const stage = stages[s];
        const activeIdx = stage.active;
        const weights = stage.weights;

        // Build active bounds
        const lowerBounds = new Float64Array(totalParamsCount);
        const upperBounds = new Float64Array(totalParamsCount);
        for (let i = 0; i < totalParamsCount; i++) {
            if (activeIdx.includes(i)) {
                const b = getBounds(i, current_x[i]);
                lowerBounds[i] = b[0];
                upperBounds[i] = b[1];
            } else {
                lowerBounds[i] = current_x[i] - 1e-9;
                upperBounds[i] = current_x[i] + 1e-9;
            }
        }

        optimizeLBFGS(
            current_x, activeIdx, weights, computeForward, lowerBounds, upperBounds, stage.iters,
            (iter, loss, currentSim) => {
                self.postMessage(JSON.stringify({
                    type: 'update',
                    stage: stage.label,
                    iter: iter,
                    loss: loss,
                    sim_current: Array.from(currentSim)
                }));
            }
        );
    }

    // Extract Final Parameters and Curves for Output
    const final_sim = new Float64Array(M);
    computeForward(current_x, global_weights, final_sim);

    const final_diffusivity = current_x[0] * mult_diff;
    const final_beta_left = current_x[1] * mult_beta;
    const final_beta_right = current_x[2] * mult_beta;
    const final_v_center = current_x[3];
    const final_baseline_offset = current_x[4] * mult_offset;

    // Generate High-Resolution Diagnostic Curves (500 pts)
    const N_DIAG = 500;
    const v_plot = new Float64Array(N_DIAG);
    const d_of_v = new Float64Array(N_DIAG);
    const dos_total = new Float64Array(N_DIAG);
    const dos_matrix = [];

    for (let k = 0; k < num_peaks; k++) {
        dos_matrix.push(new Float64Array(N_DIAG));
    }

    const diag_step = (v_max - v_min) / (N_DIAG - 1);
    for (let i = 0; i < N_DIAG; i++) {
        const v = v_min + i * diag_step;
        v_plot[i] = v;
        const beta = v < final_v_center ? final_beta_left : final_beta_right;
        d_of_v[i] = final_diffusivity * Math.exp(beta * (v - final_v_center) * (v - final_v_center));

        let tot_dos = 0;
        for (let k = 0; k < num_peaks; k++) {
            const base = NUM_GLOBALS + k * 3;
            const w = current_x[base];
            const vc = current_x[base + 1];
            const sharpness = current_x[base + 2];
            const exp_term = Math.exp(-sharpness * (v - vc));
            const denom = 1.0 + exp_term;
            const mode_val = (w * sharpness * exp_term) / (denom * denom);
            dos_matrix[k][i] = mode_val;
            tot_dos += mode_val;
        }
        dos_total[i] = tot_dos;
    }

    // Package Results
    const result_data = {
        parameters: {
            diffusivity: final_diffusivity,
            beta_left: final_beta_left,
            beta_right: final_beta_right,
            baseline_offset: final_baseline_offset,
            v_center: final_v_center
        },
        plots: {
            v_plot: Array.from(v_plot),
            d_of_v: Array.from(d_of_v),
            dos_total: Array.from(dos_total),
            dos_matrix: dos_matrix.map(row => Array.from(row)),
            exp_potential: Array.from(exp_potential),
            exp_current: Array.from(exp_current),
            sim_current: Array.from(final_sim)
        }
    };

    self.postMessage(JSON.stringify({
        type: 'done',
        data: result_data
    }));
}

// Ultra-Fast Fourier Diffusion Finite-Difference Core in Native Typed Arrays
function runFourierSimulation(
    time_array, potential_array,
    diffusivity, beta_left, beta_right, v_center,
    peaks_array, num_peaks, num_terms, thickness
) {
    const N_steps = potential_array.length - 1;
    const simulated_currents = new Float64Array(N_steps);

    // Precompute wavenumbers, fourier coefficients, sin integrals
    const wavenumbers = new Float64Array(num_terms);
    const fourier_coeffs = new Float64Array(num_terms);
    const sin_integrals = new Float64Array(num_terms);
    const PI = Math.PI;
    const two_L = 2.0 * thickness;

    for (let n = 0; n < num_terms; n++) {
        const mode = 2.0 * (n + 1) - 1.0;
        const wn = (mode * PI) / two_L;
        wavenumbers[n] = wn;
        fourier_coeffs[n] = 4.0 / (mode * PI);
        sin_integrals[n] = 1.0 / wn;
    }

    // Compute Equilibrium Occupancy theta_eq(t_j)
    const occ_eq = new Float64Array(potential_array.length);
    for (let j = 0; j < potential_array.length; j++) {
        const v = potential_array[j];
        let occ_sum = 0.0;
        for (let p = 0; p < num_peaks; p++) {
            const base = p * 3;
            const w = peaks_array[base];
            const vc = peaks_array[base + 1];
            const k = peaks_array[base + 2];
            occ_sum += w / (1.0 + Math.exp(-k * (v - vc)));
        }
        occ_eq[j] = occ_sum;
    }

    // Step 1: Precompute Decay & Forcing Matrices
    const decay_flat = new Float64Array(N_steps * num_terms);
    const forcing_flat = new Float64Array(N_steps * num_terms);

    const cum_dec = new Float64Array(num_terms);
    const acc_forc = new Float64Array(num_terms);
    for (let n = 0; n < num_terms; n++) {
        cum_dec[n] = 1.0;
        acc_forc[n] = 0.0;
    }

    for (let j = 0; j < N_steps; j++) {
        let dt = time_array[j + 1] - time_array[j];
        if (dt <= 0.0) dt = 1e-6;

        const v_next = potential_array[j + 1];
        const beta = (v_next < v_center) ? beta_left : beta_right;
        const d_val = diffusivity * Math.exp(beta * (v_next - v_center) * (v_next - v_center));
        const occ_diff = occ_eq[j] - occ_eq[j + 1];

        const rowOffset = j * num_terms;
        for (let n = 0; n < num_terms; n++) {
            const wn = wavenumbers[n];
            const k_dt = d_val * dt * wn * wn;
            const dec = Math.exp(-k_dt);
            
            let forcing_factor;
            if (k_dt < 1e-8) {
                forcing_factor = 1.0 - 0.5 * k_dt;
            } else {
                forcing_factor = (1.0 - dec) / (k_dt + 1e-15);
            }

            const forc = occ_diff * fourier_coeffs[n] * forcing_factor;
            decay_flat[rowOffset + n] = dec;
            forcing_flat[rowOffset + n] = forc;

            cum_dec[n] = cum_dec[n] * dec;
            acc_forc[n] = acc_forc[n] * dec + forc;
        }
    }

    // Initial Steady-State Fourier Modes T_m_0
    const fourier_modes = new Float64Array(num_terms);
    let total_ions_old = thickness * occ_eq[0];
    for (let n = 0; n < num_terms; n++) {
        const t0 = acc_forc[n] / (1.0 - cum_dec[n] + 1e-15);
        fourier_modes[n] = t0;
        total_ions_old += t0 * sin_integrals[n];
    }

    // Time Evolution & Current Calculation
    for (let j = 0; j < N_steps; j++) {
        let dt = time_array[j + 1] - time_array[j];
        if (dt <= 0.0) dt = 1e-6;

        const rowOffset = j * num_terms;
        let sum_fourier = 0.0;
        for (let n = 0; n < num_terms; n++) {
            const next_mode = fourier_modes[n] * decay_flat[rowOffset + n] + forcing_flat[rowOffset + n];
            fourier_modes[n] = next_mode;
            sum_fourier += next_mode * sin_integrals[n];
        }

        const total_ions_new = thickness * occ_eq[j + 1] + sum_fourier;
        simulated_currents[j] = (total_ions_new - total_ions_old) / dt;
        total_ions_old = total_ions_new;
    }

    return simulated_currents;
}

// Bounded L-BFGS & Adaptive Gradient Optimizer in Pure JavaScript
function optimizeLBFGS(x, activeIndices, weights, computeLoss, lowerBounds, upperBounds, maxIter, onIter) {
    const N = x.length;
    const numActive = activeIndices.length;
    const currentSim = new Float64Array(weights.length);

    let currentLoss = computeLoss(x, weights, currentSim);
    let iter = 0;

    // L-BFGS memory history
    const m = 6;
    const s_history = [];
    const y_history = [];
    const rho_history = [];

    const grad = new Float64Array(N);
    const grad_old = new Float64Array(N);
    const x_old = new Float64Array(N);
    const dir = new Float64Array(N);

    // Initial gradient computation
    computeGradient(x, activeIndices, weights, computeLoss, currentLoss, lowerBounds, upperBounds, grad);

    for (iter = 1; iter <= maxIter; iter++) {
        // Compute search direction via two-loop L-BFGS recursion
        const q = new Float64Array(N);
        for (let i = 0; i < N; i++) q[i] = grad[i];

        const k = s_history.length;
        const alpha = new Float64Array(k);

        for (let i = k - 1; i >= 0; i--) {
            const s_i = s_history[i];
            const y_i = y_history[i];
            const rho_i = rho_history[i];

            let dot = 0;
            for (let j = 0; j < numActive; j++) {
                const idx = activeIndices[j];
                dot += s_i[idx] * q[idx];
            }
            alpha[i] = rho_i * dot;
            for (let j = 0; j < numActive; j++) {
                const idx = activeIndices[j];
                q[idx] -= alpha[i] * y_i[idx];
            }
        }

        // Initial Hessian scaling gamma
        let gamma = 1.0;
        if (k > 0) {
            const s_last = s_history[k - 1];
            const y_last = y_history[k - 1];
            let s_dot_y = 0, y_dot_y = 0;
            for (let j = 0; j < numActive; j++) {
                const idx = activeIndices[j];
                s_dot_y += s_last[idx] * y_last[idx];
                y_dot_y += y_last[idx] * y_last[idx];
            }
            if (y_dot_y > 1e-12) gamma = s_dot_y / y_dot_y;
        }

        const r = new Float64Array(N);
        for (let j = 0; j < numActive; j++) {
            const idx = activeIndices[j];
            r[idx] = gamma * q[idx];
        }

        for (let i = 0; i < k; i++) {
            const s_i = s_history[i];
            const y_i = y_history[i];
            const rho_i = rho_history[i];

            let y_dot_r = 0;
            for (let j = 0; j < numActive; j++) {
                const idx = activeIndices[j];
                y_dot_r += y_i[idx] * r[idx];
            }
            const beta = rho_i * y_dot_r;
            for (let j = 0; j < numActive; j++) {
                const idx = activeIndices[j];
                r[idx] += s_i[idx] * (alpha[i] - beta);
            }
        }

        for (let j = 0; j < numActive; j++) {
            const idx = activeIndices[j];
            dir[idx] = -r[idx];
        }

        // Backtracking line search with Armijo condition
        for (let i = 0; i < N; i++) {
            x_old[i] = x[i];
            grad_old[i] = grad[i];
        }

        let stepSize = 1.0;
        let stepAccepted = false;
        const x_trial = new Float64Array(N);

        for (let ls = 0; ls < 10; ls++) {
            for (let i = 0; i < N; i++) {
                if (activeIndices.includes(i)) {
                    let nextVal = x_old[i] + stepSize * dir[i];
                    if (nextVal < lowerBounds[i]) nextVal = lowerBounds[i];
                    if (nextVal > upperBounds[i]) nextVal = upperBounds[i];
                    x_trial[i] = nextVal;
                } else {
                    x_trial[i] = x_old[i];
                }
            }

            const trialLoss = computeLoss(x_trial, weights, currentSim);
            if (trialLoss < currentLoss || stepSize < 1e-4) {
                currentLoss = trialLoss;
                for (let i = 0; i < N; i++) x[i] = x_trial[i];
                stepAccepted = true;
                break;
            }
            stepSize *= 0.5;
        }

        if (!stepAccepted) break;

        // Compute new gradient
        computeGradient(x, activeIndices, weights, computeLoss, currentLoss, lowerBounds, upperBounds, grad);

        // Update L-BFGS curvature pairs (s = x_new - x_old, y = grad_new - grad_old)
        const s_vec = new Float64Array(N);
        const y_vec = new Float64Array(N);
        let s_dot_y = 0;

        for (let j = 0; j < numActive; j++) {
            const idx = activeIndices[j];
            s_vec[idx] = x[idx] - x_old[idx];
            y_vec[idx] = grad[idx] - grad_old[idx];
            s_dot_y += s_vec[idx] * y_vec[idx];
        }

        if (s_dot_y > 1e-10) {
            if (s_history.length >= m) {
                s_history.shift();
                y_history.shift();
                rho_history.shift();
            }
            s_history.push(s_vec);
            y_history.push(y_vec);
            rho_history.push(1.0 / s_dot_y);
        }

        if (iter % 5 === 0 || iter === 1) {
            onIter(iter, currentLoss, currentSim);
        }
    }
}

// Numerical Central/Forward Gradient Estimator
function computeGradient(x, activeIndices, weights, computeLoss, currentLoss, lowerBounds, upperBounds, gradOut) {
    const N = x.length;
    for (let i = 0; i < N; i++) gradOut[i] = 0;

    const dummySim = null;
    for (let j = 0; j < activeIndices.length; j++) {
        const idx = activeIndices[j];
        const orig = x[idx];
        let eps = Math.abs(orig) * 1e-4;
        if (eps < 1e-6) eps = 1e-6;

        let x_plus = orig + eps;
        if (x_plus > upperBounds[idx]) x_plus = upperBounds[idx];
        const actual_h = x_plus - orig;

        if (actual_h > 1e-12) {
            x[idx] = x_plus;
            const loss_plus = computeLoss(x, weights, dummySim);
            gradOut[idx] = (loss_plus - currentLoss) / actual_h;
            x[idx] = orig;
        } else {
            gradOut[idx] = 0;
        }
    }
}

// CSV Parser and Preprocessor
function parseAndPreprocessCSV(content, potCol, curCol, scanRate, skipFactor) {
    const lines = content.split(/\r?\n/);
    const rawPot = [];
    const rawCur = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line || line.startsWith('#') || line.startsWith('//')) continue;
        const tokens = line.split(/[,\t;\s]+/).map(t => parseFloat(t));
        if (tokens.length > Math.max(potCol, curCol)) {
            const v = tokens[potCol];
            const c = tokens[curCol];
            if (!isNaN(v) && !isNaN(c)) {
                rawPot.push(v);
                rawCur.push(c);
            }
        }
    }

    const totalPts = Math.min(rawPot.length, rawCur.length);
    if (totalPts === 0) throw new Error("No numeric data rows found in CSV.");

    // Compute raw time based on triangular potential scan
    const rawTime = new Float64Array(totalPts);
    rawTime[0] = 0.0;
    for (let i = 1; i < totalPts; i++) {
        const dv = Math.abs(rawPot[i] - rawPot[i - 1]);
        rawTime[i] = rawTime[i - 1] + (dv / scanRate);
    }

    // Downsample by skip factor
    const sampledPot = [];
    const sampledCur = [];
    const sampledTime = [];

    for (let i = 0; i < totalPts; i += skipFactor) {
        sampledPot.push(rawPot[i]);
        sampledCur.push(rawCur[i]);
        sampledTime.push(rawTime[i]);
    }

    return {
        time: new Float64Array(sampledTime),
        potential: new Float64Array(sampledPot),
        current: new Float64Array(sampledCur)
    };
}

// Savitzky-Golay 2nd Derivative Weights
function computeLossWeights(currentArr, lossWeightConst) {
    const M = currentArr.length;
    const weights = new Float64Array(M);
    
    // 2nd derivative approximation
    const d2I = new Float64Array(M);
    let maxD2 = 0;
    for (let i = 1; i < M - 1; i++) {
        const val = Math.abs(currentArr[i + 1] - 2.0 * currentArr[i] + currentArr[i - 1]);
        d2I[i] = val;
        if (val > maxD2) maxD2 = val;
    }
    d2I[0] = d2I[1];
    d2I[M - 1] = d2I[M - 2];
    if (maxD2 === 0) maxD2 = 1.0;

    for (let i = 0; i < M; i++) {
        weights[i] = (d2I[i] / maxD2) + lossWeightConst;
    }
    return weights;
}
