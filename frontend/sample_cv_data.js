// Sample Cyclic Voltammetry dataset for quick testing and demonstration
// Columns: Index, Potential (V) [col 1], Current (A) [col 2]

function generateSampleCVData() {
    const rows = ["Index,Potential_V,Current_A"];
    const v_min = -0.8;
    const v_max = 0.8;
    const n_pts = 400;
    
    // Forward scan (-0.8V to 0.8V)
    for (let i = 0; i <= n_pts; i++) {
        const v = v_min + (v_max - v_min) * (i / n_pts);
        // Anodic peak around 0.15V + background charging + exponential edge
        const peak = 4.5e-5 * Math.exp(-Math.pow((v - 0.15) / 0.12, 2));
        const bg = 8e-6 * (v + 0.2) + 2e-7 * Math.exp(5 * (v - 0.7));
        const noise = (Math.random() - 0.5) * 5e-7;
        const current = peak + bg + noise;
        rows.push(`${i},${v.toFixed(5)},${current.toExponential(6)}`);
    }
    
    // Reverse scan (0.8V to -0.8V)
    for (let i = 0; i <= n_pts; i++) {
        const v = v_max - (v_max - v_min) * (i / n_pts);
        // Cathodic peak around -0.05V + background charging + negative exponential edge
        const peak = -3.8e-5 * Math.exp(-Math.pow((v - (-0.05)) / 0.14, 2));
        const bg = -6e-6 * (v + 0.2) - 2e-7 * Math.exp(-5 * (v - (-0.7)));
        const noise = (Math.random() - 0.5) * 5e-7;
        const current = peak + bg + noise;
        rows.push(`${n_pts + 1 + i},${v.toFixed(5)},${current.toExponential(6)}`);
    }
    
    return rows.join("\n");
}

const SAMPLE_CV_CSV = generateSampleCVData();
