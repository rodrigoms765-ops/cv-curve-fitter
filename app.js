// Cyclic Voltammetry Parameter Extraction & Physical Model Fitting
// High-Performance JAX Auto-Diff Engine Client

// Global State
let stagedFiles = [];
let detectedColumns = [];
let isOptimizing = false;
// One joint fit across every staged scan, not one fit per file.
let fitResult = null;

// Muted, print-legible series colours. Ordered so the first few stay separable
// in greyscale as well as in colour, which matters when a figure is reused.
const chartColors = [
    '#1f3a52', '#7a2e2e', '#3f7d4e', '#8a6d1f', '#4a3c6b',
    '#2c6e75', '#a05a2c', '#5a5f66', '#63406b', '#31556b',
    '#6b4423', '#2f6b4f', '#84343f', '#3b4a6b', '#7a6a2e'
];

// Global Modal Handler
window.toggleModal = function(modalId, show) {
    const modal = document.getElementById(modalId);
    if (modal) {
        if (show) {
            modal.classList.remove('hidden');
        } else {
            modal.classList.add('hidden');
        }
    }
};

// Global Advanced Settings Toggle
window.toggleAdvanced = function() {
    const advToggle = document.getElementById('advanced-toggle');
    const advContent = document.getElementById('advanced-content');
    const toggleIcon = document.getElementById('toggle-icon');
    if (advContent) {
        const isHidden = advContent.classList.toggle('hidden');
        if (advToggle) advToggle.setAttribute('aria-expanded', !isHidden);
        if (toggleIcon) {
            toggleIcon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
        }
    }
};

// Delimiter Detection
function detectDelimiter(line) {
    const commas = (line.match(/,/g) || []).length;
    const tabs = (line.match(/\t/g) || []).length;
    const semicolons = (line.match(/;/g) || []).length;
    if (tabs > commas && tabs > semicolons) return '\t';
    if (semicolons > commas && semicolons > tabs) return ';';
    return ',';
}

// Generalized 2-Column CSV Analysis & Dropdown Populator
function analyzeCSVAndPopulateColumns(content) {
    const lines = content.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith('#') && !l.trim().startsWith('//'));
    if (lines.length === 0) return;

    const delimiter = detectDelimiter(lines[0]);
    const firstLineFields = lines[0].split(delimiter).map(s => s.trim());
    const secondLineFields = lines.length > 1 ? lines[1].split(delimiter).map(s => s.trim()) : [];

    let hasHeader = false;
    if (firstLineFields.some(f => isNaN(parseFloat(f)) && f.length > 0)) {
        hasHeader = true;
    }

    const colCount = hasHeader ? firstLineFields.length : (secondLineFields.length || firstLineFields.length);
    detectedColumns = [];

    const colCounts = new Array(colCount).fill(0);
    const startRow = hasHeader ? 1 : 0;
    for (let i = startRow; i < lines.length; i++) {
        const tokens = lines[i].split(delimiter);
        for (let c = 0; c < colCount; c++) {
            if (c < tokens.length) {
                const s = tokens[c].trim();
                if (s !== "" && !isNaN(parseFloat(s))) {
                    colCounts[c]++;
                }
            }
        }
    }

    let defaultPot = 0;
    let defaultCur = colCount > 1 ? 1 : 0;

    for (let c = 0; c < colCount; c++) {
        let rawHeader = hasHeader && firstLineFields[c] ? firstLineFields[c] : `Column ${c}`;
        let cleanHeader = rawHeader.replace(/["']/g, '');
        let ptsStr = ` (${colCounts[c].toLocaleString()} pts)`;
        let displayName = `${cleanHeader}${ptsStr}`;

        detectedColumns.push({
            index: c,
            name: displayName,
            rawName: cleanHeader,
            count: colCounts[c]
        });

        // Smart column auto-detection based on header text
        const lower = cleanHeader.toLowerCase();
        if (lower.includes('potential') || lower.includes('volt') || lower === 'v' || lower.includes('(v)')) {
            defaultPot = c;
        } else if (lower.includes('current') || lower.includes('curr') || lower === 'i' || lower.includes('(a)') || lower.includes('amp')) {
            defaultCur = c;
        }
    }

    const potSelect = document.getElementById('pot_col');
    const curSelect = document.getElementById('cur_col');
    const metaBar = document.getElementById('column-meta-bar');
    const metaText = document.getElementById('detected-columns-text');

    if (potSelect && curSelect) {
        potSelect.innerHTML = '';
        curSelect.innerHTML = '';

        detectedColumns.forEach(col => {
            const optP = document.createElement('option');
            optP.value = col.index;
            optP.textContent = `[Col ${col.index}] ${col.name}`;
            potSelect.appendChild(optP);

            const optC = document.createElement('option');
            optC.value = col.index;
            optC.textContent = `[Col ${col.index}] ${col.name}`;
            curSelect.appendChild(optC);
        });

        potSelect.value = defaultPot;
        curSelect.value = defaultCur;

        if (metaBar && metaText) {
            metaBar.classList.add('visible');
            metaText.innerHTML = `Loaded <strong>${colCount} column${colCount > 1 ? 's' : ''}</strong> &bull; <strong>${(lines.length - startRow).toLocaleString()} rows</strong>`;
        }

        window.updateLivePreviewFromColumns();
    }
}

// Global Live Preview & Baseline Plotter
window.updateLivePreviewFromColumns = function() {
    if (stagedFiles.length === 0) return;

    const potSelect = document.getElementById('pot_col');
    const curSelect = document.getElementById('cur_col');
    if (!potSelect || !curSelect) return;

    const potCol = parseInt(potSelect.value, 10);
    const curCol = parseInt(curSelect.value, 10);

    let globalVMin = Infinity, globalVMax = -Infinity;
    let globalIMin = Infinity, globalIMax = -Infinity;
    let totalPoints = 0;

    for (let f of stagedFiles) {
        const lines = f.content.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith('#') && !l.trim().startsWith('//'));
        if (lines.length === 0) continue;

        const delimiter = detectDelimiter(lines[0]);
        const firstTokens = lines[0].split(delimiter).map(t => t.trim());
        let startIndex = 0;
        if (firstTokens.length > Math.max(potCol, curCol)) {
            if (isNaN(parseFloat(firstTokens[potCol])) || isNaN(parseFloat(firstTokens[curCol]))) {
                startIndex = 1;
            }
        }

        f.expPotential = [];
        f.expCurrent = [];

        for (let i = startIndex; i < lines.length; i++) {
            const tokens = lines[i].split(delimiter);
            if (tokens.length > Math.max(potCol, curCol)) {
                const vStr = tokens[potCol].trim();
                const cStr = tokens[curCol].trim();
                if (vStr !== "" && cStr !== "") {
                    const v = parseFloat(vStr);
                    const c = parseFloat(cStr);
                    if (!isNaN(v) && !isNaN(c)) {
                        f.expPotential.push(v);
                        f.expCurrent.push(c);
                        
                        globalVMin = Math.min(globalVMin, v);
                        globalVMax = Math.max(globalVMax, v);
                        globalIMin = Math.min(globalIMin, c);
                        globalIMax = Math.max(globalIMax, c);
                    }
                }
            }
        }
        totalPoints += f.expPotential.length;
    }

    if (totalPoints > 0) {
        const vMinInput = document.getElementById('v_min');
        const vMaxInput = document.getElementById('v_max');
        if (vMinInput && vMaxInput) {
            vMinInput.value = globalVMin.toFixed(3);
            vMaxInput.value = globalVMax.toFixed(3);
        }

        const vRangeSpan = document.getElementById('stat-v-range');
        const iRangeSpan = document.getElementById('stat-i-range');
        const ptsSpan = document.getElementById('stat-points-count');
        const statsBox = document.getElementById('col-stats-preview');

        if (vRangeSpan) vRangeSpan.innerText = `${globalVMin.toFixed(3)} V to ${globalVMax.toFixed(3)} V`;
        if (iRangeSpan) iRangeSpan.innerText = `${globalIMin.toExponential(2)} A to ${globalIMax.toExponential(2)} A`;
        if (ptsSpan) ptsSpan.innerText = `${totalPoints.toLocaleString()}`;
        if (statsBox) statsBox.classList.add('visible');

        const statusDetails = document.getElementById('status-details');
        if (statusDetails) {
            let nameText = stagedFiles.length === 1 ? stagedFiles[0].name : `${stagedFiles.length} file(s)`;
            statusDetails.innerHTML = `Loaded <strong>${nameText}</strong> &bull; Potential (Col ${potCol}) &amp; Current (Col ${curCol}) &bull; ${totalPoints.toLocaleString()} points ready for optimization.`;
        }

        renderInitialExpPlot();
    }
};

function addLoadedFiles(filesData) {
    if (filesData.length === 0) return;
    
    stagedFiles = filesData.map(f => {
        let scanRate = 0.010;
        let match = f.name.match(/(\d+(?:\.\d+)?)mVs/i);
        if (match) {
            scanRate = parseFloat(match[1]) / 1000.0;
        } else {
            match = f.name.match(/(\d+(?:\.\d+)?)V/i);
            if (match) scanRate = parseFloat(match[1]);
        }
        return {
            name: f.name,
            content: f.content,
            scanRate: scanRate,
            expPotential: [],
            expCurrent: [],
            results: null,
            status: 'pending'
        };
    });

    const fileNameDisplay = document.getElementById('file-name-display');
    if (fileNameDisplay) {
        fileNameDisplay.innerHTML = stagedFiles.map((f, i) => `
            <div class="file-row">
                <span class="file-row-name" title="${f.name}">${f.name}</span>
                <span class="file-row-rate">
                    <label for="scan_rate_${i}">V s<sup>&minus;1</sup></label>
                    <input type="number" step="any" id="scan_rate_${i}" value="${f.scanRate}" required />
                </span>
            </div>
        `).join('');
        fileNameDisplay.classList.add('has-file');
    }
    
    if (stagedFiles.length > 0) {
        analyzeCSVAndPopulateColumns(stagedFiles[0].content);
    }
}

// Global File Input Handlers
window.handleCSVFileChange = function(input) {
    if (!input || !input.files || input.files.length === 0) return;
    const filesArray = Array.from(input.files);
    Promise.all(filesArray.map(file => new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = ev => resolve({ content: ev.target.result, name: file.name });
        reader.readAsText(file);
    }))).then(filesData => {
        addLoadedFiles(filesData);
    });
};

window.handleCSVDrop = function(event) {
    if (!event || !event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length === 0) return;
    const filesArray = Array.from(event.dataTransfer.files);
    Promise.all(filesArray.map(file => new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = ev => resolve({ content: ev.target.result, name: file.name });
        reader.readAsText(file);
    }))).then(filesData => {
        addLoadedFiles(filesData);
    });
};


// Global Form Submit Handler
window.handleFormSubmit = async function(e) {
    if (e && e.preventDefault) e.preventDefault();
    
    if (stagedFiles.length === 0) {
        alert('Please select and upload a cyclic voltammetry CSV data file first.');
        return false;
    }
    if (isOptimizing) return false;

    const cvForm = document.getElementById('cv-form');
    const formData = new FormData(cvForm);
    const config = {};
    formData.forEach((value, key) => {
        config[key] = value;
    });

    config.pot_col = parseInt(document.getElementById('pot_col').value, 10);
    config.cur_col = parseInt(document.getElementById('cur_col').value, 10);

    // Every scan is sent in one request: the solver fits them against a single
    // shared D(V) and DOS, so they cannot be optimised one file at a time.
    const files = stagedFiles.map((f, i) => {
        const srInput = document.getElementById(`scan_rate_${i}`);
        return {
            name: f.name,
            content: f.content,
            scan_rate: srInput ? parseFloat(srInput.value) : f.scanRate
        };
    });

    startOptimizationUI();

    const stageEl = document.getElementById('status-stage');
    const detailsEl = document.getElementById('status-details');
    if (detailsEl) {
        detailsEl.innerText = `Fitting ${files.length} scan rate(s) simultaneously against one shared diffusivity and density of states...`;
    }

    try {
        if (stageEl) stageEl.innerText = 'Waking solver…';
        await wakeSolver();
        fitResult = await executeSolver(files, config);
        stagedFiles.forEach(f => { f.status = 'done'; });
        if (stageEl) stageEl.innerText = 'Fit Complete';
        if (detailsEl) {
            detailsEl.innerText = `Joint fit over ${files.length} scan rate(s) complete.`;
        }
        updateLivePlotProgress();
        displayExtractedResults();
    } catch (err) {
        console.error('Joint fit failed:', err);
        fitResult = null;
        stagedFiles.forEach(f => { f.status = 'error'; });
        if (stageEl) stageEl.innerText = 'Fit Failed';
        if (detailsEl) detailsEl.innerText = err.message || 'The solver could not complete this fit.';
    }

    stopOptimizationUI();
    return false;
};

// Render's free instances sleep when idle, and the first request also pays the
// JAX import and JIT compile. Waking the instance separately turns one very long
// request into two shorter ones, which is what a proxy timeout actually cares about.
async function wakeSolver() {
    const stageEl = document.getElementById('status-stage');
    const started = Date.now();
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 120000);
            const res = await fetch(window.location.origin + '/health',
                                    { signal: controller.signal, cache: 'no-store' });
            clearTimeout(timer);
            if (res.ok) return true;
        } catch (err) {
            const secs = Math.floor((Date.now() - started) / 1000);
            if (stageEl) stageEl.innerText = `Waking solver — ${secs} s elapsed`;
        }
    }
    return false;
}

// Single joint fit over every staged scan
async function executeSolver(files, config) {
    const endpoints = [
        window.location.origin + "/api/solve",
        window.location.origin + "/solve",
        "http://127.0.0.1:8000/api/solve"
    ];

    const startTime = Date.now();
    const pollInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
        const stageEl = document.getElementById('status-stage');
        if (!stageEl) return;
        // The free instance is roughly 30x slower than a laptop, so set expectations
        // rather than letting a correct-but-slow fit look like a hang.
        const hint = elapsedSec > 60
            ? '. The hosted solver is slow; several minutes is normal.'
            : '';
        stageEl.innerText = `Fitting — ${elapsedSec} s elapsed${hint}`;
    }, 500);

    try {
        let lastError = null;
        for (const endpoint of endpoints) {
            try {
                const controller = new AbortController();
                // A four-scan two-environment fit measured 326 s on the live free
                // instance, so the old 300 s ceiling aborted a job that was about to
                // succeed. Render itself served that request fine; the client was the
                // only thing giving up. Headroom here for more scans or a colder start.
                const timeoutId = setTimeout(() => controller.abort(), 900000);
                const res = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ files: files, config: config }),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                if (res.ok) {
                    const data = await res.json();
                    // The solver answered. If it reports a problem that is the real
                    // answer, so stop here rather than re-running the fit elsewhere.
                    if (data.type === 'error') {
                        throw Object.assign(new Error(data.message || 'Solver error'), { fromSolver: true });
                    }
                    return data;
                }
                lastError = new Error(`Solver returned HTTP ${res.status}`);
            } catch (err) {
                if (err && err.fromSolver) throw err;
                if (err && err.name === 'AbortError') {
                    const mins = ((Date.now() - startTime) / 60000).toFixed(1);
                    err = new Error(`The solver did not respond within ${mins} minutes. `
                        + `Try fewer scan rates, a larger downsample factor, or fewer DOS sub-bands.`);
                }
                lastError = err;
                console.warn(`Solve attempt on ${endpoint} failed:`, err);
            }
        }
        throw lastError || new Error("Could not communicate with solver engine.");
    } finally {
        clearInterval(pollInterval);
    }
}

function startOptimizationUI() {
    isOptimizing = true;
    const spinner = document.getElementById('status-spinner');
    const submitBtn = document.getElementById('submit-btn');
    if (spinner) spinner.classList.remove('hidden');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerText = 'Extracting Parameters...';
    }
}

function stopOptimizationUI() {
    isOptimizing = false;
    const spinner = document.getElementById('status-spinner');
    const submitBtn = document.getElementById('submit-btn');
    if (spinner) spinner.classList.add('hidden');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Execute Optimization';
    }
}

// Plot theme. The panel heading names each figure and the caption beneath it
// carries the interpretation, so the plots themselves take no title - only axes,
// a legend, and a faint grid.
const INK = '#1a1c1f';
const INK_MUTED = '#55595f';
const SERIF = "'Source Serif 4', Georgia, serif";

const layoutConfig = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, -apple-system, sans-serif', color: INK, size: 12 },
    margin: { l: 82, r: 24, t: 16, b: 58 },
    xaxis: {
        gridcolor: 'rgba(26, 28, 31, 0.07)',
        zerolinecolor: 'rgba(26, 28, 31, 0.22)',
        linecolor: 'rgba(26, 28, 31, 0.35)',
        tickfont: { color: INK_MUTED, size: 11 },
        titlefont: { color: INK, size: 13, family: SERIF },
        tickangle: 0
    },
    yaxis: {
        gridcolor: 'rgba(26, 28, 31, 0.07)',
        zerolinecolor: 'rgba(26, 28, 31, 0.22)',
        linecolor: 'rgba(26, 28, 31, 0.35)',
        tickfont: { color: INK_MUTED, size: 11 },
        titlefont: { color: INK, size: 13, family: SERIF },
        tickformat: '.2e'
    }
};

const legendConfig = {
    bgcolor: 'rgba(255, 255, 255, 0.88)',
    bordercolor: 'rgba(26, 28, 31, 0.18)',
    borderwidth: 1,
    font: { color: INK, size: 11 }
};

function renderInitialExpPlot() {
    if (!window.Plotly) return;
    
    const traces = [];
    stagedFiles.forEach((f, i) => {
        if (f.expPotential && f.expPotential.length > 0) {
            traces.push({
                x: f.expPotential,
                y: f.expCurrent,
                mode: 'lines',
                type: 'scatter',
                name: `Exp: ${f.name}`,
                line: { color: chartColors[i % chartColors.length], width: 2.4 }
            });
        }
    });

    const layout = Object.assign({}, layoutConfig, {
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential <i>V</i> (V vs. reference)' }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current <i>I</i> (A)' }),
        showlegend: true,
        legend: Object.assign({ x: 0.02, y: 0.98 }, legendConfig)
    });

    Plotly.react('live-chart', traces, layout, { responsive: true, displaylogo: false });
}

function updateLivePlotProgress() {
    if (!window.Plotly) return;

    const traces = [];
    stagedFiles.forEach((f, i) => {
        if (f.expPotential && f.expPotential.length > 0) {
            traces.push({
                x: f.expPotential,
                y: f.expCurrent,
                mode: 'lines',
                type: 'scatter',
                name: `Exp: ${f.name}`,
                line: { color: chartColors[i % chartColors.length], width: 2.0, dash: 'dot' }
            });

            // The solver returns one entry per scan, ordered by scan rate, so match on name.
            const s = fitResult && fitResult.scans
                ? fitResult.scans.find(sc => sc.name === f.name)
                : null;
            if (s && s.sim_current) {
                traces.push({
                    x: s.exp_potential,
                    y: s.sim_current,
                    mode: 'lines',
                    type: 'scatter',
                    name: `Fit: ${f.name}`,
                    line: { color: chartColors[i % chartColors.length], width: 2.8 }
                });
            }
        }
    });

    const layout = Object.assign({}, layoutConfig, {
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential <i>V</i> (V vs. reference)' }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current <i>I</i> (A)' }),
        showlegend: true,
        legend: Object.assign({ x: 0.02, y: 0.98 }, legendConfig)
    });

    Plotly.react('live-chart', traces, layout, { responsive: true, displaylogo: false });
}

function displayExtractedResults() {
    const resultsPanel = document.getElementById('results-panel');
    if (resultsPanel) resultsPanel.classList.remove('hidden');

    const paramsDiv = document.getElementById('params-output');
    if (paramsDiv && fitResult) {
        const p = fitResult.params || {};
        paramsDiv.innerHTML = '';
        paramsDiv.className = '';

        // Shared parameters: one film, one diffusivity, one density of states.
        const heading = document.createElement('h4');
        heading.style.marginBottom = '1rem';
        const twoSite = p.transport === 'two_site';
        const nScans = p.num_scans || stagedFiles.length;
        heading.innerText = `Shared across ${nScans} sweep rate${nScans === 1 ? '' : 's'}`
            + ` — ${twoSite ? 'two transport environments' : 'a single diffusivity'}`;
        paramsDiv.appendChild(heading);

        const grid = document.createElement('div');
        grid.className = 'stats-grid';
        const cards = [];
        if (twoSite) {
            cards.push(
                { label: 'Diffusivity, fast environment', value: `${(p.d_fast || 0).toExponential(3)} cm²/s` },
                { label: 'Diffusivity, slow environment', value: `${(p.d_slow || 0).toExponential(3)} cm²/s` },
                { label: 'Fast fraction of sites', value: `${(100 * (p.frac_fast || 0)).toFixed(1)}%` },
                { label: 'Ratio, slow / fast', value: `${(p.d_ratio || 0).toPrecision(3)}` }
            );
        } else {
            cards.push({ label: 'Diffusivity',
                         value: `${(p.d_fast || p.D0 || 0).toExponential(3)} cm²/s` });
        }
        // Only meaningful when the fit found real curvature. With both betas at zero
        // D(V) is a flat line and V_c has nothing to sit on, so it drifts anywhere
        // inside its bounds - printing it then would dress noise as a measurement.
        if (p.d_of_v_determined) {
            cards.push(
                { label: 'D(V) minimum, V_c', value: `${p.Vc.toFixed(4)} V` },
                { label: 'Exponents β, left / right', value: `${p.beta_L.toFixed(3)} / ${p.beta_R.toFixed(3)}` }
            );
        } else {
            cards.push({ label: 'D(V) shape', value: 'flat — not determined' });
        }
        cards.push(
            { label: 'DOS width (FWHM)', value: `${(p.dos_fwhm || 0).toFixed(4)} V` },
            { label: 'DOS integrated charge', value: `${(p.dos_charge || 0).toExponential(3)} C` }
        );
        cards.forEach(c => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `<span class="stat-label">${c.label}</span><span class="stat-value">${c.value}</span>`;
            grid.appendChild(card);
        });
        paramsDiv.appendChild(grid);

        // Per-scan: only the fit quality and the non-faradaic offset differ.
        if (fitResult.scans && fitResult.scans.length) {
            const perScan = document.createElement('div');
            perScan.style.marginTop = '1.5rem';
            let rows = fitResult.scans.map((s, i) => `
                <tr>
                    <td><span style="color:${chartColors[i % chartColors.length]};">&#9632;</span> ${s.name || '&mdash;'}</td>
                    <td>${(s.scan_rate * 1000).toFixed(0)}</td>
                    <td>${s.rmse_pct.toFixed(2)}</td>
                    <td>${s.baseline_offset.toExponential(2)}</td>
                    <td>${(s.anodic_charge || 0).toExponential(2)}</td>
                    <td>${(s.non_faradaic_pct || 0).toFixed(1)}</td>
                </tr>`).join('');
            perScan.innerHTML = `
                <h4>Fit Quality by Sweep Rate</h4>
                <div class="table-scroll">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Rate (mV s<sup>&minus;1</sup>)</th>
                            <th>RMSE (% range)</th>
                            <th>Offset (A)</th>
                            <th>Anodic charge (C)</th>
                            <th>Non-faradaic (%)</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
                </div>`;
            paramsDiv.appendChild(perScan);
        }

        // Bound hits and identifiability caveats reported by the solver.
        if (fitResult.notes && fitResult.notes.length) {
            const notes = document.createElement('div');
            notes.className = 'notes';
            notes.innerHTML = '<span class="notes-title">Caveats reported by the solver</span>'
                + fitResult.notes.map(n => `<p class="note-item">${n}</p>`).join('');
            paramsDiv.appendChild(notes);
        }
    }

    renderSecondaryPlots();
}

function renderSecondaryPlots() {
    if (!window.Plotly) return;

    // One shared DOS and one shared D(V) for the whole batch, not one per file.
    if (!fitResult || !fitResult.plots) return;
    const plots = fitResult.plots;

    const dosTraces = [{
        x: plots.v_plot,
        y: plots.dos_total,
        mode: 'lines',
        type: 'scatter',
        name: 'Shared DOS',
        line: { color: chartColors[0], width: 2.8 }
    }];

    const fp = (fitResult && fitResult.params) || {};
    const fastLabel = fp.transport === 'two_site'
        ? `Fast environment (${(100 * (fp.frac_fast || 0)).toFixed(0)}% of sites)`
        : 'Shared D(V)';
    const diffTraces = [{
        x: plots.v_plot,
        y: plots.d_of_v,
        mode: 'lines',
        type: 'scatter',
        name: fastLabel,
        line: { color: chartColors[1], width: 2.8 }
    }];
    if (plots.d_of_v_slow) {
        diffTraces.push({
            x: plots.v_plot,
            y: plots.d_of_v_slow,
            mode: 'lines',
            type: 'scatter',
            name: `Slow environment (${(100 * (1 - (fp.frac_fast || 0))).toFixed(0)}%)`,
            line: { color: chartColors[2 % chartColors.length], width: 2.8 }
        });
    }

    const dosLayout = Object.assign({}, layoutConfig, {
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential <i>V</i> (V vs. reference)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'DOS (arb. units)', autorange: true, tickformat: '.2e' }),
        showlegend: true,
        legend: Object.assign({ orientation: 'h', y: -0.28 }, legendConfig),
        margin: { l: 78, r: 24, t: 12, b: 92 }
    });

    Plotly.react('dos-chart', dosTraces, dosLayout, { responsive: true, displaylogo: false });

    const diffLayout = Object.assign({}, layoutConfig, {
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential <i>V</i> (V vs. reference)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Diffusivity <i>D</i> (cm²/s)', type: 'log', autorange: true, tickformat: '.1e' }),
        showlegend: true,
        legend: Object.assign({ orientation: 'h', y: -0.28 }, legendConfig),
        margin: { l: 78, r: 24, t: 12, b: 92 }
    });

    Plotly.react('diffusivity-chart', diffTraces, diffLayout, { responsive: true, displaylogo: false });
}

// Global Export Functions
window.exportResultsJson = function() {
    if (!fitResult) return;
    downloadFile(JSON.stringify(fitResult, null, 2),
                 'cv_shared_fit_parameters.json', 'application/json');
};

window.exportResultsCsv = function() {
    if (!fitResult || !fitResult.plots) return;
    const plots = fitResult.plots;
    const scans = fitResult.scans || [];

    // Shared columns first, then one Exp_V / Exp_I / Fit_I block per scan.
    const hasSlow = !!plots.d_of_v_slow;
    let header = ["V_Plot", "DOS_shared", hasSlow ? "D_V_fast" : "D_V_shared"];
    if (hasSlow) header.push("D_V_slow");
    scans.forEach(s => {
        const tag = `${(s.scan_rate * 1000).toFixed(0)}mVs`;
        header.push(`Exp_V_${tag}`, `Exp_I_${tag}`, `Fit_I_${tag}`);
    });

    let maxLen = plots.v_plot.length;
    scans.forEach(s => { maxLen = Math.max(maxLen, s.exp_potential.length); });

    const rows = [header.join(",")];
    for (let i = 0; i < maxLen; i++) {
        const row = [
            i < plots.v_plot.length ? plots.v_plot[i] : "",
            i < plots.dos_total.length ? plots.dos_total[i] : "",
            i < plots.d_of_v.length ? plots.d_of_v[i] : ""
        ];
        if (hasSlow) row.push(i < plots.d_of_v_slow.length ? plots.d_of_v_slow[i] : "");
        scans.forEach(s => {
            row.push(i < s.exp_potential.length ? s.exp_potential[i] : "",
                     i < s.exp_current.length ? s.exp_current[i] : "",
                     i < s.sim_current.length ? s.sim_current[i] : "");
        });
        rows.push(row.join(","));
    }

    downloadFile(rows.join("\n"), 'cv_shared_fit_curves.csv', 'text/csv');
};

function downloadFile(content, fileName, contentType) {
    const a = document.createElement("a");
    const file = new Blob([content], { type: contentType });
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(a.href);
    }, 100);
}

// Master Initialization Function
window.__initCVApp = function() {
    // `expPotential` lives per staged file; there is no global of that name.
    if (window.Plotly && stagedFiles.some(f => f.expPotential && f.expPotential.length > 0)) {
        Plotly.Plots.resize('live-chart');
    }
};

// Window resize observer to keep Plotly charts perfectly proportioned
window.addEventListener('resize', () => {
    if (window.Plotly) {
        const chartIds = ['live-chart', 'dos-chart', 'diffusivity-chart'];
        chartIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.data) {
                Plotly.Plots.resize(id);
            }
        });
    }
});

// Run initialization immediately and on DOM load
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.__initCVApp);
    } else {
        window.__initCVApp();
    }
}
