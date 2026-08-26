// Cyclic Voltammetry Parameter Extraction & Physical Model Fitting
// High-Performance JAX Auto-Diff Engine Client

// Global State
let stagedFiles = [];
let detectedColumns = [];
let isOptimizing = false;
// One joint fit across every staged scan, not one fit per file.
let fitResult = null;

const chartColors = [
    '#38bdf8', '#f43f5e', '#10b981', '#fbbf24', '#a855f7',
    '#fb7185', '#34d399', '#facc15', '#e879f9', '#818cf8',
    '#2dd4bf', '#f87171', '#c084fc', '#f472b6', '#3b82f6'
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
            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.05); padding: 0.5rem 1rem; border-radius: 4px; margin-bottom: 0.5rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%;">${f.name}</span>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <label for="scan_rate_${i}" style="font-size: 0.8rem; color: #cbd5e1;">Scan Rate (V/s):</label>
                    <input type="number" step="any" id="scan_rate_${i}" value="${f.scanRate}" style="width: 80px; padding: 0.25rem; border-radius: 4px; border: 1px solid #334155; background: #1e293b; color: #fff; font-size: 0.85rem;" required />
                </div>
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
        fitResult = await executeSolver(files, config);
        stagedFiles.forEach(f => { f.status = 'done'; });
        if (stageEl) stageEl.innerText = '✓ Shared Physical Model Extraction Complete';
        if (detailsEl) {
            detailsEl.innerText = `Joint fit over ${files.length} scan rate(s) complete.`;
        }
        updateLivePlotProgress();
        displayExtractedResults();
    } catch (err) {
        console.error('Joint fit failed:', err);
        fitResult = null;
        stagedFiles.forEach(f => { f.status = 'error'; });
        if (stageEl) stageEl.innerText = '✕ Optimization Failed';
        if (detailsEl) detailsEl.innerText = err.message || 'The solver could not complete this fit.';
    }

    stopOptimizationUI();
    return false;
};

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
        if (stageEl) stageEl.innerText = `⚡ Joint Multi-Scan Parameter Extraction (${elapsedSec}s)...`;
    }, 500);

    try {
        let lastError = null;
        for (const endpoint of endpoints) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 300000);
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

// Scientific Academic Plotly Layout Configuration (High Contrast)
const layoutConfig = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, -apple-system, sans-serif', color: '#f8fafc', size: 12 },
    margin: { l: 80, r: 40, t: 60, b: 60 },
    xaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.12)',
        zerolinecolor: 'rgba(255, 255, 255, 0.25)',
        tickfont: { color: '#cbd5e1', size: 12 },
        titlefont: { color: '#ffffff', size: 14 },
        tickangle: 0
    },
    yaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.12)',
        zerolinecolor: 'rgba(255, 255, 255, 0.25)',
        tickfont: { color: '#cbd5e1', size: 12 },
        titlefont: { color: '#ffffff', size: 14 },
        tickformat: '.2e'
    }
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
        title: { text: `Experimental Voltammograms (${stagedFiles.length} files)`, font: { color: '#ffffff', size: 15, family: 'Inter, sans-serif' } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Applied Potential <i>V</i> (V vs. Ref)' }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current <i>I</i> (A)' }),
        showlegend: true,
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(15, 23, 42, 0.9)', font: { color: '#ffffff', size: 12 }, bordercolor: '#334155', borderwidth: 1 }
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
        title: { text: 'Experimental vs. Fitted Voltammograms Overlay', font: { color: '#ffffff', size: 15, family: 'Inter, sans-serif' } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Applied Potential <i>V</i> (V vs. Ref)' }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current <i>I</i> (A)' }),
        showlegend: true,
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(15, 23, 42, 0.9)', font: { color: '#ffffff', size: 12 }, bordercolor: '#334155', borderwidth: 1 }
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
        heading.innerText = `Shared across ${p.num_scans || stagedFiles.length} scan rate(s)`;
        paramsDiv.appendChild(heading);

        const grid = document.createElement('div');
        grid.className = 'stats-grid';
        [
            { label: 'Diffusivity D₀ (at V_c)', value: `${(p.D0 || 0).toExponential(3)} cm²/s` },
            { label: 'D(V) Minimum Potential (V_c)', value: `${(p.Vc || 0).toFixed(4)} V` },
            { label: 'β left / right', value: `${(p.beta_L || 0).toFixed(3)} / ${(p.beta_R || 0).toFixed(3)}` },
            { label: 'DOS Width (FWHM)', value: `${(p.dos_fwhm || 0).toFixed(4)} V` }
        ].forEach(c => {
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
                    <td style="padding:0.4rem 0.75rem; color:${chartColors[i % chartColors.length]};">${s.name || '-'}</td>
                    <td style="padding:0.4rem 0.75rem; text-align:right;">${(s.scan_rate * 1000).toFixed(0)} mV/s</td>
                    <td style="padding:0.4rem 0.75rem; text-align:right;">${s.rmse_pct.toFixed(2)}%</td>
                    <td style="padding:0.4rem 0.75rem; text-align:right;">${s.baseline_offset.toExponential(2)} A</td>
                </tr>`).join('');
            perScan.innerHTML = `
                <h4 style="margin-bottom:0.75rem;">Per-Scan Fit Quality</h4>
                <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
                    <thead>
                        <tr style="border-bottom:1px solid #334155; color:#94a3b8;">
                            <th style="padding:0.4rem 0.75rem; text-align:left;">File</th>
                            <th style="padding:0.4rem 0.75rem; text-align:right;">Scan Rate</th>
                            <th style="padding:0.4rem 0.75rem; text-align:right;">RMSE (% of range)</th>
                            <th style="padding:0.4rem 0.75rem; text-align:right;">Baseline Offset</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>`;
            paramsDiv.appendChild(perScan);
        }

        // Bound hits and identifiability caveats reported by the solver.
        if (fitResult.notes && fitResult.notes.length) {
            const notes = document.createElement('div');
            notes.style.cssText = 'margin-top:1.25rem; padding:0.85rem 1rem; border-left:3px solid #fbbf24; background:rgba(251,191,36,0.08); font-size:0.87rem; color:#fde68a;';
            notes.innerHTML = fitResult.notes.map(n => `<div style="margin:0.2rem 0;">${n}</div>`).join('');
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

    const diffTraces = [{
        x: plots.v_plot,
        y: plots.d_of_v,
        mode: 'lines',
        type: 'scatter',
        name: 'Shared D(V)',
        line: { color: chartColors[1], width: 2.8 }
    }];

    const dosLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Extracted Density of States DOS(V)', font: { color: '#ffffff', size: 15, family: 'Inter, sans-serif' } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential <i>V</i> (V vs. Ref)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'DOS (a.u.)', autorange: true, tickformat: '.2e' }),
        showlegend: true,
        legend: { orientation: 'h', y: -0.3, font: { color: '#ffffff', size: 11 } },
        margin: { l: 80, r: 40, t: 60, b: 100 }
    });

    Plotly.react('dos-chart', dosTraces, dosLayout, { responsive: true, displaylogo: false });

    const diffLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Voltage-Dependent Diffusivity Profile D(V)', font: { color: '#ffffff', size: 15, family: 'Inter, sans-serif' } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential <i>V</i> (V vs. Ref)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Diffusivity <i>D</i> (cm²/s)', type: 'log', autorange: true, tickformat: '.1e' }),
        showlegend: true,
        legend: { orientation: 'h', y: -0.3, font: { color: '#ffffff', size: 11 } },
        margin: { l: 80, r: 40, t: 60, b: 100 }
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
    let header = ["V_Plot", "DOS_shared", "D_V_shared"];
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
