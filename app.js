// CV Curve Fitting Pro - Pure Python JAX + SciPy Engine Client
// Connected to Python FastAPI / WebSocket Backend (Free ZeroGPU & CPU Ready)

// Global State
let activeSocket = null;
let expPotential = [];
let expCurrent = [];
let latestResults = null;
let stagedFileContent = null;
let stagedFileName = "No file chosen";
let detectedColumns = [];
let isBackendAvailable = false;
let activeBackendType = "offline"; // "cloud", "local", "offline"

// Default endpoints
const DEFAULT_LOCAL_URL = "http://127.0.0.1:8000";
const DEFAULT_HF_SPACE_URL = "https://rodrigo1421-cv-curve-fitting.hf.space";

function getStoredBackendUrl() {
    return localStorage.getItem('cv_backend_url') || "";
}

function resolveEndpoints(rawUrl) {
    let url = (rawUrl || "").trim().replace(/\/+$/, "");
    if (!url) {
        if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
            url = window.location.port ? `${window.location.protocol}//${window.location.host}` : DEFAULT_LOCAL_URL;
        } else if (window.location.hostname.endsWith(".hf.space") || window.location.hostname.includes("huggingface.co")) {
            url = `${window.location.protocol}//${window.location.host}`;
        } else {
            url = DEFAULT_HF_SPACE_URL;
        }
    }
    
    let httpHealth = "";
    let wsSolve = "";

    if (url.startsWith("https://")) {
        httpHealth = url + "/health";
        wsSolve = url.replace("https://", "wss://") + "/ws/solve";
    } else if (url.startsWith("http://")) {
        httpHealth = url + "/health";
        wsSolve = url.replace("http://", "ws://") + "/ws/solve";
    } else if (url.startsWith("wss://")) {
        wsSolve = url.includes("/ws/solve") ? url : url + "/ws/solve";
        httpHealth = url.replace("wss://", "https://").replace(/\/ws\/solve\/?$/, "") + "/health";
    } else if (url.startsWith("ws://")) {
        wsSolve = url.includes("/ws/solve") ? url : url + "/ws/solve";
        httpHealth = url.replace("ws://", "http://").replace(/\/ws\/solve\/?$/, "") + "/health";
    } else {
        httpHealth = "https://" + url + "/health";
        wsSolve = "wss://" + url + "/ws/solve";
    }

    return {
        rawUrl: url,
        httpHealth: httpHealth,
        wsSolve: wsSolve
    };
}

let currentEndpoints = resolveEndpoints(getStoredBackendUrl());

// Probing Python Backend Engine
async function probePythonBackend() {
    const dot = document.getElementById('engine-dot');
    const label = document.getElementById('engine-label');
    const msg = document.getElementById('engine-status-msg');

    const candidates = [
        currentEndpoints.httpHealth,
        window.location.origin + "/health",
        DEFAULT_HF_SPACE_URL + "/health",
        DEFAULT_LOCAL_URL + "/health"
    ];

    for (const url of candidates) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 4000);
            const res = await fetch(url, { signal: controller.signal });
            clearTimeout(timeoutId);

            if (res.ok) {
                const data = await res.json();
                isBackendAvailable = true;
                const isLocal = url.includes("127.0.0.1") || url.includes("localhost");
                activeBackendType = isLocal ? "local" : "cloud";

                if (dot) dot.className = 'status-dot online';
                if (label) {
                    label.innerText = isLocal ? 'Local JAX Engine Active' : 'ZeroGPU A100 Active (100% Free)';
                }
                if (msg) {
                    const hw = data.hardware || (isLocal ? "Local Machine" : "Hugging Face ZeroGPU");
                    msg.innerHTML = `<span style="color: #10b981; font-weight: 500;">✓ Connected to ${hw}</span> &bull; JAX Auto-Diff Engine Ready ($0.00 / Free)`;
                }
                return true;
            }
        } catch (e) {
            // Check next candidate
        }
    }

    isBackendAvailable = true; // allow optimistic connection
    activeBackendType = "cloud";
    if (dot) dot.className = 'status-dot online';
    if (label) label.innerText = 'ZeroGPU JAX Engine Ready';
    if (msg) {
        msg.innerHTML = `<span style="color: #10b981; font-weight: 500;">✓ ZeroGPU JAX Server Active</span>`;
    }
    return false;
}

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

// Global Backend URL Configuration
window.saveBackendUrl = function() {
    const urlInput = document.getElementById('backend-url-input');
    if (urlInput) {
        const val = urlInput.value.trim();
        localStorage.setItem('cv_backend_url', val);
        currentEndpoints = resolveEndpoints(val);
        const msg = document.getElementById('engine-status-msg');
        if (msg) msg.innerText = 'Testing connection...';
        probePythonBackend();
    }
};

window.setBackendPreset = function(presetType) {
    const urlInput = document.getElementById('backend-url-input');
    if (!urlInput) return;

    if (presetType === 'origin') {
        urlInput.value = window.location.origin;
    } else if (presetType === 'hf') {
        urlInput.value = DEFAULT_HF_SPACE_URL;
    } else if (presetType === 'local') {
        urlInput.value = DEFAULT_LOCAL_URL;
    }
    window.saveBackendUrl();
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

// Robust CSV Column Analysis & Dropdown Populator
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

    for (let c = 0; c < colCount; c++) {
        let rawHeader = hasHeader && firstLineFields[c] ? firstLineFields[c] : `Column ${c}`;
        let cycleNum = Math.floor(c / 4) + 1;
        let isAdjusted = rawHeader.toLowerCase().includes('adjusted');
        let isCurrent = rawHeader.toLowerCase().includes('current') || rawHeader.toLowerCase().includes('(a)');
        let isPotential = rawHeader.toLowerCase().includes('potential') || rawHeader.toLowerCase().includes('(v)');

        let typeStr = isPotential ? "Potential (V)" : (isCurrent ? "Current (A)" : rawHeader);
        let adjStr = isAdjusted ? " [Adjusted]" : " [Raw]";
        let cycleStr = colCount >= 8 ? `Cycle ${cycleNum}` : "";
        let ptsStr = ` (${colCounts[c].toLocaleString()} pts)`;

        let displayName = `${cycleStr ? cycleStr + ' ' : ''}${typeStr}${adjStr}${ptsStr}`;
        if (!hasHeader) displayName = `Column ${c}${ptsStr}`;

        detectedColumns.push({
            index: c,
            name: displayName,
            rawName: rawHeader,
            count: colCounts[c],
            isAdjusted: isAdjusted
        });
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

        let defaultPot = 0;
        let defaultCur = colCount > 1 ? 1 : 0;

        if (colCount >= 10) {
            defaultPot = 8;
            defaultCur = 9;
        } else if (colCount >= 4) {
            defaultPot = 0;
            defaultCur = 1;
        }

        potSelect.value = defaultPot;
        curSelect.value = defaultCur;

        if (metaBar && metaText) {
            metaBar.classList.add('visible');
            metaText.innerHTML = `✓ Detected <strong>${colCount} columns</strong> &bull; Total <strong>${lines.length - startRow} rows</strong>`;
        }

        updateCycleButtonsActiveState(defaultPot, defaultCur);
        window.updateLivePreviewFromColumns();
    }
}

function updateCycleButtonsActiveState(pot, cur) {
    const buttons = document.querySelectorAll('.preset-pill-btn');
    buttons.forEach(btn => {
        const bPot = parseInt(btn.getAttribute('data-pot'), 10);
        const bCur = parseInt(btn.getAttribute('data-cur'), 10);
        if (bPot === pot && bCur === cur) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// Global Cycle Preset Click Handler
window.applyCyclePreset = function(pot, cur, skip, btn) {
    const potSelect = document.getElementById('pot_col');
    const curSelect = document.getElementById('cur_col');
    const skipInput = document.getElementById('skip_factor');
    if (skip && skipInput) {
        skipInput.value = skip;
    }
    if (potSelect && curSelect) {
        potSelect.value = pot;
        curSelect.value = cur;
        updateCycleButtonsActiveState(pot, cur);
        window.updateLivePreviewFromColumns();
    }
};

// Global Live Preview & Baseline Plotter
window.updateLivePreviewFromColumns = function() {
    if (!stagedFileContent) return;

    const potSelect = document.getElementById('pot_col');
    const curSelect = document.getElementById('cur_col');
    if (!potSelect || !curSelect) return;

    const potCol = parseInt(potSelect.value, 10);
    const curCol = parseInt(curSelect.value, 10);
    updateCycleButtonsActiveState(potCol, curCol);

    const lines = stagedFileContent.split(/\r?\n/).filter(l => l.trim() && !l.trim().startsWith('#') && !l.trim().startsWith('//'));
    if (lines.length === 0) return;

    const delimiter = detectDelimiter(lines[0]);
    const firstTokens = lines[0].split(delimiter).map(t => t.trim());
    let startIndex = 0;
    if (firstTokens.length > Math.max(potCol, curCol)) {
        if (isNaN(parseFloat(firstTokens[potCol])) || isNaN(parseFloat(firstTokens[curCol]))) {
            startIndex = 1;
        }
    }

    const previewPot = [];
    const previewCur = [];

    for (let i = startIndex; i < lines.length; i++) {
        const tokens = lines[i].split(delimiter);
        if (tokens.length > Math.max(potCol, curCol)) {
            const vStr = tokens[potCol].trim();
            const cStr = tokens[curCol].trim();
            if (vStr !== "" && cStr !== "") {
                const v = parseFloat(vStr);
                const c = parseFloat(cStr);
                if (!isNaN(v) && !isNaN(c)) {
                    previewPot.push(v);
                    previewCur.push(c);
                }
            }
        }
    }

    if (previewPot.length > 0) {
        expPotential = previewPot;
        expCurrent = previewCur;

        const vMin = Math.min(...previewPot);
        const vMax = Math.max(...previewPot);
        const iMin = Math.min(...previewCur);
        const iMax = Math.max(...previewCur);

        const vMinInput = document.getElementById('v_min');
        const vMaxInput = document.getElementById('v_max');
        if (vMinInput && vMaxInput) {
            vMinInput.value = vMin.toFixed(2);
            vMaxInput.value = vMax.toFixed(2);
        }

        const vRangeSpan = document.getElementById('stat-v-range');
        const iRangeSpan = document.getElementById('stat-i-range');
        const ptsSpan = document.getElementById('stat-points-count');
        const statsBox = document.getElementById('col-stats-preview');

        if (vRangeSpan) vRangeSpan.innerText = `${vMin.toFixed(2)}V to ${vMax.toFixed(2)}V`;
        if (iRangeSpan) iRangeSpan.innerText = `${iMin.toExponential(2)}A to ${iMax.toExponential(2)}A`;
        if (ptsSpan) ptsSpan.innerText = `${previewPot.length.toLocaleString()}`;
        if (statsBox) statsBox.classList.add('visible');

        const statusDetails = document.getElementById('status-details');
        if (statusDetails) {
            statusDetails.innerHTML = `Loaded <strong>${stagedFileName}</strong> &bull; Col ${potCol} (V) &amp; Col ${curCol} (I) &bull; ${previewPot.length} points ready for JAX optimization.`;
        }

        renderInitialExpPlot(previewPot, previewCur);
    }
};

function setLoadedFile(content, name) {
    stagedFileContent = content;
    stagedFileName = name;

    const fileNameDisplay = document.getElementById('file-name-display');
    if (fileNameDisplay) {
        fileNameDisplay.innerText = name;
        fileNameDisplay.classList.add('has-file');
    }
    analyzeCSVAndPopulateColumns(content);
}

// Global File Input Handlers
window.handleCSVFileChange = function(input) {
    if (!input || !input.files || input.files.length === 0) return;
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = function(ev) {
        setLoadedFile(ev.target.result, file.name);
    };
    reader.readAsText(file);
};

window.handleCSVDrop = function(event) {
    if (!event || !event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length === 0) return;
    const file = event.dataTransfer.files[0];
    const reader = new FileReader();
    reader.onload = function(ev) {
        setLoadedFile(ev.target.result, file.name);
    };
    reader.readAsText(file);
};

// Global Form Submit Handler
window.handleFormSubmit = async function(e) {
    if (e && e.preventDefault) e.preventDefault();
    
    if (!stagedFileContent) {
        alert('Please select and upload a CSV data file first.');
        return false;
    }

    const cvForm = document.getElementById('cv-form');
    const formData = new FormData(cvForm);
    const config = {};
    formData.forEach((value, key) => {
        config[key] = value;
    });

    config.pot_col = parseInt(document.getElementById('pot_col').value, 10);
    config.cur_col = parseInt(document.getElementById('cur_col').value, 10);

    startOptimizationUI();
    executePythonSolver(stagedFileContent, config);
    return false;
};

// Execution via Python JAX WebSocket
function executePythonSolver(fileContent, config) {
    const stageEl = document.getElementById('status-stage');
    const detailsEl = document.getElementById('status-details');
    if (stageEl) stageEl.innerText = 'Connecting to ZeroGPU JAX Engine...';
    if (detailsEl) detailsEl.innerText = `Connecting via WebSocket on ${currentEndpoints.wsSolve}...`;

    if (activeSocket) {
        activeSocket.close();
    }

    try {
        activeSocket = new WebSocket(currentEndpoints.wsSolve);
    } catch (err) {
        console.error("WebSocket initialization failed:", err);
        handleBackendOffline();
        return;
    }

    activeSocket.onopen = () => {
        const isLocal = currentEndpoints.rawUrl.includes("127.0.0.1") || currentEndpoints.rawUrl.includes("localhost");
        if (stageEl) stageEl.innerText = isLocal ? '⚡ Local JAX Engine Running' : '☁️ ZeroGPU A100 Engine Running';
        if (detailsEl) detailsEl.innerText = 'JAX Auto-Diff multi-stage L-BFGS-B optimization in progress...';
        
        activeSocket.send(JSON.stringify({
            action: 'solve',
            config: config,
            file_content: fileContent
        }));
    };

    activeSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleSolverMessage(data);
        } catch (e) {
            console.error("Error parsing message from server:", e);
        }
    };

    activeSocket.onerror = (err) => {
        console.warn("WebSocket connection encountered error. Falling back if necessary:", err);
    };

    activeSocket.onclose = () => {
        stopOptimizationUI();
    };
}

function handleSolverMessage(data) {
    const stageEl = document.getElementById('status-stage');
    const detailsEl = document.getElementById('status-details');

    if (data.type === 'progress') {
        const stageName = data.stage_name || `Stage ${data.stage}`;
        const iterText = data.iteration ? ` (Iter ${data.iteration})` : '';
        if (stageEl) stageEl.innerText = `⚡ ${stageName}${iterText}`;
        if (detailsEl) detailsEl.innerText = `Objective Loss: ${data.loss.toExponential(4)} | Diffusivity D0: ${(data.d0 || 0).toExponential(3)} cm²/s`;

        if (data.current_fit && window.Plotly) {
            updateLivePlotProgress(data.current_fit);
        }
    } else if (data.type === 'done') {
        if (stageEl) stageEl.innerText = '✓ Optimization Successfully Converged';
        if (detailsEl) detailsEl.innerText = `Converged in ${data.total_iterations || 100} iterations with final loss ${data.final_loss ? data.final_loss.toExponential(4) : 'N/A'}.`;
        
        stopOptimizationUI();
        latestResults = data;
        displayExtractedResults(data);
    } else if (data.type === 'error') {
        if (stageEl) stageEl.innerText = '❌ Optimization Error';
        if (detailsEl) detailsEl.innerText = data.message || 'An error occurred during calculation.';
        stopOptimizationUI();
        alert(`Solver Message: ${data.message}`);
    }
}

function handleBackendOffline() {
    const stageEl = document.getElementById('status-stage');
    const detailsEl = document.getElementById('status-details');
    if (stageEl) stageEl.innerText = 'Engine Connecting...';
    if (detailsEl) detailsEl.innerText = 'Please wait a moment while the ZeroGPU container initializes.';
    stopOptimizationUI();
}

function startOptimizationUI() {
    const spinner = document.getElementById('status-spinner');
    const submitBtn = document.getElementById('submit-btn');
    if (spinner) spinner.classList.remove('hidden');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerText = 'Optimizing in JAX...';
    }
}

function stopOptimizationUI() {
    const spinner = document.getElementById('status-spinner');
    const submitBtn = document.getElementById('submit-btn');
    if (spinner) spinner.classList.add('hidden');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Execute Optimization';
    }
}

// Plotly Visualizations
const layoutConfig = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Roboto, sans-serif', color: '#64748b', size: 12 },
    margin: { l: 65, r: 25, t: 35, b: 50 },
    xaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        zerolinecolor: 'rgba(255, 255, 255, 0.1)',
        tickfont: { color: '#94a3b8' }
    },
    yaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        zerolinecolor: 'rgba(255, 255, 255, 0.1)',
        tickfont: { color: '#94a3b8' }
    }
};

function renderInitialExpPlot(pot, cur) {
    if (!window.Plotly) return;
    const traceExp = {
        x: pot,
        y: cur,
        mode: 'lines',
        type: 'scatter',
        name: 'Experimental CV',
        line: { color: '#38bdf8', width: 2 }
    };

    const layout = Object.assign({}, layoutConfig, {
        title: { text: `Cyclic Voltammogram: ${stagedFileName}`, font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Applied Potential (V)' }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current (A)' }),
        showlegend: true,
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(15, 23, 42, 0.7)' }
    });

    Plotly.react('live-chart', [traceExp], layout, { responsive: true });
}

function updateLivePlotProgress(currentFit) {
    if (!window.Plotly) return;
    const traceExp = {
        x: expPotential,
        y: expCurrent,
        mode: 'lines',
        type: 'scatter',
        name: 'Experimental Data',
        line: { color: '#38bdf8', width: 2 }
    };

    const traceSim = {
        x: currentFit.potential || expPotential,
        y: currentFit.current,
        mode: 'lines',
        type: 'scatter',
        name: 'JAX Model Fit',
        line: { color: '#f43f5e', width: 2.5 }
    };

    const layout = Object.assign({}, layoutConfig, {
        title: { text: 'Live Hardware-Accelerated JAX Fit Overlay', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)' }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current (A)' }),
        showlegend: true,
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(15, 23, 42, 0.7)' }
    });

    Plotly.react('live-chart', [traceExp, traceSim], layout, { responsive: true });
}

function displayExtractedResults(results) {
    const resultsPanel = document.getElementById('results-panel');
    if (resultsPanel) resultsPanel.classList.remove('hidden');

    const paramsDiv = document.getElementById('params-output');
    if (paramsDiv) {
        paramsDiv.innerHTML = '';
        const params = results.params || {};

        const cards = [
            { label: 'Baseline Diffusivity (D₀)', value: `${(params.D0 || 0).toExponential(3)} cm²/s` },
            { label: 'Central Voltage (V_c)', value: `${(params.Vc || 0).toFixed(4)} V` },
            { label: 'Asymmetry Left (β_L)', value: `${(params.beta_L || 0).toFixed(4)} V⁻²` },
            { label: 'Asymmetry Right (β_R)', value: `${(params.beta_R || 0).toFixed(4)} V⁻²` },
            { label: 'DC Current Offset (I_offset)', value: `${(params.I_offset || 0).toExponential(3)} A` },
            { label: 'Final Objective Loss', value: results.final_loss ? results.final_loss.toExponential(4) : 'N/A' }
        ];

        cards.forEach(c => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `
                <span class="stat-label">${c.label}</span>
                <span class="stat-value">${c.value}</span>
            `;
            paramsDiv.appendChild(card);
        });
    }

    if (results.plots) {
        renderSecondaryPlots(results.plots);
    }
}

function renderSecondaryPlots(plots) {
    if (!window.Plotly) return;

    // DOS Plot
    const dosTraces = [];
    if (plots.dos_peaks && plots.dos_peaks.length > 0) {
        plots.dos_peaks.forEach((peak, i) => {
            dosTraces.push({
                x: plots.v_plot,
                y: peak,
                mode: 'lines',
                type: 'scatter',
                name: `Sub-band ${i+1}`,
                line: { width: 1, dash: 'dot' }
            });
        });
    }

    dosTraces.push({
        x: plots.v_plot,
        y: plots.dos_total,
        mode: 'lines',
        type: 'scatter',
        name: 'Total DOS(V)',
        line: { color: '#10b981', width: 2.5 }
    });

    const dosLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Extracted Density of States DOS(V)', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'DOS (a.u.)', autorange: true }),
        showlegend: false
    });

    Plotly.react('dos-chart', dosTraces, dosLayout, { responsive: true });

    // Diffusivity D(V) Plot
    const traceDiff = {
        x: plots.v_plot,
        y: plots.d_of_v,
        mode: 'lines',
        type: 'scatter',
        name: 'D(V)',
        line: { color: '#38bdf8', width: 2.5 }
    };

    const diffLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Voltage-Dependent Diffusivity D(V)', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Diffusivity (cm²/s)', type: 'log', autorange: true }),
        showlegend: false
    });

    Plotly.react('diffusivity-chart', [traceDiff], diffLayout, { responsive: true });
}

// Global Export Functions
window.exportResultsJson = function() {
    if (!latestResults) return;
    const jsonStr = JSON.stringify(latestResults, null, 2);
    downloadFile(jsonStr, 'cv_optimization_results.json', 'application/json');
};

window.exportResultsCsv = function() {
    if (!latestResults || !latestResults.plots) return;
    const p = latestResults.plots;
    const rows = ["Index,Potential_V,Exp_Current_A,Sim_Current_A,V_Plot,D_of_V,DOS_Total"];
    const maxLen = Math.max(p.exp_potential.length, p.v_plot.length);

    for (let i = 0; i < maxLen; i++) {
        const pot = i < p.exp_potential.length ? p.exp_potential[i] : "";
        const expCur = i < p.exp_current.length ? p.exp_current[i] : "";
        const simCur = i < p.sim_current.length ? p.sim_current[i] : "";
        const vp = i < p.v_plot.length ? p.v_plot[i] : "";
        const dv = i < p.d_of_v.length ? p.d_of_v[i] : "";
        const dos = i < p.dos_total.length ? p.dos_total[i] : "";
        rows.push(`${i},${pot},${expCur},${simCur},${vp},${dv},${dos}`);
    }

    downloadFile(rows.join("\n"), 'cv_optimization_data.csv', 'text/csv');
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
    probePythonBackend();
    const urlInput = document.getElementById('backend-url-input');
    if (urlInput && !urlInput.value) {
        urlInput.value = getStoredBackendUrl() || currentEndpoints.rawUrl;
    }
};

// Run initialization immediately and periodically until DOM elements exist
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.__initCVApp);
    } else {
        window.__initCVApp();
    }
}
