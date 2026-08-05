// CV Curve Fitting Pro - Dual-Mode Client Controller & Visualizer
// Seamless Hybrid: Fast Local JAX Backend (ws://127.0.0.1:8000) with In-Browser Worker Fallback
// Features: Auto CSV Column Detection, Cycle Presets, Live Pre-Fit Plotting, Multi-Engine Switching

// Global State
let solverWorker = null;
let activeSocket = null;
let expPotential = [];
let expCurrent = [];
let latestResults = null;
let stagedFileContent = null;
let stagedFileName = "No file chosen";
let detectedColumns = [];
let isLocalBackendAvailable = false;

const BACKEND_URL_HTTP = "http://127.0.0.1:8000/health";
const BACKEND_URL_WS = "ws://127.0.0.1:8000/ws/solve";

// Plotly Baseline Layout Configuration
const layoutConfig = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8FAFC',
    font: { color: '#0F172A', family: 'Roboto, sans-serif' },
    xaxis: { 
        autorange: true,
        gridcolor: '#E2E8F0', 
        zerolinecolor: '#94A3B8',
        linecolor: '#CBD5E1',
        linewidth: 1,
        mirror: true,
        ticks: 'outside',
        title: 'Potential (V)'
    },
    yaxis: { 
        autorange: true,
        gridcolor: '#E2E8F0', 
        zerolinecolor: '#94A3B8',
        linecolor: '#CBD5E1',
        linewidth: 1,
        mirror: true,
        ticks: 'outside',
        exponentformat: 'e',
        title: 'Current (A)'
    },
    margin: { t: 30, r: 30, l: 70, b: 60 }
};

// Initialize In-Browser Web Worker Fallback
function initWorker() {
    if (solverWorker) solverWorker.terminate();
    solverWorker = new Worker('solver_worker.js');
    
    solverWorker.onmessage = (e) => {
        try {
            const msg = typeof e.data === 'string' ? JSON.parse(e.data) : e.data;
            handleSolverMessage(msg);
        } catch (err) {
            console.error("Worker message parse error:", err);
        }
    };
    
    solverWorker.onerror = (err) => {
        console.error("Solver worker error:", err);
        document.getElementById('status-stage').innerText = 'Worker Error';
        document.getElementById('status-details').innerText = 'In-browser solver encountered an error.';
        resetUI();
    };
}

// Probe Local JAX Backend
async function probeLocalBackend() {
    const dot = document.getElementById('engine-dot');
    const label = document.getElementById('engine-label');
    const msg = document.getElementById('engine-status-msg');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1200);
        const res = await fetch(BACKEND_URL_HTTP, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (res.ok) {
            isLocalBackendAvailable = true;
            if (dot) dot.className = 'status-dot online';
            if (label) label.innerText = '⚡ Local JAX Active';
            if (msg) {
                msg.innerText = '⚡ Local JAX Backend Connected (Port 8000)';
                msg.className = 'engine-status-msg online';
            }
            return true;
        }
    } catch (e) {
        // Backend offline
    }

    isLocalBackendAvailable = false;
    if (dot) dot.className = 'status-dot offline';
    if (label) label.innerText = '🌐 In-Browser Engine';
    if (msg) {
        msg.innerText = 'Local server offline. Using in-browser engine.';
        msg.className = 'engine-status-msg';
    }
    return false;
}

// Check backend on load and periodically
probeLocalBackend();
setInterval(probeLocalBackend, 10000);

// Advanced Settings Accordion Toggle
const advToggle = document.getElementById('advanced-toggle');
const advContent = document.getElementById('advanced-content');
const toggleIcon = document.getElementById('toggle-icon');

if (advToggle && advContent) {
    advToggle.addEventListener('click', () => {
        const isHidden = advContent.classList.toggle('hidden');
        advToggle.setAttribute('aria-expanded', !isHidden);
        if (toggleIcon) {
            toggleIcon.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(180deg)';
        }
    });
}

// Documentation & Backend Guide Modals
const docModal = document.getElementById('doc-modal');
const navDocBtn = document.getElementById('nav-doc-btn');
const docCloseBtn = document.getElementById('doc-close-btn');
const docOkBtn = document.getElementById('doc-ok-btn');

const backendModal = document.getElementById('backend-modal');
const engineBadge = document.getElementById('engine-status-badge');
const backendHelpLink = document.getElementById('backend-help-link');
const backendCloseBtn = document.getElementById('backend-close-btn');
const backendOkBtn = document.getElementById('backend-ok-btn');

function openModal(modal) {
    if (modal) modal.classList.remove('hidden');
}

function closeModal(modal) {
    if (modal) modal.classList.add('hidden');
}

if (navDocBtn) navDocBtn.addEventListener('click', (e) => { e.preventDefault(); openModal(docModal); });
if (docCloseBtn) docCloseBtn.addEventListener('click', () => closeModal(docModal));
if (docOkBtn) docOkBtn.addEventListener('click', () => closeModal(docModal));

if (engineBadge) engineBadge.addEventListener('click', () => openModal(backendModal));
if (backendHelpLink) backendHelpLink.addEventListener('click', () => openModal(backendModal));
if (backendCloseBtn) backendCloseBtn.addEventListener('click', () => closeModal(backendModal));
if (backendOkBtn) backendOkBtn.addEventListener('click', () => closeModal(backendModal));

[docModal, backendModal].forEach(m => {
    if (m) {
        m.addEventListener('click', (e) => {
            if (e.target === m) closeModal(m);
        });
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal(docModal);
        closeModal(backendModal);
    }
});

// CSV Delimiter Detector
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

    // Check if line 0 is a string header
    let hasHeader = false;
    if (firstLineFields.some(f => isNaN(parseFloat(f)) && f.length > 0)) {
        hasHeader = true;
    }

    const colCount = hasHeader ? firstLineFields.length : (secondLineFields.length || firstLineFields.length);
    detectedColumns = [];

    // Count non-empty numeric data points per column
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

    // Track cycle numbers for standard 4-column-per-cycle pattern
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

    // Populate Select Elements
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

        // Smart Default Selection:
        // If >= 10 columns (Standard 4-cycle CV file), default to Col 8 & Col 9 (Cycle 3 Raw)
        // If fewer columns, default to Col 0 & Col 1
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
        updateCycleButtonsActiveState(defaultPot, defaultCur);

        if (metaBar && metaText) {
            metaBar.classList.add('has-data');
            metaText.innerText = `✓ Detected ${colCount} columns across ${lines.length - (hasHeader ? 1 : 0)} data rows.`;
        }

        // Immediately update preview plot with selected columns
        updateLivePreviewFromColumns();
    }
}

// Update Active Cycle Preset Button
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

// Preset Buttons Event Delegation
const presetContainer = document.getElementById('cycle-preset-container') || document.getElementById('cycle-preset-buttons');
if (presetContainer) {
    presetContainer.addEventListener('click', (e) => {
        if (e.target && e.target.classList.contains('preset-pill-btn')) {
            const pot = parseInt(e.target.getAttribute('data-pot'), 10);
            const cur = parseInt(e.target.getAttribute('data-cur'), 10);
            const skip = e.target.getAttribute('data-skip');

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
                updateLivePreviewFromColumns();
            }
        }
    });
}

// Extract and plot experimental data for chosen columns immediately
function updateLivePreviewFromColumns() {
    if (!stagedFileContent) return;

    const potCol = parseInt(document.getElementById('pot_col').value, 10);
    const curCol = parseInt(document.getElementById('cur_col').value, 10);
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
        
        // Update Stats Card
        let minV = Math.min(...previewPot);
        let maxV = Math.max(...previewPot);
        let minI = Math.min(...previewCur);
        let maxI = Math.max(...previewCur);

        const vSpan = document.getElementById('stat-v-range');
        const iSpan = document.getElementById('stat-i-range');
        const countSpan = document.getElementById('stat-points-count');

        if (vSpan) vSpan.innerText = `${minV.toFixed(2)}V to ${maxV.toFixed(2)}V`;
        if (iSpan) iSpan.innerText = `${minI.toExponential(2)}A to ${maxI.toExponential(2)}A`;
        if (countSpan) countSpan.innerText = previewPot.length.toLocaleString();

        initLiveChart(expPotential, expCurrent);
        
        document.getElementById('status-stage').innerText = 'Data Loaded & Ready';
        document.getElementById('status-details').innerText = `Previewing Col ${potCol} (V) vs Col ${curCol} (I) [${previewPot.length} points]. Click Execute Optimization to fit.`;
    }
}

// Column change event listeners
const potSelectEl = document.getElementById('pot_col');
const curSelectEl = document.getElementById('cur_col');
if (potSelectEl) potSelectEl.addEventListener('change', updateLivePreviewFromColumns);
if (curSelectEl) curSelectEl.addEventListener('change', updateLivePreviewFromColumns);

// File Upload & Drag-Drop Handling
const fileInput = document.getElementById('csv-file');
const fileNameDisplay = document.getElementById('file-name-display');
const dropZone = document.getElementById('drop-zone');

function setLoadedFile(content, name) {
    stagedFileContent = content;
    stagedFileName = name;
    if (fileNameDisplay) {
        fileNameDisplay.innerText = name;
        fileNameDisplay.classList.add('has-file');
    }
    analyzeCSVAndPopulateColumns(content);
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                setLoadedFile(ev.target.result, file.name);
            };
            reader.readAsText(file);
        }
    });
}

if (dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                setLoadedFile(ev.target.result, file.name);
            };
            reader.readAsText(file);
        }
    });
}

// Form Submission & Dual-Mode Execution Dispatcher
const cvForm = document.getElementById('cv-form');
if (cvForm) {
    cvForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!stagedFileContent) {
            alert('Please select a CSV file first.');
            return;
        }

        const formData = new FormData(cvForm);
        const config = {};
        formData.forEach((value, key) => {
            config[key] = value;
        });

        // Make sure selected columns are integers
        config.pot_col = parseInt(document.getElementById('pot_col').value, 10);
        config.cur_col = parseInt(document.getElementById('cur_col').value, 10);

        // Determine execution engine
        const engineSelection = document.getElementById('engine-select').value;
        const useLocalBackend = (engineSelection === 'local') || (engineSelection === 'auto' && isLocalBackendAvailable);

        startOptimizationUI();

        if (useLocalBackend) {
            executeLocalJAXSolver(stagedFileContent, config);
        } else {
            executeInBrowserSolver(stagedFileContent, config);
        }
    });
}

// Execution via Local JAX WebSocket
function executeLocalJAXSolver(fileContent, config) {
    document.getElementById('status-stage').innerText = 'Connecting to Local JAX Backend...';
    document.getElementById('status-details').innerText = 'Initializing WebSocket on ws://127.0.0.1:8000...';

    if (activeSocket) {
        activeSocket.close();
    }

    try {
        activeSocket = new WebSocket(BACKEND_URL_WS);
    } catch (err) {
        console.warn("WebSocket initialization failed, falling back to In-Browser:", err);
        fallbackToBrowserSolver(fileContent, config, "Could not open WebSocket. Falling back to in-browser engine.");
        return;
    }

    activeSocket.onopen = () => {
        document.getElementById('status-stage').innerText = '⚡ JAX Engine Running';
        document.getElementById('status-details').innerText = 'Hardware-accelerated XLA optimization in progress...';
        
        activeSocket.send(JSON.stringify({
            action: 'solve',
            config: config,
            file_content: fileContent
        }));
    };

    activeSocket.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleSolverMessage(msg);
        } catch (err) {
            console.error("WebSocket message parse error:", err);
        }
    };

    activeSocket.onerror = (err) => {
        console.warn("WebSocket error:", err);
        fallbackToBrowserSolver(fileContent, config, "Local JAX server connection failed. Switched to In-Browser solver.");
    };

    activeSocket.onclose = () => {
        activeSocket = null;
    };
}

// Fallback Helper
function fallbackToBrowserSolver(fileContent, config, notice) {
    document.getElementById('status-details').innerText = notice;
    executeInBrowserSolver(fileContent, config);
}

// Execution via In-Browser Web Worker
function executeInBrowserSolver(fileContent, config) {
    if (!solverWorker) initWorker();
    
    document.getElementById('status-stage').innerText = '🌐 In-Browser Solver Running';
    document.getElementById('status-details').innerText = 'Executing physics diffusion simulation in Web Worker...';

    solverWorker.postMessage({
        action: 'solve',
        file_content: fileContent,
        config: config
    });
}

// Universal Message Handler for Solver Messages (both Worker & WebSocket)
function handleSolverMessage(msg) {
    if (msg.type === 'init') {
        expPotential = msg.exp_potential;
        expCurrent = msg.exp_current;
        initLiveChart(expPotential, expCurrent);
    } else if (msg.type === 'status') {
        document.getElementById('status-details').innerText = msg.message;
    } else if (msg.type === 'update') {
        updateLiveChart(msg);
    } else if (msg.type === 'done') {
        finishOptimization(msg.data);
    } else if (msg.type === 'error') {
        document.getElementById('status-stage').innerText = 'Optimization Failed';
        document.getElementById('status-details').innerText = msg.message || 'An error occurred during calculation.';
        resetUI();
    }
}

// UI State Management
function startOptimizationUI() {
    const submitBtn = document.getElementById('submit-btn');
    const spinner = document.getElementById('status-spinner');
    const resultsPanel = document.getElementById('results-panel');

    submitBtn.disabled = true;
    submitBtn.innerText = 'Optimizing...';
    if (spinner) spinner.classList.remove('hidden');
    if (resultsPanel) resultsPanel.classList.add('hidden');
}

function resetUI() {
    const submitBtn = document.getElementById('submit-btn');
    const spinner = document.getElementById('status-spinner');

    submitBtn.disabled = false;
    submitBtn.innerText = 'Execute Optimization';
    if (spinner) spinner.classList.add('hidden');
}

// Plotly Live Charting with Connected Hysteresis Loop
function initLiveChart(potential, current) {
    const traceExp = {
        x: potential,
        y: current,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Experimental Data',
        line: { color: '#0284c7', width: 1.8 },
        marker: { color: '#0284c7', size: 3, opacity: 0.7 }
    };

    const traceSim = {
        x: potential,
        y: [],
        mode: 'lines',
        type: 'scatter',
        name: 'Physics Model Fit',
        line: { color: '#16a34a', width: 2.5 }
    };

    const layout = Object.assign({}, layoutConfig, {
        title: { text: 'Cyclic Voltammogram Live View', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current (A)', autorange: true }),
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.85)' }
    });

    Plotly.react('live-chart', [traceExp, traceSim], layout, { responsive: true, displayModeBar: false });
}

function updateLiveChart(data) {
    document.getElementById('status-stage').innerText = data.stage;
    document.getElementById('status-details').innerText = `Iteration ${data.iter} | Current Weighted Loss: ${data.loss.toExponential(4)}`;

    Plotly.update('live-chart', {
        y: [expCurrent, data.sim_current]
    }, {}, [0, 1]);
}

// Completion & Visualizations
function finishOptimization(data) {
    latestResults = data;
    resetUI();

    document.getElementById('status-stage').innerText = 'Optimization Complete';
    document.getElementById('status-details').innerText = 'Model converged successfully. Extracted physical parameters displayed below.';

    // Populate Parameter Table
    const params = data.parameters;
    document.getElementById('param-diffusivity').innerText = params.diffusivity.toExponential(4);
    document.getElementById('param-beta-left').innerText = params.beta_left.toFixed(4);
    document.getElementById('param-beta-right').innerText = params.beta_right.toFixed(4);
    document.getElementById('param-v-center').innerText = params.v_center.toFixed(4);
    document.getElementById('param-offset').innerText = params.baseline_offset.toExponential(4);

    const resultsPanel = document.getElementById('results-panel');
    if (resultsPanel) resultsPanel.classList.remove('hidden');

    renderDiagnosticPlots(data.plots);
}

function renderDiagnosticPlots(plots) {
    // 1. Final CV Fit Plot
    const traceExp = {
        x: plots.exp_potential,
        y: plots.exp_current,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Experimental Data',
        line: { color: '#64748B', width: 1.5 },
        marker: { color: '#64748B', size: 3, opacity: 0.7 }
    };

    const traceSim = {
        x: plots.exp_potential,
        y: plots.sim_current,
        mode: 'lines',
        type: 'scatter',
        name: 'Physics Fit',
        line: { color: '#0F172A', width: 2.5 }
    };

    const cvLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Final Cyclic Voltammogram Fit', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Current (A)', autorange: true }),
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.85)' }
    });

    Plotly.react('cv-fit-plot', [traceExp, traceSim], cvLayout, { responsive: true });

    // 2. Diffusivity D(V) Plot
    const traceDiff = {
        x: plots.v_plot,
        y: plots.d_of_v,
        mode: 'lines',
        type: 'scatter',
        name: 'D(V)',
        line: { color: '#2563EB', width: 2.5 }
    };

    const diffLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Voltage-Dependent Diffusivity D(V)', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'Diffusivity (cm²/s)', type: 'log', autorange: true }),
        showlegend: false
    });

    Plotly.react('diffusivity-plot', [traceDiff], diffLayout, { responsive: true });

    // 3. Density of States DOS(V) Plot with Sub-bands
    const dosTraces = [];

    if (plots.dos_matrix && plots.dos_matrix.length > 0) {
        plots.dos_matrix.forEach((band, idx) => {
            dosTraces.push({
                x: plots.v_plot,
                y: band,
                mode: 'lines',
                type: 'scatter',
                name: `Mode ${idx + 1}`,
                line: { width: 1, dash: 'dot', color: '#94A3B8' },
                hoverinfo: 'skip'
            });
        });
    }

    dosTraces.push({
        x: plots.v_plot,
        y: plots.dos_total,
        mode: 'lines',
        type: 'scatter',
        name: 'Total DOS',
        line: { color: '#059669', width: 3 }
    });

    const dosLayout = Object.assign({}, layoutConfig, {
        title: { text: 'Extracted Density of States DOS(V)', font: { size: 14 } },
        xaxis: Object.assign({}, layoutConfig.xaxis, { title: 'Potential (V)', autorange: true }),
        yaxis: Object.assign({}, layoutConfig.yaxis, { title: 'DOS (a.u.)', autorange: true }),
        showlegend: false
    });

    Plotly.react('dos-plot', dosTraces, dosLayout, { responsive: true });
}

// Data Export Utilities
const exportJsonBtn = document.getElementById('export-json-btn');
const exportCsvBtn = document.getElementById('export-csv-btn');

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

if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => {
        if (!latestResults) return;
        const jsonStr = JSON.stringify(latestResults, null, 2);
        downloadFile(jsonStr, 'cv_optimization_results.json', 'application/json');
    });
}

if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
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
    });
}

// Initial Worker Spawn
initWorker();
