// Cyclic Voltammetry Parameter Extraction & Physical Model Fitting
// High-Performance JAX Auto-Diff Engine Client

// Global State
let stagedFiles = [];
let detectedColumns = [];
let isOptimizing = false;

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
    
    stagedFiles = filesData.map(f => ({
        name: f.name,
        content: f.content,
        expPotential: [],
        expCurrent: [],
        results: null,
        status: 'pending'
    }));

    const fileNameDisplay = document.getElementById('file-name-display');
    if (fileNameDisplay) {
        if (stagedFiles.length === 1) {
            fileNameDisplay.innerText = stagedFiles[0].name;
        } else {
            fileNameDisplay.innerText = `${stagedFiles.length} files selected`;
        }
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

// Helpers for Gradio Element Discovery and Value Setting
function findGradioElement(selector) {
    let el = document.querySelector(selector);
    if (el) return el;
    const grApp = document.querySelector('gradio-app');
    if (grApp && grApp.shadowRoot) {
        return grApp.shadowRoot.querySelector(selector);
    }
    return null;
}

function setGradioInputValue(containerSelector, val) {
    const container = findGradioElement(containerSelector);
    if (!container) return false;
    const input = container.querySelector('textarea, input') || container;
    
    try {
        const proto = Object.getPrototypeOf(input);
        const desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) {
            desc.set.call(input, val);
        } else {
            input.value = val;
        }
    } catch (e) {
        input.value = val;
    }
    
    input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
    return true;
}

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

    startOptimizationUI();
    
    for (let i = 0; i < stagedFiles.length; i++) {
        const stageEl = document.getElementById('status-stage');
        const detailsEl = document.getElementById('status-details');
        if (stageEl) stageEl.innerText = `⚡ Optimizing File ${i + 1} of ${stagedFiles.length}: ${stagedFiles[i].name}`;
        if (detailsEl) detailsEl.innerText = 'Executing multi-stage non-linear L-BFGS-B optimization on JAX auto-diff engine...';
        
        try {
            const data = await executeZeroGPUSolver(stagedFiles[i].content, config, i + 1, stagedFiles.length);
            stagedFiles[i].results = data;
            stagedFiles[i].status = 'done';
            updateLivePlotProgress();
        } catch (err) {
            console.error(`Error processing ${stagedFiles[i].name}:`, err);
            stagedFiles[i].status = 'error';
            stagedFiles[i].errorMsg = err.message;
        }
    }
    
    const stageEl = document.getElementById('status-stage');
    const detailsEl = document.getElementById('status-details');
    if (stageEl) stageEl.innerText = '✓ Batch Physical Model Extraction Complete';
    const successCount = stagedFiles.filter(f => f.status === 'done').length;
    if (detailsEl) detailsEl.innerText = `Successfully processed ${successCount} of ${stagedFiles.length} files.`;
    
    stopOptimizationUI();
    displayExtractedResults();
    return false;
};

// Execution via Native ZeroGPU Pipeline & Direct HTTP API
async function executeZeroGPUSolver(fileContent, config, currentIdx, totalCount) {
    return new Promise(async (resolve, reject) => {
        // 1. Native Gradio ZeroGPU Queue Trigger
        const fileSet = setGradioInputValue('#gr_input_file', fileContent);
        const configSet = setGradioInputValue('#gr_input_config', JSON.stringify(config));
        const grBtn = findGradioElement('#gr_trigger_btn button') || findGradioElement('#gr_trigger_btn');

        if (fileSet && configSet && grBtn) {
            const startTime = Date.now();
            setGradioInputValue('#gr_output_json', '');

            const pollInterval = setInterval(() => {
                const outContainer = findGradioElement('#gr_output_json');
                const outEl = outContainer ? (outContainer.querySelector('textarea, input') || outContainer) : null;
                const textVal = (outEl ? outEl.value : "") || (outContainer ? outContainer.innerText : "");

                if (textVal && textVal.trim().startsWith('{') && textVal.trim().endsWith('}')) {
                    clearInterval(pollInterval);
                    try {
                        const data = JSON.parse(textVal.trim());
                        if (data.type === 'error') reject(new Error(data.message || 'Solver error'));
                        else resolve(data);
                    } catch (e) {
                        reject(new Error(`Failed to parse output JSON: ${e.message}`));
                    }
                    return;
                }

                const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
                const stageEl = document.getElementById('status-stage');
                if (stageEl) stageEl.innerText = `⚡ Non-Linear Parameter Extraction File ${currentIdx}/${totalCount} (${elapsedSec}s)...`;

                if (Date.now() - startTime > 180000) {
                    clearInterval(pollInterval);
                    reject(new Error("Optimization calculation timed out (3 min)."));
                }
            }, 500);

            grBtn.click();
            return;
        }

        // 2. Direct HTTP POST fallback
        const endpoints = [
            window.location.origin + "/api/solve",
            window.location.origin + "/solve",
            "http://127.0.0.1:8000/api/solve"
        ];

        for (const endpoint of endpoints) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 180000);
                const res = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        file_content: fileContent,
                        config: config
                    }),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                if (res.ok) {
                    const data = await res.json();
                    if (data.type === 'error') reject(new Error(data.message || 'Solver error'));
                    else resolve(data);
                    return;
                }
            } catch (err) {
                console.warn(`HTTP solve attempt on ${endpoint} failed:`, err);
            }
        }
        reject(new Error("Could not communicate with solver engine."));
    });
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
            
            if (f.results && f.results.plots && f.results.plots.sim_current) {
                traces.push({
                    x: f.results.plots.exp_potential || f.expPotential,
                    y: f.results.plots.sim_current,
                    mode: 'lines',
                    type: 'scatter',
                    name: `Sim: ${f.name}`,
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
    if (paramsDiv) {
        paramsDiv.innerHTML = '';
        paramsDiv.className = ''; // Remove global grid class to prevent disorganized wrapping
        
        stagedFiles.filter(f => f.status === 'done' && f.results).forEach((f, i) => {
            const params = f.results.params || {};
            
            const fileContainer = document.createElement('div');
            fileContainer.style.marginBottom = '2rem';
            
            const header = document.createElement('h4');
            header.style.color = chartColors[i % chartColors.length];
            header.style.marginBottom = '1rem';
            header.innerText = `Parameters: ${f.name}`;
            fileContainer.appendChild(header);

            const grid = document.createElement('div');
            grid.className = 'stats-grid'; // Apply grid locally per file

            const cards = [
                { label: 'Diffusivity Constant (D₀)', value: `${(params.D0 || 0).toExponential(3)} cm²/s` },
                { label: 'Thermodynamic Potential (V_c)', value: `${(params.Vc || 0).toFixed(4)} V` },
                { label: 'Baseline DC Offset (I_offset)', value: `${(params.I_offset || 0).toExponential(3)} A` }
            ];

            cards.forEach(c => {
                const card = document.createElement('div');
                card.className = 'stat-card';
                card.innerHTML = `
                    <span class="stat-label">${c.label}</span>
                    <span class="stat-value">${c.value}</span>
                `;
                grid.appendChild(card);
            });
            
            fileContainer.appendChild(grid);
            paramsDiv.appendChild(fileContainer);
        });
    }

    renderSecondaryPlots();
}

function renderSecondaryPlots() {
    if (!window.Plotly) return;

    const dosTraces = [];
    const diffTraces = [];
    
    stagedFiles.filter(f => f.status === 'done' && f.results && f.results.plots).forEach((f, i) => {
        const plots = f.results.plots;
        
        dosTraces.push({
            x: plots.v_plot,
            y: plots.dos_total,
            mode: 'lines',
            type: 'scatter',
            name: `Total DOS: ${f.name}`,
            line: { color: chartColors[i % chartColors.length], width: 2.8 }
        });
        
        diffTraces.push({
            x: plots.v_plot,
            y: plots.d_of_v,
            mode: 'lines',
            type: 'scatter',
            name: `D(V): ${f.name}`,
            line: { color: chartColors[i % chartColors.length], width: 2.8 }
        });
    });

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
    const batchResults = stagedFiles.filter(f => f.status === 'done').map(f => ({
        name: f.name,
        results: f.results
    }));
    if (batchResults.length === 0) return;
    const jsonStr = JSON.stringify(batchResults, null, 2);
    downloadFile(jsonStr, 'cv_extracted_batch_parameters.json', 'application/json');
};

window.exportResultsCsv = function() {
    const doneFiles = stagedFiles.filter(f => f.status === 'done' && f.results && f.results.plots);
    if (doneFiles.length === 0) return;

    let rows = [];
    
    // Create Header Row 1 (File Names)
    let header1 = [];
    doneFiles.forEach(f => {
        header1.push(`File: ${f.name}`, "", "", "", "", "");
    });
    rows.push(header1.join(","));
    
    // Create Header Row 2 (Column Names)
    let header2 = [];
    doneFiles.forEach(() => {
        header2.push("Exp_V", "Exp_I", "Sim_I", "V_Plot", "DOS", "D_V");
    });
    rows.push(header2.join(","));
    
    // Find absolute max length across all files for both experimental data and plots
    let globalMaxLen = 0;
    doneFiles.forEach(f => {
        const p = f.results.plots;
        globalMaxLen = Math.max(globalMaxLen, p.exp_potential.length, p.v_plot.length);
    });

    for (let i = 0; i < globalMaxLen; i++) {
        let row = [];
        doneFiles.forEach(f => {
            const p = f.results.plots;
            const expV = i < p.exp_potential.length ? p.exp_potential[i] : "";
            const expI = i < p.exp_current.length ? p.exp_current[i] : "";
            const simI = i < p.sim_current.length ? p.sim_current[i] : "";
            const vp = i < p.v_plot.length ? p.v_plot[i] : "";
            const dv = i < p.d_of_v.length ? p.d_of_v[i] : "";
            const dos = i < p.dos_total.length ? p.dos_total[i] : "";
            row.push(expV, expI, simI, vp, dos, dv);
        });
        rows.push(row.join(","));
    }

    downloadFile(rows.join("\n"), 'cv_extracted_batch_curves.csv', 'text/csv');
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
    if (window.Plotly && expPotential.length > 0) {
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
