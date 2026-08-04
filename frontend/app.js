// CV Curve Fitting Pro - Client Controller & Visualizer

// Global State
let solverWorker = null;
let expPotential = [];
let expCurrent = [];
let latestResults = null;
let stagedFileContent = null;
let stagedFileName = "No file chosen";

// Plotly Baseline Layout Configuration
const layoutConfig = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8FAFC',
    font: { color: '#0F172A', family: 'Roboto, sans-serif' },
    xaxis: { 
        gridcolor: '#E2E8F0', 
        zerolinecolor: '#94A3B8',
        linecolor: '#CBD5E1',
        linewidth: 1,
        mirror: true,
        ticks: 'outside'
    },
    yaxis: { 
        gridcolor: '#E2E8F0', 
        zerolinecolor: '#94A3B8',
        linecolor: '#CBD5E1',
        linewidth: 1,
        mirror: true,
        ticks: 'outside',
        exponentformat: 'e'
    },
    margin: { t: 30, r: 30, l: 70, b: 60 }
};

// Initialize Web Worker
function initWorker() {
    solverWorker = new Worker('solver_worker.js');
    
    solverWorker.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            handleWorkerMessage(msg);
        } catch (err) {
            console.error("Worker message parse error:", err);
        }
    };
    
    solverWorker.onerror = (err) => {
        console.error("Solver worker error:", err);
        document.getElementById('status-stage').innerText = 'Worker Error';
        document.getElementById('status-details').innerText = 'WebAssembly solver worker encountered an error.';
        resetUI();
    };
}

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

// Documentation Modal Controls
const docModal = document.getElementById('doc-modal');
const navDocBtn = document.getElementById('nav-doc-btn');
const docCloseBtn = document.getElementById('doc-close-btn');
const docOkBtn = document.getElementById('doc-ok-btn');

function openDocModal(e) {
    if (e) e.preventDefault();
    if (docModal) docModal.classList.remove('hidden');
}

function closeDocModal() {
    if (docModal) docModal.classList.add('hidden');
}

if (navDocBtn) navDocBtn.addEventListener('click', openDocModal);
if (docCloseBtn) docCloseBtn.addEventListener('click', closeDocModal);
if (docOkBtn) docOkBtn.addEventListener('click', closeDocModal);

if (docModal) {
    docModal.addEventListener('click', (e) => {
        if (e.target === docModal) closeDocModal();
    });
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && docModal && !docModal.classList.contains('hidden')) {
        closeDocModal();
    }
});

// File Upload & Drag-and-Drop Handling
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('csv-file');
const fileNameDisplay = document.getElementById('file-name-display');

function setLoadedFile(content, name) {
    stagedFileContent = content;
    stagedFileName = name;
    if (fileNameDisplay) {
        fileNameDisplay.innerText = name;
        fileNameDisplay.style.color = '#0F172A';
    }
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
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
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

}

// Form Submission & Solver Execution
const cvForm = document.getElementById('cv-form');
if (cvForm) {
    cvForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        if (!stagedFileContent) {
            alert("Please upload a CSV file or click 'Load Demo Data' first.");
            return;
        }

        const formData = new FormData(cvForm);
        const config = Object.fromEntries(formData.entries());
        delete config.file;

        document.getElementById('submit-btn').disabled = true;
        document.getElementById('status-panel').classList.remove('hidden');
        document.getElementById('results-panel').classList.add('hidden');
        
        const spinner = document.getElementById('status-spinner');
        if (spinner) spinner.classList.remove('hidden');
        
        document.getElementById('status-stage').innerText = 'Initializing...';
        document.getElementById('status-details').innerText = 'Setting up optimization problem...';
        
        if (!solverWorker) {
            initWorker();
        }
        
        solverWorker.postMessage({
            action: 'solve',
            file_content: stagedFileContent,
            config: config
        });
    });
}

function resetUI() {
    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn) submitBtn.disabled = false;
    const spinner = document.getElementById('status-spinner');
    if (spinner) spinner.classList.add('hidden');
}

// Handle Worker Messages
function handleWorkerMessage(msg) {
    if (msg.type === 'status') {
        document.getElementById('status-details').innerText = msg.message;
    } else if (msg.type === 'init') {
        expPotential = msg.exp_potential;
        expCurrent = msg.exp_current;
        
        Plotly.newPlot('live-chart', [
            {
                x: expPotential,
                y: expCurrent,
                type: 'scatter',
                mode: 'markers',
                name: 'Experimental Data',
                marker: { color: '#64748B', size: 4 }
            },
            {
                x: expPotential,
                y: new Array(expPotential.length).fill(0),
                type: 'scatter',
                mode: 'lines',
                name: 'Simulated Fit',
                line: { color: '#334155', width: 2.5 }
            }
        ], {
            ...layoutConfig,
            xaxis: { 
                ...layoutConfig.xaxis, 
                title: 'Potential (V)',
                range: [Math.min(...expPotential), Math.max(...expPotential)],
                exponentformat: 'none'
            },
            yaxis: { ...layoutConfig.yaxis, title: 'Current (A)' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#E2E8F0', borderwidth: 1 }
        }, {responsive: true});
        
    } else if (msg.type === 'update') {
        document.getElementById('status-stage').innerText = msg.stage;
        document.getElementById('status-details').innerText = `Iteration: ${msg.iter} | Loss: ${msg.loss.toFixed(4)}`;
        
        Plotly.update('live-chart', {
            y: [expCurrent, msg.sim_current]
        });
        
    } else if (msg.type === 'done') {
        document.getElementById('status-stage').innerText = 'Optimization Complete';
        document.getElementById('status-details').innerText = 'All stages converged successfully. Physical parameters extracted.';
        latestResults = msg.data;
        displayResults(msg.data);
        resetUI();
        
    } else if (msg.type === 'error') {
        alert('Error: ' + msg.message);
        document.getElementById('status-stage').innerText = "Solver Error";
        document.getElementById('status-details').innerText = msg.message;
        resetUI();
    }
}

// Display Extracted Results
function displayResults(data) {
    document.getElementById('results-panel').classList.remove('hidden');
    
    const paramsDiv = document.getElementById('params-output');
    if (paramsDiv && data.parameters) {
        paramsDiv.innerHTML = `
            <div class="stat-box">
                <div class="label">Diffusivity D₀</div>
                <div class="value">${data.parameters.diffusivity.toExponential(4)} <span class="unit">cm²/s</span></div>
            </div>
            <div class="stat-box">
                <div class="label">Beta (Left)</div>
                <div class="value">${data.parameters.beta_left.toExponential(4)}</div>
            </div>
            <div class="stat-box">
                <div class="label">Beta (Right)</div>
                <div class="value">${data.parameters.beta_right.toExponential(4)}</div>
            </div>
            <div class="stat-box">
                <div class="label">Baseline Offset</div>
                <div class="value">${data.parameters.baseline_offset.toExponential(4)} <span class="unit">A</span></div>
            </div>
            <div class="stat-box">
                <div class="label">V Center</div>
                <div class="value">${data.parameters.v_center.toFixed(4)} <span class="unit">V</span></div>
            </div>
        `;
    }
    
    // Density of States (DOS) Plot with Individual Sub-Bands
    const dosTraces = [];
    
    // Add individual Gaussian sub-peaks if available
    if (data.plots.dos_matrix && Array.isArray(data.plots.dos_matrix)) {
        data.plots.dos_matrix.forEach((modeY, idx) => {
            dosTraces.push({
                x: data.plots.v_plot,
                y: modeY,
                type: 'scatter',
                mode: 'lines',
                name: `Mode ${idx + 1}`,
                line: { color: '#94A3B8', width: 1, dash: 'dot' },
                hoverinfo: 'skip',
                showlegend: false
            });
        });
    }
    
    // Add Total DOS curve
    dosTraces.push({
        x: data.plots.v_plot,
        y: data.plots.dos_total,
        type: 'scatter',
        mode: 'lines',
        name: 'Total DOS',
        line: { color: '#0F172A', width: 2.5 },
        showlegend: true
    });
    
    Plotly.newPlot('dos-chart', dosTraces, {
        ...layoutConfig,
        xaxis: { 
            ...layoutConfig.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)]
        },
        yaxis: { ...layoutConfig.yaxis, title: 'Density of States (a.u.)' },
        legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#E2E8F0', borderwidth: 1 }
    }, {responsive: true});
    
    // Extracted Diffusivity Plot
    Plotly.newPlot('diffusivity-chart', [{
        x: data.plots.v_plot,
        y: data.plots.d_of_v,
        type: 'scatter',
        mode: 'lines',
        name: 'D(V)',
        line: { color: '#2563EB', width: 2.5 }
    }], {
        ...layoutConfig,
        xaxis: { 
            ...layoutConfig.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)],
            exponentformat: 'none'
        },
        yaxis: { ...layoutConfig.yaxis, title: 'Diffusivity D (cm²/s)', type: 'log' },
        showlegend: false
    }, {responsive: true});
}

// Export Results (JSON & CSV)
const exportJsonBtn = document.getElementById('export-json-btn');
if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => {
        if (!latestResults) return;
        const blob = new Blob([JSON.stringify(latestResults, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cv_fit_parameters_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });
}

const exportCsvBtn = document.getElementById('export-csv-btn');
if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
        if (!latestResults || !latestResults.plots) return;
        const plots = latestResults.plots;
        const rows = ["Potential_V,Exp_Current_A,Sim_Current_A"];
        const len = plots.exp_potential.length;
        for (let i = 0; i < len; i++) {
            const v = plots.exp_potential[i];
            const expI = plots.exp_current[i];
            const simI = plots.sim_current[i] !== undefined ? plots.sim_current[i] : "";
            rows.push(`${v},${expI},${simI}`);
        }
        const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cv_fitted_curves_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    });
}

// Initialize worker on page load
initWorker();
