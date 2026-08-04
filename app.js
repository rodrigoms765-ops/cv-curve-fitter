// ==========================================================================
// CV Curve Fitting Pro — Client-Side Application Controller
// ==========================================================================

let solverWorker = null;
let currentFileContent = "";
let lastResultsData = null;
let expPotential = [];
let expCurrent = [];

// Advanced settings toggle
document.getElementById('advanced-toggle').addEventListener('click', () => {
    const content = document.getElementById('advanced-content');
    const toggle = document.getElementById('advanced-toggle');
    const isHidden = content.classList.toggle('hidden');
    toggle.setAttribute('aria-expanded', !isHidden);
});

// Plotly Base Layout Configuration
const basePlotLayout = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8FAFC',
    font: { color: '#0F172A', family: 'Inter, sans-serif' },
    xaxis: { 
        gridcolor: '#E2E8F0', 
        zerolinecolor: '#CBD5E1',
        linecolor: '#CBD5E1',
        linewidth: 1,
        mirror: true,
        ticks: 'outside',
        tickfont: { family: 'JetBrains Mono, monospace', size: 11 }
    },
    yaxis: { 
        gridcolor: '#E2E8F0', 
        zerolinecolor: '#CBD5E1',
        linecolor: '#CBD5E1',
        linewidth: 1,
        mirror: true,
        ticks: 'outside',
        exponentformat: 'e',
        tickfont: { family: 'JetBrains Mono, monospace', size: 11 }
    },
    margin: { t: 25, r: 25, l: 65, b: 50 },
    hovermode: 'closest'
};

// Initialize Web Worker
function initSolverWorker() {
    solverWorker = new Worker('solver_worker.js');
    
    solverWorker.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            handleWorkerMessage(msg);
        } catch (err) {
            console.error("Worker message parse error:", err, e.data);
        }
    };

    solverWorker.onerror = (err) => {
        console.error("Worker error:", err);
        updateEngineBadge('error', 'Worker Error');
        alert("Failed to run WebAssembly solver. Please ensure your browser supports modern WebAssembly.");
    };
}

function updateEngineBadge(state, label) {
    const badge = document.getElementById('engine-status-badge');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('engine-status-text');
    const submitBtn = document.getElementById('submit-btn');
    const submitBtnText = document.getElementById('submit-btn-text');

    if (state === 'ready') {
        badge.classList.add('ready');
        dot.classList.remove('pulsing');
        dot.classList.add('ready');
        text.innerText = label || 'WASM Engine Ready';
        submitBtn.disabled = false;
        submitBtnText.innerText = 'Run Multi-Stage Optimization';
    } else if (state === 'status') {
        text.innerText = label;
        submitBtnText.innerText = label;
    } else if (state === 'error') {
        text.innerText = label || 'Engine Failed';
        submitBtnText.innerText = 'Engine Error';
    }
}

function handleWorkerMessage(msg) {
    if (msg.type === 'status') {
        updateEngineBadge('status', msg.message);
    } else if (msg.type === 'ready') {
        updateEngineBadge('ready', 'WASM Engine Ready');
    } else if (msg.type === 'init') {
        expPotential = msg.exp_potential;
        expCurrent = msg.exp_current;
        
        document.getElementById('chart-placeholder').classList.add('hidden');
        
        Plotly.newPlot('live-chart', [
            {
                x: expPotential,
                y: expCurrent,
                type: 'scatter',
                mode: 'markers',
                name: 'Experimental CV',
                marker: { color: '#64748B', size: 4.5, opacity: 0.8 }
            },
            {
                x: expPotential,
                y: new Array(expPotential.length).fill(0),
                type: 'scatter',
                mode: 'lines',
                name: 'Fitted Model',
                line: { color: '#4F46E5', width: 2.5 }
            }
        ], {
            ...basePlotLayout,
            xaxis: { 
                ...basePlotLayout.xaxis, 
                title: 'Potential (V)',
                range: [Math.min(...expPotential), Math.max(...expPotential)]
            },
            yaxis: { ...basePlotLayout.yaxis, title: 'Current (A)' },
            legend: { x: 0.03, y: 0.96, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#E2E8F0', borderwidth: 1 }
        }, { responsive: true });
        
    } else if (msg.type === 'update') {
        document.getElementById('stage-badge').innerText = msg.stage;
        document.getElementById('status-details').innerText = `Iteration ${msg.iter}`;
        document.getElementById('loss-badge').innerText = `Loss: ${msg.loss.toFixed(4)}`;
        
        // Dynamic progress bar calculation
        let progress = 25;
        if (msg.stage.includes('1.5')) progress = 45;
        else if (msg.stage.includes('Stage 2')) progress = 70;
        else if (msg.stage.includes('Stage 3')) progress = 90;
        document.getElementById('progress-bar').style.width = `${progress}%`;
        
        Plotly.update('live-chart', {
            y: [expCurrent, msg.sim_current]
        });
        
    } else if (msg.type === 'done') {
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('stage-badge').innerText = 'Complete';
        document.getElementById('status-details').innerText = 'Optimization Converged Successfully';
        
        lastResultsData = msg.data;
        displayResults(msg.data);
        resetSubmitButton();
        
    } else if (msg.type === 'error') {
        alert('Optimization Error: ' + msg.message);
        document.getElementById('stage-badge').innerText = 'Error';
        document.getElementById('status-details').innerText = msg.message;
        resetSubmitButton();
    }
}

function resetSubmitButton() {
    const btn = document.getElementById('submit-btn');
    const spinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('submit-btn-text');
    
    btn.disabled = false;
    spinner.classList.add('hidden');
    btnText.innerText = 'Run Multi-Stage Optimization';
}

function displayResults(data) {
    document.getElementById('results-panel').classList.remove('hidden');
    
    const paramsDiv = document.getElementById('params-output');
    paramsDiv.innerHTML = `
        <div class="param-card">
            <div class="param-label">Diffusivity (D₀)</div>
            <div class="param-value">${data.parameters.diffusivity.toExponential(4)}<span class="param-unit">cm²/s</span></div>
        </div>
        <div class="param-card">
            <div class="param-label">Beta (Left / Anodic)</div>
            <div class="param-value">${data.parameters.beta_left.toFixed(4)}</div>
        </div>
        <div class="param-card">
            <div class="param-label">Beta (Right / Cathodic)</div>
            <div class="param-value">${data.parameters.beta_right.toFixed(4)}</div>
        </div>
        <div class="param-card">
            <div class="param-label">Baseline Offset</div>
            <div class="param-value">${data.parameters.baseline_offset.toExponential(4)}<span class="param-unit">A</span></div>
        </div>
        <div class="param-card">
            <div class="param-label">Center Potential (V₀)</div>
            <div class="param-value">${data.parameters.v_center.toFixed(4)}<span class="param-unit">V</span></div>
        </div>
    `;
    
    // Density of States Chart
    Plotly.newPlot('dos-chart', [{
        x: data.plots.v_plot,
        y: data.plots.dos_total,
        type: 'scatter',
        mode: 'lines',
        name: 'Total DOS',
        line: { color: '#0F172A', width: 2 }
    }], {
        ...basePlotLayout,
        xaxis: { 
            ...basePlotLayout.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)]
        },
        yaxis: { ...basePlotLayout.yaxis, title: 'DOS (a.u.)' },
        showlegend: false
    }, { responsive: true });
    
    // Diffusivity vs Potential Profile Chart
    Plotly.newPlot('diffusivity-chart', [{
        x: data.plots.v_plot,
        y: data.plots.d_of_v,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#06B6D4', width: 2.5 }
    }], {
        ...basePlotLayout,
        xaxis: { 
            ...basePlotLayout.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)]
        },
        yaxis: { ...basePlotLayout.yaxis, title: 'D(V) (cm²/s)', type: 'log' },
        showlegend: false
    }, { responsive: true });
}

// File Upload & Drag-and-Drop
const dropzone = document.getElementById('csv-dropzone');
const fileInput = document.getElementById('csv-file');
const fileNameDisplay = document.getElementById('file-name-display');

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelection(fileInput.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelection(e.target.files[0]);
    }
});

function handleFileSelection(file) {
    if (!file) return;
    fileNameDisplay.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileNameDisplay.classList.remove('hidden');
    
    const reader = new FileReader();
    reader.onload = (e) => {
        currentFileContent = e.target.result;
    };
    reader.readAsText(file);
}

// Load Sample CV Data Button
document.getElementById('btn-load-sample').addEventListener('click', () => {
    currentFileContent = SAMPLE_CV_CSV;
    fileNameDisplay.innerText = "sample_ferrocene_cv.csv (Generated Demo)";
    fileNameDisplay.classList.remove('hidden');
    
    // Set matching configuration for sample data
    document.getElementById('pot_col').value = 1;
    document.getElementById('cur_col').value = 2;
    document.getElementById('scan_rate').value = "0.010";
    document.getElementById('film_thickness').value = "0.0001";
    document.getElementById('v_min').value = "-0.8";
    document.getElementById('v_max').value = "0.8";
    document.getElementById('skip_factor').value = "5";
    document.getElementById('num_peaks').value = "25";
    
    alert("Sample CV dataset loaded! Click 'Run Multi-Stage Optimization' to execute.");
});

// Form Submission
document.getElementById('cv-form').addEventListener('submit', (e) => {
    e.preventDefault();
    
    if (!currentFileContent) {
        alert("Please upload a CSV file or click 'Load Sample CV Data' first.");
        return;
    }
    
    const form = e.target;
    const formData = new FormData(form);
    const config = Object.fromEntries(formData.entries());
    delete config.file;
    
    // UI state updates
    const btn = document.getElementById('submit-btn');
    const spinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('submit-btn-text');
    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.innerText = 'Optimizing in WASM...';
    
    document.getElementById('status-card').classList.remove('hidden');
    document.getElementById('results-panel').classList.add('hidden');
    document.getElementById('progress-bar').style.width = '10%';
    document.getElementById('stage-badge').innerText = 'Initializing...';
    document.getElementById('status-details').innerText = 'Starting solver...';
    document.getElementById('loss-badge').innerText = 'Loss: --';
    
    // Dispatch to Web Worker
    solverWorker.postMessage({
        action: 'solve',
        file_content: currentFileContent,
        config: config
    });
});

// Export JSON
document.getElementById('btn-export-json').addEventListener('click', () => {
    if (!lastResultsData) return;
    const blob = new Blob([JSON.stringify(lastResultsData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cv_curve_fit_results_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Start Worker on Page Load
initSolverWorker();
