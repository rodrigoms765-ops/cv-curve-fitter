// ==========================================================================
// Cyclic Voltammetry Model Fitting — Application Controller
// ==========================================================================

let solverWorker = null;
let currentFileContent = "";
let lastResultsData = null;
let expPotential = [];
let expCurrent = [];

// Publication-standard plot layout
const scientificPlotLayout = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#FFFFFF',
    font: { color: '#111827', family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif', size: 12 },
    xaxis: { 
        gridcolor: '#F3F4F6', 
        zerolinecolor: '#E5E7EB',
        linecolor: '#111827',
        linewidth: 1,
        mirror: true,
        ticks: 'inside',
        showline: true
    },
    yaxis: { 
        gridcolor: '#F3F4F6', 
        zerolinecolor: '#E5E7EB',
        linecolor: '#111827',
        linewidth: 1,
        mirror: true,
        ticks: 'inside',
        showline: true,
        exponentformat: 'e'
    },
    margin: { t: 20, r: 20, l: 65, b: 50 },
    hovermode: 'closest'
};

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
        document.getElementById('engine-status').innerText = 'Worker error';
        alert("Failed to load solver worker. Ensure your browser supports WebAssembly.");
    };
}

function handleWorkerMessage(msg) {
    const statusElem = document.getElementById('engine-status');
    const submitBtn = document.getElementById('submit-btn');

    if (msg.type === 'status') {
        statusElem.innerText = msg.message;
    } else if (msg.type === 'ready') {
        statusElem.innerText = 'Engine ready';
        statusElem.classList.add('ready');
        submitBtn.disabled = false;
    } else if (msg.type === 'init') {
        expPotential = msg.exp_potential;
        expCurrent = msg.exp_current;
        
        Plotly.newPlot('live-chart', [
            {
                x: expPotential,
                y: expCurrent,
                type: 'scatter',
                mode: 'markers',
                name: 'Experimental',
                marker: { color: '#6B7280', size: 4 }
            },
            {
                x: expPotential,
                y: new Array(expPotential.length).fill(0),
                type: 'scatter',
                mode: 'lines',
                name: 'Fitted Model',
                line: { color: '#1E40AF', width: 2 }
            }
        ], {
            ...scientificPlotLayout,
            xaxis: { 
                ...scientificPlotLayout.xaxis, 
                title: 'Potential (V)',
                range: [Math.min(...expPotential), Math.max(...expPotential)]
            },
            yaxis: { ...scientificPlotLayout.yaxis, title: 'Current (A)' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.9)', bordercolor: '#E5E7EB', borderwidth: 1 }
        }, { responsive: true });
        
    } else if (msg.type === 'update') {
        document.getElementById('status-stage').innerText = `${msg.stage} (Iteration ${msg.iter})`;
        document.getElementById('status-metrics').innerText = `Loss: ${msg.loss.toFixed(5)}`;
        
        Plotly.update('live-chart', {
            y: [expCurrent, msg.sim_current]
        });
        
    } else if (msg.type === 'done') {
        document.getElementById('status-stage').innerText = 'Optimization Converged';
        lastResultsData = msg.data;
        displayResults(msg.data);
        submitBtn.disabled = false;
        submitBtn.innerText = 'Run Optimization';
        
    } else if (msg.type === 'error') {
        alert('Optimization error: ' + msg.message);
        document.getElementById('status-stage').innerText = 'Error: ' + msg.message;
        submitBtn.disabled = false;
        submitBtn.innerText = 'Run Optimization';
    }
}

function displayResults(data) {
    document.getElementById('results-card').classList.remove('hidden');
    document.getElementById('diagnostics-grid').classList.remove('hidden');
    
    const tbody = document.getElementById('params-tbody');
    tbody.innerHTML = `
        <tr>
            <td>Baseline Diffusivity</td>
            <td>D₀</td>
            <td>${data.parameters.diffusivity.toExponential(4)}</td>
            <td>cm²/s</td>
        </tr>
        <tr>
            <td>Exponential Coefficient (Anodic)</td>
            <td>β_L</td>
            <td>${data.parameters.beta_left.toFixed(4)}</td>
            <td>V⁻²</td>
        </tr>
        <tr>
            <td>Exponential Coefficient (Cathodic)</td>
            <td>β_R</td>
            <td>${data.parameters.beta_right.toFixed(4)}</td>
            <td>V⁻²</td>
        </tr>
        <tr>
            <td>Current Baseline Offset</td>
            <td>I_offset</td>
            <td>${data.parameters.baseline_offset.toExponential(4)}</td>
            <td>A</td>
        </tr>
        <tr>
            <td>Center Transition Potential</td>
            <td>E₀</td>
            <td>${data.parameters.v_center.toFixed(4)}</td>
            <td>V</td>
        </tr>
    `;
    
    // Density of States
    Plotly.newPlot('dos-chart', [{
        x: data.plots.v_plot,
        y: data.plots.dos_total,
        type: 'scatter',
        mode: 'lines',
        name: 'DOS',
        line: { color: '#111827', width: 1.75 }
    }], {
        ...scientificPlotLayout,
        xaxis: { 
            ...scientificPlotLayout.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)]
        },
        yaxis: { ...scientificPlotLayout.yaxis, title: 'Density of States (a.u.)' },
        showlegend: false
    }, { responsive: true });
    
    // Diffusivity vs Potential
    Plotly.newPlot('diffusivity-chart', [{
        x: data.plots.v_plot,
        y: data.plots.d_of_v,
        type: 'scatter',
        mode: 'lines',
        name: 'D(E)',
        line: { color: '#2563EB', width: 1.75 }
    }], {
        ...scientificPlotLayout,
        xaxis: { 
            ...scientificPlotLayout.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)]
        },
        yaxis: { ...scientificPlotLayout.yaxis, title: 'D(E) (cm²/s)', type: 'log' },
        showlegend: false
    }, { responsive: true });
}

// File Selection
const fileInput = document.getElementById('csv-file');
const fileStatusText = document.getElementById('file-status-text');

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    fileStatusText.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    const reader = new FileReader();
    reader.onload = (ev) => {
        currentFileContent = ev.target.result;
    };
    reader.readAsText(file);
});

// Load Sample Data
document.getElementById('btn-load-sample').addEventListener('click', () => {
    currentFileContent = SAMPLE_CV_CSV;
    fileStatusText.innerText = "Example dataset loaded (synthetic CV)";
    document.getElementById('pot_col').value = 1;
    document.getElementById('cur_col').value = 2;
    document.getElementById('scan_rate').value = "0.010";
    document.getElementById('film_thickness').value = "0.0001";
    document.getElementById('v_min').value = "-0.8";
    document.getElementById('v_max').value = "0.8";
    document.getElementById('skip_factor').value = "5";
    document.getElementById('num_peaks').value = "25";
});

// Submit Optimization
document.getElementById('cv-form').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!currentFileContent) {
        alert("Please select a CSV file or load the example dataset.");
        return;
    }
    
    const formData = new FormData(e.target);
    const config = Object.fromEntries(formData.entries());
    delete config.file;
    
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = true;
    submitBtn.innerText = 'Calculating...';
    
    document.getElementById('status-banner').classList.remove('hidden');
    document.getElementById('results-card').classList.add('hidden');
    document.getElementById('diagnostics-grid').classList.add('hidden');
    document.getElementById('status-stage').innerText = 'Initializing optimization stages...';
    document.getElementById('status-metrics').innerText = 'Loss: --';
    
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
    a.download = `cv_model_parameters_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Start Worker
initSolverWorker();
