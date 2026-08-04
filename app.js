document.getElementById('advanced-toggle').addEventListener('click', () => {
    document.getElementById('advanced-content').classList.toggle('hidden');
});

const layoutConfig = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#F8F9FA',
    font: { color: '#212529', family: 'Roboto, sans-serif' },
    xaxis: { 
        gridcolor: '#DEE2E6', 
        zerolinecolor: '#ADB5BD',
        linecolor: '#DEE2E6',
        linewidth: 1,
        mirror: true,
        ticks: 'outside'
    },
    yaxis: { 
        gridcolor: '#DEE2E6', 
        zerolinecolor: '#ADB5BD',
        linecolor: '#DEE2E6',
        linewidth: 1,
        mirror: true,
        ticks: 'outside',
        exponentformat: 'e'
    },
    margin: { t: 30, r: 30, l: 70, b: 60 }
};

let solverWorker = null;
let expPotential = [];
let expCurrent = [];

function initWorker() {
    solverWorker = new Worker('solver_worker.js');
    solverWorker.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            handleWorkerMessage(msg);
        } catch (err) {
            console.error("Parse error:", err);
        }
    };
    solverWorker.onerror = (err) => {
        console.error("Worker error:", err);
        alert("Worker initialization failed. Please ensure WebAssembly is supported.");
        resetUI();
    };
}

document.getElementById('cv-form').addEventListener('submit', (e) => {
    e.preventDefault();
    
    const form = e.target;
    const fileInput = document.getElementById('csv-file');
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData(form);
    const config = Object.fromEntries(formData.entries());
    delete config.file;

    document.getElementById('submit-btn').disabled = true;
    document.getElementById('status-panel').classList.remove('hidden');
    document.getElementById('results-panel').classList.add('hidden');
    document.querySelector('.status-header').classList.remove('hidden');
    document.querySelector('.status-header .spinner').classList.remove('hidden');
    document.getElementById('status-stage').innerText = 'Initializing...';
    document.getElementById('status-details').innerText = 'Starting solver...';
    
    const reader = new FileReader();
    reader.onload = (ev) => {
        const fileContent = ev.target.result;
        
        if (!solverWorker) {
            initWorker();
        }
        
        solverWorker.postMessage({
            action: 'solve',
            file_content: fileContent,
            config: config
        });
    };
    reader.readAsText(file);
});

function resetUI() {
    document.getElementById('submit-btn').disabled = false;
}

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
                name: 'Raw Data',
                marker: { color: '#6C757D', size: 4 }
            },
            {
                x: expPotential,
                y: new Array(expPotential.length).fill(0),
                type: 'scatter',
                mode: 'lines',
                name: 'Simulated',
                line: { color: '#334155', width: 2 }
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
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)', bordercolor: '#DEE2E6', borderwidth: 1 }
        }, {responsive: true});
        
    } else if (msg.type === 'update') {
        document.getElementById('status-stage').innerText = msg.stage;
        document.getElementById('status-details').innerText = `Iteration: ${msg.iter} | Loss: ${msg.loss.toFixed(4)}`;
        
        Plotly.update('live-chart', {
            y: [expCurrent, msg.sim_current]
        });
        
    } else if (msg.type === 'done') {
        document.querySelector('.status-header').classList.add('hidden');
        displayResults(msg.data);
        resetUI();
        
    } else if (msg.type === 'error') {
        alert('Error: ' + msg.message);
        document.querySelector('.status-header .spinner').classList.add('hidden');
        document.getElementById('status-stage').innerText = "Error";
        document.getElementById('status-details').innerText = msg.message;
        resetUI();
    }
}

function displayResults(data) {
    document.getElementById('results-panel').classList.remove('hidden');
    
    const paramsDiv = document.getElementById('params-output');
    paramsDiv.innerHTML = `
        <div class="stat-box">
            <div class="label">Diffusivity</div>
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
    
    const dosTraces = [{
        x: data.plots.v_plot,
        y: data.plots.dos_total,
        type: 'scatter',
        mode: 'lines',
        name: 'Total DOS',
        line: { color: '#0f172a', width: 2 }
    }];
    
    Plotly.newPlot('dos-chart', dosTraces, {
        ...layoutConfig,
        xaxis: { 
            ...layoutConfig.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)]
        },
        yaxis: { ...layoutConfig.yaxis, title: 'Density of States (a.u.)' },
        showlegend: false
    }, {responsive: true});
    
    Plotly.newPlot('diffusivity-chart', [{
        x: data.plots.v_plot,
        y: data.plots.d_of_v,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#475569', width: 2 }
    }], {
        ...layoutConfig,
        xaxis: { 
            ...layoutConfig.xaxis, 
            title: 'Potential (V)',
            range: [Math.min(...data.plots.v_plot), Math.max(...data.plots.v_plot)],
            exponentformat: 'none'
        },
        yaxis: { ...layoutConfig.yaxis, title: 'D (cm²/s)', type: 'log' }
    }, {responsive: true});
}

// Custom file input logic
document.getElementById('csv-file').addEventListener('change', function(e) {
    const fileName = e.target.files[0] ? e.target.files[0].name : "No file chosen";
    document.getElementById('file-name-display').innerText = fileName;
});

// Initialize worker on page load
initWorker();
