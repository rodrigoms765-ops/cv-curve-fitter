import json

notebook_path = r"C:\Users\rodri\OneDrive\Documents\Chem Project\FD_solver.ipynb"
output_path = r"C:\Users\rodri\.gemini\antigravity\scratch\cv_curve_ui\FD_solver.py"

with open(notebook_path, 'r', encoding='utf-8') as f:
    d = json.load(f)

with open(output_path, 'w', encoding='utf-8') as f:
    for cell in d['cells']:
        if cell['cell_type'] == 'code':
            f.write(''.join(cell['source']) + '\n\n')
