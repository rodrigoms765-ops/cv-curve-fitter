"""Regenerate the stored example result shown by "Show example result".

The website loads manifest.json, stages the example scans with its settings, and
renders result.json through the same code path as a live fit. That file is only
honest while it matches what the solver would compute now, so rerun this after
any change to cv_solver.py or to the example settings:

    python examples/build_example.py

The window is taken from the data exactly as the browser takes it - the extremes
of the potential column across every file, to three decimals - and written back
into the manifest, so a live run from the loaded example reproduces the stored one.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import solve_cv_api  # noqa: E402

manifest_path = HERE / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = manifest["config"]

files = []
pot = []
for f in manifest["files"]:
    content = (HERE / f["path"]).read_text(encoding="utf-8")
    files.append({"name": f["name"], "content": content, "scan_rate": f["scan_rate"]})
    df = pd.read_csv(HERE / f["path"])
    v = pd.to_numeric(df.iloc[:, config["pot_col"]], errors="coerce")
    i = pd.to_numeric(df.iloc[:, config["cur_col"]], errors="coerce")
    pot.append(v[v.notna() & i.notna()].to_numpy())

pot = np.concatenate(pot)
config["v_min"] = float(f"{pot.min():.3f}")
config["v_max"] = float(f"{pot.max():.3f}")

result = json.loads(solve_cv_api(files, json.dumps(config)))
if result.get("type") != "done":
    sys.exit(f"solver failed: {result.get('message')}")

(HERE / manifest["result"]).write_text(json.dumps(result), encoding="utf-8")
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

p = result["params"]
print(f"window {config['v_min']} to {config['v_max']} V")
print(f"D_fast {p['d_fast']:.3e}  D_slow {p['d_slow']:.3e}  fast fraction {p['frac_fast']:.3f}")
print("RMSE " + "  ".join(f"{s['rmse_pct']:.2f}%" for s in result["scans"]))
