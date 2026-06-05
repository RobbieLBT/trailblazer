"""
test.py — full pipeline test runner

Builds transmission + cell graphs, then launches the Streamlit app.
Run from the repo root with the venv active:

    python test.py
    python test.py --wx none          # skip weather fetch (faster)
    python test.py --no-streamlit     # build only, no app
"""

import argparse
import subprocess
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--wx", default="gairmet", choices=["gairmet", "none"],
                    help="Weather provider for graph build (default: gairmet)")
parser.add_argument("--no-streamlit", action="store_true",
                    help="Skip launching Streamlit after build")
args = parser.parse_args()

graphs = [
    ("cell",         ["--graph", "cell",         "--elevation", "--airspace", "--wx", args.wx, "--cell-grid-km", "12.5", "--cell-max-edge", "40", "--tiles", "--landscan", "data/population/landscan-mosaic-unitedstates-v1-assets/landscan-mosaic-unitedstates-v1.tif"]),
    ("transmission", ["--graph", "transmission", "--elevation", "--airspace", "--wx", args.wx, "--tiles", "--landscan", "data/population/landscan-mosaic-unitedstates-v1-assets/landscan-mosaic-unitedstates-v1.tif"]),
]

results = {}

for name, extra_args in graphs:
    print(f"\n{'='*60}")
    print(f"  Building {name} graph — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    t0 = datetime.now()

    ret = subprocess.run(
        [sys.executable, "build_graph.py"] + extra_args,
        check=False,
    )

    elapsed = (datetime.now() - t0).total_seconds()
    status  = "✅ OK" if ret.returncode == 0 else f"❌ FAILED (exit {ret.returncode})"
    results[name] = (elapsed, status)
    print(f"\n  {name}: {status} — {elapsed:.1f}s")

print(f"\n{'='*60}  SUMMARY")
for name, (elapsed, status) in results.items():
    print(f"  {name:<16} {status}  ({elapsed:.1f}s)")
print(f"{'='*60}\n")

if not args.no_streamlit:
    print("  Launching Streamlit app… (Ctrl+C to quit)\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])