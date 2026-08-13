"""One-click 1-hour Crafter experiment for a Kaggle T4 GPU.

Paste this file's contents into a Kaggle notebook cell (see kaggle/README.md
for the 2-minute setup). It installs crafter, runs BOTH arms (RSSM baseline
and fixed-metriplectic treatment) with a hard per-arm wall-clock budget
totaling under an hour, and writes the results JSONs to
/kaggle/working/crafter_results/ for download.

The experiment is seeded and deterministic given the seed: both arms see
identical environment seeds, identical optimizer configs, and differ ONLY
in the stochastic prior (DreamerV3 categorical RSSM vs. the EB-H-JEPA
fixed-metriplectic map). Any sample-efficiency difference is attributable
to predictor structure alone.
"""

import subprocess
import sys

# --- 0. install crafter (open-world benchmark; no GPU needed for env) ------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "crafter"],
               check=True)

# --- 1. download this repo's experiment files ---------------------------------
import os
import shutil
import urllib.request

EXPERIMENT_URL = ("https://raw.githubusercontent.com/"
                  "sehajr-singhs/ebh-jepa/main/experiments/crafter/")
for fname in ("agent.py", "train.py"):
    urllib.request.urlretrieve(EXPERIMENT_URL + fname, fname)
    print(f"fetched {fname}")

# --- 2. run both arms, 27 min each, hard stop before the 1h budget ----------
OUT = "/kaggle/working/crafter_results"
os.makedirs(OUT, exist_ok=True)
os.environ["KAGGLE_WORKING"] = OUT

ARMS = [
    ("rssm",         "DreamerV3 categorical RSSM (baseline)"),
    ("metriplectic", "fixed-metriplectic prior (E-arm)"),
]
for predictor, label in ARMS:
    print(f"\n===== arm: {label} =====")
    subprocess.run([sys.executable, "train.py",
                    "--predictor", predictor,
                    "--env", "crafter",
                    "--train-steps", "20000",
                    "--warmup", "500",
                    "--wm-updates", "10",
                    "--eval-every", "5000",
                    "--eval-episodes", "2",
                    "--max-seconds", "1620",      # 27 min hard stop
                    "--outdir", OUT,
                    "--seed", "0"],
                   check=True)

print("\nDONE. Download these files from the notebook's Output tab:")
for f in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT, f))
