"""EB-H-JEPA Crafter experiment — controlled 1-hour T4 run (Kaggle kernel).

Runs BOTH arms of the DreamerV3 latent-stability study inside one GPU
session with hard per-arm wall-clock stops:

  arm 1  rssm          DreamerV3 categorical RSSM (baseline)
  arm 2  metriplectic  fixed-metriplectic prior (EB-H-JEPA E-arm)

Everything else in the agent is identical, so any difference in sample
efficiency or latent stability is attributable to the predictor structure.
Results JSONs (per-update losses + (step, eval_return) pairs) are written to
/kaggle/working/crafter_results/ and fetched with:
  kaggle kernels output sehajrsingh/ebh-jepa-crafter -p results/kaggle/
"""

import os
import shutil
import subprocess
import sys
import time

# --- 0. install Crafter (the open-world env; needs internet) ---------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "crafter"],
               check=True)
print("crafter installed")

# --- 1. fetch the experiment code from the repo (single source of truth) ---
import urllib.request

EXPERIMENT_URL = ("https://raw.githubusercontent.com/"
                  "sehajr-singhs/ebh-jepa/main/experiments/crafter/")
for fname in ("agent.py", "train.py"):
    urllib.request.urlretrieve(EXPERIMENT_URL + fname, fname)
    print("fetched", fname)

# --- 2. run both arms, 26 min each, hard stop before the 1h budget --------
OUT = "/kaggle/working/crafter_results"
os.makedirs(OUT, exist_ok=True)

ARMS = [
    ("rssm",         "DreamerV3 categorical RSSM (baseline)"),
    ("metriplectic", "fixed-metriplectic prior (E-arm)"),
]
for predictor, label in ARMS:
    print(f"\n===== arm: {label} =====")
    t0 = time.time()
    subprocess.run([sys.executable, "train.py",
                    "--predictor", predictor,
                    "--env", "crafter",
                    "--train-steps", "20000",
                    "--warmup", "500",
                    "--wm-updates", "10",
                    "--eval-every", "5000",
                    "--eval-episodes", "2",
                    "--max-seconds", "1560",      # 26 min hard stop
                    "--outdir", OUT,
                    "--seed", "0"],
                   check=True)
    print(f"arm {predictor} finished in {time.time()-t0:.0f}s")

print("\nDONE. Files in /kaggle/working/crafter_results/:")
for f in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT, f), os.path.getsize(os.path.join(OUT, f)))
