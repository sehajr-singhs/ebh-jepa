"""EB-H-JEPA Crafter experiment — controlled 1-hour T4 run (Kaggle kernel).

Runs BOTH arms of the DreamerV3 latent-stability study inside one GPU
session with hard per-arm wall-clock stops:

  arm 1  rssm          DreamerV3 categorical RSSM (baseline)
  arm 2  metriplectic  fixed-metriplectic prior (EB-H-JEPA E-arm)

Everything else in the agent is identical, so any difference in sample
efficiency or latent stability is attributable to the predictor structure.
Results JSONs (per-update losses + (step, eval_return) pairs + stability
audits) are written to /kaggle/working/crafter_results/ and fetched with:
  kaggle kernels output sehajrsingh/eb-h-jepa-crafter-1h -p results/kaggle/

Network bootstrap: Kaggle boot DNS is flaky; we wait for connectivity with
retries before installing / fetching, so a transient DNS failure does not
burn the GPU session.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request


# --- 0. network bootstrap (retry until pypi is reachable) ------------------
def net_ready(host=("pypi.org", 443), retries=8, delay=20):
    for i in range(retries):
        try:
            socket.create_connection(host, timeout=10).close()
            print(f"net ready (attempt {i + 1})")
            return True
        except OSError as e:
            print(f"net not ready (attempt {i + 1}): {e}")
            time.sleep(delay)
    return False


assert net_ready(), "no network connectivity — aborting before burning the GPU hour"

# --- 1. install Crafter (PyPI, fallback: git checkout) ----------------------
def pip_install(target):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--timeout", "60", "--retries", "5", target], check=True)


try:
    pip_install("crafter")
except subprocess.CalledProcessError:
    print("PyPI install failed; trying git checkout")
    pip_install("git+https://github.com/danijar/crafter.git")
print("crafter installed")

# --- 2. fetch the experiment code from the repo (single source of truth) ---
def fetch(url, dst, retries=5, delay=15):
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, dst)
            print("fetched", dst)
            return
        except Exception as e:
            print(f"fetch {dst} failed (attempt {i + 1}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"could not fetch {url}")


EXPERIMENT_URL = ("https://raw.githubusercontent.com/"
                  "sehajr-singhs/ebh-jepa/main/experiments/crafter/")
for fname in ("agent.py", "train.py"):
    fetch(EXPERIMENT_URL + fname, fname)

# --- 3. run both arms, 26 min each, hard stop before the 1h budget ---------
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
                    "--audit-every", "2000",
                    "--max-seconds", "1560",      # 26 min hard stop
                    "--outdir", OUT,
                    "--seed", "0"],
                   check=True)
    print(f"arm {predictor} finished in {time.time() - t0:.0f}s")

print("\nDONE. Files in /kaggle/working/crafter_results/:")
for f in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT, f), os.path.getsize(os.path.join(OUT, f)))
