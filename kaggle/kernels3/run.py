"""EB-H-JEPA Crafter experiment — FINAL 3-seed run, fully OFFLINE (Kaggle kernel).

The paper's headline numbers: 3 seeds x 2 arms (rssm, metriplectic) of the
DreamerV3 latent-stability study, with matched free-bit budgets and the
stability audit (drift, raw KL, latent norm, dissipation tr(R)/n).

No kernel internet needed: crafter + deps are vendored as wheels and the
experiment code ships in the dataset sehajrsingh/ebh-jepa-crafter-env,
mounted at /kaggle/input/. (Kaggle's kernel internet is silently disabled
for unverified accounts; this sidesteps it entirely.)

Per run: 20k env steps, hard 23-min wall-clock stop. 6 runs x ~23 min =
~2.3h GPU. Outputs land in /kaggle/working/crafter_results/ and are fetched
with:
  kaggle kernels output sehajrsingh/eb-h-jepa-crafter-3seed -p results/kaggle/
"""

import glob
import os
import shutil
import subprocess
import sys
import time

OUT = "/kaggle/working/crafter_results"
SEEDS = [int(s) for s in os.environ.get("CRAFTER_SEEDS", "0 1 2").split()]
MAX_SECONDS = int(os.environ.get("CRAFTER_MAX_SECONDS", "1380"))  # 23 min/run

# Kaggle's mount layout changed: datasets now live under
# /kaggle/input/datasets/<owner>/<slug>/ (older kernels used /kaggle/input/<slug>/).
CANDIDATES = [
    "/kaggle/input/datasets/sehajrsingh/ebh-jepa-crafter-env",
    "/kaggle/input/ebh-jepa-crafter-env",
]
INPUT = next((p for p in CANDIDATES if os.path.isdir(p)), None)
assert INPUT, f"dataset not mounted (tried {CANDIDATES})"
print("dataset mounted at", INPUT, "->", sorted(os.listdir(INPUT))[:8], "...")

# --- 1. offline install of crafter + vendored deps --------------------------
# Kaggle auto-extracts .tar.gz on upload, so the sdist arrives as a directory
# (crafter-1.8.3/crafter-1.8.3/ with setup.py) rather than the tarball.
def find_sdist():
    for p in glob.glob(f"{INPUT}/crafter-*/**/setup.py", recursive=True):
        return os.path.dirname(p)
    for p in glob.glob(f"{INPUT}/crafter-*.tar.gz"):
        return p
    raise FileNotFoundError(f"no crafter sdist under {INPUT}")


def offline_install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-index", "--find-links", INPUT,
                    "--no-build-isolation",
                    find_sdist()],
                   check=True)


try:
    offline_install()
    print("crafter installed OFFLINE")
except Exception as e:
    print(f"offline install failed ({e}); trying network as fallback")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "crafter"],
                   check=True)

# --- 2. ship the experiment code from the dataset (not the network) ---------
for fname in ("agent.py", "train.py"):
    shutil.copy(os.path.join(INPUT, fname), fname)
    print("shipped", fname)

# --- 3. runs: 3 seeds x 2 arms ----------------------------------------------
os.makedirs(OUT, exist_ok=True)
for seed in SEEDS:
    for predictor, label in (("rssm", "DreamerV3 categorical RSSM"),
                             ("metriplectic", "fixed-metriplectic prior")):
        print(f"\n===== seed {seed} | arm: {label} =====")
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
                        "--max-seconds", str(MAX_SECONDS),
                        "--outdir", OUT,
                        "--seed", str(seed)],
                       check=True)
        print(f"seed {seed} {predictor} finished in {time.time() - t0:.0f}s")

print("\nDONE. Files in /kaggle/working/crafter_results/:")
for f in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT, f), os.path.getsize(os.path.join(OUT, f)))
