#!/usr/bin/env python3
"""Crafter 300k scale-up: 3 seeds x 2 arms on Kaggle T4 GPU.

Full DreamerV3-style training at 300k environment steps to test whether
the metriplectic latent-stability advantage translates into sample efficiency.
Offline (no kernel internet): all deps from the mounted dataset.

Per arm: ~45 min at 300k steps on T4. 6 runs x ~45 min = ~4.5h total.
Outputs land in /kaggle/working/crafter_300k/.
"""
import glob, os, shutil, subprocess, sys, time

OUT = "/kaggle/working/crafter_300k"
SEEDS = [0, 1, 2]
MAX_SECONDS = 2700  # 45 min per arm
TRAIN_STEPS = 300000

# Dataset mount (same layout as kernels3)
CANDIDATES = [
    "/kaggle/input/datasets/sehajrsingh/ebh-jepa-crafter-env",
    "/kaggle/input/ebh-jepa-crafter-env",
]
INPUT = next((p for p in CANDIDATES if os.path.isdir(p)), None)
assert INPUT, f"dataset not mounted (tried {CANDIDATES})"
print("Dataset:", INPUT)

# --- 1. Offline install ---
def find_sdist():
    for p in glob.glob(f"{INPUT}/crafter-*/**/setup.py", recursive=True):
        return os.path.dirname(p)
    raise FileNotFoundError(f"no crafter sdist under {INPUT}")

def offline_install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-index", "--find-links", INPUT,
                    "imageio", "opensimplex", "ruamel.yaml"], check=True)
    import site
    sdist = find_sdist()
    pkg_src = os.path.join(sdist, "crafter")
    dst = os.path.join(site.getsitepackages()[0], "crafter")
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(pkg_src, dst)
    print(f"crafter -> {dst}")

try:
    offline_install()
except Exception as e:
    print(f"offline failed ({e}); network fallback")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "crafter"], check=True)

# --- 2. Ship experiment code from dataset ---
for fname in ("agent.py", "train.py", "analyze.py"):
    src = os.path.join(INPUT, fname)
    if os.path.exists(src):
        shutil.copy(src, fname)
        print(f"shipped {fname}")

# --- 3. Run 3 seeds x 2 arms ---
os.makedirs(OUT, exist_ok=True)
for seed in SEEDS:
    for predictor in ("rssm", "metriplectic"):
        label = "RSSM" if predictor == "rssm" else "metriplectic"
        print(f"\n{'='*50}")
        print(f"seed={seed} | arm={label} | steps={TRAIN_STEPS}")
        print(f"{'='*50}")
        t0 = time.time()
        subprocess.run([
            sys.executable, "train.py",
            "--predictor", predictor,
            "--env", "crafter",
            "--train-steps", str(TRAIN_STEPS),
            "--warmup", "500",
            "--wm-updates", "10",
            "--eval-every", "10000",
            "--eval-episodes", "3",
            "--audit-every", "5000",
            "--max-seconds", str(MAX_SECONDS),
            "--outdir", OUT,
            "--seed", str(seed),
        ], check=True)
        elapsed = time.time() - t0
        print(f"FINISHED {label} seed={seed} in {elapsed:.0f}s ({elapsed/60:.1f}min)")

print(f"\n{'='*50}")
print("ALL DONE. Results:")
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f} ({sz} bytes)")

# Copy to working dir for Kaggle download
os.makedirs("/kaggle/working/crafter_300k_results", exist_ok=True)
for f in glob.glob(f"{OUT}/*.json"):
    shutil.copy(f, "/kaggle/working/crafter_300k_results/")
