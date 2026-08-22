#!/usr/bin/env python3
"""Kaggle GPU kernel: EBH-JEPA Kuramoto-Sivashinsky atlas (5 arms × 3 seeds).

Everything is offline: no pip install, no git clone, no network at all.
Code + wheels come from the dataset mount.
"""
import os, sys, json, time, subprocess, glob, shutil, re

# Ensure the working directory and src/ are on the path
os.chdir('/kaggle/working')
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

# ── 1. Locate the mounted dataset ──────────────────────────────────────────
CANDIDATES = glob.glob("/kaggle/input/*ebh*") + glob.glob("/kaggle/input/datasets/*/*ebh*")
if not CANDIDATES:
    raise RuntimeError("Dataset not found in /kaggle/input")
DATASET = CANDIDATES[0]
print(f"Dataset mounted at {DATASET}")

# ── 2. Install offline wheels (zero network) ───────────────────────────────
WHEELS = glob.glob(os.path.join(DATASET, "*.whl"))
if WHEELS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index",
                           "--find-links", DATASET] + WHEELS, stdout=subprocess.DEVNULL)

# ── 3. Set up src/ebhjepa package from dataset files ───────────────────────
os.makedirs("src/ebhjepa", exist_ok=True)
PKG = "src/ebhjepa"
for f in ["__init__.py", "ebhjepa.py", "ks.py"]:
    src = os.path.join(DATASET, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(PKG, f))
        print(f"  copied {f}")

# Also copy the benchmark runner
os.makedirs("benchmarks", exist_ok=True)
for f in ["ks_benchmark.py", "run_benchmark.py"]:
    src = os.path.join(DATASET, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join("benchmarks", f))

# Copy crafter files too (needed by train.py if referenced)
for f in ["agent.py", "train.py"]:
    src = os.path.join(DATASET, f)
    if os.path.exists(src):
        shutil.copy(src, f)

# ── 4. Verify imports work ─────────────────────────────────────────────────
import torch
print(f"torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

sys.path.insert(0, "src")
from ebhjepa import EBHJepa, LossWeights
from ebhjepa.ks import KS_MODES, ks_divergence
from benchmarks.ks_benchmark import build_arms, train_stress_ks, ks_stress_eval

print(f"KS: {KS_MODES} modes, div F = {ks_divergence():.4f}")

# ── 5. Run the full 5-arm × 3-seed atlas ───────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dt, obs_dim, latent_dim = 0.01, 32, 64
STEPS = 250
SEEDS = [0, 1, 2]
cfg = (device, obs_dim, latent_dim, dt, 128)

print(f"\n{'='*60}")
print(f"KS ATLAS: 5 arms × {len(SEEDS)} seeds, {STEPS} steps each")
print(f"device={device}, batch=128")
print(f"{'='*60}\n")

t_start = time.time()
arms = build_arms(device, obs_dim, latent_dim, dt, 128, float("nan"))
os.makedirs("results", exist_ok=True)

summary = []
for tag, model in arms:
    for seed in SEEDS:
        torch.manual_seed(seed)
        short = tag.split("/")[0]
        out_path = f"results/ks_{short}_seed{seed}.json"
        try:
            model2 = train_stress_ks(model, f"{tag} s={seed}", 6, STEPS, cfg)
            r = ks_stress_eval(model2, f"{tag} s={seed}", cfg, eval_horizon=200)
            r["status"] = "ok"
        except Exception as exc:
            import traceback
            print(f"FAILED: {tag} seed={seed}: {exc}")
            traceback.print_exc()
            r = {"tag": tag, "seed": seed, "status": "failed", "error": str(exc),
                 "finite": False, "lam_err": float("nan"), "contraction_err": float("nan")}
        r["seed"] = seed
        r["steps"] = STEPS
        r["wall_seconds"] = round(time.time() - t_start, 1)
        json.dump(r, open(out_path, "w"), indent=2, default=str)
        summary.append(r)
        print(f"[saved {out_path}]")
        torch.cuda.empty_cache()

# ── 6. Aggregate results ───────────────────────────────────────────────────
out = {
    "meta": {
        "system": "kuramoto-sivashinsky", "L": 22.0, "modes": KS_MODES,
        "dt": dt, "true_divergence": ks_divergence(),
        "device": device, "torch": torch.__version__,
        "seeds": SEEDS, "steps_per_arm": STEPS,
        "wall_seconds": round(time.time() - t_start, 1),
    },
    "arms": summary,
}
stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
path = f"results/ks_benchmark_{stamp}.json"
json.dump(out, open(path, "w"), indent=2, default=str)
print(f"\nKS ATLAS COMPLETE in {time.time()-t_start:.0f}s")
print(f"Results: {path}")

# Copy results to output dir for Kaggle download
os.makedirs("/kaggle/working/results", exist_ok=True)
for f in glob.glob("results/ks_*.json"):
    shutil.copy(f, "/kaggle/working/results/")
    print(f"  -> /kaggle/working/results/{os.path.basename(f)}")
