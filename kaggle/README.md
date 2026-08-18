# Crafter experiment on a free Kaggle T4 (fully offline)

This is the scale-up run that produces the headline numbers for
`papers/neurips2026/neurips2026.tex` (Table 2): a controlled comparison of
two latent predictors — DreamerV3's categorical RSSM vs. the EB-H-JEPA
fixed-metriplectic prior — inside an identical DreamerV3-style agent on
Crafter, the world-model benchmark introduced by Danijar Hafner's group.

**Budget: ~2.3h of T4 compute for the 3-seed run** (6 runs × 23 min), or
~55 min for a 1-seed pilot. Free tier: 30 GPU hours/week.

## How the offline run works (why it is built this way)

Kaggle silently disables kernel internet for un-verified accounts: every
`pip install crafter` in a kernel dies with `Temporary failure in name
resolution` even with `enable_internet: true`. This repo sidesteps that
entirely — the kernel needs **zero network**:

1. **`kaggle/dataset/`** is a private Kaggle dataset vendoring everything
   the run needs: `agent.py` + `train.py` (the exact committed experiment
   code), the `crafter` sdist, and Linux/py3.12 wheels for its deps
   (imageio, opensimplex, ruamel.yaml + C ext).
   Rebuild/re-upload after code changes:
   ```bash
   cp experiments/crafter/agent.py kaggle/dataset/agent.py
   cp experiments/crafter/train.py kaggle/dataset/train.py
   cd kaggle/dataset && kaggle datasets version -m "sync code"
   ```
2. **`kaggle/kernels3/`** is the kernel: it mounts the dataset, installs
   the wheel deps with `pip install --no-index --find-links`, and copies
   the pure-Python `crafter` package straight into site-packages (no sdist
   build — Kaggle's pip build step is flaky), then runs the 6-run grid.
   Push with:
   ```bash
   kaggle kernels push -p kaggle/kernels3
   ```

Notes learned the hard way (all handled in `run.py`):
- Kaggle now mounts datasets at
  `/kaggle/input/datasets/<owner>/<slug>/`, **not** `/kaggle/input/<slug>/`.
- Kaggle auto-extracts `.tar.gz` uploads, so the crafter sdist arrives as a
  directory — `run.py` finds it by globbing for `setup.py`.
- `dist.mode` on torch distributions is a property, not a method, and the
  `cont_head` logits must be sigmoid'd before the TD(λ) recursion or
  `gamma·cont·lambda` exceeds 1 and returns diverge. Both fixed in
  `experiments/crafter/agent.py` (see git log).

## Setup (2 minutes, once)

1. **Wire Kaggle on your machine** (so the `kaggle` CLI works):
   - `https://www.kaggle.com/settings/account` → *Create New Token*
     → download `kaggle.json`.
   - Put it at `~/.kaggle/kaggle.json` (Windows:
     `C:\Users\<you>\.kaggle\kaggle.json`).
2. **Push the kernel** (uploads `kaggle/kernels3/run.py` + metadata to
   Kaggle, starts the run): `kaggle kernels push -p kaggle/kernels3`.
3. The run needs the private dataset to exist — first time, create it from
   `kaggle/dataset/` (`kaggle datasets create -p kaggle/dataset`), then
   update it as in step 1 above.

## Collect results

```bash
kaggle kernels output sehajrsingh/eb-h-jepa-crafter-3seed -p results/kaggle/
mkdir -p results/crafter
cp results/kaggle/crafter_results/crafter_*.json results/crafter/
python experiments/crafter/analyze.py results/crafter   # → Table 2 + curves
```

Every run JSON carries seed, device, wall time, per-update losses, eval
returns, and the latent-stability audit — each paper claim traces to an
artifact.

## Why this is the right run

- **Crafter** is *the* benchmark of the RSSM/DreamerV3 group; a structure
  result on it speaks their language.
- The two arms are **identical except the stochastic prior**, with matched
  total KL free-bit budgets (32 bits/step) — clean attribution.
- The **latent-stability audit** (raw KL pre-free-bits, scale-free
  prior–posterior drift, latent boundedness, dissipation tr(R)/n) turns
  the run into a mechanism study, not just a return comparison.
- Either outcome is publishable: structure helping (first positive evidence
  for physical priors in an RL world model) or hurting (the honest negative
  the field needs).
