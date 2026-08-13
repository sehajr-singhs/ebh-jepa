# One-hour Crafter experiment on a free Kaggle T4

This is the scale-up run that produces the headline numbers for
`papers/neurips2026/neurips2026.tex` (Table 2): a controlled comparison of
two latent predictors — DreamerV3's categorical RSSM vs. the EB-H-JEPA
fixed-metriplectic prior — inside an identical DreamerV3-style agent on
Crafter, the world-model benchmark introduced by Danijar Hafner's group.

**Budget: under 1 hour of T4 compute** (2 × 27 min arms + install), which
fits in a single Kaggle GPU session. Free tier: 30 GPU hours/week.

## Setup (2 minutes, once)

1. **Wire Kaggle on your machine** (so `kaggle` CLI works — optional but
   handy for downloading results):
   - `https://www.kaggle.com/settings/account` → *Create New Token*
     → download `kaggle.json`.
   - Put it at `~/.kaggle/kaggle.json` (Windows:
     `C:\Users\<you>\.kaggle\kaggle.json`).
2. **Create the notebook.** On Kaggle: *New Notebook* → paste the entire
   contents of [`crafter_experiment.py`](crafter_experiment.py) into the
   first cell.
3. **Enable the GPU:** Settings → Accelerator → **GPU T4 x2** (free).
4. **Run all.** Two arms run sequentially, ~30 min each, with hard
   wall-clock stops — the session cannot blow past an hour.

## Collect results

The JSONs land in `/kaggle/working/crafter_results/` (Output tab →
download, or `kaggle kernels output <owner>/<notebook> -p results/`).

Copy them into this repo as the paper's evidence:

```bash
mkdir -p results/crafter
cp <download-dir>/crafter_*.json results/crafter/
```

Then run the analysis + fill Table 2:

```bash
python experiments/crafter/analyze.py results/crafter
```

(If `analyze.py` does not exist yet, the per-step return curves are already
in each JSON under `eval_return` — plot them and transcribe.)

## Why this is the right run

- **Crafter** is *the* benchmark of the RSSM/DreamerV3 group; a structure
  result on it speaks their language.
- The two arms are **identical except the stochastic prior** — clean
  attribution.
- Every run JSON carries seed, device, wall time, and per-update losses —
  each paper claim traces to an artifact.
- Either outcome is publishable: structure helping (first positive evidence
  for physical priors in an RL world model) or hurting (the honest negative
  the field needs).

## What to do while it runs

Update the placeholder Table 2 in `papers/neurips2026/neurips2026.tex`,
re-run the Lorenz atlas on a GPU for fresh timestamps, and (if you want
extra signal) add a third arm with a *smaller* metriplectic latent to test
the capacity confound.
