"""Generate colab/EBH_JEPA_Crafter_1h.ipynb (one-click 1-hour T4 run).

Reproducible: edit the cells below, re-run `python build_notebook.py`,
commit the .ipynb alongside. The notebook fetches agent.py / train.py from
the repo at run time, so the cells never drift from the code they run.
"""

import nbformat as nbf

NB_PATH = "colab/EBH_JEPA_Crafter_1h.ipynb"

MD_HEADER = """\
# EB-H-JEPA: Crafter experiment — controlled 1-hour run

A controlled comparison of two latent predictors inside an identical
DreamerV3-style agent on Crafter (Hafner 2021):

| arm | predictor | role |
|---|---|---|
| `rssm` | DreamerV3 categorical RSSM (32×32 codes) | baseline |
| `metriplectic` | fixed-metriplectic map `z' = z + dt(J∇H − R∇S)` | treatment (E-arm) |

Everything else — encoder, decoder, GRU, KL free bits, TD(λ) imagination,
actor-critic — is byte-identical, so any sample-efficiency difference is
attributable to predictor structure alone.

**Before running:**
1. Runtime → Change runtime type → **T4 GPU**.
2. Run all cells in order (total ~55 min; the session stays busy so it will
   not idle-disconnect). Keep this tab open.
3. The final cell saves the results JSONs to Google Drive.
"""

C_SETUP = """\
# 0) GPU check + install Crafter (the environment itself)
import subprocess, sys
get_ipython().system('nvidia-smi')
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "crafter"], check=True)
print("crafter installed; torch:", __import__("torch").__version__)"""

C_FETCH = """\
# 1) fetch the experiment code from the repo (single source of truth)
import urllib.request
BASE = "https://raw.githubusercontent.com/sehajr-singhs/ebh-jepa/main/experiments/crafter/"
for f in ("agent.py", "train.py"):
    urllib.request.urlretrieve(BASE + f, f)
    print("fetched", f)
get_ipython().system('ls -la agent.py train.py')"""

C_ARM = """\
# {n}) ARM {k} — {label} ({m} min hard stop)
import subprocess, sys, os
os.makedirs("crafter_results", exist_ok=True)
subprocess.run([sys.executable, "train.py",
    "--predictor", "{p}", "--env", "crafter",
    "--train-steps", "20000", "--warmup", "500", "--wm-updates", "10",
    "--eval-every", "5000", "--eval-episodes", "2",
    "--max-seconds", "1620", "--outdir", "crafter_results", "--seed", "0"],
    check=True)
print("arm {k} ({p}) finished")"""

C_SAVE = """\
# 4) save results to Google Drive + print download list
import glob, shutil, os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    dst = "/content/drive/MyDrive/ebh_jepa_crafter_results"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob("crafter_results/*.json"):
        shutil.copy(f, dst)
    print("saved to Drive:", dst)
except Exception as e:
    print("Drive mount skipped:", e)
print("")
print("results in /content/crafter_results/:")
for f in sorted(glob.glob("crafter_results/*.json")):
    print("  ", f, os.path.getsize(f), "bytes")
print("")
print("Download them (Files panel → right-click → Download), copy into the")
print("repo as results/crafter/, then fill Table 2 in papers/neurips2026/neurips2026.tex")"""

MD_FOOTER = """\
## What the results mean

Each JSON has `(step, eval_return)` pairs under `eval_return` plus per-update
losses. Compare the return curves at 10k / 20k env steps:

- **metriplectic wins** → first positive evidence that physical inductive
  biases pay off inside an RL world model → strengthen the paper's claim.
- **rssm wins / tie** → the honest negative the field needs → the paper
  reports a controlled null result.

Either way the numbers go into Table 2 of
`papers/neurips2026/neurips2026.tex` and the run JSONs ship as evidence in
`results/crafter/`. See `kaggle/README.md` for the same protocol via Kaggle.
"""


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": [], "gpuType": "T4",
                  "name": "EB-H-JEPA Crafter experiment (1h)"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }
    cells = [nbf.v4.new_markdown_cell(MD_HEADER),
             nbf.v4.new_code_cell(C_SETUP),
             nbf.v4.new_code_cell(C_FETCH),
             nbf.v4.new_code_cell(C_ARM.format(n=2, k=1, p="rssm",
                                               label="DreamerV3 categorical RSSM baseline",
                                               m=27)),
             nbf.v4.new_code_cell(C_ARM.format(n=3, k=2, p="metriplectic",
                                               label="fixed-metriplectic prior",
                                               m=27)),
             nbf.v4.new_code_cell(C_SAVE),
             nbf.v4.new_markdown_cell(MD_FOOTER)]
    nb.cells = cells
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    nbf.validate(nb)
    print(f"wrote {NB_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
