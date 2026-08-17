"""Summarize Crafter run JSONs -> paper Table 2 + latent-stability comparison.

Usage:
    python analyze.py <results_dir...>

Each JSON is written by train.py and carries per-update losses, (step,
eval_return) pairs, and (if the audit was enabled) latent-stability metrics:
drift (prior-vs-posterior), raw KL, latent norm, dissipation tr(R)/n.

Prints a Markdown table ready to paste into papers/neurips2026/neurips2026.tex.
"""

import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(paths):
    runs = []
    for p in paths:
        for f in sorted(glob.glob(p.rstrip("/\\") + "/*.json")):
            with open(f) as fh:
                d = json.load(fh)
            if "predictor" in d and "eval_return" in d:
                runs.append((f, d))
    return runs


def eval_at(d, steps):
    """Return at the last eval <= steps, else None."""
    best = None
    for s, r in d["eval_return"]:
        if s <= steps:
            best = r
        else:
            break
    return best


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    runs = load_runs(sys.argv[1:])
    if not runs:
        print("no run JSONs found")
        return 1
    by_predictor = {}
    for f, d in runs:
        by_predictor.setdefault(d["predictor"], []).append((f, d))

    print("# Crafter results — sample efficiency (Table 2 source)\n")
    print("| arm | seed | return@5k | return@10k | return@15k | return@20k | wall_s |")
    print("|---|---|---|---|---|---|---|")
    for pred in sorted(by_predictor):
        for f, d in sorted(by_predictor[pred]):
            r5 = eval_at(d, 5000)
            r10 = eval_at(d, 10000)
            r15 = eval_at(d, 15000)
            r20 = eval_at(d, 20000)
            fmt = lambda v: f"{v:.2f}" if v is not None else "—"
            print(f"| {pred} | {d['config'].get('seed', '?')} | {fmt(r5)} | "
                  f"{fmt(r10)} | {fmt(r15)} | {fmt(r20)} | {d.get('wall_seconds', '?')} |")

    # stability: mean over the last audit entries per arm.
    # NOTE: raw drift is NOT comparable across arms (1024-dim one-hot
    # categorical vs 64-dim continuous); the scale-free quantity is
    # drift / latent_norm (relative predictive error of the prior).
    print("\n# Latent stability (last audit per run; drift_norm = drift/latent_norm)\n")
    print("| arm | seed | drift | drift_norm | kl_raw | latent_norm | dissipation |")
    print("|---|---|---|---|---|---|---|")
    for pred in sorted(by_predictor):
        for f, d in sorted(by_predictor[pred]):
            stab = d.get("stability", [])
            if not stab:
                print(f"| {pred} | {d['config'].get('seed', '?')} | (no audit) | | | | |")
                continue
            last = stab[-1]
            diss = last.get("dissipation", "—")
            diss = f"{diss:.4f}" if isinstance(diss, float) else diss
            dn = (last["drift"] / last["latent_norm"]) if last["latent_norm"] else float("nan")
            print(f"| {pred} | {d['config'].get('seed', '?')} | "
                  f"{last['drift']:.4f} | {dn:.3f} | {last['kl_raw']:.3f} | "
                  f"{last['latent_norm']:.3f} | {diss} |")

    # plots: eval return + normalized drift curves per arm
    plot_curves(by_predictor)

    # head-to-head at matched steps
    print("\n# Head-to-head at matched env steps\n")
    if "rssm" in by_predictor and "metriplectic" in by_predictor:
        a = by_predictor["rssm"][0][1]
        b = by_predictor["metriplectic"][0][1]
        for s in (5000, 10000, 15000, 20000):
            ra, rb = eval_at(a, s), eval_at(b, s)
            if ra is not None and rb is not None:
                diff = rb - ra
                print(f"  step {s:5d}: rssm {ra:6.2f} | metriplectic {rb:6.2f} | "
                      f"delta {diff:+.2f} {'(structure helps)' if diff > 0 else '(structure hurts/neutral)'}")
    return 0


def plot_curves(by_predictor):
    """Eval return and normalized drift vs env steps -> PNGs in results/."""
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        colors = {"rssm": "#1f77b4", "metriplectic": "#d62728"}
        for pred, runs in by_predictor.items():
            color = colors.get(pred, None)
            for f, d in runs:
                xs = [s for s, _ in d["eval_return"]]
                ys = [r for _, r in d["eval_return"]]
                if xs:
                    ax[0].plot(xs, ys, "o-", color=color, label=pred if color else None)
                stab = d.get("stability", [])
                if stab:
                    sx = [a["step"] for a in stab]
                    sy = [a["drift"] / a["latent_norm"] if a["latent_norm"] else float("nan")
                          for a in stab]
                    ax[1].plot(sx, sy, "s-", color=color,
                               label=pred + " (drift_norm)" if color else None)
        ax[0].set_xlabel("env steps"); ax[0].set_ylabel("eval return")
        ax[0].set_title("Crafter sample efficiency"); ax[0].legend()
        ax[1].set_xlabel("env steps"); ax[1].set_ylabel("drift / ||z||")
        ax[1].set_title("Latent stability (prior-vs-posterior drift)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig("results/crafter_curves.png", dpi=150)
        print("\nwrote results/crafter_curves.png")
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    sys.exit(main())
