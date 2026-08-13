"""Reproducible runner for the EB-H-JEPA 5-arm benchmark (B/C/D/E/F).

Mirrors the `main()` entry point of the core module exactly (same arms, same
seeds, same config), but additionally dumps every per-arm metric to
`results/benchmark_<timestamp>.json` so claims in the paper can be traced to a
concrete run. Use `--steps N` for a fast smoke run.

Usage:
    python -m benchmarks.run_benchmark [--steps 250] [--train-horizon 6]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebhjepa import EBHJepa, LossWeights  # noqa: E402
from ebhjepa.ebhjepa import (  # noqa: E402
    LORENZ_BETA,
    LORENZ_DIVERGENCE,
    LORENZ_LAMBDA1,
    LORENZ_RHO,
    LORENZ_SIGMA,
    stress_eval,
    train_stress,
    true_lorenz_lyapunov,
)


def build_arms(device, obs_dim, latent_dim, dt, batch, lam_true):
    """Construct the five benchmark arms identically to `main()`."""
    common = dict(latent_dim=latent_dim, dt=dt, mode="vector", in_dim=obs_dim,
                  integrator="symplectic")
    return [
        ("B/UNCONSTRAINED-MLP",
         EBHJepa(predictor_type="mlp", conserve_mode="none",
                 weights=LossWeights(1.0, 1.0, 0.0, 0.0, 2.0, 0.0), **common)),
        ("C/RIGID-HAMILTONIAN",
         EBHJepa(predictor_type="hamiltonian", conserve_mode="flat",
                 weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.0), **common)),
        ("D/METRIPLECTIC-NAIVE",
         EBHJepa(predictor_type="metriplectic", conserve_mode="dissipative",
                 weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.0), **common)),
        ("E/METRIPLECTIC-FIXED",
         EBHJepa(predictor_type="metriplectic", conserve_mode="dissipative",
                 weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.02),
                 metriplectic_fixed_J=True, metriplectic_r_init_std=0.05,
                 **common)),
        ("F/SPECTRAL-ALIGNED",
         EBHJepa(predictor_type="metriplectic", conserve_mode="dissipative",
                 weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.02, 0.10),
                 metriplectic_fixed_J=True, metriplectic_r_init_std=0.05,
                 spectral_lam1_target=lam_true, **common)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=250, help="train steps per arm")
    ap.add_argument("--batch", type=int, default=128,
                    help="batch size (reduce to ~32 on CPU for fast smoke runs)")
    ap.add_argument("--train-horizon", type=int, default=6,
                    help="final multi-step alignment horizon K")
    ap.add_argument("--eval-horizon", type=int, default=200,
                    help="rollout horizon for evaluation")
    ap.add_argument("--outdir", type=str, default=str(ROOT / "results"))
    ap.add_argument("--warp", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt, obs_dim, latent_dim = 0.01, 32, 64
    batch = args.batch
    cfg = (device, obs_dim, latent_dim, dt, batch)

    print(f"device={device}  torch={torch.__version__}  "
          f"python={platform.python_version()}")
    print(f"CHAOTIC Lorenz-63 stress test (sigma={LORENZ_SIGMA}, "
          f"rho={LORENZ_RHO}, beta={LORENZ_BETA:.3f}) | nonlinear warp={args.warp}")
    print(f"reference invariants: div f = {LORENZ_DIVERGENCE:.4f}, "
          f"textbook lambda1 = {LORENZ_LAMBDA1}")

    lam_true = true_lorenz_lyapunov(dt, 3000, device)
    print(f"empirical TRUE lambda1 (Benettin, dt={dt}) = {lam_true:+.4f}")

    arms = build_arms(device, obs_dim, latent_dim, dt, batch, lam_true)
    results, log = [], []
    t_start = time.time()
    for tag, model in arms:
        torch.manual_seed(0)  # identical seed per arm, as in main()
        try:
            train_stress(model, tag, args.train_horizon, args.steps, cfg,
                         warp=args.warp)
            r = stress_eval(model, tag, cfg, eval_horizon=args.eval_horizon,
                            warp=args.warp, true_lambda=lam_true)
            r["status"] = "ok"
        except Exception as exc:  # benchmark robustness: record, continue
            print(f"[{tag}] ARM FAILED: {type(exc).__name__}: {exc}")
            r = {"tag": tag, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                 "finite": False, "motion_ratio": float("nan"),
                 "lam_model": float("nan"), "lam_err": float("nan"),
                 "contraction": float("nan"), "contraction_err": float("nan"),
                 "r_mag": None, "chamfer": float("nan")}
        results.append(r)
        log.append(r)

    wall = time.time() - t_start
    out = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "seed": 0,
            "steps_per_arm": args.steps,
            "train_horizon": args.train_horizon,
            "eval_horizon": args.eval_horizon,
            "warp": args.warp,
            "obs_dim": obs_dim, "latent_dim": latent_dim, "dt": dt, "batch": batch,
            "true_lambda1": lam_true,
            "wall_seconds": round(wall, 1),
        },
        "arms": results,
    }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(args.outdir) / f"benchmark_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nresults written to {path}")

    # Compact summary table (same reading as main()).
    print("\n===== B/C/D/E/F SUMMARY =====")
    print(f"{'model':<22}{'finite':>7}{'mratio':>8}{'lam_model':>11}"
          f"{'lam_err':>9}{'contract':>10}{'|d-13.7|':>10}{'tr(R)/n':>10}"
          f"{'chamfer':>9}")
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['tag']:<22}{'FAILED':>7}"); continue
        fin = "yes" if r["finite"] else "NO"
        rm = " n/a" if r.get("r_mag") is None else f"{r['r_mag']:.4f}"
        print(f"{r['tag']:<22}{fin:>7}{r['motion_ratio']:>8.3f}"
              f"{r['lam_model']:>11.3f}{r['lam_err']:>9.3f}"
              f"{r['contraction']:>10.3f}{r['contraction_err']:>10.3f}"
              f"{rm:>10}{r['chamfer']:>9.2f}")
    print(f"\nreference: true lambda1={lam_true:+.3f}, "
          f"true div f={LORENZ_DIVERGENCE:.3f}")
    print("pipeline OK")


if __name__ == "__main__":
    main()
