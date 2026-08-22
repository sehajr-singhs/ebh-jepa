"""Reproducible 5-arm benchmark on the SECOND chaotic system: Kuramoto--Sivashinsky.

Mirrors benchmarks/run_benchmark.py (same arms, same loss weights, same seeds,
same chaos-aware metrics) but on the Galerkin-truncated KS system of
src/ebhjepa/ks.py, whose exact phase-space contraction is the closed form
div F = sum_j (j^2 kappa^2 - j^4 kappa^4) and whose largest Lyapunov exponent
is estimated by the same Benettin scheme used for Lorenz. A one-system atlas
could be a fluke of Lorenz's geometry; KS is a genuinely different dissipative
chaotic system (scale-dependent spectrum, unstable long waves + stable short-wave
bath), so failure modes that reproduce on both are structural.

Usage:
    python -m benchmarks.ks_benchmark [--steps 250] [--arm E] [--seeds 0,1,2]
    python -m benchmarks.ks_benchmark --steps 40 --batch 32   # CPU smoke

Writes results/ks_<ARM>_seed<s>.json per arm (immediately, so a crash cannot
lose finished arms) plus results/ks_benchmark_<ts>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebhjepa import EBHJepa, LossWeights  # noqa: E402
from ebhjepa.ebhjepa import (  # noqa: E402
    _chamfer,
    _hausdorff,
    latent_contraction_rate,
    nonlinear_warp,
    rollout_lyapunov,
    train_stress,
)
from ebhjepa.ks import (  # noqa: E402
    KS_MODES,
    ks_divergence,
    ks_lambda1,
    make_ks_sequence,
)

KS_TRUE_DIVERGENCE = ks_divergence(KS_MODES)


def build_arms(device, obs_dim, latent_dim, dt, batch, lam_true):
    """Identical arm constructions to the Lorenz benchmark (same losses, same
    fixed-J / dead-saddle / spectral-alignment treatments)."""
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


def train_stress_ks(model, tag, train_horizon, steps, cfg, warp=True):
    """train_stress on KS data: same progressive-horizon curriculum, same loss,
    swapped data generator."""
    device, obs_dim, latent_dim, dt, batch = cfg
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"\n=== KS | {tag} | predictor={model.predictor_type} "
          f"conserve_mode={model.conserve_mode} train_horizon={train_horizon} ===")
    model.train()
    for step in range(1, steps + 1):
        frac = (step - 1) / steps
        K = 1 if frac < 0.30 else (3 if frac < 0.60 else train_horizon)
        x_seq, _ = make_ks_sequence(batch, obs_dim, dt, K, device, warp=warp)
        out = model(x_seq)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | K={K} | total {out['loss'].item():.4f} "
                  f"| align {out['align'].item():.4f} "
                  f"| motion {out['motion'].item():.4f} "
                  f"| contract {out['contract'].item():.4f} "
                  f"| div {out['div'].item():+.3f} "
                  f"| mratio {out['motion_ratio'].item():.3f}")
    return model


def ks_stress_eval(model, tag, cfg, eval_horizon=200, warp=True,
                   true_lambda=None, true_div=None):
    """Chaos-aware evaluation on KS: same metrics as the Lorenz stress_eval,
    judged against the KS truth invariants (closed-form divergence, Benettin
    lambda1)."""
    device, obs_dim, latent_dim, dt, batch = cfg
    model.eval()
    x_seq, _ = make_ks_sequence(batch, obs_dim, dt, eval_horizon, device,
                                warp=warp, seed=999)
    B, Kp1, X = x_seq.shape
    with torch.enable_grad():
        z_seq = model.encoder(x_seq.reshape(B * Kp1, X)).reshape(B, Kp1, -1)
        traj = model.predictor.rollout(
            z_seq[:, 0], steps=eval_horizon, dt=dt,
            integrator=model.integrator, detach_steps=True, substeps=3,
        )
        finite = bool(torch.isfinite(traj).all().item())
        max_norm = traj.norm(dim=-1).max().item() if finite else float("inf")
        short = min(20, eval_horizon)
        align_short = (F.mse_loss(traj[:, 1:1 + short],
                                  z_seq[:, 1:1 + short].detach()).item()
                       if finite else float("inf"))
        if finite:
            Ppred, Ptrue = traj[0].detach(), z_seq[0].detach()
            hausd = _hausdorff(Ppred, Ptrue)
            chamf = _chamfer(Ppred, Ptrue)
            diam = torch.cdist(Ptrue, Ptrue).max().item() + 1e-8
            chamf_rel = chamf / diam
        else:
            hausd = chamf = chamf_rel = float("inf")
        lam_model = (rollout_lyapunov(model, z_seq[:, 0].detach(), dt, eval_horizon)
                     if finite else float("nan"))
        zpts = z_seq[:min(64, B), 0].detach()
        contr = latent_contraction_rate(model, zpts, dt) if finite else float("nan")
        r_mag = (model.predictor.diss_trace(traj).mean().item()
                 if (finite and model.predictor_type == "metriplectic") else None)
        pred_speed = ((traj[:, 1:] - traj[:, :-1]).norm(dim=-1).mean().item()
                      if finite else float("inf"))
        targ_speed = (z_seq[:, 1:] - z_seq[:, :-1]).norm(dim=-1).mean().item()

    true_lambda = true_lambda if true_lambda is not None else float("nan")
    true_div = true_div if true_div is not None else KS_TRUE_DIVERGENCE
    lam_err = abs(lam_model - true_lambda) if not math.isnan(lam_model) else float("nan")
    contr_err = abs(contr - true_div) if not math.isnan(contr) else float("nan")

    print(f"\n[{tag}] KS 200-step CHAOTIC rollout  (finite={finite}, "
          f"max||z||={max_norm:.3g})")
    print(f"[{tag}] short-{short} align MSE = {align_short:.4f}   "
          f"per-step motion: pred={pred_speed:.4f} target={targ_speed:.4f}")
    print(f"[{tag}] (1) attractor overlap: Hausdorff={hausd:.4f}  "
          f"Chamfer={chamf:.4f}  (rel={chamf_rel:.3f})")
    print(f"[{tag}] (2) Lyapunov: model lambda1={lam_model:+.4f}  "
          f"vs true={true_lambda:+.4f}  |err|={lam_err:.4f}")
    print(f"[{tag}] (3) contraction tr(J-I)/dt = {contr:+.4f}  "
          f"vs true div f={true_div:.4f}  |err|={contr_err:.4f}")
    if r_mag is not None:
        print(f"[{tag}]     dissipation engagement tr(R)/n = {r_mag:.5f}  "
              f"({'R ENGAGED' if r_mag > 1e-3 else 'R ~OFF'})")
    return {"tag": tag, "finite": finite, "max_norm": max_norm,
            "align_short": align_short, "hausdorff": hausd, "chamfer": chamf,
            "chamfer_rel": chamf_rel, "lam_model": lam_model,
            "lam_err": lam_err, "contraction": contr,
            "contraction_err": contr_err, "r_mag": r_mag,
            "pred_speed": pred_speed, "targ_speed": targ_speed,
            "true_lambda": true_lambda, "true_div": true_div}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train-horizon", type=int, default=6)
    ap.add_argument("--eval-horizon", type=int, default=200)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "results"))
    ap.add_argument("--warp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--arm", type=str, default=None,
                    help="restrict to one arm tag substring, e.g. E")
    ap.add_argument("--no-true-lam", action="store_true",
                    help="skip the Benettin lambda1 (slow) and use NaN truth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt, obs_dim, latent_dim = 0.01, 32, 64
    cfg = (device, obs_dim, latent_dim, dt, args.batch)
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"device={device}  torch={torch.__version__}  python={platform.python_version()}")
    print(f"KS: L={22.0}, {KS_MODES} complex modes ({2 * KS_MODES} real dof), "
          f"dt={dt}, warp={args.warp}")
    print(f"true div F = {KS_TRUE_DIVERGENCE:.4f} (closed form: "
          f"sum_j (j^2 k^2 - j^4 k^4), nonlinearity trace-free)")
    lam_true = float("nan")
    if not args.no_true_lam:
        lam_true = ks_lambda1(dt, 3000, device)
        print(f"empirical TRUE lambda1 (Benettin, dt={dt}) = {lam_true:+.4f}")

    arms = build_arms(device, obs_dim, latent_dim, dt, args.batch, lam_true)
    if args.arm:
        arms = [(t, m) for (t, m) in arms if args.arm.upper() in t]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    summary = []
    t_start = time.time()
    for tag, model in arms:
        for seed in seeds:
            torch.manual_seed(seed)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = Path(args.outdir) / f"ks_{tag.split('/')[0]}_seed{seed}.json"
            try:
                train_stress_ks(model, f"{tag} seed={seed}", args.train_horizon,
                                args.steps, cfg, warp=args.warp)
                r = ks_stress_eval(model, f"{tag} seed={seed}", cfg,
                                   eval_horizon=args.eval_horizon, warp=args.warp,
                                   true_lambda=lam_true)
                r["status"] = "ok"
            except Exception as exc:
                print(f"[{tag} seed={seed}] ARM FAILED: {type(exc).__name__}: {exc}")
                r = {"tag": tag, "seed": seed, "status": "failed",
                     "error": f"{type(exc).__name__}: {exc}",
                     "finite": False, "motion_ratio": float("nan"),
                     "lam_model": float("nan"), "lam_err": float("nan"),
                     "contraction": float("nan"), "contraction_err": float("nan"),
                     "r_mag": None, "chamfer": float("nan")}
            r["seed"] = seed
            r["steps"] = args.steps
            r["train_horizon"] = args.train_horizon
            r["wall_seconds"] = round(time.time() - t_start, 1)
            out_path.write_text(json.dumps(r, indent=2, default=str))
            summary.append(r)
            print(f"[saved {out_path.name}]")
            torch.cuda.empty_cache() if device == "cuda" else None

    out = {
        "meta": {
            "system": "kuramoto-sivashinsky",
            "L": 22.0, "modes": KS_MODES, "dt": dt,
            "true_divergence": KS_TRUE_DIVERGENCE,
            "true_lambda1": lam_true,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device, "torch": torch.__version__,
            "seeds": seeds, "steps_per_arm": args.steps,
            "train_horizon": args.train_horizon,
            "eval_horizon": args.eval_horizon, "warp": args.warp,
            "wall_seconds": round(time.time() - t_start, 1),
        },
        "arms": summary,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(args.outdir) / f"ks_benchmark_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
