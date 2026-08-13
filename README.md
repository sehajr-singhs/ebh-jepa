# EB-H-JEPA — Energy-Based Hamiltonian Latent World Models

**Every constraint class fails — on a property disjoint from the one it enforces.**

A five-arm failure atlas for the claim that imposing physical structure on a
JEPA world-model predictor yields physically faithful latent dynamics. We
train structurally distinct latent predictors — unconstrained MLP, rigid
Hamiltonian, naive metriplectic, fixed (divergence-gated) metriplectic, and
spectrally aligned metriplectic — on the nonlinearly lifted Lorenz-63
attractor, and audit each under a chaos-aware protocol: rollout boundedness,
leading Lyapunov exponent, phase-space contraction, attractor overlap, and
dissipation engagement.

> **Why this matters for world models.** JEPA-style world models (LeCun 2022;
> Assran 2023; Garrido 2024) and model-based RL (Hafner 2023 DreamerV3;
> Hafner 2024 DreamerV4; Bruce 2024 Genie) all bet that a latent predictor can
> learn dynamics that stay stable and meaningful over long horizons. This
> project isolates *which structural bets actually engage* and *which ones
> silently fail* — with a reproducible benchmark, not vibes.

## The verdict (real numbers, 3 seeds)

Benchmark: Lorenz-63 under a nonlinear diffeomorphic lift into a 64-d latent
space. True divergence `div f = −13.667` (sum of Lyapunov spectrum); true
largest exponent `λ₁ ≈ 0.9`. Full per-seed traces in [`results/`](results/).

| arm | rollout finite | motion ratio | model λ₁ | \|λ₁ − 0.9\| | contraction | tr(R)/n |
|---|---|---|---|---|---|---|
| B unconstrained MLP | sometimes | — | explodes | — | — | — |
| C rigid Hamiltonian | yes | ~0 (frozen) | ~0 | — | ≈ 0 (can't dissipate) | — |
| D naive metriplectic | yes | low | overshoot | — | wrong | 0.00000 (dead saddle) |
| E fixed metriplectic | yes | 1.1–2.9 | 1.0–4.3 | 0.3–3.4 | **−13.80 ± 0.17** | 2.3–6.2 |
| F + spectral alignment | yes | 1.2–3.8 | 1.4–5.0 | 0.5–4.1 | −13.25 ± 0.19 | 2.5–8.3 |

Reading the table honestly:

- **The dissipation gate works.** Once the two structural bugs are fixed
  (constant divergence-free J closes the skew loophole; nonzero LLᵀ init
  breaks the dead saddle), R genuinely engages and the learned phase-space
  contraction matches the true value to ~1% (E: −13.80 ± 0.17 vs −13.667).
- **Spectral shaping does not.** Pinning the *sum* of the spectrum cannot pin
  any individual exponent: λ₁ still runs 1.5–4× high in every seed. The
  differentiable QR proxy shapes only a finite-horizon statistic, not the
  asymptotic Oseledets exponent.
- **Overlap scores reward dead models.** A frozen predictor sits inside the
  encoded attractor and posts the *best* Chamfer/Hausdorff overlap while
  being dynamically dead. Geometric overlap must be gated on the rollout
  being alive and chaotic (λ₁ > 0 and sane motion ratio), or replaced by a
  motion-invariant statistic.

The full mechanistic analysis (five isolated exploits, each annotated at its
code site) lives in the module docstring: `src/ebhjepa/ebhjepa.py`, section
*STRUCTURAL FAILURE ATLAS*.

## Reproduce

```bash
pip install -r requirements.txt

# Full benchmark (5 arms × 250 steps). ~15 min on one GPU, ~90 min on CPU.
python benchmarks/run_benchmark.py --steps 250

# Fast smoke run (pipeline check, ~5 min CPU):
python benchmarks/run_benchmark.py --steps 40 --batch 32

# Tests:
python -m pytest tests/ -q
```

Every run writes `results/benchmark_<timestamp>.json` with full metadata
(device, torch version, seed, wall time) so paper claims trace to a concrete
run. Recorded 3-seed evidence from the paper is in `results/`.

## The constructive experiment (new): Crafter sample efficiency

The Lorenz atlas is a negative result; the constructive question is whether
the E-arm fixes pay off inside a real RL world model. `experiments/crafter/`
contains a compact DreamerV3-style agent whose RSSM stochastic prior is
pluggable: **DreamerV3's categorical RSSM (baseline)** vs. **the
fixed-metriplectic map (treatment)** — identical everything else. Both arms
run on Crafter (Hafner 2021), the benchmark of the world-model group that
introduced RSSM. Either outcome is publishable: structure helping (first
positive evidence for physical priors in an RL world model) or hurting (the
honest negative the field needs).

A one-hour T4 protocol (both arms, hard budget) is ready in two flavors:
- [`colab/`](colab/README.md) — **zero setup**: upload the ready notebook,
  pick T4 GPU, Run all (~55 min), results land in Google Drive.
- [`kaggle/`](kaggle/README.md) — same protocol after a 2-minute
  `kaggle.json` setup.

## Repository layout

```
src/ebhjepa/ebhjepa.py    # the entire method: encoders, Hamiltonian /
                          #   metriplectic predictors, SIGReg anti-collapse,
                          #   chaos-aware evaluation (single-file, like
                          #   DreamerV3's dreamer.py)
benchmarks/               # reproducible 5-arm runner -> results JSON
experiments/crafter/      # DreamerV3-style agent, pluggable predictor,
                          #   for the Crafter sample-efficiency test
colab/                    # zero-setup 1-hour T4 notebook (free GPU)
kaggle/                   # same 1-hour run via Kaggle (free GPU)
tests/                    # fast CPU smoke tests
paper/                    # manuscript drafts v1 (NMI-Letters, IEEE Trans.)
papers/neurips2026/       # consolidated paper v2 (NeurIPS format)
docs/figs/                # paper figures
docs/OUTREACH.md          # why this project, who it's for, next steps
```

## Papers

- `paper/nmi_letters.tex` — Nature Machine Intelligence (Letters) format,
  149-word abstract, 3.4k-word main text, 5 display items. *(v1 draft)*
- `paper/ieee_transactions.tex` — IEEE Transactions format, 10 pp. *(v1 draft)*
- `papers/neurips2026/neurips2026.tex` — consolidated v2: the failure atlas
  plus the Crafter controlled experiment (Table 2 filled by the T4 run).

Author: Sehaj Randhir Singh (independent researcher; partial affiliation with
NYU Tandon School of Engineering).

## Honest framing

Lorenz is not a *generic* Hamiltonian/GENERIC system; in this setup H, S and
R are learned proxies in a latent chart, not physical energy and entropy. We
claim on-attractor stability and invariant recovery — **not** recovery of true
thermodynamics. That is precisely the point: if the constraints can't even
recover Lorenz's invariants here, they will not be recovered by adding
structure alone in a real world model.

## License

MIT — see [LICENSE](LICENSE).
