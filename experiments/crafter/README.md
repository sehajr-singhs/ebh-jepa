# Crafter experiment: does physical structure pay off in an RL world model?

A controlled test of the Part-1 question in a *real* model-based RL loop.
Two arms of an otherwise identical compact DreamerV3-style agent differ in
exactly one component — the stochastic prior of the RSSM:

| arm | prior | notes |
|---|---|---|
| `rssm` | DreamerV3 categorical RSSM (32×32 codes) | the baseline |
| `metriplectic` | Gaussian mean = fixed-metriplectic map `z' = z + dt(J∇H − R∇S)` | the E-arm from the paper |

Everything else — CNN encoder/decoder, GRU memory, KL free bits, symlog
reward/continue heads, TD(λ) imagination, actor-critic — is byte-identical.

## Run

```bash
# Real Crafter env on GPU (this is the paper's protocol):
python train.py --predictor rssm         --env crafter --train-steps 20000
python train.py --predictor metriplectic --env crafter --train-steps 20000

# Hermetic CPU smoke (no crafter installed — synthetic image env):
python train.py --env fake --predictor metriplectic --train-steps 120 --warmup 30 \
    --batch 4 --seq-len 8 --wm-updates 1 --max-seconds 240
```

Each run writes `results/crafter_<arm>_<ts>.json` with per-update losses and
`(step, eval_return)` pairs.

## Scale-up

The 1-hour T4 protocol (both arms, hard budget) is one click away:
see [`../../kaggle/README.md`](../../kaggle/README.md).

## Design honesty

- The metriplectic arm is the **E-arm**: constant divergence-free `J`
  (closes the learned-skew loophole) and `R = LLᵀ` with `L` init `0.05`
  (breaks the dead saddle). These are the two fixes the Lorenz atlas shows
  actually make dissipation engage.
- The baseline is **DreamerV3's exact categorical prior** — not a strawman.
- The agent is small (this is a 1-hour study, not a 10-day run); the paper
  claims a *controlled comparison*, not state-of-the-art scores.
