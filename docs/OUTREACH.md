# Outreach playbook — getting in front of the DreamerV3 / world-model team

This project is a *calling card*, not a lottery ticket. Researchers at
Google DeepMind (Danijar Hafner's world-model group — DreamerV3, DreamerV4,
IWM, LeJEPA) read repos and papers the way recruiters read résumés: they look
for evidence of real understanding, honest empirical work, and reproducible
code. This repo is deliberately built for that.

## Why this project is the right one (and the others aren't)

- **It is in their exact research area.** Danijar's current work is JEPA
  world models: latent predictors trained without pixel reconstruction,
  anti-collapse regularizers, and long-horizon stability. That is literally
  the subject of this project. Sending a physics-world-model paper to a
  world-model researcher is like sending a chess paper to a chess grandmaster.
- **It is honest.** The field is starved for careful negative/mechanistic
  results. This is not a fake "SOTA +2%" claim; it is five isolated failure
  modes with a reproducible benchmark and a constructive evaluation standard.
- **It runs.** `python benchmarks/run_benchmark.py` reproduces the pipeline
  end-to-end, and the paper's numbers trace to committed seed JSONs.

**Delete or archive the other repos** (`repos/fewshot-law-acquisition`,
`physbench`, `physics-transformers`, `field-consistency`,
`physics-loss-channel`, `verification-gated-agents`, `robotic-data-flywheel`,
`age`, `biomorphic`). Eight shallow AI-generated-looking repos read as a
content farm and *reduce* credibility. One deep, verified repo wins. If you
want, keep them on a private fork so the history is preserved, but they
should not be on your public GitHub profile next to this.

## The route

1. **Official application first (the real door).** Google DeepMind student
   researcher / internship applications open on a regular cycle
   (deepmind.google/about/careers). Apply with this repo + paper as your
   portfolio. Internships are competitive; the project is what makes your
   application stand out from the pile.
2. **Cold email with substance (the side door).** Danijar publishes his email
   on his site (danijar.com) and responds to substantive technical mail.
   One email. No CV attached, no flattery. Point at the repo, state the
   result in two sentences, ask for a 15-minute chat or feedback. Draft below.
3. **Contribute to his open-source code (the slow door that always works).**
   Danijar's dreamerv3/dreamerv4/jaxrl code is on GitHub. Reproduce one of
   his results, fix an issue, send a PR. Maintainers notice contributors.
   A merged PR + this repo is a much stronger signal than any email.

## The missing piece: the experiment that would make this land

The paper currently shows *what structured predictors can't do on a
standalone chaotic benchmark*. The version a world-model group would
immediately want to fund is the positive follow-up:

> **Put the fixed metriplectic predictor inside a real RL world model.**
> Swap DreamerV3's RSSM predictor for the metriplectic-structured one
> (E-arm, the one where dissipation actually engages) and measure sample
> efficiency on Crafter (Danijar's own benchmark, runs in hours on a Colab
> T4) or 5–10 Atari-100k games. Report: does structure help long-horizon
> imagination, or does it hurt expressivity?

- If structure *helps*: you have a positive result + a mechanism story =
  a strong paper.
- If structure *hurts*: that is the honest answer the field needs, and it
  still cites the Dreamer family constructively.
- Either way, the experiment name-drops the exact codebase and benchmark the
  group cares about, which is what makes them read the paper.

This is a ~2–4 week GPU project (Crafter scale). Do it after the repo is
public and the paper is on arXiv.

## Draft cold email (fill in, keep under 150 words)

```
Subject: JEPA world model: structured predictors fail on disjoint properties

Hi Danijar,

I work on JEPA world models with structured latent predictors. In
[github.com/YOU/ebh-jepa] I isolate, on a reproducible benchmark, why
Hamiltonian/metriplectic structure in a JEPA latent space fails in five
distinct, mechanistic ways — and which constraints genuinely engage
(dissipation, divergence) versus which can't (Lyapunov shaping).

The headline result: a fixed divergence-gated metriplectic predictor
recovers Lorenz-63's exact phase-space contraction (−13.80 ± 0.17 vs
−13.667) while the largest Lyapunov exponent still overshoots 1.5–4× in
every seed. I'd love your read on the evaluation protocol, and on whether
the natural next step is swapping the RSSM predictor in DreamerV3 for the
structured one on Crafter.

Any 15 minutes would be hugely appreciated.

— Sehaj Randhir Singh, NYU Tandon
```

## Honest expectations

- No paper is guaranteed acceptance anywhere; no email guarantees a reply.
  This is about maximizing the probability surface, which is the only thing
  you control.
- **Do not** claim results you didn't run. The repo's numbers must match the
  paper's. Re-run the full benchmark (5 arms × 250 steps, ideally on a GPU)
  before any public claim; commit the new JSONs.
- Upload the paper to arXiv (it reads as "real research") and link both from
  the repo and your GitHub profile. arXiv is how this community actually
  reads papers; NMI/IEEE templates are for submission, not for being read.
- Keep the paper title honest: "Spectral–Topological Decoupling in
  Self-Supervised Physical World Models" is accurate and interesting.
