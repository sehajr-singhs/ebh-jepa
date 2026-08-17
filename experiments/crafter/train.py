"""Train the compact DreamerV3-style agent on Crafter (or a hermetic fake env).

Usage:
    python train.py --predictor rssm          # DreamerV3 categorical baseline
    python train.py --predictor metriplectic  # EB-H-JEPA fixed-metriplectic prior
    python train.py --env fake --train-steps 300 --wm-updates 2    # CPU smoke
    python train.py --max-seconds 3300        # hard stop before a 1h budget

Writes results/crafter_<predictor>_<ts>.json (metrics) and, if crafter is
available, evaluation episode returns. Seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]          # repo root
sys.path.insert(0, str(ROOT / "experiments" / "crafter"))

from agent import CONFIG, ActorCritic, WorldModel  # noqa: E402

try:
    import crafter
    HAS_CRAFTER = True
except Exception:  # not installed (e.g., local Python 3.14): use fake env
    HAS_CRAFTER = False


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

class CrafterEnv:
    """Thin wrapper over crafter.Env (64x64x3 uint8, 17 discrete actions)."""

    def __init__(self, seed=0):
        self.env = crafter.Env(seed=seed)
        self.action_dim = 17

    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(np.asarray(obs, np.float32) / 255.0).permute(2, 0, 1)

    def step(self, action: int):
        obs, rew, done, info = self.env.step(int(action))
        obs = torch.from_numpy(np.asarray(obs, np.float32) / 255.0).permute(2, 0, 1)
        return obs, float(rew), bool(done), info


class FakeEnv:
    """Hermetic stand-in: moving bright square, reward for keeping it centered.
    Same interface as CrafterEnv so the pipeline is testable with no deps."""

    def __init__(self, seed=0, size=64):
        self.rng = random.Random(seed)
        self.size = size
        self.action_dim = 4
        self.pos = [size // 2, size // 2]
        self.vel = [0.0, 0.0]
        self.t = 0

    def reset(self):
        self.pos = [self.size // 2, self.size // 2]
        self.vel = [0.0, 0.0]
        self.t = 0
        return self._frame()

    def _frame(self):
        img = np.zeros((self.size, self.size, 3), np.float32)
        x, y = int(self.pos[0]), int(self.pos[1])
        img[max(0, y - 4):y + 4, max(0, x - 4):x + 4] = (1.0, 0.8, 0.2)
        return torch.from_numpy(img).permute(2, 0, 1)

    def step(self, action: int):
        # actions: 0 keep, 1 left, 2 right, 3 random jitter
        if action == 1:
            self.pos[0] -= 2.0
        elif action == 2:
            self.pos[0] += 2.0
        elif action == 3:
            self.pos[0] += self.rng.uniform(-3, 3)
        self.pos[0] = min(max(self.pos[0], 2), self.size - 2)
        self.vel[0] = 0.8 * self.vel[0] + self.rng.uniform(-0.5, 0.5)
        self.pos[0] += self.vel[0]
        # reward for staying in the middle band
        rew = 1.0 if self.size * 0.35 <= self.pos[0] <= self.size * 0.65 else 0.0
        self.t += 1
        done = self.t >= 40  # cap episodes so the buffer fills
        return self._frame(), float(rew), bool(done), {}


def make_env(name, seed=0):
    if name == "crafter":
        assert HAS_CRAFTER, "crafter not installed: pip install crafter"
        return CrafterEnv(seed)
    return FakeEnv(seed)


# ---------------------------------------------------------------------------
# Episode buffer
# ---------------------------------------------------------------------------

class EpisodeBuffer:
    def __init__(self, max_episodes=500):
        self.episodes = []
        self.max_episodes = max_episodes

    def start_episode(self):
        self.current = {"obs": [], "act": [], "rew": [], "done": []}

    def add(self, obs, act, rew, done):
        c = self.current
        c["obs"].append(obs); c["act"].append(act)
        c["rew"].append(rew); c["done"].append(done)

    def end_episode(self):
        if len(self.current["obs"]) >= 2:
            self.episodes.append(self.current)
            if len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)
        self.current = None

    def sample_batch(self, batch, T, device, action_dim):
        """(obs, acts, rews, conts): (T,B,3,H,W), (T,B,A), (T,B,1), (T,B,1)."""
        obs, acts, rews, conts = [], [], [], []
        for _ in range(batch):
            ep = random.choice(self.episodes)
            n = len(ep["obs"])
            # pick a start s in [0, n-T]; pad the head with the first frame
            s = random.randrange(0, max(1, n - T + 1))
            idx = list(range(s, min(s + T, n)))
            while len(idx) < T:
                idx = [idx[0]] + idx
            seg_obs = [ep["obs"][i] for i in idx]
            seg_act = [ep["act"][i] for i in idx]
            seg_rew = [ep["rew"][i] for i in idx]
            seg_done = [ep["done"][i] for i in idx]
            obs.append(torch.stack(seg_obs))
            acts.append(F.one_hot(torch.tensor(seg_act), action_dim).float())
            rews.append(torch.tensor(seg_rew).unsqueeze(-1))
            conts.append(torch.tensor([0.0 if d else 1.0 for d in seg_done]).unsqueeze(-1))
        return (torch.stack(obs).permute(1, 0, 2, 3, 4).to(device),
                torch.stack(acts).permute(1, 0, 2).to(device),
                torch.stack(rews).permute(1, 0, 2).to(device),
                torch.stack(conts).permute(1, 0, 2).to(device))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", choices=["rssm", "metriplectic"], default="rssm")
    ap.add_argument("--env", choices=["crafter", "fake"], default=None)
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=None, help="override WM batch")
    ap.add_argument("--seq-len", type=int, default=None, help="override WM seq len")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wm-updates", type=int, default=10,
                    help="WM+AC update batches per env step")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--audit-every", type=int, default=2000,
                    help="latent-stability audit interval (0 = off)")
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="hard stop (0 = no budget). 3300 fits inside 1h.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args()

    env_name = args.env or ("crafter" if HAS_CRAFTER else "fake")
    if args.env == "crafter" and not HAS_CRAFTER:
        raise SystemExit("crafter not installed: pip install crafter")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    cfg = {**CONFIG, "seed": args.seed}
    if args.batch:
        cfg["batch"] = args.batch
    if args.seq_len:
        cfg["seq_len"] = args.seq_len
    cfg["action_dim"] = make_env(env_name).action_dim  # 17 or 4
    print(f"[{env_name}] predictor={args.predictor} device={device} "
          f"action_dim={cfg['action_dim']} crafter={HAS_CRAFTER}")

    wm = WorldModel(predictor=args.predictor, config=cfg).to(device)
    ac = ActorCritic(wm).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg["wm_lr"])
    ac_opt = torch.optim.Adam(ac.actor.parameters(), lr=cfg["actor_lr"])
    cr_opt = torch.optim.Adam(ac.critic.parameters(), lr=cfg["critic_lr"])

    buffer = EpisodeBuffer()
    env = make_env(env_name, seed=args.seed)

    metrics = {"predictor": args.predictor, "env": env_name, "config": cfg,
               "steps": [], "eval_return": [], "wm_rec": [], "wm_kl": [],
               "reward": [], "actor_loss": [], "critic_loss": [],
               "stability": []}
    t_start = time.time()
    step = 0
    obs = env.reset()
    buffer.start_episode()
    train_ret = 0.0
    train_ret_history = []
    h = torch.zeros(1, cfg["gru_dim"], device=device)   # running latent state
    z_prev = None

    def act_one(obs_t, h, z_prev, sample=True):
        """Encode, posterior-sample the latent, pick an action, advance GRU.
        Returns (action, h_next, z)."""
        with torch.no_grad():
            feat = wm.encoder(obs_t)
            if wm.predictor_type == "rssm":
                z, _ = wm.prior.sample(h, feat)
            else:
                z = wm.prior.posterior_sample(h, feat, sample=sample)
            a, _ = ac.act(z, h, sample=sample)
            a_i = int(a.item())
            h_next = wm.gru(wm.z_in(torch.cat(
                [F.one_hot(torch.tensor([a_i], device=device),
                           cfg["action_dim"]).float(), z], -1)), h)
        return a_i, h_next, z

    def finish():
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = Path(args.outdir) / f"crafter_{args.predictor}_{stamp}.json"
        metrics["wall_seconds"] = round(time.time() - t_start, 1)
        path.write_text(json.dumps(metrics, indent=2, default=str))
        print(f"\nmetrics -> {path}")
        if train_ret_history:
            print(f"final mean train return (last 10 eps): "
                  f"{np.mean(train_ret_history[-10:]):.2f}")
        raise SystemExit(0)

    try:
        while step < args.train_steps:
            # ---- act -------------------------------------------------------
            if step < args.warmup:
                action = random.randrange(cfg["action_dim"])
                h = torch.zeros(1, cfg["gru_dim"], device=device)
                z_prev = None
            else:
                action, h, z_prev = act_one(obs.unsqueeze(0), h, z_prev,
                                            sample=step % 20 != 0)
            new_obs, rew, done, _ = env.step(action)
            # store (obs_t, a_t, r_t, d_t); the next entry is obs_{t+1}
            buffer.add(obs, action, rew, done)
            obs = new_obs
            train_ret += rew
            step += 1

            if done:
                buffer.end_episode()
                train_ret_history.append(train_ret)
                train_ret = 0.0
                obs = env.reset()
                buffer.start_episode()
                h = torch.zeros(1, cfg["gru_dim"], device=device)
                z_prev = None

            # ---- train -----------------------------------------------------
            if step >= args.warmup and len(buffer.episodes) >= 2 and \
                    step % max(1, args.train_steps // 50) == 0:
                for _ in range(args.wm_updates):
                    obs_b, act_b, rew_b, cont_b = buffer.sample_batch(
                        cfg["batch"], cfg["seq_len"], device, cfg["action_dim"])
                    wm_loss = wm.compute_loss(obs_b, act_b, rew_b, cont_b)
                    wm_opt.zero_grad(set_to_none=True)
                    wm_loss["total"].backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg["grad_clip"])
                    wm_opt.step()
                    # imagination actor-critic (single backward: both losses
                    # share the imagination graph)
                    states = ac_states(wm, buffer, cfg, device)
                    il = ac.imagine_loss(states, train=True)
                    ac_opt.zero_grad(set_to_none=True)
                    cr_opt.zero_grad(set_to_none=True)
                    (il["actor"] + il["critic"]).backward()
                    ac_opt.step()
                    cr_opt.step()
                    metrics["steps"].append(step)
                    metrics["wm_rec"].append(wm_loss["rec"].item())
                    metrics["wm_kl"].append(wm_loss["kl"].item())
                    metrics["reward"].append(wm_loss["reward"].item())
                    metrics["actor_loss"].append(il["actor"].item())
                    metrics["critic_loss"].append(il["critic"].item())
                if len(metrics["steps"]) % 10 == 0:
                    print(f"step {step:6d} rec={wm_loss['rec'].item():.4f} "
                          f"kl={wm_loss['kl'].item():.3f} "
                          f"rew={wm_loss['reward'].item():.4f} "
                          f"actor={il['actor'].item():.3f} "
                          f"critic={il['critic'].item():.3f} "
                          f"[{time.time()-t_start:.0f}s]")

            # ---- latent-stability audit -------------------------------------
            if args.audit_every and step % args.audit_every == 0 and step > 0 \
                    and len(buffer.episodes) >= 2:
                aud = stability_audit(wm, buffer, cfg, device)
                aud["step"] = step
                metrics["stability"].append(aud)
                print(f"  audit @ {step}: drift={aud['drift']:.4f} "
                      f"kl_raw={aud['kl_raw']:.3f} norm={aud['latent_norm']:.3f}"
                      + (f" diss={aud['dissipation']:.4f}"
                         if "dissipation" in aud else ""))

            # ---- eval + budget --------------------------------------------
            if step % args.eval_every == 0 and step > 0:
                ret = evaluate(wm, ac, env_name, args.eval_episodes, device, cfg)
                metrics["eval_return"].append((step, ret))
                print(f"  eval @ {step}: mean return {ret:.2f}")
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                print(f"time budget ({args.max_seconds}s) reached at step {step}")
                finish()
    except KeyboardInterrupt:
        pass
    finish()


def stability_audit(wm, buffer, cfg, device, T=8, B=8):
    """Latent-stability metrics for one batch, no_grad.

    The DreamerV3 latent-stability question: how far does the learned
    PRIOR drift from where the POSTERIOR lands after the real observation
    arrives? Reports, per timestep averaged over the batch:
      drift       mean squared distance prior-sample vs posterior latent
      kl_raw      prior/posterior KL WITHOUT free-bit clamping (predictive
                  error the free-bit loss hides)
      latent_norm mean squared latent norm (boundedness of the predictor)
      dissipation mean tr(R)/dim for the metriplectic arm (does dissipation
                  engage inside the RL loop — mirrors the Lorenz atlas)
    """
    obs_b, act_b, _, _ = buffer.sample_batch(B, T, device, cfg["action_dim"])
    with torch.no_grad():
        zs, _ = wm.observe(obs_b, act_b)   # posterior latents (T,B,D)
        feat = wm.encoder(obs_b.reshape(-1, 3, cfg["obs_size"], cfg["obs_size"])
                          ).reshape(T, B, -1)
        h = torch.zeros(B, cfg["gru_dim"], device=device)
        z_prev = None
        drift = kl_raw = norm = diss = 0.0
        for t in range(T):
            if wm.predictor_type == "rssm":
                p = torch.distributions.Categorical(logits=wm.prior.prior(h)
                                                    .view(-1, wm.prior.codes, wm.prior.classes))
                q = torch.distributions.Categorical(logits=wm.prior.posterior(
                    torch.cat([h, feat[t]], -1))
                    .view(-1, wm.prior.codes, wm.prior.classes))
                kl_raw += torch.distributions.kl.kl_divergence(q, p).sum(-1).mean()
                z, _ = wm.prior.sample(h, detach_grad=True)  # pure prior
            else:
                mu_p = wm.prior.step_from(z_prev, h, detach_grad=True)
                var_p = torch.exp(wm.prior.logvar).expand_as(mu_p)
                mu_q, lv_q = wm.prior.posterior_params(h, feat[t])
                var_q = torch.exp(lv_q)
                kl_raw += (0.5 * (lv_q - wm.prior.logvar.unsqueeze(0)
                                  + (var_p + (mu_p - mu_q).square()) / var_q
                                  - 1.0)).sum(-1).mean()
                z = wm.prior.posterior_sample(h, feat[t])
                R = wm.prior._R(z, h)
                diss += (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                         / wm.prior.z_dim).mean()
            drift += (z - zs[t]).square().mean(-1).mean()
            norm += z.square().mean(-1).mean()
            z_prev = z
            h = wm.gru(wm.z_in(torch.cat([act_b[t], z], -1)), h)
        out = dict(drift=float((drift / T).item()),
                   kl_raw=float((kl_raw / T).item()),
                   latent_norm=float((norm / T).item()))
        if wm.predictor_type == "metriplectic":
            out["dissipation"] = float((diss / T).item())
        return out


def ac_states(wm, buffer, cfg, device):
    """Random starting latent states for imagination, from the buffer."""
    B = cfg["imag_batch"]
    if wm.predictor_type == "rssm":
        return (torch.zeros(B, cfg["gru_dim"], device=device), None)
    # metriplectic: start from random continuous latents
    z0 = torch.randn(B, wm._zdim(), device=device) * 0.3
    return (torch.zeros(B, cfg["gru_dim"], device=device), z0)


def evaluate(wm, ac, env_name, episodes, device, cfg):
    env = make_env(env_name, seed=999)
    returns = []
    for _ in range(episodes):
        obs = env.reset().unsqueeze(0).to(device)
        h = torch.zeros(1, cfg["gru_dim"], device=device)
        z_prev = None
        ret = 0.0
        done = False
        for _ in range(1000):
            with torch.no_grad():
                feat = wm.encoder(obs)
                if wm.predictor_type == "rssm":
                    z, _ = wm.prior.sample(h, feat, sample=False)
                else:
                    z = wm.prior.posterior_sample(h, feat, sample=False)
                a, _ = ac.act(z, h, sample=False)
                a_i = int(a.item())
                h = wm.gru(wm.z_in(torch.cat(
                    [F.one_hot(torch.tensor([a_i], device=device),
                               cfg["action_dim"]).float(), z], -1)), h)
            obs, rew, done, _ = env.step(a_i)
            obs = obs.unsqueeze(0).to(device)
            ret += rew
            if done:
                break
        returns.append(ret)
    return float(np.mean(returns))


if __name__ == "__main__":
    main()
