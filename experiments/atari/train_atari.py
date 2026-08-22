"""Train the compact DreamerV3-style agent on Atari 100k games.

Standard protocol (Hafner et al., 2023):
  - 84x84 grayscale, frame skip 4, sticky actions (p=0.25)
  - 100k environment steps per game (400k frames)
  - Two arms: RSSM baseline vs. metriplectic prior
  - Reward clipping: sign(reward)

Usage:
    python train_atari.py --game Pong --predictor rssm
    python train_atari.py --game Breakout --predictor metriplectic --train-steps 100000

Writes results/atari_<game>_<predictor>_<ts>.json.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "crafter"))
sys.path.insert(0, str(ROOT / "experiments" / "atari"))

from agent import ActorCritic, WorldModel  # noqa: E402
from atari_env import make_atari, ATARI_100K_GAMES  # noqa: E402

# Atari-specific CONFIG: 84x84 grayscale, variable action dim
ATARI_CONFIG = dict(
    obs_size=84,
    in_channels=1,  # grayscale
    cnn_channels=(32, 64, 128, 256),
    feat_dim=512,
    gru_dim=256,
    z_classes=32,
    z_codes=32,
    z_dim=64,
    batch=16,
    seq_len=50,        # longer sequences for Atari (temporal structure)
    imag_horizon=15,
    imag_batch=8,
    wm_lr=3e-4,        # higher LR for faster convergence on Atari
    actor_lr=1e-4,
    critic_lr=1e-4,
    kl_free_bits_total=32.0,
    grad_clip=100.0,
    gamma=0.997,
    lambd=0.95,
)


class EpisodeBuffer:
    """Episode buffer for Atari (same structure as Crafter but stores uint8)."""

    def __init__(self, max_episodes=500):
        self.episodes = []
        self.max_episodes = max_episodes

    def start_episode(self):
        self.current = {"obs": [], "act": [], "rew": [], "done": []}

    def add(self, obs, act, rew, done):
        self.current["obs"].append(obs)
        self.current["act"].append(act)
        self.current["rew"].append(float(rew))
        self.current["done"].append(bool(done))

    def end_episode(self):
        if len(self.current["obs"]) >= 2:
            self.episodes.append(self.current)
            if len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)
        self.current = None

    def sample_batch(self, batch, T, device, action_dim):
        obs, acts, rews, conts = [], [], [], []
        for _ in range(batch):
            ep = random.choice(self.episodes)
            n = len(ep["obs"])
            s = random.randrange(0, max(1, n - T + 1))
            idx = list(range(s, min(s + T, n)))
            while len(idx) < T:
                idx = [idx[0]] + idx
            obs.append(torch.stack([ep["obs"][i] for i in idx]))
            acts.append(F.one_hot(torch.tensor([ep["act"][i] for i in idx]), action_dim).float())
            rews.append(torch.tensor([ep["rew"][i] for i in idx]).unsqueeze(-1))
            conts.append(torch.tensor([0.0 if ep["done"][i] else 1.0 for i in idx]).unsqueeze(-1))
        return (torch.stack(obs).permute(1, 0, 2, 3, 4).to(device),
                torch.stack(acts).permute(1, 0, 2).to(device),
                torch.stack(rews).permute(1, 0, 2).to(device),
                torch.stack(conts).permute(1, 0, 2).to(device))


def make_env(game, seed=0):
    return make_atari(game, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="Atari game name (e.g., Pong)")
    ap.add_argument("--predictor", choices=["rssm", "metriplectic"], default="rssm")
    ap.add_argument("--train-steps", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wm-updates", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--audit-every", type=int, default=5000)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args()

    game = args.game
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    cfg = {**ATARI_CONFIG, "seed": args.seed}

    env = make_env(game, seed=args.seed)
    cfg["action_dim"] = env.action_space.n
    print(f"[{game}] predictor={args.predictor} device={device} "
          f"action_dim={cfg['action_dim']}")

    wm = WorldModel(predictor=args.predictor, config=cfg).to(device)
    ac = ActorCritic(wm).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg["wm_lr"])
    ac_opt = torch.optim.Adam(ac.actor.parameters(), lr=cfg["actor_lr"])
    cr_opt = torch.optim.Adam(ac.critic.parameters(), lr=cfg["critic_lr"])

    buffer = EpisodeBuffer()
    metrics = {
        "predictor": args.predictor, "game": game, "config": cfg,
        "steps": [], "eval_return": [], "wm_rec": [], "wm_kl": [],
        "reward": [], "actor_loss": [], "critic_loss": [], "stability": [],
    }
    t_start = time.time()
    step = 0
    obs, _ = env.reset()
    buffer.start_episode()
    train_ret = 0.0
    train_ret_history = []
    h = torch.zeros(1, cfg["gru_dim"], device=device)
    z_prev = None

    def act_one(obs_t, h, z_prev, sample=True):
        with torch.no_grad():
            feat = wm.encoder(obs_t)
            if wm.predictor_type == "rssm":
                z, _ = wm.prior.sample(h, feat)
            else:
                z = wm.prior.posterior_sample(h, feat, sample=sample)
            a, _ = ac.act(z, h, sample=sample)
            a_i = int(a.item())
            h_next = wm.gru(wm.z_in(torch.cat([
                F.one_hot(torch.tensor([a_i], device=device), cfg["action_dim"]).float(), z], -1)), h)
        return a_i, h_next, z

    def finish():
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = Path(args.outdir) / f"atari_{game}_{args.predictor}_{stamp}.json"
        metrics["wall_seconds"] = round(time.time() - t_start, 1)
        path.write_text(json.dumps(metrics, indent=2, default=str))
        print(f"\nmetrics -> {path}")
        if train_ret_history:
            print(f"final mean train return (last 10 eps): {np.mean(train_ret_history[-10:]):.2f}")
        raise SystemExit(0)

    try:
        while step < args.train_steps:
            if step < args.warmup:
                action = random.randrange(cfg["action_dim"])
                h = torch.zeros(1, cfg["gru_dim"], device=device)
                z_prev = None
            else:
                action, h, z_prev = act_one(obs.unsqueeze(0), h, z_prev,
                                            sample=step % 20 != 0)

            new_obs, rew, done, truncated, _ = env.step(action)
            rew_clipped = float(np.sign(rew))  # sign reward clipping
            buffer.add(obs, action, rew_clipped, done or truncated)
            obs = new_obs
            train_ret += rew  # unclipped for logging
            step += 1

            if done or truncated:
                buffer.end_episode()
                train_ret_history.append(train_ret)
                train_ret = 0.0
                obs, _ = env.reset()
                buffer.start_episode()
                h = torch.zeros(1, cfg["gru_dim"], device=device)
                z_prev = None

            # Train world model + actor-critic
            if step >= args.warmup and len(buffer.episodes) >= 2 and \
                    step % 10 == 0:
                for _ in range(args.wm_updates):
                    wm.train()
                    obs_b, acts_b, rews_b, conts_b = buffer.sample_batch(
                        cfg["batch"], cfg["seq_len"], device, cfg["action_dim"])
                    wm_out = wm(obs_b)
                    # World model loss
                    wm_loss = (wm_out["rec_loss"] + wm_out["kl_loss"] +
                               wm_out["reward_loss"] + wm_out["continue_loss"])
                    wm_opt.zero_grad(set_to_none=True)
                    wm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg["grad_clip"])
                    wm_opt.step()

                # Actor-critic imagination
                wm.eval()
                with torch.no_grad():
                    z0 = wm.prior.sample(h, wm.encoder(obs.unsqueeze(0)))[0] \
                         if wm.predictor_type == "rssm" \
                         else wm.prior.posterior_sample(h, wm.encoder(obs.unsqueeze(0)), sample=True)
                imag = wm.predictor.rollout(z0, cfg["imag_horizon"], wm.dt,
                                            integrator=wm.integrator,
                                            detach_steps=True)
                ac_out = ac.imagine_loss(imag, wm)
                ac_opt.zero_grad(set_to_none=True)
                ac_out["actor_loss"].backward()
                torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), 100.0)
                ac_opt.step()

                cr_opt.zero_grad(set_to_none=True)
                ac_out["critic_loss"].backward()
                torch.nn.utils.clip_grad_norm_(ac.critic.parameters(), 100.0)
                cr_opt.step()

            # Log metrics
            if step % 1000 == 0:
                metrics["steps"].append(step)
                metrics["reward"].append(train_ret_history[-1] if train_ret_history else 0.0)
                metrics["actor_loss"].append(float(ac_out.get("actor_loss", 0)))
                metrics["critic_loss"].append(float(ac_out.get("critic_loss", 0)))
                print(f"  step {step:>6d}/{args.train_steps} | "
                      f"eps {len(train_ret_history)} | "
                      f"last_ret {train_ret_history[-1]:.1f}" if train_ret_history else "")

            # Hard time budget
            if args.max_seconds > 0 and time.time() - t_start > args.max_seconds:
                print(f"\nTIME LIMIT ({args.max_seconds}s) reached at step {step}")
                finish()

    except KeyboardInterrupt:
        finish()

    finish()


if __name__ == "__main__":
    main()
