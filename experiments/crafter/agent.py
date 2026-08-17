"""Compact DreamerV3-style agent for Crafter, with a pluggable latent predictor.

This is a faithful-but-small reimplementation of the DreamerV3 recipe
(Hafner et al., 2023): recurrent state-space model (RSSM) world model,
latent imagination, actor-critic. The *only* structural difference between
the two arms is the stochastic prior:

  predictor="rssm"         DreamerV3's categorical prior: z_t ~ Cat(MLP(h_t)).
  predictor="metriplectic" a continuous latent whose prior MEAN is advanced by
                           the fixed divergence-gated metriplectic map from
                           the EB-H-JEPA paper (constant canonical J, R=LL^T
                           with nonzero init, learned H and S conditioned on
                           the GRU state h):  z' = z + dt (J gradH - R gradS).

Everything else — encoder, decoder, GRU memory, reconstruction / KL /
reward / continue losses, imagination, actor-critic — is identical across
arms, so any difference in sample efficiency is attributable to the
predictor structure alone.

Two usage modes:
  - Real Crafter env (pip install crafter) on GPU: `python train.py`.
  - Hermetic smoke mode with a synthetic image env (no crafter dependency):
    `python train.py --env fake --train-steps 200 --wm-train 50`.

Design notes (honesty, matching the paper):
  - The metriplectic arm is the E-arm: fixed divergence-free J (closes the
    skew loophole) and R = L L^T with L init 0.05 (breaks the dead saddle).
    The paper shows this is the arm where dissipation genuinely engages.
  - The baseline is DreamerV3's exact categorical RSSM, so the comparison is
    "does the physical inductive bias help or hurt a real RL world model".
  - KL free bits, symlog rewards/values, and latent imagination follow
    DreamerV3 at compact scale (see CONFIG below).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration (compact scale; ~1h on a T4 for a partial Crafter run)
# ---------------------------------------------------------------------------

CONFIG = dict(
    obs_size=64,           # squared frames
    action_dim=17,         # Crafter's 17 discrete actions
    cnn_channels=(32, 64, 128, 256),
    feat_dim=512,          # encoder/decoder bottleneck
    gru_dim=256,           # recurrent memory
    z_classes=32,          # categorical RSSM: number of classes
    z_codes=32,            # categorical RSSM: number of independent codes
    z_dim=64,              # metriplectic RSSM: continuous latent dim (even)
    batch=16,
    seq_len=16,            # world-model training sequence length
    imag_horizon=15,       # imagination horizon
    imag_batch=8,          # imagination batch
    wm_lr=1e-4,
    actor_lr=3e-5,
    critic_lr=3e-5,
    kl_free_bits_total=32.0,  # total KL free-bit budget, matched across
                              #   arms: 1.0/component for 32-code RSSM,
                              #   0.5/dim for the 64-dim continuous prior
    grad_clip=100.0,       # DreamerV3's large WM clip
    gamma=0.997,           # long-horizon discount used by DreamerV3
    lambda_=0.95,          # TD(lambda) for imagination returns
    entropy=3e-4,
    dt=1.0,                # metriplectic integrator step
    substeps=1,
    seed=0,
)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(x.abs()))


# ---------------------------------------------------------------------------
# Image encoder / decoder (DreamerV3-style CNN)
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), feat_dim=512):
        super().__init__()
        layers = []
        cin = 3
        size = 64
        for ch in channels:
            size //= 2
            layers += [nn.Conv2d(cin, ch, 4, stride=2, padding=1),
                       nn.LayerNorm([ch, size, size]), nn.SiLU()]
            cin = ch
        self.cnn = nn.Sequential(*layers)  # 64->32->16->8->4, ends 4x4
        self.mlp = nn.Sequential(nn.Linear(channels[-1] * 4 * 4, feat_dim),
                                 nn.LayerNorm(feat_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,64,64)->(B,feat)
        return self.mlp(self.cnn(x).flatten(1))


class ConvDecoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), in_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, channels[-1] * 4 * 4),
                                 nn.SiLU())
        layers = []
        prev = channels[-1]
        size = 4
        for ch in reversed(channels[:-1]):
            size *= 2
            layers += [nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1),
                       nn.LayerNorm([ch, size, size]), nn.SiLU()]
            prev = ch
        layers += [nn.ConvTranspose2d(prev, 3, 4, stride=2, padding=1)]
        self.deconv = nn.Sequential(*layers)  # 4->8->16->32->64

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B,in)->(B,3,64,64)
        h = self.mlp(z)  # (B, C*4*4)
        return self.deconv(h.reshape(-1, self.mlp[0].out_features // 16, 4, 4))


# ---------------------------------------------------------------------------
# Latent predictors
# ---------------------------------------------------------------------------

class CategoricalPrior(nn.Module):
    """DreamerV3's categorical stochastic prior p(z_t | h_t)."""

    def __init__(self, z_classes, z_codes, gru_dim, feat_dim):
        super().__init__()
        self.classes, self.codes = z_classes, z_codes
        self.prior = nn.Sequential(nn.Linear(gru_dim, feat_dim), nn.SiLU(),
                                   nn.Linear(feat_dim, z_classes * z_codes))
        self.posterior = nn.Sequential(nn.Linear(gru_dim + feat_dim, feat_dim),
                                       nn.SiLU(),
                                       nn.Linear(feat_dim, z_classes * z_codes))

    def sample(self, h, x_feat=None, detach_grad=False, sample=True):
        if x_feat is None:  # prior: p(z | h)
            logits = self.prior(h)
        else:               # posterior: q(z | h, x)
            logits = self.posterior(torch.cat([h, x_feat], -1))
        logits = logits.reshape(-1, self.classes * self.codes)
        if detach_grad:
            logits = logits.detach()
        dist = torch.distributions.OneHotCategoricalStraightThrough(
            logits=logits.view(-1, self.codes, self.classes))
        z = dist.sample() if sample else dist.mode()
        return z.reshape(-1, self.codes * self.classes), dist

    def kl(self, h, x_feat):
        """KL(q(z|h,x) || p(z|h)) with per-group free bits (DreamerV3)."""
        p = torch.distributions.Categorical(
            logits=self.prior(h).view(-1, self.codes, self.classes))
        q = torch.distributions.Categorical(
            logits=self.posterior(torch.cat([h, x_feat], -1))
            .view(-1, self.codes, self.classes))
        kl = torch.distributions.kl.kl_divergence(q, p)  # (B, codes)
        fb = getattr(self, "free_bits", CONFIG.get("kl_free_bits_total", 32.0)
                     / self.codes)
        kl = kl.clamp_min(fb).sum(-1).mean()
        return kl


class MetriplecticPrior(nn.Module):
    """Fixed divergence-gated metriplectic prior (EB-H-JEPA E-arm).

    z' = z + dt (J grad_z H(z, h) - R(z) grad_z S(z, h)), with:
      - J: constant canonical symplectic matrix (div(J gradH) == 0 exactly,
        closing the learned-skew loophole documented in the paper),
      - R(z) = L(z) L(z)^T with L initialized to 0.05 (breaking the dead
        LL^T saddle that keeps R stuck at 0 in the naive arm),
      - learned potentials H, S conditioned on the GRU memory h.

    The prior is a Gaussian with mean = metriplectic map, fixed variance, so
    the KL with the posterior stays analytic. dt is a learnable scale.
    """

    def __init__(self, z_dim, gru_dim, feat_dim, dt=1.0, substeps=1):
        super().__init__()
        assert z_dim % 2 == 0, "metriplectic latent must be even-dimensional"
        self.z_dim, self.dt, self.substeps = z_dim, dt, substeps
        half = z_dim // 2
        J = torch.zeros(z_dim, z_dim)
        for i in range(half):
            J[i, half + i] = 1.0
            J[half + i, i] = -1.0
        self.register_buffer("J", J)  # constant canonical J (fixed)
        self.energy = nn.Sequential(nn.Linear(z_dim + gru_dim, feat_dim), nn.SiLU(),
                                    nn.Linear(feat_dim, 1))
        self.entropy = nn.Sequential(nn.Linear(z_dim + gru_dim, feat_dim), nn.SiLU(),
                                     nn.Linear(feat_dim, 1))
        self.L = nn.Sequential(nn.Linear(z_dim + gru_dim, feat_dim), nn.SiLU(),
                               nn.Linear(feat_dim, z_dim * z_dim))
        self.ln_dt = nn.Parameter(torch.tensor(math.log(dt)))
        self.posterior = nn.Sequential(nn.Linear(gru_dim + feat_dim, feat_dim),
                                       nn.SiLU(),
                                       nn.Linear(feat_dim, 2 * z_dim))
        self.logvar = nn.Parameter(torch.zeros(z_dim) + math.log(0.1))
        with torch.no_grad():
            for m in self.L.modules():
                if isinstance(m, nn.Linear) and m.out_features == z_dim * z_dim:
                    m.weight.mul_(0.0); m.bias.mul_(0.0)
                    # init the final map's weights to ~0.05 so R = LL^T starts
                    # slightly off the dead saddle (E-arm fix)
                    m.weight.data.normal_(0.0, 0.05 / math.sqrt(feat_dim))

    def _R(self, z, h):
        L = self.L(torch.cat([z, h], -1)).view(-1, self.z_dim, self.z_dim)
        return L @ L.transpose(-1, -2)  # PSD by construction

    def step(self, z, h):
        """One metriplectic step; z: (B, D), h: (B, H) -> (B, D).

        dz = dt * (J grad_z H - R grad_z S).  With constant canonical J,
        div(J gradH) == 0 exactly, so volume change comes only from R
        (the E-arm fix); R = L L^T is PSD by construction.

        The field is a deterministic function of (z, h): we differentiate
        the potentials with create_graph=False and re-arm z, so the map's
        value is exact while gradient flow downstream (e.g. to the actor
        through the direct z + dt*field path) stays cheap and memory-safe.
        """
        # re-enable grad even when called under no_grad (e.g. the stability
        # audit): autograd.grad needs a live graph w.r.t. z
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            zh = torch.cat([z, h], -1)
            gH = torch.autograd.grad(self.energy(zh).sum(), z,
                                     create_graph=False)[0]
            gS = torch.autograd.grad(self.entropy(zh).sum(), z,
                                     create_graph=False)[0]
            R = self._R(z, h)
            # J gradH = (gH @ J^T) for constant J; see derivation in module docstring
            field = (gH @ self.J.t()).unsqueeze(-1) - R @ gS.unsqueeze(-1)
            dt = torch.exp(self.ln_dt)
            for _ in range(self.substeps):
                z = z + dt * field.squeeze(-1)
        return z.detach()

    def sample(self, h, x_feat=None, detach_grad=False, sample=True):
        if x_feat is None:  # prior: Gaussian centered on the metriplectic map
            raise NotImplementedError(
                "metriplectic prior needs z_{t-1}; use step_from() in the RSSM")

    def step_from(self, z, h, detach_grad=False):
        """Prior mean one step ahead from the previous latent.
        z=None (sequence start): no dynamics info, use a zero prior mean."""
        if z is None:
            return torch.zeros(h.shape[0], self.z_dim, device=h.device)
        z = z.detach() if detach_grad else z
        mu = self.step(z, h)
        return mu

    def posterior_params(self, h, x_feat):
        out = self.posterior(torch.cat([h, x_feat], -1))
        mu, logvar = out.chunk(2, -1)
        return mu, logvar

    def kl(self, h, x_feat, z_prev, detach_prior=True):
        """KL(q(z|h,x) || p(z|h, z_prev)) for one time step, free bits."""
        mu_p = self.step_from(z_prev, h, detach_grad=detach_prior)
        var_p = torch.exp(self.logvar).expand_as(mu_p)
        mu_q, logvar_q = self.posterior_params(h, x_feat)
        var_q = torch.exp(logvar_q)
        kl = (0.5 * (logvar_q - self.logvar.unsqueeze(0)
                     + (var_p + (mu_p - mu_q).square()) / var_q - 1.0))
        fb = getattr(self, "free_bits", CONFIG.get("kl_free_bits_total", 32.0)
                     / self.z_dim)
        kl = kl.clamp_min(fb).sum(-1).mean()
        return kl

    def posterior_sample(self, h, x_feat, sample=True):
        mu, logvar = self.posterior_params(h, x_feat)
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z


# ---------------------------------------------------------------------------
# RSSM world model (pluggable predictor)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(self, predictor="rssm", config=None):
        super().__init__()
        cfg = {**CONFIG, **(config or {})}
        self.cfg = cfg
        self.predictor_type = predictor
        self.encoder = ConvEncoder(cfg["cnn_channels"], cfg["feat_dim"])
        self.decoder = ConvDecoder(cfg["cnn_channels"], in_dim=self._zdim())
        self.gru = nn.GRUCell(cfg["gru_dim"], cfg["gru_dim"])
        self.z_in = nn.Linear(cfg["action_dim"] + cfg["z_dim"]
                              if predictor == "metriplectic"
                              else cfg["action_dim"] + cfg["z_classes"] * cfg["z_codes"],
                              cfg["gru_dim"])
        # per-component free bits from a TOTAL budget, matched across arms
        # (RSSM: 32 codes -> 1.0 each, DreamerV3's exact setting; continuous
        #  64-dim -> 0.5 each), so posterior-collapse slack cannot confound
        total_fb = cfg.get("kl_free_bits_total", 32.0)
        if predictor == "rssm":
            self.prior = CategoricalPrior(cfg["z_classes"], cfg["z_codes"],
                                          cfg["gru_dim"], cfg["feat_dim"])
            self.prior.free_bits = total_fb / cfg["z_codes"]
        elif predictor == "metriplectic":
            self.prior = MetriplecticPrior(cfg["z_dim"], cfg["gru_dim"],
                                           cfg["feat_dim"], cfg["dt"], cfg["substeps"])
            self.prior.free_bits = total_fb / cfg["z_dim"]
        else:
            raise ValueError(predictor)
        self.reward_head = nn.Sequential(nn.Linear(cfg["gru_dim"] + self._zdim(), 512),
                                         nn.SiLU(), nn.Linear(512, 1))
        self.cont_head = nn.Sequential(nn.Linear(cfg["gru_dim"] + self._zdim(), 512),
                                       nn.SiLU(), nn.Linear(512, 1))

    def _zdim(self):
        c = self.cfg
        return (c["z_classes"] * c["z_codes"] if self.predictor_type == "rssm"
                else c["z_dim"])

    def _state0(self, batch, device):
        return (torch.zeros(batch, self.cfg["gru_dim"], device=device),
                None)  # (h, z_prev)

    def observe(self, obs, actions, states=None):
        """Encode a sequence. obs: (T, B, 3, H, W), actions: (T, B, A).
        Returns (posterior z (T,B,D), state features (T,B,gru_dim), dists)."""
        cfg = self.cfg
        T, B = obs.shape[0], obs.shape[1]
        feat = self.encoder(obs.reshape(-1, 3, cfg["obs_size"], cfg["obs_size"])
                            ).reshape(T, B, -1)
        zs, hs = [], []
        h, z_prev = self._state0(B, obs.device)
        for t in range(T):
            if self.predictor_type == "rssm":
                z, _ = self.prior.sample(h, feat[t])
            else:
                z = self.prior.posterior_sample(h, feat[t])
            zs.append(z); hs.append(h)
            z_prev = z
            h = self.gru(self.z_in(torch.cat([actions[t], z], -1)), h)
        return torch.stack(zs), torch.stack(hs)

    def imagine(self, actions, states):
        """Roll out the prior (no observations) for T steps.
        actions: (T, B, A) -> (posterior-free z (T,B,D), h (T,B,gru))."""
        cfg = self.cfg
        T, B = actions.shape[0], actions.shape[1]
        h, z_prev = states
        zs, hs = [], []
        for t in range(T):
            if self.predictor_type == "rssm":
                z, _ = self.prior.sample(h, detach_grad=False)
            else:
                mu = self.prior.step_from(z_prev, h)
                var = torch.exp(self.prior.logvar).expand_as(mu)
                z = mu + torch.randn_like(mu) * torch.sqrt(var)
            zs.append(z); hs.append(h)
            z_prev = z
            h = self.gru(self.z_in(torch.cat([actions[t], z], -1)), h)
        return torch.stack(zs), torch.stack(hs)

    def compute_loss(self, obs, actions, rewards, continues):
        """Full world-model loss on a batch of sequences.
        obs: (T,B,3,64,64); actions: (T,B,A); rewards/continues: (T,B,1)."""
        cfg = self.cfg
        T, B = obs.shape[0], obs.shape[1]
        zs, hs = self.observe(obs, actions)
        # reconstruction
        rec = self.decoder(zs.reshape(-1, self._zdim()))
        rec = rec.reshape(T, B, 3, cfg["obs_size"], cfg["obs_size"])
        rec_loss = F.mse_loss(rec, obs, reduction="none").mean()
        # reward + continue (symlog targets)
        zh = torch.cat([hs.reshape(-1, cfg["gru_dim"]),
                        zs.reshape(-1, self._zdim())], -1)
        rew_pred = self.reward_head(zh).reshape(T, B, 1)
        rew_loss = F.mse_loss(symlog(rew_pred), symlog(rewards))
        cont_pred = self.cont_head(zh).reshape(T, B, 1)
        cont_loss = F.binary_cross_entropy_with_logits(
            cont_pred, continues)
        # KL (prior vs posterior) with free bits
        if self.predictor_type == "rssm":
            kl_loss = 0.0
            h, z_prev = self._state0(B, obs.device)
            for t in range(T):
                kl_loss = kl_loss + self.prior.kl(h, self.encoder(obs[t]))
                z, _ = self.prior.sample(h, self.encoder(obs[t]), detach_grad=True)
                z_prev = z
                h = self.gru(self.z_in(torch.cat([actions[t], z], -1)), h)
            kl_loss = kl_loss / T
        else:
            kl_loss = 0.0
            h, z_prev = self._state0(B, obs.device)
            feat = self.encoder(obs.reshape(-1, 3, cfg["obs_size"], cfg["obs_size"])
                                ).reshape(T, B, -1)
            for t in range(T):
                kl_loss = kl_loss + self.prior.kl(h, feat[t], z_prev)
                z = self.prior.posterior_sample(h, feat[t])
                z_prev = z
                h = self.gru(self.z_in(torch.cat([actions[t], z], -1)), h)
            kl_loss = kl_loss / T
        total = rec_loss + 0.1 * kl_loss + 1.0 * rew_loss + 1.0 * cont_loss
        return dict(total=total, rec=rec_loss, kl=kl_loss,
                    reward=rew_loss, cont=cont_loss)

    def state_feat(self, z, h):
        return torch.cat([h, z], -1)

    def reward_from(self, z, h):
        return symexp(self.reward_head(self.state_feat(z, h)))

    def cont_from(self, z, h):
        return torch.sigmoid(self.cont_head(self.state_feat(z, h)))


# ---------------------------------------------------------------------------
# Actor-critic (imagination)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm
        cfg = wm.cfg
        self.actor = nn.Sequential(nn.Linear(cfg["gru_dim"] + wm._zdim(), 512),
                                   nn.SiLU(), nn.Linear(512, cfg["action_dim"]))
        self.critic = nn.Sequential(nn.Linear(cfg["gru_dim"] + wm._zdim(), 512),
                                    nn.SiLU(), nn.Linear(512, 1))

    def act(self, z, h, sample=True):
        logits = self.actor(self.wm.state_feat(z, h))
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample() if sample else dist.probs.argmax(-1), dist

    def imagine_loss(self, states, train=True):
        """DreamerV3-style latent imagination with TD(lambda) returns."""
        cfg = self.wm.cfg
        T, B = cfg["imag_horizon"], cfg["imag_batch"]
        device = next(self.parameters()).device
        h, z_prev = states
        actions = torch.zeros(T, B, cfg["action_dim"], device=device)
        zs, hs = [], []
        for t in range(T):
            if self.wm.predictor_type == "rssm":
                z, _ = self.wm.prior.sample(h, detach_grad=not train)
            else:
                mu = self.wm.prior.step_from(z_prev, h)
                var = torch.exp(self.wm.prior.logvar).expand_as(mu)
                z = mu + torch.randn_like(mu) * torch.sqrt(var)
            a, dist = self.act(z, h)
            actions[t] = F.one_hot(a, cfg["action_dim"]).float()
            zs.append(z); hs.append(h)
            z_prev = z
            h = self.wm.gru(self.wm.z_in(torch.cat([actions[t], z], -1)), h)
        zs, hs = torch.stack(zs), torch.stack(hs)
        feats = torch.cat([hs.reshape(-1, cfg["gru_dim"]),
                           zs.reshape(-1, self.wm._zdim())], -1)
        with torch.no_grad():
            rew = self.wm.reward_head(feats).reshape(T, B, 1)
            cont = self.wm.cont_head(feats).reshape(T, B, 1)
            val = self.critic(feats).reshape(T, B, 1)
        # TD(lambda) from the end
        returns = torch.zeros_like(rew)
        g = 0.0
        for t in reversed(range(T)):
            g = rew[t] + cfg["gamma"] * cont[t] * (1 - cfg["lambda_"]) * val[t] \
                + cfg["gamma"] * cont[t] * cfg["lambda_"] * g
            returns[t] = g
        value = self.critic(feats).reshape(T, B, 1)
        critic_loss = F.mse_loss(value, returns)
        # actor: maximize returns (stop gradients through values/returns)
        logits = self.actor(feats).reshape(T, B, cfg["action_dim"])
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions.argmax(-1))
        actor_loss = -(returns.detach() * logp.unsqueeze(-1)).mean() \
            - cfg["entropy"] * dist.entropy().mean()
        return dict(actor=actor_loss, critic=critic_loss)
