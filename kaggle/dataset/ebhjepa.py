# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 <AUTHOR NAME -- fill in before submission>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
EB-H-JEPA : Energy-Based Hamiltonian Latent World Model
=======================================================

A JEPA-style world model whose *predictor* is not a free-form neural net but a
learned Hamiltonian. The latent state is split into generalized position `q` and
momentum `p`. A scalar network H(q, p) defines an energy; the state is advanced
in time by a *symplectic* integrator that follows Hamilton's equations exactly:

        dq/dt = + dH/dp
        dp/dt = - dH/dq

Why bother? An unconstrained latent predictor is free to contract, explode, or
slowly drift over a long rollout because nothing ties consecutive steps together.
A symplectic integrator of a learned H approximately conserves that H along the
trajectory and preserves phase-space volume (Liouville's theorem), which removes
the two easy failure modes of long rollouts: monotone contraction to a collapsed
point and monotone energy growth. That is the mechanism we are betting on for
long-horizon stability. It is an inductive bias, not a proof.

Anti-collapse for the *encoder* is handled by SIGReg (Sketched Isotropic Gaussian
Regularization, the LeJEPA / "Lea-style" idea): push every 1-D random projection
of the embeddings toward a standard Gaussian. If all 1-D marginals are N(0,1),
the joint is isotropic Gaussian, which is full-rank by construction, so no teacher
network, no stop-gradient, no moving average is needed to avoid trivial collapse.

Shape convention used throughout:
    B  = batch size
    C, H_img, W_img = image channels / height / width  (image mode)
    Xdim = flat observation dim                         (vector mode)
    D  = latent dim (must be even);  q,p each have dim D//2

Run this file directly to execute the 5-arm (B/C/D/E/F) benchmark end to end.


STRUCTURAL FAILURE ATLAS  (empirical; Lorenz-63 under a nonlinear coordinate warp)
=================================================================================
This module doubles as the reproducible evidence base for a NEGATIVE-RESULTS study:
imposing physical / thermodynamic constraints inside a SELF-SUPERVISED LATENT space
does not, by itself, yield physical dynamics, because the optimizer routes around
each scalar constraint through a geometric exploit. Five distinct exploits were
isolated; each is annotated at its mechanism site below and summarized here. NONE of
this claims to recover physics: Lorenz is not a natural GENERIC system, the latent
chart is an un-invertible network invention (no decoder), so H/S/R are proxies.

(1) TRIVIAL REPRESENTATION FREEZE  (Model C; early metriplectic arms).
    Multi-step alignment MSE is minimized by a predictor that DOES NOT MOVE: a frozen
    field ties the persistence baseline, and compounding K-step error makes any
    motion locally costlier than none, so the field gradients collapse to ~0 (per-step
    motion ~1e-3 vs target ~0.4; lambda1 ~ 0). Partial mitigations: the K=1->3->6
    horizon curriculum and the DETACHED motion-magnitude penalty (EBHJepa.forward #5).

(2) SCALE-CONDITIONING PARADOX  (the energy anchor).
    A raw kinetic anchor MSE(H, 0.5||dx||^2) is scale-STARVED: the raw proxy spread
    (~0.036) is ~30x below the ||dH/dp|| (~3.2) the integrator needs for dq=dt*dH/dp,
    so matching magnitudes clamps the dynamics and refreezes the model (motion ratio
    ~0.01). Fix: compare Z-SCORES of both sides (z_score_tensor), keeping only the
    shape/correlation constraint; MSE(zscore a, zscore b) = 2(1-corr(a,b)), O(1).

(3) COORDINATE-SKEW LOOPHOLE + R DEAD-SADDLE  (why dissipation would not engage).
    Two stacked mechanisms kept tr(R)/n == 0.00000 in every NAIVE metriplectic run:
      (a) a state-dependent skew J(z) is NOT divergence-free -- div(J gradH) != 0 --
          so it can supply the demanded volume contraction "for free", leaving R off
          (the divergence gate is satisfied with no dissipation matrix at all);
      (b) more fundamentally, R = L L^T with L ZERO-INIT has dR/dL = 0 at L=0, a dead
          saddle where NO gradient can lift R off zero. Even after closing (a) by
          fixing J to a constant canonical skew (div(J gradH)=0 exactly), R stayed
          stuck at 0 until (b) was broken with a small nonzero L init.
    With BOTH fixed (Model E): R engages (tr(R)/n ~ 5.6), contraction hits -13.667.

(4) SPECTRUM-SUM vs LAMBDA1 DECOUPLING  (Model E).
    Pinning the divergence pins the SUM of the Lyapunov spectrum (an invariant), but
    the sum does not pin any INDIVIDUAL exponent. E reaches contraction -12.8 while
    lambda1 BLOWS UP to +3.5 (true ~0.88): the negative exponents compensate.
    Confirmed 4x (lambda1 in {-0.42, +1.44, +3.5, +4.29}, all with the "correct" sum).
    A scalar volume constraint is structurally incapable of shaping lambda1.

(5) FINITE-TIME SPECTRAL TRAP  (Model F), two regimes:
    (5a) SHORT horizon (K=6 = 0.06 Lyapunov times; the IN-REPO Model F): the 6-step
         finite-time QR exponent is DECOUPLED from the asymptotic lambda1 -- during
         training lambda1_hat drifts to -0.17 while the true 200-step lambda1 is +1.44
         (OPPOSITE SIGN). Shaping the short proxy does not move the real exponent;
         eval lambda1 lands at 1.44 (== naive), not 0.88.
    (5b) LONG horizon (100-step differentiable QR; GPU, pathB_spectral_colab.ipynb):
         the long proxy IS trainable toward target (lambda1_hat -> +0.85) and its
         100-step backprop does NOT explode gradients (all modes ~3.9e-3). BUT nothing
         constrains GLOBAL boundedness: the 200-step eval rollout diverges to +inf.
         The trap moves from "wrong proxy" (5a) to "right proxy, unbounded global
         trajectory" (5b). Local-exponent + divergence supervision still leaves the
         long rollout free to explode.

EVALUATION-STANDARD CONTRIBUTION -- the OVERLAP TRAP. Chamfer/Hausdorff distance to
the encoded attractor REWARDS a frozen predictor: a stationary point inside the
attractor cloud scores near-perfect (Model C posts the BEST chamfer while being
dynamically DEAD). Geometric overlap must be GATED on the rollout being alive and
chaotic (lambda1 > 0 and a sane motion ratio), or replaced by a motion-invariant
statistic (correlation dimension / latent power spectrum). This is the evaluation
standard these negative results argue for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ENCODER  (x -> z)  with an anti-collapse projector head
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Maps a raw observation x to a latent embedding z.

    Two backbones are supported so the same model works for physical state
    vectors and for image frames:
      - mode="vector": an MLP over a flat state of size `in_dim`.
      - mode="image" : a small CNN over (C, H_img, W_img) frames.

    A separate projector head maps backbone features -> the `latent_dim` space
    that SIGReg regularizes. Keeping backbone and projector separate is the
    standard JEPA trick: the regularizer shapes the projected space while the
    backbone stays free to keep information the projector might discard.
    """

    def __init__(
        self,
        latent_dim: int,
        mode: str = "vector",
        in_dim: int = 32,
        in_channels: int = 3,
        hidden: int = 256,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even so z splits into (q, p)"
        self.mode = mode
        self.latent_dim = latent_dim

        if mode == "vector":
            # x: (B, in_dim) -> features: (B, hidden)
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            feat_dim = hidden
        elif mode == "image":
            # x: (B, C, H_img, W_img) -> features: (B, feat_dim)
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # H/2
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),           # H/4
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),          # H/8
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),                              # (B,128,1,1)
                nn.Flatten(),                                         # (B,128)
            )
            feat_dim = 128
        else:
            raise ValueError(f"unknown mode {mode!r}")

        # Projector head: backbone features -> latent (this is what SIGReg sees).
        # features: (B, feat_dim) -> z: (B, latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) or (B, C, H_img, W_img)  ->  z: (B, latent_dim)
        feats = self.backbone(x)
        z = self.projector(feats)
        return z


# ---------------------------------------------------------------------------
# 2. LATENT SPLIT   z = [q, p]
# ---------------------------------------------------------------------------
def split_latent(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split z evenly into position q and momentum p.

    z: (B, D)  ->  q: (B, D//2), p: (B, D//2)
    """
    d = z.shape[-1]
    assert d % 2 == 0, "latent dim must be even"
    q, p = z[..., : d // 2], z[..., d // 2 :]
    return q, p


def join_latent(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Inverse of split_latent.  q,p: (B, D//2)  ->  z: (B, D)"""
    return torch.cat([q, p], dim=-1)


# ---------------------------------------------------------------------------
# 3. ENERGY-BASED PREDICTOR  (Hamiltonian Neural Network + symplectic integrator)
# ---------------------------------------------------------------------------
class HamiltonianSystem(nn.Module):
    """Learns a scalar Hamiltonian H(q, p) and integrates Hamilton's equations.

    The network maps the full phase point [q, p] to a single scalar energy.
    The *dynamics* are never learned directly; they are derived from H by exact
    autograd differentiation, which is what makes them symplectic-compatible:

        dq/dt = + dH/dp
        dp/dt = - dH/dq

    `create_graph=True` on the grad calls is essential: it keeps the derivative
    graph differentiable so the alignment loss can backprop through the whole
    integrator into H's weights.
    """

    def __init__(self, half_dim: int, hidden: int = 128):
        super().__init__()
        self.half_dim = half_dim
        # [q, p]: (B, 2*half_dim) -> H: (B, 1)
        self.net = nn.Sequential(
            nn.Linear(2 * half_dim, hidden),
            nn.Softplus(),          # smooth (C-inf) so second-order grads are well behaved
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # q,p: (B, half_dim)  ->  H: (B, 1)
        return self.net(torch.cat([q, p], dim=-1))

    def time_derivative(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (dq/dt, dp/dt) from Hamilton's equations via autograd.

        q,p: (B, half_dim)  ->  dqdt,dpdt: (B, half_dim)
        """
        # We need gradients of H w.r.t. q and p, so make sure they carry grad.
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.energy(q, p).sum()  # sum over batch: grad then gives per-sample dH/d.

        dHdq, dHdp = torch.autograd.grad(
            H, (q, p), create_graph=True, retain_graph=True
        )
        dqdt = dHdp           # + dH/dp
        dpdt = -dHdq          # - dH/dq
        return dqdt, dpdt

    # --- integrators ------------------------------------------------------
    def symplectic_euler_step(
        self, q: torch.Tensor, p: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One semi-implicit (symplectic) Euler step.

        Update p first using dH/dq at the current point, then update q using
        dH/dp at the *new* momentum. This ordering is what makes the map
        symplectic (volume preserving), unlike naive explicit Euler.
        """
        dHdq, _ = self._grads(q, p)
        p_next = p - dt * dHdq
        _, dHdp = self._grads(q, p_next)   # dH/dp evaluated at (q, p_next)
        q_next = q + dt * dHdp
        return q_next, p_next

    def rk4_step(
        self, q: torch.Tensor, p: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One classical Runge-Kutta 4 step of the coupled (q, p) ODE.

        Higher local accuracy than symplectic Euler but not exactly symplectic;
        useful when you want small per-step error over shorter horizons.
        """
        def f(qq, pp):
            return self.time_derivative(qq, pp)  # (dqdt, dpdt)

        k1q, k1p = f(q, p)
        k2q, k2p = f(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
        k3q, k3p = f(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
        k4q, k4p = f(q + dt * k3q, p + dt * k3p)

        q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
        p_next = p + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        return q_next, p_next

    def _grads(self, q, p):
        """Helper: raw (dH/dq, dH/dp) with graph kept for backprop."""
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.energy(q, p).sum()
        dHdq, dHdp = torch.autograd.grad(H, (q, p), create_graph=True, retain_graph=True)
        return dHdq, dHdp

    def rollout(
        self,
        z0: torch.Tensor,
        steps: int,
        dt: float,
        integrator: str = "symplectic",
        detach_steps: bool = False,
        substeps: int = 1,   # ignored here; kept for a uniform rollout signature
    ) -> torch.Tensor:
        """Advance an initial latent z0 forward `steps` times.

        z0: (B, D)  ->  traj: (B, steps+1, D)   (includes the initial state)

        detach_steps: if True, detach the carried state after each step so the
        autograd graph does not grow with the horizon. Use it for long eval
        rollouts (we only need the values); leave False for training so the
        alignment loss can backprop through the whole unroll.
        """
        q, p = split_latent(z0)
        traj = [join_latent(q, p)]
        step_fn = self.symplectic_euler_step if integrator == "symplectic" else self.rk4_step
        for _ in range(steps):
            q, p = step_fn(q, p, dt)
            if detach_steps:
                q, p = q.detach(), p.detach()
            traj.append(join_latent(q, p))
        return torch.stack(traj, dim=1)  # (B, steps+1, D)


# ---------------------------------------------------------------------------
# 3b. PORT-HAMILTONIAN PREDICTOR  (conservative energy + learned dissipation)
# ---------------------------------------------------------------------------
class PortHamiltonianSystem(nn.Module):
    """A dissipative upgrade of the Hamiltonian predictor.

    A purely conservative Hamiltonian cannot represent a system that loses
    energy. The port-Hamiltonian form adds a learned, positive-semidefinite
    dissipation matrix R(q, p) alongside the scalar energy H(q, p):

        dq/dt = + dH/dp
        dp/dt = - dH/dq - R(q, p) @ dH/dp

    R is built as R = L Lᵀ from a network that emits the factor L, which
    guarantees R ⪰ 0. That sign guarantee matters: along a trajectory,

        dH/dt = (dH/dq)·(dq/dt) + (dH/dp)·(dp/dt)
              = (dH/dp)·(-dH/dq - R dH/dp) + (dH/dq)·(dH/dp)
              = -(dH/dp)ᵀ R (dH/dp)  <= 0

    so the learned energy can only decay, never spontaneously grow. That is
    exactly the bias a damped system needs, and it is strictly more general than
    the conservative model (R = 0 recovers it).
    """

    def __init__(self, half_dim: int, hidden: int = 128):
        super().__init__()
        self.half_dim = half_dim
        # H head:  [q,p] (B, 2*half_dim) -> H (B, 1)
        self.h_net = nn.Sequential(
            nn.Linear(2 * half_dim, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1),
        )
        # L head:  [q,p] (B, 2*half_dim) -> vec(L) (B, half_dim*half_dim)
        self.l_net = nn.Sequential(
            nn.Linear(2 * half_dim, hidden), nn.Softplus(),
            nn.Linear(hidden, half_dim * half_dim),
        )
        # Zero-init the L head so R = 0 at start: the model begins as a purely
        # conservative Hamiltonian (which is numerically stable, like model A) and
        # learns dissipation gently, instead of starting with a large random R that
        # makes the explicit integrator stiff and blow up over long rollouts.
        nn.init.zeros_(self.l_net[-1].weight)
        nn.init.zeros_(self.l_net[-1].bias)

    def energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # q,p: (..., half_dim) -> H: (..., 1)
        return self.h_net(torch.cat([q, p], dim=-1))

    def dissipation(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Positive-semidefinite dissipation matrix R = L Lᵀ.

        q,p: (..., half_dim) -> R: (..., half_dim, half_dim)
        """
        lead = q.shape[:-1]
        L = self.l_net(torch.cat([q, p], dim=-1))          # (..., half_dim**2)
        L = L.view(*lead, self.half_dim, self.half_dim)     # (..., half_dim, half_dim)
        # R = L Lᵀ is PSD by construction; divide by half_dim to bound its scale so
        # the explicit dissipation term dt*R can't dominate the step and diverge.
        R = (L @ L.transpose(-1, -2)) / self.half_dim
        return R

    def _grads(self, q, p):
        """(dH/dq, dH/dp, R) at (q, p), graph kept for backprop."""
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.energy(q, p).sum()
        dHdq, dHdp = torch.autograd.grad(H, (q, p), create_graph=True, retain_graph=True)
        R = self.dissipation(q, p)
        return dHdq, dHdp, R

    def port_step(self, q, p, dt, substeps: int = 1):
        """Semi-implicit port-Hamiltonian Euler step, optionally sub-stepped.

        Splitting one dt into `substeps` smaller steps reduces the stiffness of
        the explicit dissipation term (the classic explicit-Euler stability
        limit scales with step size), which is what keeps long rollouts finite.

        q,p: (B, half_dim) -> q_next, p_next: (B, half_dim)
        """
        h = dt / substeps
        for _ in range(substeps):
            dHdq, dHdp, R = self._grads(q, p)
            Rp = torch.matmul(R, dHdp.unsqueeze(-1)).squeeze(-1)  # R @ dH/dp : (B, half_dim)
            p = p - h * (dHdq + Rp)                               # -dH/dq - R dH/dp
            _, dHdp2, _ = self._grads(q, p)                       # dH/dp at new momentum
            q = q + h * dHdp2                                     # +dH/dp
        return q, p

    def rollout(self, z0, steps, dt, integrator=None, detach_steps=False, substeps=1):
        # signature mirrors HamiltonianSystem.rollout; `integrator` is ignored
        # (the port step is fixed). More `substeps` = more stable long rollouts at
        # higher cost; training uses 1, the long eval rollout uses more.
        # z0: (B, D) -> traj: (B, steps+1, D)
        q, p = split_latent(z0)
        traj = [join_latent(q, p)]
        for _ in range(steps):
            q, p = self.port_step(q, p, dt, substeps=substeps)
            if detach_steps:
                q, p = q.detach(), p.detach()
            traj.append(join_latent(q, p))
        return torch.stack(traj, dim=1)

    def dissipation_residual(self, traj: torch.Tensor, dt: float) -> torch.Tensor:
        """Dissipative Consistency Penalty (adaptive replacement for var-flatness).

        Force the learned energy drop between consecutive steps to match the work
        removed by the learned dissipation matrix:

            L = mean | H(z_{t+1}) - H(z_t) + dt * (dH/dpᵀ R dH/dp) |²

        The bracket is exactly the first-order port-Hamiltonian energy balance:
        it is zero when the energy decay equals the dissipated power, so this
        does not force energy flat (that would fight the true decay); it forces
        energy to decay *at the right rate*.  Inputs are detached so the penalty
        shapes H and R at the visited states without perturbing the rollout path.

        traj: (B, T, D) -> scalar
        """
        B, T, _ = traj.shape
        q, p = split_latent(traj)                              # (B, T, half_dim)

        H = self.energy(q.detach(), p.detach()).squeeze(-1)    # (B, T)

        # Evaluate dH/dp and R at steps 0..T-2 (flatten batch and time).
        qf = q[:, :-1].reshape(-1, self.half_dim).detach().requires_grad_(True)
        pf = p[:, :-1].reshape(-1, self.half_dim).detach().requires_grad_(True)
        Hf = self.energy(qf, pf).sum()
        dHdp = torch.autograd.grad(Hf, pf, create_graph=True, retain_graph=True)[0]  # (M, hd)
        R = self.dissipation(qf, pf)                           # (M, hd, hd)
        # dissipated power  dH/dpᵀ R dH/dp  >= 0  (R is PSD)
        power = torch.einsum("mi,mij,mj->m", dHdp, R, dHdp).view(B, T - 1)  # (B, T-1)

        residual = H[:, 1:] - H[:, :-1] + dt * power           # (B, T-1)
        return (residual ** 2).mean()


# ---------------------------------------------------------------------------
# 3c. UNCONSTRAINED MLP PREDICTOR  (pure-JEPA baseline, no physics)
# ---------------------------------------------------------------------------
class MLPPredictor(nn.Module):
    """A free-form residual MLP latent predictor: z_{t+1} = z_t + f(z_t).

    No energy, no symplectic structure, no dissipation. This is the standard
    JEPA-style predictor and serves as the control arm that shows what the
    physical inductive biases actually buy over long rollouts.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def step_z(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)  # residual update, z: (B, D) -> (B, D)

    def rollout(self, z0, steps, dt=None, integrator=None, detach_steps=False, substeps=1):
        # z0: (B, D) -> traj: (B, steps+1, D)
        z = z0
        traj = [z]
        for _ in range(steps):
            z = self.step_z(z)
            if detach_steps:
                z = z.detach()
            traj.append(z)
        return torch.stack(traj, dim=1)


# ---------------------------------------------------------------------------
# 3d. METRIPLECTIC PREDICTOR  (learnable coordinates + dissipation, no q/p split)
# ---------------------------------------------------------------------------
class MetriplecticSystem(nn.Module):
    """Coordinate-free dissipative dynamics on the FULL latent z (no hardcoded
    [q, p] split). The state evolves as

        dz/dt = J(z) @ grad_z H(z)  -  R(z) @ grad_z S(z)

    with two learned scalar potentials (H = energy, S = entropy/dissipation
    potential) and two learned state-dependent matrices:
        J(z) = A - Aᵀ            skew-symmetric  (the "reversible" coordinate map)
        R(z) = L Lᵀ / dim        positive semidefinite (the dissipation metric)

    The skew part contributes nothing to dH/dt (∇Hᵀ J ∇H = 0 for any skew J), so
    it is a pure coordinate/geometry operator the network is free to invent. The
    R part is what removes energy.

    HONESTY LABEL: this is metriplectic-*inspired*, NOT thermodynamically
    consistent GENERIC. True GENERIC also imposes the degeneracy conditions
    J∇S = 0 and R∇H = 0, which force dH/dt = 0 (exact energy conservation) and
    dS/dt >= 0. We deliberately do NOT impose them, because our oscillator is an
    OPEN system that must LOSE mechanical energy; forcing dH/dt = 0 would re-break
    the dissipation we spent the thread building. So J and R here are free learned
    operators, not certified GENERIC brackets.
    """

    def __init__(self, dim: int, hidden: int = 128, fixed_J: bool = False,
                 r_init_std: float = 0.0):
        super().__init__()
        self.dim = dim
        # fixed_J: replace the learned skew J(z) with a CONSTANT canonical symplectic
        # matrix J0 = [[0, I], [-I, 0]]. For any constant skew J, div(J grad H) =
        # sum_ij J_ij Hess(H)_ij = 0 (skew contracted with the symmetric Hessian),
        # so the reversible term is EXACTLY divergence-free. That closes the loophole
        # from milestone 3, where the state-dependent J supplied the volume
        # contraction and left R at zero: with J0 fixed, div(f) = -div(R grad S), so
        # R is the ONLY thing that can contract latent volume.
        self.fixed_J = fixed_J
        if fixed_J:
            assert dim % 2 == 0, "fixed_J needs an even latent dim for [[0,I],[-I,0]]"
            h = dim // 2
            J0 = torch.zeros(dim, dim)
            J0[:h, h:] = torch.eye(h)
            J0[h:, :h] = -torch.eye(h)
            self.register_buffer("J0", J0)
        self.h_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1),
        )
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1),
        )
        self.j_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, dim * dim),
        )
        self.l_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Softplus(),
            nn.Linear(hidden, dim * dim),
        )
        # R = L Lᵀ init. DEAD-SADDLE WARNING (milestone 4): with L zero-init,
        # dR/dL = 0 at L=0, so NO gradient can ever lift R off zero and the
        # dissipation matrix stays EXACTLY 0 forever (the real reason R never
        # engaged in every earlier run). r_init_std > 0 breaks that saddle with a
        # small nonzero L so R can actually be learned; r_init_std = 0 keeps the old
        # dissipation-free (and, as we found, dissipation-STUCK) behavior.
        if r_init_std > 0:
            nn.init.normal_(self.l_net[-1].weight, std=r_init_std)
            nn.init.normal_(self.l_net[-1].bias, std=r_init_std)
        else:
            nn.init.zeros_(self.l_net[-1].weight)
            nn.init.zeros_(self.l_net[-1].bias)

    def energy(self, z):    # z: (..., dim) -> (..., 1)
        return self.h_net(z)

    def entropy(self, z):   # z: (..., dim) -> (..., 1)
        return self.s_net(z)

    def J_mat(self, z):
        lead = z.shape[:-1]
        if self.fixed_J:
            return self.J0.expand(*lead, self.dim, self.dim)   # constant, div-free
        A = self.j_net(z).view(*lead, self.dim, self.dim)
        return A - A.transpose(-1, -2)               # strictly skew-symmetric

    def R_mat(self, z):
        lead = z.shape[:-1]
        L = self.l_net(z).view(*lead, self.dim, self.dim)
        return (L @ L.transpose(-1, -2)) / self.dim  # PSD, scale-bounded

    def time_derivative(self, z):
        """dz/dt = J ∇H - R ∇S, gradients via autograd.  z:(B,dim)->(B,dim)"""
        z = z.requires_grad_(True)
        H = self.energy(z).sum()
        S = self.entropy(z).sum()
        gH = torch.autograd.grad(H, z, create_graph=True, retain_graph=True)[0]
        gS = torch.autograd.grad(S, z, create_graph=True, retain_graph=True)[0]
        J = self.J_mat(z)
        R = self.R_mat(z)
        dz = (torch.matmul(J, gH.unsqueeze(-1))
              - torch.matmul(R, gS.unsqueeze(-1))).squeeze(-1)
        return dz

    def step(self, z, dt, substeps=1):
        """RK2 (midpoint) step, sub-stepped. Explicit Euler on the skew part grows
        energy; midpoint is far better behaved on the oscillatory (skew) flow."""
        h = dt / substeps
        for _ in range(substeps):
            k1 = self.time_derivative(z)
            k2 = self.time_derivative(z + 0.5 * h * k1)
            z = z + h * k2
        return z

    def rollout(self, z0, steps, dt, integrator=None, detach_steps=False, substeps=1):
        # z0: (B, dim) -> traj: (B, steps+1, dim)
        z = z0
        traj = [z]
        for _ in range(steps):
            z = self.step(z, dt, substeps=substeps)
            if detach_steps:
                z = z.detach()
            traj.append(z)
        return torch.stack(traj, dim=1)

    def diss_trace(self, z):
        """Mean dissipation rate tr(R(z))/dim along a (B, T, dim) trajectory -> (B,T)."""
        R = self.R_mat(z)
        return R.diagonal(dim1=-2, dim2=-1).sum(-1) / self.dim

    def field_divergence(self, z, n_probe: int = 2):
        """Divergence of the learned latent field  div f = tr(d f / d z),
        f(z) = J(z) gradH - R(z) gradS, as a DIFFERENTIABLE Hutchinson estimate:

            tr(A) = E_eps[ eps^T A eps ],   eps Rademacher,
            A = df/dz,   eps^T (df/dz) eps = eps . grad_z( (f . eps).sum() ).

        create_graph=True everywhere keeps the estimate differentiable (it is a
        SECOND derivative of H/S, so training on it backprops third order into the
        H/S/J/R nets), which is what lets the invariant-dissipation loss pull R up.

        z: (M, dim) -> (M,)  per-point local volume-contraction rate (1/time).

        NOTE on the target: the TIME-AVERAGE of this divergence is the coordinate-
        invariant sum of the Lyapunov spectrum (-13.667 for Lorenz). The physical
        Lorenz divergence is constant -13.667 everywhere, so a per-point target is
        exactly right in physical coordinates; under the nonlinear warp the latent
        field's divergence is no longer spatially constant, so a per-point MSE is a
        slightly stricter (variance-penalizing) surrogate for the true invariant.
        """
        z = z.requires_grad_(True)
        f = self.time_derivative(z)                       # (M, dim), graph to params + z
        est = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for _ in range(n_probe):
            eps = (torch.randint(0, 2, z.shape, device=z.device) * 2 - 1).to(z.dtype)
            g = torch.autograd.grad((f * eps).sum(), z,
                                    create_graph=True, retain_graph=True)[0]  # (df/dz)^T eps
            est = est + (g * eps).sum(-1)                 # eps^T (df/dz) eps
        return est / n_probe

    def finite_time_spectrum(self, z0, dt, steps: int = 6, m: int = 4, h: float = 1e-3):
        """DIFFERENTIABLE continuous-QR (Benettin) estimate of the top-m finite-time
        Lyapunov exponents of the learned latent field, tracked along the rollout.

        Method (classic Benettin with reorthonormalization, made differentiable):
          - carry an orthonormal tangent frame Q in (B, dim, m);
          - at each base state z_t, form the Jacobian action J_local @ Q by
            FINITE-DIFFERENCE JVP  (f(z + h q_j) - f(z)) / h  per column j -- this
            avoids nesting autograd inside autograd (no third-order graph) while
            staying differentiable w.r.t. the H/S/R/J parameters, which is the CPU
            trick that makes spectral supervision affordable;
          - advance the tangent one linearized step  V = Q + dt * (J_local @ Q),
            then reorthonormalize with a DIFFERENTIABLE QR: Q_new, R = qr(V);
          - accumulate log|diag(R)| -- those are the per-step local expansion rates;
          - lambda_hat_i = (1/(steps*dt)) * sum_t log|R_ii^(t)|.

        z0: (B, dim) base states (positions detached upstream so we shape the FIELD,
        not the trajectory). Returns lyap: (B, m) finite-time exponents, differentiable.

        HONEST CAVEAT: over steps=6 (0.06 Lyapunov-time units) these are NOISY
        finite-time exponents, not converged asymptotic ones. They are a usable
        training proxy for lambda1; the decisive check is the 200-step eval lambda1.
        """
        B, dim = z0.shape
        # Random orthonormal starting frame (B, dim, m).
        Q, _ = torch.linalg.qr(torch.randn(B, dim, m, device=z0.device, dtype=z0.dtype))
        z = z0
        log_growth = torch.zeros(B, m, device=z0.device, dtype=z0.dtype)
        for _ in range(steps):
            f0 = self.time_derivative(z)                          # (B, dim)
            cols = []
            for j in range(m):
                qj = Q[:, :, j]                                   # (B, dim)
                fj = self.time_derivative(z + h * qj)            # (B, dim)
                cols.append((fj - f0) / h)                        # ~ J_local @ q_j
            JQ = torch.stack(cols, dim=-1)                        # (B, dim, m)
            V = Q + dt * JQ                                       # linearized tangent step
            Q, R = torch.linalg.qr(V)                             # differentiable QR
            diagR = torch.diagonal(R, dim1=-2, dim2=-1).abs().clamp_min(1e-12)  # (B, m)
            log_growth = log_growth + torch.log(diagR)
            z = self.step(z, dt).detach()                         # advance base state (no graph)
        return log_growth / (steps * dt)                          # (B, m)

    def dissipation_residual(self, traj, dt):
        """Metriplectic energy-balance consistency (analogue of the port penalty):

            dH/dt = -∇Hᵀ R ∇S   =>   residual = H_{t+1} - H_t + dt*(∇Hᵀ R ∇S).

        Same caveat as the port version: R = 0 is a trivial minimizer whenever H
        is already ~conserved, so this term cannot by itself force dissipation on.
        """
        B, T, D = traj.shape
        H = self.energy(traj.detach()).squeeze(-1)                 # (B, T)
        zf = traj[:, :-1].reshape(-1, D).detach().requires_grad_(True)
        Hf = self.energy(zf).sum()
        Sf = self.entropy(zf).sum()
        gH = torch.autograd.grad(Hf, zf, create_graph=True, retain_graph=True)[0]
        gS = torch.autograd.grad(Sf, zf, create_graph=True, retain_graph=True)[0]
        R = self.R_mat(zf)
        power = torch.einsum("mi,mij,mj->m", gH, R, gS).view(B, T - 1)  # ∇Hᵀ R ∇S
        residual = H[:, 1:] - H[:, :-1] + dt * power
        return (residual ** 2).mean()


# ---------------------------------------------------------------------------
# 4a. SIGReg  (Sketched Isotropic Gaussian Regularization)  — anti-collapse
# ---------------------------------------------------------------------------
def z_score_tensor(t: torch.Tensor, dim=None, eps: float = 1e-5) -> torch.Tensor:
    """Standardize a tensor to zero mean / unit std.

    Used to make the energy anchor SCALE-FREE. Matching raw magnitudes forced H to
    stay as small as the raw kinetic proxy (spread ~0.036), which clamped dH/dp and
    froze the dynamics (motion needs ||dH/dp||~3.2, a ~30x conflict). Comparing
    z-scores keeps only the SHAPE constraint -- H's peaks and valleys must line up
    with when the raw data moves fast or slow -- while leaving H's absolute scale
    free to grow to whatever dq = dt * dH/dp requires.
    """
    if dim is None:
        return (t - t.mean()) / (t.std() + eps)
    mean = t.mean(dim=dim, keepdim=True)
    std = t.std(dim=dim, keepdim=True)
    return (t - mean) / (std + eps)


def sigreg_loss(
    z: torch.Tensor,
    n_projections: int = 64,
    n_freqs: int = 16,
    freq_scale: float = 2.0,
) -> torch.Tensor:
    """Push the embedding distribution toward an isotropic standard Gaussian.

    Idea (LeJEPA / SIGReg): if *every* 1-D projection u^T z of the embeddings is
    distributed as N(0, 1), then the joint distribution is isotropic Gaussian.
    We sketch the distribution with many random unit directions u, and for each
    direction compare the empirical characteristic function (ECF) of the
    projected samples to the known Gaussian characteristic function
    phi(t) = exp(-t^2 / 2), an Epps-Pulley style goodness-of-fit test. Matching
    the full CF simultaneously constrains mean, variance, and all higher moments,
    so the only fixed point is a true standard Gaussian -- collapse to a point or
    to a low-rank subspace is heavily penalized. No teacher / EMA required.

    z: (B, D)  ->  scalar loss
    """
    B, D = z.shape

    # Random unit projection directions.  u: (D, n_projections)
    u = torch.randn(D, n_projections, device=z.device, dtype=z.dtype)
    u = F.normalize(u, dim=0)

    proj = z @ u  # (B, n_projections)  each column is a 1-D sketch of the batch

    # Evaluation frequencies for the characteristic function.  t: (n_freqs,)
    t = torch.linspace(0.1, freq_scale * math.pi, n_freqs, device=z.device, dtype=z.dtype)

    # Empirical characteristic function per projection:
    #   ECF(t) = mean_b exp(i t x_b) = <cos(t x)> + i <sin(t x)>
    # proj: (B, P) ; t: (F,)  ->  angles: (B, P, F)
    angles = proj.unsqueeze(-1) * t.view(1, 1, -1)
    ecf_re = torch.cos(angles).mean(dim=0)  # (P, F)
    ecf_im = torch.sin(angles).mean(dim=0)  # (P, F)

    # Target Gaussian CF is real: phi(t) = exp(-t^2 / 2).  (F,)
    gauss_re = torch.exp(-0.5 * t ** 2).view(1, -1)  # (1, F)

    # Squared distance between ECF and Gaussian CF, averaged over freqs & projections.
    loss = (ecf_re - gauss_re) ** 2 + ecf_im ** 2  # (P, F)
    return loss.mean()


# ---------------------------------------------------------------------------
# 4b. The full model wrapper + losses
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    align: float = 1.0
    sigreg: float = 1.0
    conserve: float = 0.1   # weight on the energy-conservation / dissipation penalty
    anchor: float = 0.0     # weight on the kinetic-proxy energy anchor
    motion: float = 0.0     # weight on the motion-matching (anti-freeze) penalty
    contract: float = 0.0   # weight on the divergence/dissipation GATE (metriplectic)
    spectral: float = 0.0   # weight on the leading-Lyapunov (lambda1) alignment (metriplectic)


class EBHJepa(nn.Module):
    """Ties the encoder and Hamiltonian predictor together.

    Training objective per (x_t, x_{t+1}) pair:
      - alignment: MSE( predict(encode(x_t)) , encode(x_{t+1}) )
      - sigreg   : SIGReg on the encoder outputs (both t and t+1)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        dt: float = 0.1,
        integrator: str = "symplectic",
        predictor_type: str = "hamiltonian",   # "hamiltonian"|"port"|"mlp"|"metriplectic"
        conserve_mode: str = "flat",           # "flat" | "dissipative" | "none"
        weights: LossWeights | None = None,
        contract_target: float = -(10.0 + 1.0 + 8.0 / 3.0),  # -13.667 (Lorenz div f)
        metriplectic_fixed_J: bool = False,   # constant div-free J (closes the loophole)
        metriplectic_r_init_std: float = 0.0,  # >0 breaks the R=LLᵀ dead-saddle init
        spectral_lam1_target: float = 0.884,  # target leading Lyapunov exponent (Lorenz)
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, **encoder_kwargs)
        half = latent_dim // 2
        if predictor_type == "hamiltonian":
            self.predictor = HamiltonianSystem(half_dim=half)
        elif predictor_type == "port":
            self.predictor = PortHamiltonianSystem(half_dim=half)
        elif predictor_type == "mlp":
            self.predictor = MLPPredictor(dim=latent_dim)
        elif predictor_type == "metriplectic":
            self.predictor = MetriplecticSystem(dim=latent_dim, fixed_J=metriplectic_fixed_J,
                                                r_init_std=metriplectic_r_init_std)
        else:
            raise ValueError(f"unknown predictor_type {predictor_type!r}")
        self.predictor_type = predictor_type
        self.conserve_mode = conserve_mode
        self.dt = dt
        self.integrator = integrator
        self.weights = weights or LossWeights()
        self.contract_target = contract_target  # invariant contraction target (sum of Lyap spectrum)
        self.spectral_lam1_target = spectral_lam1_target  # target leading exponent lambda1

    def energy_curve(self, traj: torch.Tensor) -> torch.Tensor:
        """Learned energy H at each step of a trajectory, dispatching on predictor.
        traj: (B, T, D) -> (B, T). Metriplectic reads H off the full z; the q/p
        models split first. (The MLP baseline has no energy and never calls this.)"""
        if self.predictor_type == "metriplectic":
            return self.predictor.energy(traj).squeeze(-1)
        q, p = split_latent(traj)
        return self.predictor.energy(q, p).squeeze(-1)

    def forward(self, x_seq: torch.Tensor) -> dict:
        """Multi-step (rollout-consistency) objective.

        Instead of matching a single next step, we encode a whole window of
        K+1 consecutive frames, unroll the Hamiltonian predictor K steps from the
        first latent, and align EVERY predicted step to its encoded target. This
        shapes H along trajectories, not just at isolated points, which is what a
        long-horizon model actually needs.

        x_seq: (B, K+1, Xdim)  ->  dict of losses
        """
        B, Kp1, *obs = x_seq.shape
        K = Kp1 - 1

        # Encode every frame in the window.  z_seq: (B, K+1, D)
        z_seq = self.encoder(x_seq.reshape(B * Kp1, *obs)).reshape(B, Kp1, -1)

        # Roll the latent forward K steps starting from z_seq[:, 0].
        # traj: (B, K+1, D)  (index 0 is the initial state, kept for the energy trace)
        traj = self.predictor.rollout(z_seq[:, 0], steps=K, dt=self.dt, integrator=self.integrator)

        # (1) Alignment across the whole rollout: predicted steps 1..K vs targets 1..K.
        align = F.mse_loss(traj[:, 1:], z_seq[:, 1:])            # scalar

        # (2) SIGReg on all encoded frames at once (flatten batch and time).
        reg = sigreg_loss(z_seq.reshape(B * Kp1, -1))            # scalar

        # (3) Physics-consistency penalty, adaptive to the assumed dynamics:
        #   - "flat"       : force energy constant along the rollout (var penalty).
        #                    Correct only for conservative systems.
        #   - "dissipative": force the energy drop to match the dissipated work
        #                    (port-Hamiltonian energy balance). Correct for damped
        #                    systems; does NOT force energy flat.
        #   - "none"       : no physics prior (the pure-MLP baseline).
        if self.conserve_mode == "flat":
            energies = self.energy_curve(traj)                  # (B, K+1)
            conserve = energies.var(dim=1).mean()
        elif self.conserve_mode == "dissipative":
            conserve = self.predictor.dissipation_residual(traj, self.dt)
        else:  # "none"
            conserve = torch.zeros((), device=traj.device, dtype=traj.dtype)

        # (4) DATA-SPACE kinetic anchor (replaces the failed latent-space anchor).
        #   The earlier version anchored H to 0.5*||z_{t+1}-z_t||^2, a target built
        #   from the ENCODER's own outputs. That target is shrinkable: the network
        #   minimized it by collapsing H to a constant (measured std 0.00000), so R
        #   never engaged. Here the proxy is computed from the RAW observations,
        #   under no_grad, so it is immutable -- the network cannot move the target.
        #
        #       dx/dt          = x_{t+1} - x_t                     (B, K, Xdim)
        #       data_kinetic_t = 0.5 * sum_i (dx/dt)_i^2           (B, K)
        #       L_anchor       = MSE( H(traj_t) , data_kinetic_t )
        #
        #   H is tied to the PREDICTED rollout energies (not encoder states), so the
        #   constraint acts directly on the quantity whose decay we care about. On
        #   damped data the raw frame-to-frame motion decays, so a constant H can no
        #   longer satisfy this, and the port model's only route is to raise R.
        #   NOTE: this anchors H to a KINETIC proxy, while a Hamiltonian is total
        #   energy (kinetic + potential). Kinetic energy oscillates within an orbit,
        #   so expect H to oscillate at orbital frequency, not decay monotonically.
        if self.predictor_type in ("hamiltonian", "port", "metriplectic") and self.weights.anchor > 0:
            with torch.no_grad():
                dx = (x_seq[:, 1:] - x_seq[:, :-1]).reshape(B, K, -1)   # (B, K, Xdim)
                data_kinetic = 0.5 * (dx ** 2).sum(dim=-1)              # (B, K)
            E_pred = self.energy_curve(traj)[:, :-1]                    # (B, K)
            # Z-SCORE STANDARDIZED anchor (scale-free).
            # History: raw MSE was scale-starved (H collapsed to a constant).
            # Variance-normalizing fixed the starvation but kept the MAGNITUDE
            # constraint, which then fought the motion penalty: the anchor demanded
            # H change <=0.036 per step while motion demanded ~1.0, a ~30x conflict
            # that froze the dynamics (measured motion ratio ~0.013).
            # Standardizing BOTH sides drops magnitude entirely and keeps only the
            # correlation/shape content, which is all the anchor was ever meant to
            # supply. Note MSE(zscore(a), zscore(b)) = 2*(1 - corr(a,b)), so this
            # term is O(1): ~2.0 when uncorrelated, ~0 when perfectly correlated.
            anchor = F.mse_loss(z_score_tensor(E_pred), z_score_tensor(data_kinetic))
        else:
            anchor = torch.zeros((), device=traj.device, dtype=traj.dtype)

        # (5) MOTION-MATCHING penalty -- the anti-freeze lever.
        #   Diagnosis it targets: the predictor collapsed into a no-op (per-step
        #   motion 0.003 vs target 0.4), because alignment alone made "don't move"
        #   a cheap local optimum (it ties the persistence baseline). Matching the
        #   MAGNITUDE of per-step displacement makes freezing explicitly expensive.
        #
        #   NOTE: the target magnitude is DETACHED. If it were not, the encoder
        #   could satisfy this loss by slowing its OWN motion down to meet the
        #   frozen predictor -- the same shrinkable-target failure that made the
        #   latent kinetic anchor collapse. Detaching forces the predictor to speed
        #   up rather than letting the target come down to it.
        pred_disp = traj[:, 1:] - traj[:, :-1]                  # (B, K, D)
        targ_disp = (z_seq[:, 1:] - z_seq[:, :-1]).detach()     # (B, K, D)
        pred_motion = pred_disp.norm(p=2, dim=-1)               # (B, K)
        targ_motion = targ_disp.norm(p=2, dim=-1)               # (B, K)
        motion = ((pred_motion - targ_motion) ** 2).mean()

        # (6) INVARIANT DISSIPATION GATE (metriplectic only).
        #   Pull the divergence of the learned latent field toward the coordinate-
        #   invariant contraction target (-13.667 = sum of the Lorenz Lyapunov
        #   spectrum). This is the ONLY term that structurally pressures R(z) to grow:
        #   J is skew (trace-free) so the reversible part contributes ~0 to div f, and
        #   div f = -tr(dR/dz-weighted gradS ...) is dominated by the dissipative part.
        #   Positions are DETACHED so the loss reshapes the FIELD (H/S/R) at the
        #   visited states without dragging the trajectory. Points are subsampled to
        #   cap the cost of the third-order autograd graph.
        #   HONEST NOTE: this pins the divergence, i.e. the SUM of the spectrum. It
        #   does NOT pin lambda_1 individually -- the run below tests whether there is
        #   any secondary pull on lambda_1 or whether the axes stay decoupled.
        if self.predictor_type == "metriplectic" and self.weights.contract > 0:
            Dz = traj.shape[-1]
            zc = traj[:, :-1].reshape(-1, Dz).detach()
            if zc.shape[0] > 32:
                sel = torch.randperm(zc.shape[0], device=zc.device)[:32]
                zc = zc[sel]
            div = self.predictor.field_divergence(zc, n_probe=1)     # (M,)
            contract = ((div - self.contract_target) ** 2).mean()    # per-point MSE
            div_mean = div.mean().detach()
        else:
            contract = torch.zeros((), device=traj.device, dtype=traj.dtype)
            div_mean = torch.zeros((), device=traj.device, dtype=traj.dtype)

        # (7) SPECTRAL ALIGNMENT (metriplectic): pin the LEADING Lyapunov exponent
        #   lambda1 via the differentiable continuous-QR tracker. This is the piece the
        #   scalar divergence gate structurally could not supply -- it constrains an
        #   INDIVIDUAL exponent, not just the spectrum sum, which is the only way to
        #   close the lambda1-decoupling gap (E hit the sum with lambda1=+3.5). The
        #   sum term is left to the `contract` gate above; together they realize
        #   L = alpha (lam1_hat - 0.884)^2 + beta (sum_i lam_i - (-13.667))^2.
        if self.predictor_type == "metriplectic" and self.weights.spectral > 0:
            z0s = traj[:, 0].detach()
            if z0s.shape[0] > 16:
                z0s = z0s[torch.randperm(z0s.shape[0], device=z0s.device)[:16]]
            ft = self.predictor.finite_time_spectrum(z0s, self.dt, steps=6, m=4)  # (b, 4)
            lam1_hat = ft.max(dim=-1).values                     # leading finite-time exponent
            spectral = ((lam1_hat - self.spectral_lam1_target) ** 2).mean()
            lam1_mean = lam1_hat.mean().detach()
        else:
            spectral = torch.zeros((), device=traj.device, dtype=traj.dtype)
            lam1_mean = torch.zeros((), device=traj.device, dtype=traj.dtype)

        total = (
            self.weights.align * align
            + self.weights.sigreg * reg
            + self.weights.conserve * conserve
            + self.weights.anchor * anchor
            + self.weights.motion * motion
            + self.weights.contract * contract
            + self.weights.spectral * spectral
        )

        return {
            "loss": total,
            "align": align.detach(),
            "sigreg": reg.detach(),
            "conserve": conserve.detach(),
            "anchor": anchor.detach(),
            "motion": motion.detach(),
            "contract": contract.detach(),
            "div": div_mean,          # live mean latent divergence (target -13.667)
            "spectral": spectral.detach(),
            "lam1_hat": lam1_mean,    # live leading finite-time exponent (target 0.884)
            # live motion ratio: ~1.0 means the predictor moves at the right speed
            "motion_ratio": (pred_motion.mean() / targ_motion.mean().clamp_min(1e-8)).detach(),
            "z_t": z_seq[:, 0].detach(),
        }


# ---------------------------------------------------------------------------
# Mock training loop with synthetic data
# ---------------------------------------------------------------------------
def make_synthetic_sequence(
    batch: int, obs_dim: int, dt: float, horizon: int, device: str
) -> torch.Tensor:
    """Generate windows of `horizon`+1 consecutive frames from a real 2-D
    harmonic oscillator lifted into a high-dim observation. The ground-truth
    dynamics ARE Hamiltonian, a fair target for a Hamiltonian predictor.

    returns x_seq: (batch, horizon+1, obs_dim)
    """
    theta = torch.rand(batch, 1, device=device) * 2 * math.pi
    r = 0.5 + torch.rand(batch, 1, device=device)
    pos = r * torch.cos(theta)
    vel = r * torch.sin(theta)

    # Fixed lift so the observation map is identical across runs.
    gen = torch.Generator(device="cpu").manual_seed(0)
    W = torch.randn(2, obs_dim, generator=gen).to(device)

    def lift(pos, vel):
        s = torch.cat([pos, vel], dim=-1)  # (batch, 2)
        return torch.tanh(s @ W)           # (batch, obs_dim)

    frames = [lift(pos, vel)]
    for _ in range(horizon):
        # Advance the true oscillator by a phase-space rotation of angle dt.
        pos, vel = (
            pos * math.cos(dt) + vel * math.sin(dt),
            -pos * math.sin(dt) + vel * math.cos(dt),
        )
        frames.append(lift(pos, vel))
    return torch.stack(frames, dim=1)  # (batch, horizon+1, obs_dim)


# ---------------------------------------------------------------------------
# 5. CHAOTIC DATA: dissipative Lorenz-63 attractor + nonlinear lift + nonlinear warp
# ---------------------------------------------------------------------------
# Standard chaotic Lorenz parameters (sigma, rho, beta).
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
# Phase-space volume-contraction rate of the Lorenz flow:
#     div f = d(dx)/dx + d(dy)/dy + d(dz)/dz = -sigma - 1 - beta
# It is CONSTANT (state-independent) and, crucially, equals the SUM of the Lyapunov
# spectrum (l1 + l2 + l3 ~ 0.906 + 0 - 14.57 = -13.66). The sum-of-spectrum form is
# COORDINATE-INVARIANT; the raw divergence is NOT. This is the exact macroscopic
# dissipation rate the "thermodynamic consistency" metric is checked against.
LORENZ_DIVERGENCE = -(LORENZ_SIGMA + 1.0 + LORENZ_BETA)   # = -13.6667
LORENZ_LAMBDA1 = 0.9056  # textbook largest Lyapunov exponent of Lorenz-63

# Fixed affine map raw attractor (x,y ~ [-20,20], z ~ [0,50]) -> normalized ~[-1,1],
# so the nonlinear lift/warp act on a BOUNDED domain. That bound is not cosmetic: it
# is exactly what keeps the sinusoidal warp an honest diffeomorphism (see below).
_LZ_SCALE = torch.tensor([20.0, 25.0, 25.0])
_LZ_SHIFT = torch.tensor([0.0, 0.0, 25.0])


def lorenz_rhs(state: torch.Tensor) -> torch.Tensor:
    """Lorenz-63 vector field.  state: (..., 3) -> (..., 3).

        dx/dt = sigma (y - x)
        dy/dt = x (rho - z) - y
        dz/dt = x y - beta z
    """
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = LORENZ_SIGMA * (y - x)
    dy = x * (LORENZ_RHO - z) - y
    dz = x * y - LORENZ_BETA * z
    return torch.stack([dx, dy, dz], dim=-1)


def _lorenz_rk4(state: torch.Tensor, dt: float) -> torch.Tensor:
    """One classical RK4 step of Lorenz (small local error; the true flow is
    dissipative so no symplectic integrator is needed for the ground truth)."""
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _lift_params(obs_dim: int, seed: int = 0):
    """Fixed random parameters of the nonlinear observation lift (3 -> obs_dim)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(3, obs_dim, generator=g) * 0.8       # random mixing of x,y,z
    b = torch.randn(obs_dim, generator=g) * 0.5          # random phases/offsets
    freq = 1.0 + 2.0 * torch.rand(obs_dim, generator=g)  # random sinusoid freqs
    # Per-channel CONVEX weights over {sin, cubic, exp}: each lifted feature is a
    # different random nonlinear combination, and because the weights sum to 1 and
    # every basis is bounded in [-1,1], each feature stays in [-1,1].
    mix = torch.rand(obs_dim, 3, generator=g)
    mix = mix / mix.sum(dim=-1, keepdim=True)
    return W, b, freq, mix


def nonlinear_lift(state_norm: torch.Tensor, params) -> torch.Tensor:
    """Lift normalized 3-D Lorenz state to obs_dim via random NON-LINEAR functions
    (sinusoid + cubic + exponential), replacing the old linear tanh lift.

    state_norm: (..., 3) in ~[-1,1]  ->  (..., obs_dim) in [-1, 1]

    channel_j = mix_j0 * sin(freq_j * pre) + mix_j1 * tanh(pre^3)
              + mix_j2 * tanh(exp(pre) - 1),      pre = state_norm @ W + b

    All three bases are bounded and mix is a convex combination, so every feature
    lands in [-1, 1]. That bound is what makes the downstream warp a diffeomorphism.
    """
    W, b, freq, mix = params
    W = W.to(state_norm); b = b.to(state_norm)
    freq = freq.to(state_norm); mix = mix.to(state_norm)
    pre = state_norm @ W + b                              # (..., obs_dim)
    f_sin = torch.sin(freq * pre)                         # sinusoid, bounded
    f_cub = torch.tanh(pre ** 3)                          # cubic,     bounded
    f_exp = torch.tanh(torch.exp(pre) - 1.0)             # exponential, bounded
    return mix[:, 0] * f_sin + mix[:, 1] * f_cub + mix[:, 2] * f_exp


def nonlinear_warp(x: torch.Tensor) -> torch.Tensor:
    """Element-wise nonlinear coordinate warp (a diffeomorphism on the bounded
    lifted domain):   x_warped = x + 0.2 * sin(2x) * cos(x^2).

    WHY: a *linear* scramble x @ M is undone by the encoder's first Linear layer
    (it just learns M^{-1}), so it tests nothing. This warp is nonlinear and
    non-separable, so no single linear map inverts it -- the encoder is forced to
    use its nonlinear layers to build a local coordinate chart before J/R/H/S can
    balance anything.

    HONEST INVERTIBILITY NOTE: g'(x) = 1 + 0.2[2 cos2x cos x^2 - 2x sin2x sin x^2].
    The 2x term drives g' below 0 once |x| >~ 1.6, so g is NOT globally invertible.
    We keep it a genuine diffeomorphism by bounding the lift to [-1,1]
    (nonlinear_lift), where g' > 0 throughout. Invertible ON OUR DOMAIN is the
    honest and sufficient statement -- we do not claim a global diffeomorphism.
    """
    return x + 0.2 * torch.sin(2.0 * x) * torch.cos(x ** 2)


def make_lorenz_sequence(
    batch: int,
    obs_dim: int,
    dt: float,
    horizon: int,
    device: str = "cpu",
    warp: bool = True,
    burn_in: int = 500,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows of a DISSIPATIVE CHAOTIC Lorenz-63 trajectory, nonlinearly lifted to
    obs_dim and (optionally) nonlinearly warped -- the chaos stress test that
    replaces the linear damped oscillator.

    Per sample:
      1. random initial condition, integrated `burn_in` RK4 steps to land ON the
         attractor (transient discarded), then horizon+1 frames are recorded;
      2. each 3-D state is normalized to ~[-1,1] and lifted to obs_dim by a FIXED
         random NONLINEAR map (sin/cubic/exp), not a linear matrix;
      3. if warp=True, the bounded lifted vector is pushed through the nonlinear
         diffeomorphism nonlinear_warp (the "hard" coordinate distortion).

    returns:
        x_seq    : (batch, horizon+1, obs_dim)  lifted (and warped) observations
        true_xyz : (batch, horizon+1, 3)        raw Lorenz states (for chaos metrics)
    """
    g = torch.Generator(device="cpu")
    if seed is None:
        g.seed()                     # fresh nondeterministic ICs each training call
    else:
        g.manual_seed(seed)          # reproducible ICs for evaluation
    # Spread initial conditions across the attractor's typical extent, then burn in.
    span = torch.tensor([15.0, 20.0, 20.0])
    centre = torch.tensor([0.0, 0.0, 25.0])
    state = ((torch.rand(batch, 3, generator=g) * 2 - 1) * span + centre).to(device)
    for _ in range(burn_in):
        state = _lorenz_rk4(state, dt)

    params = _lift_params(obs_dim)
    scale = _LZ_SCALE.to(device); shift = _LZ_SHIFT.to(device)

    frames, xyz = [], []
    s = state
    for _ in range(horizon + 1):
        xyz.append(s)
        s_norm = (s - shift) / scale                     # (batch, 3) ~[-1,1]
        feat = nonlinear_lift(s_norm, params)            # (batch, obs_dim) in [-1,1]
        if warp:
            feat = nonlinear_warp(feat)                  # nonlinear diffeomorphism
        frames.append(feat)
        s = _lorenz_rk4(s, dt)                           # next true state (RK4)
    x_seq = torch.stack(frames, dim=1)                   # (batch, horizon+1, obs_dim)
    true_xyz = torch.stack(xyz, dim=1)                   # (batch, horizon+1, 3)
    return x_seq, true_xyz


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scale/shift-invariant correlation between two 1-D curves over time."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a @ b / denom).item())


def train_stress(model, tag, train_horizon, steps, cfg, warp=True):
    """Train one model on the CHAOTIC Lorenz data and log its losses."""
    device, obs_dim, latent_dim, dt, batch = cfg
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"\n=== {tag} | predictor={model.predictor_type} "
          f"conserve_mode={model.conserve_mode} train_horizon={train_horizon} ===")
    model.train()
    for step in range(1, steps + 1):
        # --- PROGRESSIVE HORIZON CURRICULUM ---
        # Starting straight at K=6 rewards the safe frozen solution: compounding
        # multi-step error makes "don't move" the cheapest option. Begin at K=1 so
        # the predictor's per-step map can grow large enough to match a single step,
        # then extend the horizon once real motion has been learned.
        frac = (step - 1) / steps
        if frac < 0.30:
            K = 1
        elif frac < 0.60:
            K = 3
        else:
            K = train_horizon
        x_seq, _ = make_lorenz_sequence(batch, obs_dim, dt, K, device, warp=warp)
        out = model(x_seq)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | K={K} | total {out['loss'].item():.4f} "
                  f"| align {out['align'].item():.4f} "
                  f"| sigreg {out['sigreg'].item():.4f} "
                  f"| conserve {out['conserve'].item():.4f} "
                  f"| anchor {out['anchor'].item():.4f} "
                  f"| motion {out['motion'].item():.4f} "
                  f"| contract {out['contract'].item():.4f} "
                  f"| div {out['div'].item():+.3f} "
                  f"| spec {out['spectral'].item():.3f} "
                  f"| lam1h {out['lam1_hat'].item():+.3f} "
                  f"| mratio {out['motion_ratio'].item():.3f}")
    return model


# ---------------------------------------------------------------------------
# 6a. CHAOS-AWARE EVALUATION METRICS
# ---------------------------------------------------------------------------
# On a chaotic system a 200-step alignment MSE is meaningless: positive Lyapunov
# exponents blow up any tiny error exponentially, so even a PERFECT model diverges
# pointwise. We therefore judge the rollout on ATTRACTOR-level invariants that a
# good chaotic emulator preserves even after the pointwise trajectory has decorrelated.

def _hausdorff(P: torch.Tensor, Q: torch.Tensor) -> float:
    """Symmetric Hausdorff distance between two point clouds P:(n,d), Q:(m,d):
    the worst-case nearest-neighbor gap in either direction. Large if the predicted
    manifold leaves the true manifold anywhere (spurs, blow-ups, missing lobes)."""
    D = torch.cdist(P, Q)                                   # (n, m)
    return torch.maximum(D.min(dim=1).values.max(),
                         D.min(dim=0).values.max()).item()


def _chamfer(P: torch.Tensor, Q: torch.Tensor) -> float:
    """Symmetric MEAN nearest-neighbor distance ("attractor overlap" proxy). Less
    outlier-sensitive than Hausdorff: measures typical manifold-to-manifold gap."""
    D = torch.cdist(P, Q)
    return 0.5 * (D.min(dim=1).values.mean() + D.min(dim=0).values.mean()).item()


def true_lorenz_lyapunov(dt: float, n_steps: int, device: str, eps: float = 1e-8) -> float:
    """Benettin estimate of the LARGEST Lyapunov exponent of the TRUE Lorenz flow:
    evolve a reference and a shadow trajectory, renormalize their separation back
    to `eps` each step, and average the per-step log growth rate / dt. Should land
    near the textbook 0.906 -- a self-check that our dt and horizon resolve chaos."""
    # Compute in float64: the shadow separation eps=1e-8 is far below float32
    # precision against Lorenz states of magnitude ~20, which would collapse the
    # separation to numerical zero and produce a garbage (hugely negative) estimate.
    g = torch.Generator(device="cpu").manual_seed(7)
    s = ((torch.rand(64, 3, generator=g) * 2 - 1) * torch.tensor([12., 15., 15.])
         + torch.tensor([0., 0., 25.])).to(device).double()
    for _ in range(500):
        s = _lorenz_rk4(s, dt)
    delta = torch.randn(64, 3, generator=g).to(device).double()
    delta = delta / delta.norm(dim=-1, keepdim=True) * eps
    sp = s + delta
    acc = 0.0
    for _ in range(n_steps):
        s = _lorenz_rk4(s, dt); sp = _lorenz_rk4(sp, dt)
        diff = sp - s
        dist = diff.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        acc += torch.log(dist / eps).mean().item()
        sp = s + diff * (eps / dist)                        # renormalize to eps
    return acc / (n_steps * dt)


def rollout_lyapunov(model, z0: torch.Tensor, dt: float, n_steps: int,
                     eps: float = 1e-5) -> float:
    """Largest Lyapunov exponent of the MODEL's LATENT rollout, same Benettin
    scheme, using the predictor's one-step map uniformly across model types.

    The largest Lyapunov exponent is INVARIANT under smooth invertible coordinate
    changes, so this latent-space number is directly comparable to the true Lorenz
    value even though the encoder/warp reshuffle the coordinates. A model that has
    learned the real chaotic scaling should match it; a frozen or over-damped model
    reads ~0 (or negative), an unstable one reads large-positive / non-finite."""
    def one_step(z):
        return model.predictor.rollout(z, steps=1, dt=dt, integrator=model.integrator,
                                        detach_steps=True, substeps=1)[:, -1]
    z = z0.clone()
    delta = torch.randn_like(z)
    delta = delta / delta.norm(dim=-1, keepdim=True) * eps
    zp = z + delta
    acc, valid = 0.0, 0
    for _ in range(n_steps):
        z = one_step(z); zp = one_step(zp)
        diff = zp - z
        dist = diff.norm(dim=-1, keepdim=True)
        m = torch.isfinite(dist).squeeze(-1) & (dist.squeeze(-1) > 1e-20)
        if m.any():
            acc += torch.log(dist[m] / eps).mean().item(); valid += 1
        zp = z + diff * (eps / dist.clamp_min(1e-20))       # renormalize to eps
    return acc / (valid * dt) if valid else float("nan")


def latent_contraction_rate(model, z_points: torch.Tensor, dt: float,
                            n_probe: int = 12, h: float = 1e-3) -> float:
    """Learned phase-space contraction rate = (1/dt) * tr(J_map - I), where J_map is
    the Jacobian of the predictor's one-step latent map. tr(J_map - I)/dt approximates
    div(dz/dt), the analogue of the Lorenz -13.667. Estimated by FINITE-DIFFERENCE
    HUTCHINSON  tr(A) = E[eps^T A eps]  with Rademacher eps and JVP by forward
    difference, so it works UNIFORMLY for every predictor (symplectic Hamiltonian,
    MLP, metriplectic) without nesting autograd inside autograd.

    COORDINATE CAVEAT (stated loudly on purpose): this divergence is measured in the
    LEARNED latent chart, not physical Lorenz coordinates, so under the nonlinear
    warp it is NOT expected to equal -13.667 exactly; the coordinate-invariant target
    is the SUM of the Lyapunov spectrum. We report it as a proxy and read its SIGN and
    scale. Clean built-in check: a ~symplectic Hamiltonian map is volume-preserving,
    so it should read ~0 -- i.e. the closed-system model structurally CANNOT contract."""
    def one_step(z):
        return model.predictor.rollout(z, steps=1, dt=dt, integrator=model.integrator,
                                        detach_steps=True, substeps=1)[:, -1]
    base = one_step(z_points)
    if not torch.isfinite(base).all():
        return float("nan")
    tr = 0.0
    for _ in range(n_probe):
        eps = (torch.randint(0, 2, z_points.shape, device=z_points.device) * 2 - 1
               ).to(z_points.dtype)                          # Rademacher +/-1
        jvp = (one_step(z_points + h * eps) - base) / h      # ~ J_map @ eps
        tr += (eps * (jvp - eps)).sum(dim=-1).mean().item()  # eps^T (J_map - I) eps
    return (tr / n_probe) / dt


# ---------------------------------------------------------------------------
# 6b. THE CHAOS STRESS EVALUATION
# ---------------------------------------------------------------------------
def stress_eval(model, tag, cfg, eval_horizon=200, warp=True, true_lambda=None):
    """200-step rollout on CHAOTIC Lorenz data judged by attractor-level invariants:
    boundedness, attractor overlap (Hausdorff/Chamfer in latent space vs the ENCODED
    ground truth), largest-Lyapunov match, and learned contraction rate vs -13.667."""
    device, obs_dim, latent_dim, dt, batch = cfg
    model.eval()
    x_seq, true_xyz = make_lorenz_sequence(batch, obs_dim, dt, eval_horizon, device,
                                           warp=warp, seed=999)  # fixed eval trajectories
    B, Kp1, X = x_seq.shape
    with torch.enable_grad():  # physics predictors need autograd inside the rollout
        z_seq = model.encoder(x_seq.reshape(B * Kp1, X)).reshape(B, Kp1, -1)  # (B, T, D)
        traj = model.predictor.rollout(
            z_seq[:, 0], steps=eval_horizon, dt=dt,
            integrator=model.integrator, detach_steps=True, substeps=3,
        )
        finite = bool(torch.isfinite(traj).all().item())
        max_norm = traj.norm(dim=-1).max().item() if finite else float("inf")

        # Short-horizon alignment: still meaningful over a few steps (before chaos
        # decorrelates the pointwise trajectory); the LONG MSE is deliberately dropped.
        short = min(20, eval_horizon)
        align_short = (F.mse_loss(traj[:, 1:1 + short], z_seq[:, 1:1 + short].detach()).item()
                       if finite else float("inf"))

        # (1) ATTRACTOR GEOMETRY in latent space: predicted rollout manifold vs the
        #     ENCODED ground-truth manifold (both (T, D) clouds from one sample).
        #     Measured in latent space against encoded truth -- NOT in raw 3-D Lorenz
        #     space, because the JEPA model has no decoder back to (x, y, z).
        if finite:
            Ppred, Ptrue = traj[0].detach(), z_seq[0].detach()   # (T, D) each
            hausd = _hausdorff(Ppred, Ptrue)
            chamf = _chamfer(Ppred, Ptrue)
            # normalize by the true manifold's own diameter for a scale-free overlap
            diam = torch.cdist(Ptrue, Ptrue).max().item() + 1e-8
            chamf_rel = chamf / diam
        else:
            hausd = chamf = chamf_rel = float("inf")

        # (2) LARGEST LYAPUNOV EXPONENT of the model rollout (invariant; comparable
        #     to the true Lorenz value across the coordinate warp).
        lam_model = (rollout_lyapunov(model, z_seq[:, 0].detach(), dt, eval_horizon)
                     if finite else float("nan"))

        # (3) LEARNED CONTRACTION RATE vs -13.667 (proxy; see helper caveat).
        zpts = z_seq[:min(64, B), 0].detach()                    # on-attractor points
        contr = latent_contraction_rate(model, zpts, dt) if finite else float("nan")

        # Dissipation-matrix engagement tr(R)/n (metriplectic only): the direct read
        # on whether R(z) turned on, independent of the divergence proxy.
        r_mag = (model.predictor.diss_trace(traj).mean().item()
                 if (finite and model.predictor_type == "metriplectic") else None)

        pred_speed = ((traj[:, 1:] - traj[:, :-1]).norm(dim=-1).mean().item()
                      if finite else float("inf"))
        targ_speed = (z_seq[:, 1:] - z_seq[:, :-1]).norm(dim=-1).mean().item()

    true_lambda = LORENZ_LAMBDA1 if true_lambda is None else true_lambda
    lam_err = abs(lam_model - true_lambda) if not math.isnan(lam_model) else float("nan")
    contr_err = abs(contr - LORENZ_DIVERGENCE) if not math.isnan(contr) else float("nan")

    print(f"\n[{tag}] 200-step CHAOTIC rollout  (finite={finite}, max||z||={max_norm:.3g})")
    print(f"[{tag}] short-{short} align MSE = {align_short:.4f}   "
          f"per-step motion: pred={pred_speed:.4f} target={targ_speed:.4f}")
    print(f"[{tag}] (1) attractor overlap: Hausdorff={hausd:.4f}  Chamfer={chamf:.4f}  "
          f"(rel={chamf_rel:.3f})   [lower = stays on the encoded attractor]")
    print(f"[{tag}] (2) Lyapunov: model lambda1={lam_model:+.4f}  vs true={true_lambda:+.4f}  "
          f"|err|={lam_err:.4f}   [match = same chaotic divergence rate]")
    print(f"[{tag}] (3) contraction rate (latent proxy) tr(J-I)/dt = {contr:+.4f}  "
          f"vs true div f={LORENZ_DIVERGENCE:.4f}  |err|={contr_err:.4f}")
    if r_mag is not None:
        print(f"[{tag}]     dissipation engagement tr(R)/n = {r_mag:.5f}  "
              f"({'R ENGAGED' if r_mag > 1e-3 else 'R ~OFF'})")
    if model.predictor_type == "hamiltonian":
        print(f"[{tag}]     ^ closed-system Hamiltonian is ~volume-preserving: expect "
              f"~0, i.e. it structurally CANNOT match the -13.667 contraction.")
    elif model.predictor_type == "mlp":
        print(f"[{tag}]     ^ unconstrained MLP: no volume law at all; value is whatever "
              f"the free map happened to learn (often unbounded).")

    return {"tag": tag, "finite": finite, "max_norm": max_norm,
            "align_short": align_short, "hausdorff": hausd, "chamfer": chamf,
            "chamfer_rel": chamf_rel, "lam_model": lam_model, "lam_err": lam_err,
            "contraction": contr, "contraction_err": contr_err, "r_mag": r_mag,
            "motion_ratio": pred_speed / max(targ_speed, 1e-8)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 0.01                              # Lorenz frame spacing (resolves the chaos)
    obs_dim, latent_dim, batch = 32, 64, 128
    cfg = (device, obs_dim, latent_dim, dt, batch)  # dev, OBS, LATENT, DT, BATCH
    STEPS, TRAIN_H, EVAL_H = 250, 6, 200
    WARP = True                            # apply the nonlinear diffeomorphism warp

    print(f"device={device}  CHAOTIC Lorenz-63 stress test "
          f"(sigma={LORENZ_SIGMA}, rho={LORENZ_RHO}, beta={LORENZ_BETA:.3f})  "
          f"| nonlinear warp = {WARP}")
    print(f"reference invariants:  div f = {LORENZ_DIVERGENCE:.4f} "
          f"(= sum of Lyapunov spectrum),  textbook lambda1 = {LORENZ_LAMBDA1}")
    # Self-check: recover the true largest Lyapunov exponent at THIS dt from data.
    lam_true = true_lorenz_lyapunov(dt, 3000, device)
    print(f"empirical TRUE lambda1 (Benettin, dt={dt}) = {lam_true:+.4f}  "
          f"(sanity: should sit near {LORENZ_LAMBDA1})")

    common = dict(latent_dim=latent_dim, dt=dt, mode="vector", in_dim=obs_dim,
                  integrator="symplectic")

    # Weights: (align, sigreg, conserve, anchor, motion). Three arms on the
    # NONLINEARLY WARPED chaotic data:
    # B: unconstrained MLP  (pure JEPA control, no physics)          -- may blow up.
    # C: RIGID Cartesian Hamiltonian (hardcoded q/p, no S/R)         -- MISSPECIFIED:
    #    assumes a CLOSED, volume-preserving system; conserve_mode="flat" bakes in
    #    the "energy is constant" prior. Structurally cannot represent contraction.
    # D: METRIPLECTIC (learnable skew J + PSD R + H + S, NO q/p split) -- the full
    #    GENERIC-style arm; must self-discover a coordinate chart under the warp.
    # LossWeights = (align, sigreg, conserve, anchor, motion, contract).
    model_b = EBHJepa(predictor_type="mlp", conserve_mode="none",
                      weights=LossWeights(1.0, 1.0, 0.0, 0.0, 2.0, 0.0), **common).to(device)
    model_c = EBHJepa(predictor_type="hamiltonian", conserve_mode="flat",
                      weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.0), **common).to(device)
    # D = NAIVE metriplectic: learned J, R zero-init, no gate. The "before" arm --
    # R stays stuck at exactly 0 (dead LLᵀ saddle) and lambda1 overshoots.
    model_d = EBHJepa(predictor_type="metriplectic", conserve_mode="dissipative",
                      weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.0), **common).to(device)
    # E = FIXED metriplectic: constant divergence-free J (closes the J loophole) +
    # nonzero R init (breaks the dead saddle) + invariant dissipation gate. The
    # "after" arm -- R now genuinely engages and the contraction hits -13.667. It
    # does NOT fix lambda1 (a sum constraint cannot), which E's overshoot documents.
    model_e = EBHJepa(predictor_type="metriplectic", conserve_mode="dissipative",
                      weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.02),
                      metriplectic_fixed_J=True, metriplectic_r_init_std=0.05,
                      **common).to(device)
    # F = E plus SPECTRAL ALIGNMENT on the leading exponent via the differentiable
    # continuous-QR tracker (spectral weight > 0). This IN-REPO F uses the SHORT K=6
    # horizon (see MetriplecticSystem.finite_time_spectrum) and demonstrates FAILURE
    # (5a): the 6-step finite-time exponent is decoupled from the asymptotic lambda1,
    # so eval lambda1 stays ~1.44 (== naive), NOT 0.88. The long-horizon (100-step)
    # variant that fixes the training proxy but then blows the global rollout to +inf
    # (failure 5b) is the GPU study in pathB_spectral_colab.ipynb, not this arm.
    model_f = EBHJepa(predictor_type="metriplectic", conserve_mode="dissipative",
                      weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.02, 0.10),
                      metriplectic_fixed_J=True, metriplectic_r_init_std=0.05,
                      spectral_lam1_target=lam_true, **common).to(device)

    results = []
    for model, tag in [(model_b, "B/UNCONSTRAINED-MLP"),
                       (model_c, "C/RIGID-HAMILTONIAN"),
                       (model_d, "D/METRIPLECTIC-NAIVE"),
                       (model_e, "E/METRIPLECTIC-FIXED"),
                       (model_f, "F/SPECTRAL-ALIGNED")]:
        # Reproducibility: identical seed per arm, and isolate each arm so a rare
        # numerical hiccup in one (e.g. a QR/degenerate-tangent event in F) records a
        # failed row instead of aborting the whole benchmark.
        torch.manual_seed(0)
        try:
            train_stress(model, tag, TRAIN_H, STEPS, cfg, warp=WARP)
            results.append(stress_eval(model, tag, cfg, eval_horizon=EVAL_H,
                                       warp=WARP, true_lambda=lam_true))
        except Exception as exc:  # noqa: BLE001 -- benchmark robustness, report and continue
            print(f"[{tag}] ARM FAILED: {type(exc).__name__}: {exc}")
            results.append({"tag": tag, "finite": False, "motion_ratio": float("nan"),
                            "lam_model": float("nan"), "lam_err": float("nan"),
                            "contraction": float("nan"), "contraction_err": float("nan"),
                            "r_mag": None, "chamfer": float("nan")})

    print("\n===== B/C/D/E/F SUMMARY (200-step Lorenz rollout, nonlinear warp) =====")
    print("(matched data/steps across arms; NOTE F's QR tracker is far costlier per "
          "step -- steps are matched, wall-clock is not.)")
    print(f"{'model':<22}{'finite':>7}{'mratio':>8}{'lam_model':>11}{'lam_err':>9}"
          f"{'contract':>10}{'|d-13.7|':>10}{'tr(R)/n':>10}{'chamfer':>9}")
    for r in results:
        fin = "yes" if r["finite"] else "NO"
        rm = " n/a" if r.get("r_mag") is None else f"{r['r_mag']:.4f}"
        print(f"{r['tag']:<22}{fin:>7}{r['motion_ratio']:>8.3f}{r['lam_model']:>11.3f}"
              f"{r['lam_err']:>9.3f}{r['contraction']:>10.3f}{r['contraction_err']:>10.3f}"
              f"{rm:>10}{r['chamfer']:>9.2f}")
    print(f"\nreference: true lambda1={lam_true:+.3f},  true div f={LORENZ_DIVERGENCE:.3f}")
    print("Read the table as: (finite & small chamfer/hausdorff) => rollout stays ON")
    print("the encoded attractor; (lam_model ~ true lambda1) => it reproduced the")
    print("chaotic divergence rate; (contract ~ -13.7) => learned volume law matches")
    print("-- but contract is a COORDINATE-DEPENDENT proxy under the warp, and the")
    print("Hamiltonian arm is EXPECTED near 0 (volume-preserving => cannot dissipate).")
    print("HONEST FRAMING: Lorenz is not a natural GENERIC system, so H/S here are")
    print("learned proxies, not physical energy/entropy. We claim on-attractor")
    print("stability + invariant recovery, NOT recovery of true thermodynamics.")
    print("OVERLAP TRAP: read chamfer ONLY together with mratio -- a frozen predictor")
    print("(mratio ~ 0, lam ~ 0) posts a small chamfer while being dynamically DEAD.")
    print("Per-mechanism analysis: see the STRUCTURAL FAILURE ATLAS in the module")
    print("docstring (freeze / scale paradox / skew+dead-R / sum-lambda1 / spectral).")
    print("pipeline OK")


if __name__ == "__main__":
    main()
