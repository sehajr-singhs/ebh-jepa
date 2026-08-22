"""Kuramoto--Sivashinsky (KS) Galerkin system: the second chaotic test bed.

Why a second system? The Lorenz atlas shows *where* each structural constraint
class fails on one chaotic system; a one-system atlas could in principle be a
fluke of Lorenz's geometry (3-D, constant divergence, two lobes). KS is as far
from that as a dissipative chaotic PDE gets: an infinite-dimensional field on a
periodic domain, with a *scale-dependent* spectrum -- a handful of linearly
unstable long-wave modes and a large stable short-wave bath -- and phase-space
contraction driven entirely by the linear part.

The truncated Galerkin system used here is exactly integrable in its invariants:

    u_t = -u u_x - u_xx - u_xxxx,   x in [0, L],  u = sum_j c_j e^{i kappa j x}

with L = 22 (the classical chaotic regime), kappa = 2 pi / L, and a SYMMETRIC
mode band j = -J .. J with J = 8 (17 complex modes; conjugate symmetry c_{-j} =
conj(c_j) is preserved exactly by the truncation, so the dynamics stays on the
real-field subspace). In Fourier space the vector field is

    F_j(c) = lambda_j c_j - i kappa sum_{m+n=j} n c_m c_n,
    lambda_j = j^2 kappa^2 - j^4 kappa^4.

Two exact facts make KS as clean a test bed as Lorenz:

1. TRACE-FREE NONLINEARITY. The quadratic term is energy-conserving
   (int u u_x u dx = 0), so tr(DF) = sum_j lambda_j is exact and
   state-independent. Phase-space contraction is therefore the closed form

       div F = sum_{j=-J}^{J} (j^2 kappa^2 - j^4 kappa^4).

2. Unstable modes are few. For L = 22, |lambda_j| > 0 only for |j| <= 3, so the
   system is genuinely chaotic (positive largest Lyapunov exponent, estimated
   here by a Benettin shadow-separation scheme exactly like the Lorenz one) yet
   strongly dissipative in total volume.

The observation map into the latent chart mirrors the Lorenz pipeline: a fixed
random affine embedding of the 17-D real state into obs_dim, tanh-bounded, then
optionally the same nonlinear diffeomorphic warp. The model arms never see the
raw modes; they train and are evaluated in the lifted chart exactly as for
Lorenz, so the atlas comparison is apples-to-apples.

Note on the numerical scheme: the quadratic term is a band-limited convolution
computed exactly by FFT on zero-padded length-2M arrays (the band max |m+n| =
2J = 16 < 2M-1, so the relevant output region is alias-free), with the output
slice for mode j located at padded index j + 2J. The field matches a brute-force
O(M^3) evaluation to float32 precision.
"""

from __future__ import annotations

import math

import torch

# --- System constants -------------------------------------------------------
KS_L = 22.0                          # domain length (classic chaos regime)
KS_J = 8                             # max |mode index| -> 17 complex modes
KS_MODES = 2 * KS_J + 1              # 17 complex Fourier modes (17 real dof)


def _kappa() -> float:
    return 2.0 * math.pi / KS_L


def ks_eigenvalues(j_max: int | None = None) -> torch.Tensor:
    """Linear growth rates lambda_j = j^2 kappa^2 - j^4 kappa^4, j = -J..J."""
    j_max = j_max or KS_J
    j = torch.arange(-j_max, j_max + 1).float()
    k = _kappa()
    return j.square() * k * k - j.square().square() * k ** 4


def ks_divergence(j_max: int | None = None) -> float:
    """Exact phase-space contraction of the truncated system: sum of lambda_j.

    The quadratic term is trace-free (energy-conserving), so tr(DF) = sum_j
    lambda_j exactly -- the analogue of Lorenz's -13.667.
    """
    return float(ks_eigenvalues(j_max).sum().item())


_KS_CONST_CACHE: dict = {}


def _ks_consts(device, dtype, j_max: int):
    key = (j_max, str(device), str(dtype))
    if key not in _KS_CONST_CACHE:
        lam = ks_eigenvalues(j_max).to(device=device, dtype=dtype)
        idx = torch.arange(-j_max, j_max + 1, device=device).to(dtype=dtype)
        _KS_CONST_CACHE[key] = (lam, idx)
    return _KS_CONST_CACHE[key]


def ks_field(c: torch.Tensor, j_max: int | None = None) -> torch.Tensor:
    """Truncated KS vector field in Fourier space. c: (..., 2J+1) complex.

    F_j = lambda_j c_j - i kappa sum_{m+n=j} n c_m c_n.

    Position p in the array holds mode j = p - J, so the linear (zero-padded)
    convolution output for mode j sits at padded index j + 2J; the slice
    [2J : 2J + (2J+1)] is alias-free because the band max |m+n| = 2J < 2M-1.
    """
    j_max = j_max or KS_J
    lam, idx = _ks_consts(c.device, c.dtype, j_max)
    k = _kappa()
    m = 2 * j_max + 1
    n2 = 2 * m
    a = idx * c                                  # A_n = n c_n
    cc = torch.fft.ifft(torch.fft.fft(c, n=n2, dim=-1) ** 2, n=n2, dim=-1)
    ac = torch.fft.ifft(torch.fft.fft(a, n=n2, dim=-1)
                        * torch.fft.fft(c, n=n2, dim=-1), n=n2, dim=-1)
    # mode j lives at padded index j + 2J, so the window for j in [-J, J] is
    # [J, 3J+1] (2J = m - 1 for odd m)
    sl = slice(j_max, 3 * j_max + 1)
    q = idx * cc[..., sl] - ac[..., sl]
    return lam * c - 1j * k * q


def ks_rk4(c: torch.Tensor, dt: float, j_max: int | None = None) -> torch.Tensor:
    """One classical RK4 step of the truncated KS system (complex arithmetic)."""
    k1 = ks_field(c, j_max)
    k2 = ks_field(c + 0.5 * dt * k1, j_max)
    k3 = ks_field(c + 0.5 * dt * k2, j_max)
    k4 = ks_field(c + dt * k3, j_max)
    return c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def ks_real_state(c: torch.Tensor, j_max: int | None = None) -> torch.Tensor:
    """Real representation of a conjugate-symmetric state.

    Positions 0..J-1 are the conjugate mirror of J+1..2J, so the real degrees of
    freedom are [c_0 (DC), Re c_1, Im c_1, ..., Re c_J, Im c_J]: 2J+1 numbers.
    """
    j_max = j_max or KS_J
    dc = c[..., j_max:j_max + 1].real
    hi = c[..., j_max + 1:]                      # j = 1..J (complex)
    return torch.cat([dc, hi.real, hi.imag], dim=-1)


def _ks_init(batch: int, j_max: int, generator: torch.Generator) -> torch.Tensor:
    """Random conjugate-symmetric Fourier data on the attractor basin."""
    J = j_max
    amp = torch.exp(-0.5 * torch.arange(0, J + 1).float())       # low-mode energy
    hi = torch.randn(batch, J, generator=generator) * amp[1:] \
        + 1j * torch.randn(batch, J, generator=generator) * amp[1:]   # j = 1..J
    dc = torch.randn(batch, 1, generator=generator) * amp[0]
    c = torch.zeros(batch, 2 * J + 1, dtype=torch.complex64)
    c[..., J:J + 1] = dc
    c[..., J + 1:] = hi
    c[..., :J] = torch.conj(hi.flip(-1))          # c_{-j} = conj(c_j)
    return c


def ks_lambda1(dt: float, n_steps: int, device: str = "cpu",
               eps: float = 1e-6) -> float:
    """Benettin estimate of the largest Lyapunov exponent of the truncated KS
    system: reference + conjugate-symmetric shadow trajectory, renormalized each
    step (the shadow stays on the real-field subspace, so we measure the real
    dynamics' exponent)."""
    g = torch.Generator(device="cpu").manual_seed(7)
    J = KS_J
    c = _ks_init(64, J, g).to(device)
    for _ in range(2000):
        c = ks_rk4(c, dt)
    delta = torch.zeros(64, 2 * J + 1, dtype=torch.complex64).to(device)
    delta[..., J + 1:] = (torch.randn(64, J, generator=g)
                          + 1j * torch.randn(64, J, generator=g))
    delta[..., :J] = torch.conj(delta[..., J + 1:].flip(-1))
    delta = delta / (delta.abs().norm(dim=-1, keepdim=True) + 1e-30) * eps
    cp = c + delta
    acc = 0.0
    for _ in range(n_steps):
        c = ks_rk4(c, dt)
        cp = ks_rk4(cp, dt)
        diff = cp - c
        dist = diff.abs().norm(dim=-1, keepdim=True).clamp_min(1e-30)
        acc += torch.log(dist / eps).mean().item()
        cp = c + diff * (eps / dist)
    return acc / (n_steps * dt)


# --- Observation lift (mirrors the Lorenz nonlinear pipeline) ---------------
def make_ks_sequence(
    batch: int,
    obs_dim: int,
    dt: float,
    horizon: int,
    device: str = "cpu",
    warp: bool = True,
    burn_in: int = 800,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows of the truncated KS attractor, nonlinearly lifted to obs_dim.

    Per sample: random low-mode IC, burn-in on the attractor, then horizon+1
    frames. Each real state is normalized and embedded by a FIXED random
    affine-tanh map into obs_dim, then (optionally) pushed through the same
    nonlinear diffeomorphic warp as the Lorenz pipeline.

    returns:
        x_seq   : (batch, horizon+1, obs_dim)
        true_ks : (batch, horizon+1, 2J+1) raw normalized real states
    """
    g = torch.Generator(device="cpu")
    if seed is None:
        g.seed()
    else:
        g.manual_seed(seed)
    J = KS_J
    c = _ks_init(batch, J, g).to(device)
    for _ in range(burn_in):
        c = ks_rk4(c, dt)
    ndim = 2 * J + 1
    gen = torch.Generator(device="cpu").manual_seed(0)
    W = (torch.randn(ndim, obs_dim, generator=gen) / math.sqrt(ndim)).to(device)
    scale = 8.0  # KS amplitudes are ~O(1-8); normalize before the embedding

    frames, raw = [], []
    s = c
    for _ in range(horizon + 1):
        u = ks_real_state(s) / scale               # (batch, 2J+1) ~O(1)
        raw.append(u)
        feat = torch.tanh(u @ W)                   # (batch, obs_dim) in [-1,1]
        if warp:
            from .ebhjepa import nonlinear_warp
            feat = nonlinear_warp(feat)
        frames.append(feat)
        s = ks_rk4(s, dt)
    x_seq = torch.stack(frames, dim=1)             # (batch, horizon+1, obs_dim)
    true_ks = torch.stack(raw, dim=1)              # (batch, horizon+1, 2J+1)
    return x_seq, true_ks
