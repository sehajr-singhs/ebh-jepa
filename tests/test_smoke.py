"""Fast CPU smoke tests. Run with:  python -m pytest tests/ -q

These verify the pipeline runs (forward, backward, benchmark harness) without
training anything to convergence.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebhjepa import EBHJepa, LossWeights  # noqa: E402
from ebhjepa.ebhjepa import make_synthetic_sequence, sigreg_loss  # noqa: E402


def _small_model(predictor_type, **extra):
    return EBHJepa(
        predictor_type=predictor_type,
        conserve_mode={"mlp": "none", "hamiltonian": "flat",
                       "metriplectic": "dissipative"}[predictor_type],
        latent_dim=16, dt=0.01, mode="vector", in_dim=8,
        integrator="symplectic",
        weights=LossWeights(1.0, 1.0, 1.0, 0.1, 2.0, 0.02, 0.1),
        **extra,
    )


def test_forward_backward_all_predictors():
    x_seq = make_synthetic_sequence(batch=8, obs_dim=8, dt=0.01, horizon=2,
                                    device="cpu")
    for ptype in ["mlp", "hamiltonian", "metriplectic"]:
        model = _small_model(ptype)
        out = model(x_seq)
        assert torch.isfinite(out["loss"]), ptype
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in grads), ptype


def test_sigreg_finite():
    z = torch.randn(16, 16) * 0.7  # well-spread embeddings, not collapsed
    loss = sigreg_loss(z)
    assert torch.isfinite(loss) and loss >= 0


def test_rollout_shape():
    model = _small_model("hamiltonian")
    z0 = torch.randn(4, 16)
    traj = model.predictor.rollout(z0, steps=5, dt=0.01, integrator="symplectic")
    # (B, steps+1, D): the initial state is included
    assert traj.shape == (4, 6, 16)
    assert torch.isfinite(traj).all()


def test_benchmark_harness_imports():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import run_benchmark  # noqa: F401
    arms = run_benchmark.build_arms("cpu", 32, 64, 0.01, 8, 0.9)
    assert len(arms) == 5
    assert arms[0][1].predictor_type == "mlp"
    assert arms[-1][1].predictor_type == "metriplectic"
