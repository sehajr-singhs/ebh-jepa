"""EB-H-JEPA: Energy-Based Hamiltonian Latent World Models.

A JEPA-style world model whose latent *predictor* is a learned Hamiltonian
advanced by a symplectic integrator (plus metriplectic and MLP ablations),
trained with SIGReg anti-collapse and evaluated against chaotic attractor
invariants (Lyapunov spectrum, phase-space contraction).

The full implementation, the 5-arm benchmark, and the structural-failure
analysis live in `ebhjepa.ebhjepa`. Run the benchmark with:

    python -m benchmarks.run_benchmark

See README.md for the reproducible pipeline.
"""

from .ebhjepa import (
    EBHJepa,
    Encoder,
    HamiltonianSystem,
    LossWeights,
    MetriplecticSystem,
    MLPPredictor,
    PortHamiltonianSystem,
    make_lorenz_sequence,
    make_synthetic_sequence,
    sigreg_loss,
)

__version__ = "0.1.0"

__all__ = [
    "EBHJepa",
    "Encoder",
    "HamiltonianSystem",
    "LossWeights",
    "MetriplecticSystem",
    "MLPPredictor",
    "PortHamiltonianSystem",
    "make_lorenz_sequence",
    "make_synthetic_sequence",
    "sigreg_loss",
]
