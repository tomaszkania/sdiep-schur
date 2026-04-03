"""Public API for the ``sdiep_schur`` package."""

from .bases import CycleBasis, OrthogonalBasis, PaleyHadamardBasis, PhaseOptimisedCycleBasis, WalshHadamardBasis
from .experiments import computational_cost, convergence_study, sharpness_curve, success_rate_grid, systematic_coherence, threshold_landscape
from .sampling import dirichlet_suleimanova, wilson_interval
from .theory import delta_cycle, delta_phase, hadamard_orders_up_to, is_prime, rho_mod8

__all__ = [
    "CycleBasis",
    "OrthogonalBasis",
    "PaleyHadamardBasis",
    "PhaseOptimisedCycleBasis",
    "WalshHadamardBasis",
    "computational_cost",
    "convergence_study",
    "sharpness_curve",
    "success_rate_grid",
    "systematic_coherence",
    "threshold_landscape",
    "dirichlet_suleimanova",
    "wilson_interval",
    "delta_cycle",
    "delta_phase",
    "hadamard_orders_up_to",
    "is_prime",
    "rho_mod8",
]
