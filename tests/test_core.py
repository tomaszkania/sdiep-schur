from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

from sdiep_schur import (
    CycleBasis,
    PhaseOptimisedCycleBasis,
    WalshHadamardBasis,
    delta_cycle,
    delta_phase,
    rho_mod8,
)


def test_cycle_coherence_matches_formula() -> None:
    for n in (5, 8, 12, 17):
        basis = CycleBasis.create(n)
        theory = 2.0 * math.cos(math.pi * rho_mod8(n) / (4.0 * n)) ** 2
        assert math.isclose(basis.coherence(), theory, rel_tol=1e-10, abs_tol=1e-10)


def test_phase_coherence_matches_formula() -> None:
    for n in (6, 8, 15):
        basis = PhaseOptimisedCycleBasis.create(n)
        theta = math.pi / (4.0 * n) if n % 2 else (math.pi / n if n % 4 == 0 else math.pi / (2.0 * n))
        theory = 2.0 * math.cos(theta) ** 2
        assert math.isclose(basis.coherence(), theory, rel_tol=1e-8, abs_tol=1e-8)


def test_walsh_coherence_is_one() -> None:
    basis = WalshHadamardBasis.create(16)
    assert math.isclose(basis.coherence(), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_delta_values_are_below_half() -> None:
    for n in range(3, 30):
        assert delta_phase(n) < 0.5 + 1e-12
        if n % 8 == 0:
            assert math.isclose(delta_cycle(n), 0.5, rel_tol=1e-12, abs_tol=1e-12)
