"""Reproducible experiments for the SDIEP paper."""

from __future__ import annotations

from dataclasses import dataclass
import math
import pathlib
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .bases import CycleBasis, OrthogonalBasis, PaleyHadamardBasis, PhaseOptimisedCycleBasis, WalshHadamardBasis
from .sampling import dirichlet_suleimanova, wilson_interval
from .theory import delta_cycle, delta_phase, hadamard_orders_up_to, is_prime, rho_mod8


def systematic_coherence(out_dir: pathlib.Path) -> pd.DataFrame:
    """Compute empirical and theoretical coherence values for all basis families.

    Parameters
    ----------
    out_dir:
        Output directory.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``n``, ``basis``, ``M_empirical``, ``M_theoretical``,
        and ``rel_error``.
    """
    rows: list[dict[str, float | int | str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in range(3, 101):
        b_can = CycleBasis.create(n)
        m_can_theory = 2.0 * math.cos(math.pi * rho_mod8(n) / (4.0 * n)) ** 2
        rows.append({"n": n, "basis": "Cycle", "M_empirical": b_can.coherence(), "M_theoretical": m_can_theory, "rel_error": abs(b_can.coherence() - m_can_theory) / m_can_theory})

        b_ph = PhaseOptimisedCycleBasis.create(n)
        theta = math.pi / (4.0 * n) if n % 2 else (math.pi / n if n % 4 == 0 else math.pi / (2.0 * n))
        m_ph_theory = 2.0 * math.cos(theta) ** 2
        rows.append({"n": n, "basis": "Phase", "M_empirical": b_ph.coherence(), "M_theoretical": m_ph_theory, "rel_error": abs(b_ph.coherence() - m_ph_theory) / m_ph_theory})

        if n > 1 and (n & (n - 1)) == 0:
            b_w = WalshHadamardBasis.create(n)
            rows.append({"n": n, "basis": "Walsh", "M_empirical": b_w.coherence(), "M_theoretical": 1.0, "rel_error": abs(b_w.coherence() - 1.0)})

        q = n - 1
        if q >= 3 and q % 4 == 3 and is_prime(q):
            b_p = PaleyHadamardBasis.create(q)
            rows.append({"n": n, "basis": "Paley", "M_empirical": b_p.coherence(), "M_theoretical": 1.0, "rel_error": abs(b_p.coherence() - 1.0)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp1_coherence_systematic.csv", index=False)
    df.groupby("basis")["rel_error"].agg(["count", "mean", "std", "max"]).to_csv(out_dir / "exp1_coherence_summary.csv")
    return df


def success_rate_grid(
    out_dir: pathlib.Path,
    n_values: Sequence[int] = (8, 12, 16, 24),
    trace_resolution: int = 50,
    trials_per_point: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate success rates over a grid of trace sums.

    Parameters
    ----------
    out_dir:
        Output directory.
    n_values:
        Tested dimensions.
    trace_resolution:
        Number of trace-sum grid points per dimension.
    trials_per_point:
        Monte-Carlo trials per grid point.
    seed:
        Random seed.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in n_values:
        t_min = max(0.01, min(delta_cycle(n), delta_phase(n)) - 0.15)
        t_max = min(0.99, max(delta_cycle(n), delta_phase(n)) + 0.15)
        trace_grid = np.linspace(t_min, t_max, trace_resolution)
        bases: list[tuple[str, OrthogonalBasis]] = [("Cycle", CycleBasis.create(n)), ("Phase", PhaseOptimisedCycleBasis.create(n))]
        if n > 1 and (n & (n - 1)) == 0:
            bases.append(("Walsh", WalshHadamardBasis.create(n)))
        q = n - 1
        if q >= 3 and q % 4 == 3 and is_prime(q):
            bases.append(("Paley", PaleyHadamardBasis.create(q)))
        for basis_name, basis in bases:
            for trace_sum in trace_grid:
                successes = 0
                for _ in range(trials_per_point):
                    lambdas = dirichlet_suleimanova(n, trace_sum, rng=rng)
                    if basis.min_entry(lambdas) >= -1e-12:
                        successes += 1
                low, high = wilson_interval(successes, trials_per_point)
                rows.append({
                    "n": n,
                    "basis": basis_name,
                    "trace_sum": trace_sum,
                    "rate": successes / trials_per_point,
                    "ci_lower": low,
                    "ci_upper": high,
                    "trials": trials_per_point,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp2_success_rates.csv", index=False)
    return df


def sharpness_curve(out_dir: pathlib.Path, n_test: int = 12, t_resolution: int = 250) -> pd.DataFrame:
    """Compute the one-spike sharpness curve for the canonical cycle basis."""
    basis = CycleBasis.create(n_test)
    delta = math.pi * rho_mod8(n_test) / (4.0 * n_test)
    t_star = -1.0 / (2.0 * math.cos(delta) ** 2)
    t_values = np.linspace(-1.5, 0.1, t_resolution)
    rows: list[dict[str, float | int | str]] = []
    for t in t_values:
        lambdas = np.zeros(n_test)
        lambdas[0] = 1.0
        lambdas[1] = t
        rows.append({"n": n_test, "basis": "Cycle", "t": t, "min_entry": basis.min_entry(lambdas), "t_star": t_star})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"exp3_sharpness_n{n_test}.csv", index=False)
    return df


def computational_cost(out_dir: pathlib.Path, n_values: Sequence[int] = (4, 8, 16, 32, 64, 128), repeats: int = 30) -> pd.DataFrame:
    """Measure construction and evaluation times."""
    rng = np.random.default_rng(123)
    rows: list[dict[str, float | int | str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in n_values:
        constructors: list[tuple[str, callable]] = [
            ("Cycle", lambda n=n: CycleBasis.create(n)),
            ("Phase", lambda n=n: PhaseOptimisedCycleBasis.create(n)),
        ]
        if n > 1 and (n & (n - 1)) == 0:
            constructors.append(("Walsh", lambda n=n: WalshHadamardBasis.create(n)))
        for basis_name, ctor in constructors:
            build_times: list[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                basis = ctor()
                build_times.append((time.perf_counter() - t0) * 1000.0)
            basis = ctor()
            lambdas = dirichlet_suleimanova(n, 0.5, rng=rng)
            eval_times: list[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = basis.compute_P(lambdas)
                eval_times.append((time.perf_counter() - t0) * 1000.0)
            rows.append({
                "n": n,
                "basis": basis_name,
                "construction_time_ms": float(np.mean(build_times)),
                "construction_std_ms": float(np.std(build_times)),
                "evaluation_time_ms": float(np.mean(eval_times)),
                "evaluation_std_ms": float(np.std(eval_times)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp4_computational_cost.csv", index=False)
    return df


def convergence_study(
    out_dir: pathlib.Path,
    n: int = 16,
    trace_sum: float = 0.4,
    max_trials: int = 10000,
    checkpoints: Sequence[int] = (100, 200, 500, 1000, 2000, 5000, 10000),
) -> pd.DataFrame:
    """Study convergence of the success-rate estimate."""
    rng = np.random.default_rng(999)
    basis = PhaseOptimisedCycleBasis.create(n)
    trials = [dirichlet_suleimanova(n, trace_sum, rng=rng) for _ in range(max_trials)]
    outcomes = [basis.min_entry(lam) >= -1e-12 for lam in trials]
    rows: list[dict[str, float | int | str]] = []
    for m in checkpoints:
        successes = sum(outcomes[:m])
        low, high = wilson_interval(successes, m)
        rows.append({"basis": "Phase", "n_trials": m, "rate": successes / m, "ci_lower": low, "ci_upper": high, "ci_width": high - low})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp5_convergence.csv", index=False)
    return df


def threshold_landscape(out_dir: pathlib.Path) -> pd.DataFrame:
    """Save the canonical and phase-optimised thresholds up to n=100."""
    n_values = np.arange(3, 101)
    df = pd.DataFrame({"n": n_values, "delta_canonical": [delta_cycle(int(n)) for n in n_values], "delta_phase": [delta_phase(int(n)) for n in n_values]})
    df.to_csv(out_dir / "supp_thresholds.csv", index=False)
    return df
