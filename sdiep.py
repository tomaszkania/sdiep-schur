#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced numerical experiments for dimension-dependent SDIEP bounds.

This script provides comprehensive numerical validation including:
  • Systematic coherence verification across dimensions 3-100
  • High-resolution success rate mapping with error bars
  • Sharpness demonstration via critical spectrum analysis
  • Computational cost profiling
  • Convergence analysis for Monte Carlo estimates
  • Comparative analysis across all basis families

Usage
-----
$ python sdiep_enhanced_numerics.py

Requirements
------------
numpy>=1.26, pandas>=2.1, matplotlib>=3.8, scipy>=1.11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

# ============================================================================
# Core utilities
# ============================================================================

def is_prime(n: int) -> bool:
    """Deterministic primality test for n < 2^32."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for a in [2, 7, 61]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        witness = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                witness = False
                break
        if witness:
            return False
    return True


def dirichlet_suleimanova(
    n: int,
    trace_sum: float,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample Suleĭmanova spectrum with prescribed trace sum."""
    if not (0.0 < trace_sum <= 1.0):
        raise ValueError("trace_sum must be in (0, 1].")
    S = 1.0 - trace_sum
    if rng is None:
        rng = np.random.default_rng()
    weights = rng.dirichlet(alpha * np.ones(n - 1))
    lambdas = np.empty(n, dtype=float)
    lambdas[0] = 1.0
    lambdas[1:] = -S * weights
    return lambdas


def _rho_mod8(n: int) -> int:
    """Return ρ(n) ∈ {0,1,2,4} determined by n mod 8."""
    r = n % 8
    if r == 0:
        return 0
    if r in (1, 3, 5, 7):
        return 1
    if r in (2, 6):
        return 2
    if r == 4:
        return 4
    raise RuntimeError("Unreachable")


def delta_cycle(n: int) -> float:
    """δ_n for canonical cycle basis."""
    rho = _rho_mod8(n)
    theta = math.pi * rho / (4.0 * n)
    c2 = math.cos(theta) ** 2
    return 1.0 - 1.0 / (2.0 * c2)


def delta_phase(n: int) -> float:
    """δ_n^{(ph)} for phase-optimised cycle basis."""
    if n % 2 == 1:
        theta = math.pi / (4.0 * n)
    elif n % 4 == 0:
        theta = math.pi / n
    else:
        theta = math.pi / (2.0 * n)
    c2 = math.cos(theta) ** 2
    return 1.0 - 1.0 / (2.0 * c2)


# ============================================================================
# Basis classes
# ============================================================================

@dataclass
class OrthogonalBasis:
    """Real orthogonal basis Q with Perron column 1/√n."""
    n: int
    Q: np.ndarray

    def compute_P(self, lambdas: np.ndarray) -> np.ndarray:
        return self.Q @ np.diag(lambdas) @ self.Q.T

    def coherence(self) -> float:
        col_norms = np.max(np.abs(self.Q[:, 1:]), axis=0)
        return float(self.n * np.max(col_norms**2))

    def min_entry(self, lambdas: np.ndarray) -> float:
        return float(self.compute_P(lambdas).min())


@dataclass
class CycleBasis(OrthogonalBasis):
    """Canonical cycle-walk eigenbasis."""
    @staticmethod
    def _build_Q(n: int) -> np.ndarray:
        Q = np.zeros((n, n))
        Q[:, 0] = 1.0 / math.sqrt(n)
        j = np.arange(n)
        for k in range(1, n):
            Q[:, k] = math.sqrt(2.0/n) * np.sin(2*math.pi*k*j/n + math.pi/4)
        return Q

    @classmethod
    def create(cls, n: int) -> "CycleBasis":
        return cls(n=n, Q=cls._build_Q(n))


@dataclass
class PhaseOptimisedCycleBasis(OrthogonalBasis):
    """Phase-optimised cycle basis."""
    @staticmethod
    def _phi_optimal(n: int, j: int) -> float:
        from math import gcd as _g
        nprime = n // _g(n, j)
        r = nprime // math.gcd(nprime, 4)
        L = math.pi / 2.0
        s_eff = L / r
        return s_eff / 2.0

    @staticmethod
    def _build_Q(n: int) -> np.ndarray:
        cols = [np.ones(n) / math.sqrt(n)]
        k = np.arange(n)
        for j in range(1, (n-1)//2 + 1):
            phi = PhaseOptimisedCycleBasis._phi_optimal(n, j)
            cols.append(math.sqrt(2/n) * np.sin(2*math.pi*j*k/n + phi))
            cols.append(math.sqrt(2/n) * np.cos(2*math.pi*j*k/n + phi))
        if n % 2 == 0:
            phi_mid = math.pi / 4.0
            cols.append(math.sqrt(2/n) * np.sin(math.pi*k + phi_mid))
        Q = np.column_stack(cols)
        # Re-orthonormalize
        Q_orth = np.zeros_like(Q)
        Q_orth[:, 0] = Q[:, 0]
        for c in range(1, n):
            v = Q[:, c].copy()
            for i in range(c):
                v -= np.dot(Q_orth[:, i], v) * Q_orth[:, i]
            Q_orth[:, c] = v / np.linalg.norm(v)
        return Q_orth

    @classmethod
    def create(cls, n: int) -> "PhaseOptimisedCycleBasis":
        return cls(n=n, Q=cls._build_Q(n))


@dataclass
class WalshHadamardBasis(OrthogonalBasis):
    """Walsh–Hadamard (Sylvester) basis for n=2^d."""
    @staticmethod
    def _hadamard_sylvester(n: int) -> np.ndarray:
        if n < 1 or (n & (n-1)) != 0:
            raise ValueError("n must be power of two")
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H

    @classmethod
    def create(cls, n: int) -> "WalshHadamardBasis":
        H = cls._hadamard_sylvester(n)
        Q = H / math.sqrt(n)
        if not np.allclose(Q[:, 0], np.ones(n)/math.sqrt(n), atol=1e-12):
            for j in range(n):
                if np.allclose(Q[:, j], np.ones(n)/math.sqrt(n), atol=1e-12):
                    Q[:, [0, j]] = Q[:, [j, 0]]
                    break
        return cls(n=n, Q=Q)


@dataclass
class PaleyHadamardBasis(OrthogonalBasis):
    """Paley type-I Hadamard for q≡3(mod 4) prime."""
    @staticmethod
    def _legendre_symbol(a: int, p: int) -> int:
        a %= p
        if a == 0:
            return 0
        ls = pow(a, (p-1)//2, p)
        return 1 if ls == 1 else -1

    @staticmethod
    def _build_hadamard_paley(q: int) -> np.ndarray:
        if q < 3 or q % 4 != 3 or not is_prime(q):
            raise ValueError("q must be prime ≡3(mod 4)")
        A = np.empty((q, q))
        for x in range(q):
            for y in range(q):
                if x == y:
                    A[x,y] = 1.0
                else:
                    A[x,y] = 1.0 if PaleyHadamardBasis._legendre_symbol((x-y)%q, q)==1 else -1.0
        H = np.empty((q+1, q+1))
        H[0,0] = 1.0
        H[0,1:] = 1.0
        H[1:,0] = -1.0
        H[1:,1:] = A
        return H

    @classmethod
    def create(cls, q: int) -> "PaleyHadamardBasis":
        H = cls._build_hadamard_paley(q)
        n = q + 1
        D = np.diag(np.concatenate(([1.0], -np.ones(q))))
        Q = (D @ H) / math.sqrt(n)
        return cls(n=n, Q=Q)


# ============================================================================
# ENHANCED EXPERIMENTS
# ============================================================================

def experiment_1_systematic_coherence(outdir: pathlib.Path) -> pd.DataFrame:
    """
    Experiment 1: Systematic coherence verification.
    
    Compute M(Q) for n=3..100 for all applicable bases and verify:
    - Canonical: M(Q) = 2cos²(πρ(n)/(4n))
    - Phase: M(Q) follows piecewise formula
    - Hadamard: M(Q) = 1
    
    Returns DataFrame with columns: n, basis, M_empirical, M_theoretical, rel_error
    """
    print("Experiment 1: Systematic coherence verification (n=3..100)")
    rows = []
    
    for n in range(3, 101):
        # Canonical cycle
        try:
            B_can = CycleBasis.create(n)
            M_emp = B_can.coherence()
            M_thy = 2 * math.cos(math.pi * _rho_mod8(n) / (4*n))**2
            rows.append({
                'n': n,
                'basis': 'Cycle',
                'M_empirical': M_emp,
                'M_theoretical': M_thy,
                'rel_error': abs(M_emp - M_thy) / M_thy if M_thy > 0 else 0
            })
        except Exception as e:
            print(f"  Cycle n={n} failed: {e}")
        
        # Phase-optimised
        try:
            B_ph = PhaseOptimisedCycleBasis.create(n)
            M_emp = B_ph.coherence()
            if n % 2 == 1:
                theta = math.pi / (4*n)
            elif n % 4 == 0:
                theta = math.pi / n
            else:
                theta = math.pi / (2*n)
            M_thy = 2 * math.cos(theta)**2
            rows.append({
                'n': n,
                'basis': 'Phase',
                'M_empirical': M_emp,
                'M_theoretical': M_thy,
                'rel_error': abs(M_emp - M_thy) / M_thy
            })
        except Exception as e:
            print(f"  Phase n={n} failed: {e}")
        
        # Walsh-Hadamard (powers of 2)
        if n > 1 and (n & (n-1)) == 0:
            try:
                B_wh = WalshHadamardBasis.create(n)
                M_emp = B_wh.coherence()
                rows.append({
                    'n': n,
                    'basis': 'Walsh',
                    'M_empirical': M_emp,
                    'M_theoretical': 1.0,
                    'rel_error': abs(M_emp - 1.0)
                })
            except Exception as e:
                print(f"  Walsh n={n} failed: {e}")
        
        # Paley (q+1 where q≡3 mod 4 prime)
        q = n - 1
        if q >= 3 and q % 4 == 3 and is_prime(q):
            try:
                B_p = PaleyHadamardBasis.create(q)
                M_emp = B_p.coherence()
                rows.append({
                    'n': n,
                    'basis': 'Paley',
                    'M_empirical': M_emp,
                    'M_theoretical': 1.0,
                    'rel_error': abs(M_emp - 1.0)
                })
            except Exception as e:
                print(f"  Paley q={q} failed: {e}")
    
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'exp1_coherence_systematic.csv', index=False)
    
    # Statistical summary
    summary = df.groupby('basis')['rel_error'].agg(['count', 'mean', 'std', 'max'])
    summary.to_csv(outdir / 'exp1_coherence_summary.csv')
    print(f"  Completed {len(df)} coherence calculations")
    print(f"  Max relative error: {df['rel_error'].max():.2e}")
    
    return df


def experiment_2_success_rate_heatmap(
    outdir: pathlib.Path,
    n_values: List[int] = [8, 12, 16, 24],
    trace_resolution: int = 40,
    trials_per_point: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Experiment 2: High-resolution success rate mapping.
    
    For selected dimensions, compute P[P(Λ)≥0] on fine grid of trace sums.
    Include Wilson score confidence intervals.
    
    Returns DataFrame with: n, basis, trace_sum, rate, ci_lower, ci_upper, trials
    """
    print(f"Experiment 2: Success rate heatmap (n={n_values}, {trace_resolution} × {trials_per_point})")
    
    rows = []
    rng = np.random.default_rng(seed)
    
    for n in n_values:
        # Determine trace_sum grid based on theoretical thresholds
        delta_can = delta_cycle(n)
        delta_ph = delta_phase(n)
        t_min = max(0.01, min(delta_can, delta_ph) - 0.15)
        t_max = min(0.99, max(delta_can, delta_ph) + 0.15)
        trace_grid = np.linspace(t_min, t_max, trace_resolution)
        
        bases_to_test = [
            ('Cycle', CycleBasis.create(n)),
            ('Phase', PhaseOptimisedCycleBasis.create(n))
        ]
        
        # Add Hadamard if applicable
        if n > 1 and (n & (n-1)) == 0:
            bases_to_test.append(('Walsh', WalshHadamardBasis.create(n)))
        q = n - 1
        if q >= 3 and q % 4 == 3 and is_prime(q):
            bases_to_test.append(('Paley', PaleyHadamardBasis.create(q)))
        
        for basis_name, B in bases_to_test:
            print(f"  Testing {basis_name} n={n}")
            for ts in trace_grid:
                successes = 0
                for _ in range(trials_per_point):
                    lam = dirichlet_suleimanova(n, ts, rng=rng)
                    if B.min_entry(lam) >= -1e-12:
                        successes += 1
                
                # Wilson score interval
                p_hat = successes / trials_per_point
                z = 1.959963984540054
                denom = 1 + z**2 / trials_per_point
                centre = (p_hat + z**2/(2*trials_per_point)) / denom
                half = z * math.sqrt(p_hat*(1-p_hat)/trials_per_point + z**2/(4*trials_per_point**2)) / denom
                
                rows.append({
                    'n': n,
                    'basis': basis_name,
                    'trace_sum': ts,
                    'rate': p_hat,
                    'ci_lower': max(0, centre - half),
                    'ci_upper': min(1, centre + half),
                    'trials': trials_per_point
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'exp2_success_rates.csv', index=False)
    print(f"  Completed {len(df)} success rate evaluations")
    
    return df


def experiment_3_sharpness_analysis(
    outdir: pathlib.Path,
    n_test: int = 12,
    t_resolution: int = 200
) -> pd.DataFrame:
    """
    Experiment 3: Sharpness at theoretical threshold.
    
    For n=12 canonical basis, test one-spike spectra λ_{j*}=t near critical t*.
    Verify min entry crosses zero exactly at t* = -1/(2cos²Δ_n).
    
    Returns DataFrame with: t, min_entry, basis, n
    """
    print(f"Experiment 3: Sharpness analysis (n={n_test})")
    
    B = CycleBasis.create(n_test)
    
    # Theoretical threshold for one-spike
    rho = _rho_mod8(n_test)
    Delta_n = math.pi * rho / (4 * n_test)
    t_star = -1 / (2 * math.cos(Delta_n)**2)
    
    # Dense grid around t_star
    t_vals = np.linspace(-1.5, 0.1, t_resolution)
    
    rows = []
    for t in t_vals:
        lam = np.zeros(n_test)
        lam[0] = 1.0
        lam[1] = t  # spike at j=1 (coprime to n)
        min_val = B.min_entry(lam)
        
        rows.append({
            't': t,
            'min_entry': min_val,
            'basis': 'Cycle',
            'n': n_test
        })
    
    df = pd.DataFrame(rows)
    df['t_star'] = t_star
    df.to_csv(outdir / f'exp3_sharpness_n{n_test}.csv', index=False)
    
    # Find numerical zero crossing
    sign_changes = np.where(np.diff(np.sign(df['min_entry'].values)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        t_numerical = df.iloc[idx]['t']
        error = abs(t_numerical - t_star)
        print(f"  Theoretical t*: {t_star:.8f}")
        print(f"  Numerical zero: {t_numerical:.8f}")
        print(f"  Absolute error: {error:.2e}")
    
    return df


def experiment_4_computational_cost(
    outdir: pathlib.Path,
    n_values: List[int] = [4, 8, 16, 32, 64, 128],
    repeats: int = 20
) -> pd.DataFrame:
    """
    Experiment 4: Computational cost analysis.
    
    Time basis construction and P(Λ) computation for various n.
    
    Returns DataFrame with: n, basis, construction_time_ms, evaluation_time_ms
    """
    print(f"Experiment 4: Computational cost (n={n_values}, {repeats} repeats)")
    
    rows = []
    rng = np.random.default_rng(123)
    
    for n in n_values:
        # Only test bases applicable to this n
        basis_constructors = [
            ('Cycle', lambda: CycleBasis.create(n)),
            ('Phase', lambda: PhaseOptimisedCycleBasis.create(n))
        ]
        
        if n > 1 and (n & (n-1)) == 0:
            basis_constructors.append(('Walsh', lambda: WalshHadamardBasis.create(n)))
        
        for basis_name, constructor in basis_constructors:
            # Construction time
            times_construct = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                B = constructor()
                t1 = time.perf_counter()
                times_construct.append((t1 - t0) * 1000)  # ms
            
            # Evaluation time (one P(Λ) computation)
            B = constructor()
            lam = dirichlet_suleimanova(n, 0.5, rng=rng)
            times_eval = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = B.compute_P(lam)
                t1 = time.perf_counter()
                times_eval.append((t1 - t0) * 1000)
            
            rows.append({
                'n': n,
                'basis': basis_name,
                'construction_time_ms': np.mean(times_construct),
                'construction_std_ms': np.std(times_construct),
                'evaluation_time_ms': np.mean(times_eval),
                'evaluation_std_ms': np.std(times_eval)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'exp4_computational_cost.csv', index=False)
    print(f"  Completed timing for {len(df)} configurations")
    
    return df


def experiment_5_convergence_analysis(
    outdir: pathlib.Path,
    n: int = 16,
    trace_sum: float = 0.4,
    max_trials: int = 10000,
    checkpoints: List[int] = [100, 200, 500, 1000, 2000, 5000, 10000]
) -> pd.DataFrame:
    """
    Experiment 5: Monte Carlo convergence.
    
    Show that success rate estimate converges as number of trials increases.
    
    Returns DataFrame with: basis, n_trials, rate, ci_width
    """
    print(f"Experiment 5: Convergence analysis (n={n}, T={trace_sum})")
    
    B = PhaseOptimisedCycleBasis.create(n)
    rng = np.random.default_rng(999)
    
    # Generate all spectra upfront
    all_spectra = [dirichlet_suleimanova(n, trace_sum, rng=rng) for _ in range(max_trials)]
    results = [B.min_entry(lam) >= -1e-12 for lam in all_spectra]
    
    rows = []
    for n_trials in checkpoints:
        successes = sum(results[:n_trials])
        p_hat = successes / n_trials
        
        # Wilson CI
        z = 1.959963984540054
        denom = 1 + z**2 / n_trials
        centre = (p_hat + z**2/(2*n_trials)) / denom
        half = z * math.sqrt(p_hat*(1-p_hat)/n_trials + z**2/(4*n_trials**2)) / denom
        
        rows.append({
            'basis': 'Phase',
            'n_trials': n_trials,
            'rate': p_hat,
            'ci_lower': max(0, centre - half),
            'ci_upper': min(1, centre + half),
            'ci_width': 2 * half
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'exp5_convergence.csv', index=False)
    print(f"  Convergence width at {max_trials}: {df.iloc[-1]['ci_width']:.4f}")
    
    return df


# ============================================================================
# ENHANCED PLOTTING
# ============================================================================

def plot_exp1_coherence_comparison(df: pd.DataFrame, outdir: pathlib.Path):
    """Plot M(Q) vs n for all bases with theoretical curves."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for basis in df['basis'].unique():
        data = df[df['basis'] == basis].sort_values('n')
        ax.plot(data['n'], data['M_empirical'], 'o', label=f'{basis} (empirical)', 
                markersize=3, alpha=0.7)
        ax.plot(data['n'], data['M_theoretical'], '-', linewidth=1.5, 
                label=f'{basis} (theory)', alpha=0.8)
    
    ax.set_xlabel('Dimension $n$', fontsize=11)
    ax.set_ylabel('Coherence $M(Q)$', fontsize=11)
    ax.set_ylim([0.9, 2.1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig_exp1_coherence.pdf', dpi=300)
    plt.close(fig)


def plot_exp2_success_heatmaps(df: pd.DataFrame, outdir: pathlib.Path):
    """Create separate heatmap for each dimension."""
    for n in df['n'].unique():
        bases_for_n = sorted(df[df['n']==n]['basis'].unique())
        fig, axes = plt.subplots(1, len(bases_for_n), 
                                 figsize=(4*len(bases_for_n), 3), sharey=True)
        if len(bases_for_n) == 1:
            axes = [axes]
        
        data_n = df[df['n'] == n]
        
        for idx, basis in enumerate(bases_for_n):
            ax = axes[idx]
            data_b = data_n[data_n['basis'] == basis].sort_values('trace_sum')
            
            # Color by success rate
            scatter = ax.scatter(data_b['trace_sum'], [0]*len(data_b), 
                                 c=data_b['rate'], s=80, cmap='RdYlGn', 
                                 vmin=0, vmax=1, edgecolors='k', linewidths=0.5)
            ax.set_xlabel('Trace sum $1+\\sum_{j\\geq 2}\\lambda_j$')
            ax.set_title(f'{basis} basis')
            ax.set_yticks([])
            ax.set_ylim([-0.5, 0.5])
            
            if idx == len(axes) - 1:
                plt.colorbar(scatter, ax=ax, label='Success rate')
        
        fig.suptitle(f'Realisation probability: $n={n}$', fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f'fig_exp2_heatmap_n{n}.pdf', dpi=300)
        plt.close(fig)


def plot_exp3_sharpness(df: pd.DataFrame, outdir: pathlib.Path):
    """Plot min entry vs t with vertical line at theoretical threshold."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['t'], df['min_entry'], 'k-', linewidth=1.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Zero line')
    ax.axvline(df['t_star'].iloc[0], color='blue', linestyle='--', linewidth=1, 
               alpha=0.7, label=f"$t^* = {df['t_star'].iloc[0]:.4f}$")
    
    ax.set_xlabel('$t = \\lambda_{j^*}$', fontsize=11)
    ax.set_ylabel('$\\min_{k,\\ell} P(\\Lambda)_{k\\ell}$', fontsize=11)
    ax.set_title(f"Sharpness demonstration: $n={df['n'].iloc[0]}$, canonical cycle basis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig_exp3_sharpness.pdf', dpi=300)
    plt.close(fig)


def plot_exp4_timing(df: pd.DataFrame, outdir: pathlib.Path):
    """Plot construction and evaluation times vs n (log-log scale)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for basis in df['basis'].unique():
        data = df[df['basis'] == basis].sort_values('n')
        
        # Construction time
        ax1.loglog(data['n'], data['construction_time_ms'], 'o-', 
                   label=basis, markersize=6)
        
        # Evaluation time
        ax2.loglog(data['n'], data['evaluation_time_ms'], 's-', 
                   label=basis, markersize=6)
    
    # Reference lines
    n_ref = np.array([4, 128])
    ax1.loglog(n_ref, 0.01 * (n_ref/4)**2, 'k:', alpha=0.5, label='O(n^2)')
    ax2.loglog(n_ref, 0.001 * (n_ref/4)**2.5, 'k:', alpha=0.5, label='O(n^{2.5})')
    
    ax1.set_xlabel('Dimension $n$', fontsize=11)
    ax1.set_ylabel('Construction time (ms)', fontsize=11)
    ax1.set_title('Basis construction cost')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    ax2.set_xlabel('Dimension $n$', fontsize=11)
    ax2.set_ylabel('Evaluation time (ms)', fontsize=11)
    ax2.set_title('$P(\\Lambda)$ computation cost')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(outdir / 'fig_exp4_timing.pdf', dpi=300)
    plt.close(fig)



def plot_exp5_convergence(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    """Plot convergence of success rate estimate with error bars.

    Ensures non-negative asymmetric error bars for Matplotlib by clipping
    tiny negative values caused by rounding.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    data = df.sort_values('n_trials').reset_index(drop=True)

    # Build asymmetric yerr = [lower, upper] with non-negative entries
    lower = (data['rate'] - data['ci_lower']).astype(float).clip(lower=0).to_numpy()
    upper = (data['ci_upper'] - data['rate']).astype(float).clip(lower=0).to_numpy()
    yerr = np.vstack([lower, upper])

    ax.errorbar(
        data['n_trials'].to_numpy(),
        data['rate'].to_numpy(),
        yerr=yerr,
        fmt='o-',
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
    )

    # Final estimate as a horizontal line
    final_rate = float(data.iloc[-1]['rate'])
    ax.axhline(
        final_rate, color='red', linestyle='--', alpha=0.5,
        label=f'Final estimate: {final_rate:.4f}'
    )

    ax.set_xlabel('Number of Monte Carlo trials', fontsize=11)
    ax.set_ylabel('Success rate $P[P(\\Lambda) \\geq 0]$', fontsize=11)
    ax.set_title('Convergence of Monte Carlo estimate (95% CI)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / 'fig_exp5_convergence.pdf', dpi=300)
    plt.close(fig)




# ============================================================================
# SUPPLEMENTARY ANALYSIS
# ============================================================================

def supplementary_threshold_landscape(outdir: pathlib.Path):
    """
    Create comprehensive comparison of δ_n vs n for all formulas.
    """
    print("Supplementary: Threshold landscape")
    
    n_vals = np.arange(3, 101)
    delta_can = [delta_cycle(n) for n in n_vals]
    delta_ph = [delta_phase(n) for n in n_vals]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(n_vals, delta_can, 'o', markersize=3, label='$\\delta_n$ (canonical)', alpha=0.7)
    ax.plot(n_vals, delta_ph, 's', markersize=3, label='$\\delta_n^{\\mathrm{(ph)}}$ (phase)', alpha=0.7)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5,
               label='Baseline $\\delta = 1/2$', alpha=0.7)
    
    # Highlight 8|n points
    n_mult8 = [n for n in n_vals if n % 8 == 0]
    delta_mult8 = [delta_cycle(n) for n in n_mult8]
    ax.plot(n_mult8, delta_mult8, 'rx', markersize=8, label='$8 \\mid n$ (canonical)',
            markeredgewidth=2)
    
    ax.set_xlabel('Dimension $n$', fontsize=11)
    ax.set_ylabel('Sufficient threshold $\\delta$', fontsize=11)
    ax.set_title('Comparison of dimension-dependent bounds')
    ax.set_ylim([0, 0.55])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig_supp_threshold_landscape.pdf', dpi=300)
    plt.close(fig)
    
    # Save data
    df = pd.DataFrame({
        'n': n_vals,
        'delta_canonical': delta_can,
        'delta_phase': delta_ph
    })
    df.to_csv(outdir / 'supp_thresholds.csv', index=False)



def supplementary_hadamard_coverage(outdir: pathlib.Path, max_n: int = 100):
    """
    Identify which n ≤ max_n admit Hadamard matrices via Walsh or Paley.
    """
    print(f"Supplementary: Hadamard coverage up to n={max_n}")
    
    rows = []
    for n in range(1, max_n + 1):
        has_walsh = (n > 1 and (n & (n-1)) == 0)
        has_paley = False
        
        q = n - 1
        if q >= 3 and q % 4 == 3 and is_prime(q):
            has_paley = True
        
        rows.append({
            'n': n,
            'Walsh': has_walsh,
            'Paley': has_paley,
            'Any_Hadamard': has_walsh or has_paley
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'supp_hadamard_coverage.csv', index=False)
    
    coverage = df['Any_Hadamard'].sum() / len(df) * 100
    print(f"  Hadamard coverage: {coverage:.1f}% of n ≤ {max_n}")
    
    return df


# ============================================================================
# MAIN DRIVER
# ============================================================================

def main():
    """Run complete enhanced experimental suite."""
    print("=" * 70)
    print("ENHANCED NUMERICAL EXPERIMENTS FOR SDIEP PAPER")
    print("=" * 70)
    
    outdir = pathlib.Path("out_enhanced")
    outdir.mkdir(exist_ok=True)
    
    # Experiment 1: Systematic coherence (n=3..100)
    print("\n" + "="*70)
    df1 = experiment_1_systematic_coherence(outdir)
    plot_exp1_coherence_comparison(df1, outdir)
    
    # Experiment 2: Success rate heatmaps
    print("\n" + "="*70)
    df2 = experiment_2_success_rate_heatmap(
        outdir, 
        n_values=[8, 12, 16, 24],
        trace_resolution=50,
        trials_per_point=1000
    )
    plot_exp2_success_heatmaps(df2, outdir)
    
    # Experiment 3: Sharpness analysis
    print("\n" + "="*70)
    df3 = experiment_3_sharpness_analysis(outdir, n_test=12, t_resolution=250)
    plot_exp3_sharpness(df3, outdir)
    
    # Experiment 4: Computational cost
    print("\n" + "="*70)
    df4 = experiment_4_computational_cost(
        outdir,
        n_values=[4, 8, 16, 32, 64, 128],
        repeats=30
    )
    plot_exp4_timing(df4, outdir)
    
    # Experiment 5: Convergence analysis
    print("\n" + "="*70)
    df5 = experiment_5_convergence_analysis(
        outdir,
        n=16,
        trace_sum=0.4,
        max_trials=10000
    )
    plot_exp5_convergence(df5, outdir)
    
    # Supplementary analyses
    print("\n" + "="*70)
    print("Supplementary analyses")
    supplementary_threshold_landscape(outdir)
    supplementary_hadamard_coverage(outdir, max_n=100)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results written to: {outdir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()