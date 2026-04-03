"""Plotting helpers for the SDIEP experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_coherence(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot empirical and theoretical coherence values."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for basis in df["basis"].unique():
        g = df[df["basis"] == basis].sort_values("n")
        ax.plot(g["n"], g["M_empirical"], "o", markersize=3, alpha=0.7, label=f"{basis} (empirical)")
        ax.plot(g["n"], g["M_theoretical"], "-", linewidth=1.5, alpha=0.8, label=f"{basis} (theory)")
    ax.set_xlabel(r"Dimension $n$")
    ax.set_ylabel(r"Coherence $M(Q)$")
    ax.set_ylim(0.9, 2.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp1_coherence.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_success_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    """Create one success-rate panel per tested dimension."""
    for n in sorted(df["n"].unique()):
        bases = sorted(df[df["n"] == n]["basis"].unique())
        fig, axes = plt.subplots(1, len(bases), figsize=(4 * len(bases), 3), sharey=True)
        if len(bases) == 1:
            axes = [axes]
        data_n = df[df["n"] == n]
        for idx, basis in enumerate(bases):
            ax = axes[idx]
            g = data_n[data_n["basis"] == basis].sort_values("trace_sum")
            scatter = ax.scatter(g["trace_sum"], np.zeros(len(g)), c=g["rate"], s=80, cmap="RdYlGn", vmin=0, vmax=1, edgecolors="k", linewidths=0.5)
            ax.set_xlabel(r"Trace sum $1+\sum_{j\geq 2}\lambda_j$")
            ax.set_title(f"{basis} basis")
            ax.set_yticks([])
            ax.set_ylim([-0.5, 0.5])
            if idx == len(axes) - 1:
                plt.colorbar(scatter, ax=ax, label="Success rate")
        fig.suptitle(rf"Realisation probability: $n={n}$", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_exp2_heatmap_n{n}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_sharpness(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot the sharpness curve with the theoretical crossing."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["t"], df["min_entry"], "k-", linewidth=1.5)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="Zero line")
    t_star = float(df["t_star"].iloc[0])
    ax.axvline(t_star, color="blue", linestyle="--", linewidth=1.0, alpha=0.7, label=rf"$t^*={t_star:.4f}$")
    ax.set_xlabel(r"$t = \lambda_{j^*}$")
    ax.set_ylabel(r"$\min_{k,\ell} P(\Lambda)_{k\ell}$")
    ax.set_title(rf"Sharpness demonstration: $n={int(df['n'].iloc[0])}$, canonical cycle basis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp3_sharpness.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timing(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot construction and evaluation timings with asymptotic guides."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    for basis in df["basis"].unique():
        g = df[df["basis"] == basis].sort_values("n")
        ax1.loglog(g["n"], g["construction_time_ms"], "o-", label=basis, markersize=5)
        ax2.loglog(g["n"], g["evaluation_time_ms"], "s-", label=basis, markersize=5)
    ref_n = np.array([4, 128], dtype=float)
    cycle = df[df["basis"] == "Cycle"]
    c_anchor = float(cycle[cycle["n"] == 16]["construction_time_ms"].iloc[0])
    e_anchor = float(cycle[cycle["n"] == 16]["evaluation_time_ms"].iloc[0])
    ax1.loglog(ref_n, c_anchor * (ref_n / 16.0) ** 2, ":", alpha=0.6, label=r"$O(n^2)$")
    ax2.loglog(ref_n, e_anchor * (ref_n / 16.0) ** 3, ":", alpha=0.6, label=r"$O(n^3)$")
    ax1.set_xlabel(r"Dimension $n$")
    ax1.set_ylabel("Construction time (ms)")
    ax1.set_title("Basis construction cost")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()
    ax2.set_xlabel(r"Dimension $n$")
    ax2.set_ylabel("Evaluation time (ms)")
    ax2.set_title(r"$P(\Lambda)$ computation cost")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp4_timing.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot convergence of the success-rate estimate."""
    fig, ax = plt.subplots(figsize=(7, 5))
    g = df.sort_values("n_trials").reset_index(drop=True)
    lower = (g["rate"] - g["ci_lower"]).clip(lower=0.0).to_numpy()
    upper = (g["ci_upper"] - g["rate"]).clip(lower=0.0).to_numpy()
    yerr = np.vstack([lower, upper])
    ax.errorbar(g["n_trials"], g["rate"], yerr=yerr, fmt="o-", capsize=5, capthick=2, linewidth=2, markersize=8)
    final = float(g.iloc[-1]["rate"])
    ax.axhline(final, color="red", linestyle="--", alpha=0.5, label=rf"Final estimate: {final:.4f}")
    ax.set_xlabel("Number of Monte-Carlo trials")
    ax.set_ylabel(r"Success rate $\mathbb{P}[P(\Lambda)\ge 0]$")
    ax.set_title("Convergence of Monte-Carlo estimate (95% CI)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp5_convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_landscape(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot the canonical and phase-optimised thresholds up to n=100."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n"], df["delta_canonical"], "o", markersize=3, alpha=0.7, label=r"$\delta_n$ (canonical)")
    ax.plot(df["n"], df["delta_phase"], "s", markersize=3, alpha=0.7, label=r"$\delta_n^{(\mathrm{ph})}$ (phase)")
    ax.axhline(0.5, linestyle="--", linewidth=1.5, alpha=0.7, label=r"baseline $\delta=1/2$")
    n_mult8 = [int(n) for n in df["n"] if n % 8 == 0]
    d_mult8 = [float(df[df["n"] == n]["delta_canonical"].iloc[0]) for n in n_mult8]
    ax.plot(n_mult8, d_mult8, "x", markersize=8, markeredgewidth=2, label=r"$8\mid n$ (canonical)")
    ax.set_xlabel(r"Dimension $n$")
    ax.set_ylabel(r"Sufficient threshold $\delta$")
    ax.set_ylim(0.0, 0.55)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_supp_threshold_landscape.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
