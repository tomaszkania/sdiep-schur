#!/usr/bin/env python3
"""Regenerate the CSV data and figures used in the SDIEP paper."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdiep_schur.experiments import (
    computational_cost,
    convergence_study,
    sharpness_curve,
    success_rate_grid,
    systematic_coherence,
    threshold_landscape,
)
from sdiep_schur.plotting import (
    plot_coherence,
    plot_convergence,
    plot_sharpness,
    plot_success_heatmaps,
    plot_threshold_landscape,
    plot_timing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Directory for CSV outputs.")
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"), help="Directory for figure PDFs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    df1 = systematic_coherence(args.out_dir)
    plot_coherence(df1, args.figure_dir)

    df2 = success_rate_grid(args.out_dir)
    plot_success_heatmaps(df2, args.figure_dir)

    df3 = sharpness_curve(args.out_dir)
    plot_sharpness(df3, args.figure_dir)

    df4 = computational_cost(args.out_dir)
    plot_timing(df4, args.figure_dir)

    df5 = convergence_study(args.out_dir)
    plot_convergence(df5, args.figure_dir)

    df_supp = threshold_landscape(args.out_dir)
    plot_threshold_landscape(df_supp, args.figure_dir)


if __name__ == "__main__":
    main()
