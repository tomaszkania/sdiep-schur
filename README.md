# sdiep-schur

Reference implementation and reproducible experiments for **Schur-template constructions in the symmetric doubly stochastic inverse eigenvalue problem (SDIEP)**.

This repository accompanies the paper

> *Dimension-dependent bounds for the SDIEP via phase optimisation and Paley-type constructions*.

It provides:

- explicit orthogonal bases with Perron column `1/sqrt(n)`;
- constructive realisation via `P(Λ) = Q Λ Qᵀ`;
- coherence computations and threshold formulae;
- reproducible numerical experiments and publication figures.

The repository is organised in the same spirit as the companion project `quadratic_diagonal`: the core implementation lives under `src/`, experiments under `scripts/`, regression checks under `tests/`, generated data under `data/`, and the manuscript sources under `paper/`.

## Recommended GitHub name

Recommended repository name: **`sdiep-schur`**.

A suitable one-line repository description is:

> Reference implementation and reproducible experiments for Schur-template constructions in the symmetric doubly stochastic inverse eigenvalue problem (SDIEP), including cycle, phase-optimised, and Hadamard bases.

## What the software does

Given a basis `Q` and a Suleĭmanova spectrum `Λ = diag(1, λ₂, …, λ_n)`, the package can:

- build the candidate matrix `P(Λ) = Q Λ Qᵀ`;
- compute the coherence `M(Q)`;
- evaluate the explicit sufficient thresholds `δ_n` and `δ_n^(ph)`;
- sample random Suleĭmanova lists with a prescribed trace sum;
- reproduce the numerical experiments from the paper.

## Repository layout

| Path | Purpose |
|---|---|
| `src/sdiep_schur/` | Core package |
| `scripts/reproduce_paper.py` | Regenerates all CSV data and figures used in the paper |
| `tests/test_core.py` | Lightweight regression tests |
| `data/` | Generated CSV outputs |
| `paper/` | Clean and tracked manuscript sources and PDFs |
| `paper/figures/` | Figure files used by the manuscript |
| `paper/data/` | CSV tables used in the manuscript |
| `notebooks/` | Optional exploratory notebooks |

## Installation

```bash
pip install -e .
pip install -e .[test]
```

## Quick start

```python
from sdiep_schur import CycleBasis, delta_cycle
import numpy as np

n = 12
Q = CycleBasis.create(n)
lambda_diag = np.zeros(n)
lambda_diag[0] = 1.0
lambda_diag[1] = -0.5
P = Q.compute_P(lambda_diag)
print(Q.coherence())
print(delta_cycle(n))
print(P.min())
```

## Reproducibility

To regenerate the paper data and figures:

```bash
python scripts/reproduce_paper.py --out-dir data
```

This will write CSV outputs to `data/` and figures to `paper/figures/`.
