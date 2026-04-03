"""Sampling routines for numerical experiments."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def dirichlet_suleimanova(
    n: int,
    trace_sum: float,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a Suleĭmanova list with prescribed trace sum.

    Parameters
    ----------
    n:
        Dimension.
    trace_sum:
        Value of ``1 + sum(lambda_j)`` in ``(0, 1]``.
    alpha:
        Dirichlet concentration parameter.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Array ``(lambda_0, ..., lambda_{n-1})`` with ``lambda_0 = 1`` and
        all remaining entries in ``[-1, 0]``.
    """
    if not (0.0 < trace_sum <= 1.0):
        raise ValueError("trace_sum must lie in (0, 1].")
    rng = np.random.default_rng() if rng is None else rng
    deficit = 1.0 - trace_sum
    weights = rng.dirichlet(alpha * np.ones(n - 1))
    lambdas = np.empty(n, dtype=float)
    lambdas[0] = 1.0
    lambdas[1:] = -deficit * weights
    return lambdas


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    r"""Return the Wilson score interval for a Bernoulli proportion.

    Parameters
    ----------
    successes:
        Number of successes.
    trials:
        Number of Bernoulli trials.
    z:
        Gaussian quantile; defaults to a 95\% interval.

    Returns
    -------
    tuple[float, float]
        Lower and upper endpoints.
    """
    if trials <= 0:
        raise ValueError("trials must be positive.")
    p_hat = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (p_hat + z**2 / (2.0 * trials)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / trials + z**2 / (4.0 * trials**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
