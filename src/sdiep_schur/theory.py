r"""Closed-form threshold formulae and small arithmetic helpers for SDIEP."""

from __future__ import annotations

import math
from typing import Iterable


def is_prime(n: int) -> bool:
    """Return whether ``n`` is prime.

    Parameters
    ----------
    n:
        Integer to test.

    Returns
    -------
    bool
        ``True`` if ``n`` is prime, otherwise ``False``.

    Notes
    -----
    This is a deterministic Miller--Rabin implementation for 32-bit inputs,
    sufficient for all dimensions used in the accompanying experiments.
    """
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
    for a in (2, 7, 61):
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


def rho_mod8(n: int) -> int:
    r"""Return the arithmetic factor :math:`\rho(n)\in\{0,1,2,4\}`.

    Parameters
    ----------
    n:
        Dimension.

    Returns
    -------
    int
        The mod-8 factor from Lemma 2.2 of the paper.
    """
    residue = n % 8
    if residue == 0:
        return 0
    if residue in (1, 3, 5, 7):
        return 1
    if residue in (2, 6):
        return 2
    if residue == 4:
        return 4
    raise RuntimeError("Unreachable residue class.")


def delta_cycle(n: int) -> float:
    r"""Return the exact canonical-cycle threshold :math:`\delta_n`.

    Parameters
    ----------
    n:
        Dimension.

    Returns
    -------
    float
        The sufficient trace-sum threshold for the canonical cycle basis.
    """
    theta = math.pi * rho_mod8(n) / (4.0 * n)
    return 1.0 - 1.0 / (2.0 * math.cos(theta) ** 2)


def delta_phase(n: int) -> float:
    r"""Return the phase-optimised threshold :math:`\delta_n^{(\mathrm{ph})}`.

    Parameters
    ----------
    n:
        Dimension.

    Returns
    -------
    float
        The phase-optimised sufficient threshold.
    """
    if n % 2 == 1:
        theta = math.pi / (4.0 * n)
    elif n % 4 == 0:
        theta = math.pi / n
    else:
        theta = math.pi / (2.0 * n)
    return 1.0 - 1.0 / (2.0 * math.cos(theta) ** 2)


def hadamard_orders_up_to(max_n: int) -> list[int]:
    """Return orders up to ``max_n`` covered by Walsh or Paley constructions.

    Parameters
    ----------
    max_n:
        Upper bound.

    Returns
    -------
    list[int]
        Covered orders in ``{1, ..., max_n}`` for the implemented families.
    """
    orders: list[int] = []
    for n in range(1, max_n + 1):
        if n > 1 and (n & (n - 1)) == 0:
            orders.append(n)
            continue
        q = n - 1
        if q >= 3 and q % 4 == 3 and is_prime(q):
            orders.append(n)
    return orders
