"""Structured orthogonal bases for Schur-template SDIEP constructions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np

from .theory import is_prime


@dataclass
class OrthogonalBasis:
    """Orthogonal matrix with Perron column ``1/sqrt(n)``.

    Parameters
    ----------
    n:
        Dimension.
    Q:
        Real orthogonal matrix.
    """

    n: int
    Q: np.ndarray

    def __post_init__(self) -> None:
        if self.Q.shape != (self.n, self.n):
            raise ValueError("Q must have shape (n, n).")
        if not np.allclose(self.Q.T @ self.Q, np.eye(self.n), atol=1e-10):
            raise ValueError("Q must be orthogonal.")
        perron = np.ones(self.n) / math.sqrt(self.n)
        if not np.allclose(self.Q[:, 0], perron, atol=1e-10):
            raise ValueError("The first column of Q must be the Perron vector.")

    def compute_P(self, lambdas: np.ndarray) -> np.ndarray:
        """Return ``P(Λ) = Q diag(lambdas) Qᵀ``."""
        if lambdas.shape != (self.n,):
            raise ValueError("lambdas must have shape (n,).")
        return self.Q @ np.diag(lambdas) @ self.Q.T

    def coherence(self) -> float:
        """Return ``M(Q) = n max_{j>=1} ||q_j||_inf^2``."""
        col_norms = np.max(np.abs(self.Q[:, 1:]), axis=0)
        return float(self.n * np.max(col_norms**2))

    def min_entry(self, lambdas: np.ndarray) -> float:
        """Return the minimum entry of ``P(Λ)``."""
        return float(self.compute_P(lambdas).min())


@dataclass
class CycleBasis(OrthogonalBasis):
    """Canonical real cycle basis used in the paper."""

    @staticmethod
    def _build_Q(n: int) -> np.ndarray:
        q = np.zeros((n, n), dtype=float)
        q[:, 0] = 1.0 / math.sqrt(n)
        grid = np.arange(n)
        for k in range(1, n):
            q[:, k] = math.sqrt(2.0 / n) * np.sin(2.0 * math.pi * k * grid / n + math.pi / 4.0)
        return q

    @classmethod
    def create(cls, n: int) -> "CycleBasis":
        return cls(n=n, Q=cls._build_Q(n))


@dataclass
class PhaseOptimisedCycleBasis(OrthogonalBasis):
    """Cycle basis with phase chosen to minimise the peak sup norm."""

    @staticmethod
    def _phi_optimal(n: int, j: int) -> float:
        n_prime = n // math.gcd(n, j)
        order = n_prime // math.gcd(n_prime, 4)
        spacing = (math.pi / 2.0) / order
        return spacing / 2.0

    @staticmethod
    def _build_Q(n: int) -> np.ndarray:
        cols = [np.ones(n) / math.sqrt(n)]
        grid = np.arange(n)
        for j in range(1, (n - 1) // 2 + 1):
            phi = PhaseOptimisedCycleBasis._phi_optimal(n, j)
            cols.append(math.sqrt(2.0 / n) * np.sin(2.0 * math.pi * j * grid / n + phi))
            cols.append(math.sqrt(2.0 / n) * np.cos(2.0 * math.pi * j * grid / n + phi))
        if n % 2 == 0:
            cols.append(math.sqrt(2.0 / n) * np.sin(math.pi * grid + math.pi / 4.0))
        q = np.column_stack(cols)
        q_orth = np.zeros_like(q)
        q_orth[:, 0] = q[:, 0]
        for col in range(1, n):
            v = q[:, col].copy()
            for prev in range(col):
                v -= np.dot(q_orth[:, prev], v) * q_orth[:, prev]
            q_orth[:, col] = v / np.linalg.norm(v)
        return q_orth

    @classmethod
    def create(cls, n: int) -> "PhaseOptimisedCycleBasis":
        return cls(n=n, Q=cls._build_Q(n))


@dataclass
class WalshHadamardBasis(OrthogonalBasis):
    """Walsh--Hadamard basis for powers of two."""

    @staticmethod
    def _hadamard_sylvester(n: int) -> np.ndarray:
        if n < 1 or (n & (n - 1)) != 0:
            raise ValueError("n must be a power of two.")
        h = np.array([[1.0]])
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        return h

    @classmethod
    def create(cls, n: int) -> "WalshHadamardBasis":
        h = cls._hadamard_sylvester(n)
        q = h / math.sqrt(n)
        if not np.allclose(q[:, 0], np.ones(n) / math.sqrt(n), atol=1e-12):
            for j in range(n):
                if np.allclose(q[:, j], np.ones(n) / math.sqrt(n), atol=1e-12):
                    q[:, [0, j]] = q[:, [j, 0]]
                    break
        return cls(n=n, Q=q)


@dataclass
class PaleyHadamardBasis(OrthogonalBasis):
    """Paley type-I Hadamard basis for ``q ≡ 3 (mod 4)`` prime."""

    @staticmethod
    def _legendre_symbol(a: int, p: int) -> int:
        a %= p
        if a == 0:
            return 0
        ls = pow(a, (p - 1) // 2, p)
        return 1 if ls == 1 else -1

    @staticmethod
    def _build_hadamard_paley(q: int) -> np.ndarray:
        if q < 3 or q % 4 != 3 or not is_prime(q):
            raise ValueError("q must be prime and congruent to 3 modulo 4.")
        a = np.empty((q, q), dtype=float)
        for x in range(q):
            for y in range(q):
                if x == y:
                    a[x, y] = 1.0
                else:
                    a[x, y] = 1.0 if PaleyHadamardBasis._legendre_symbol((x - y) % q, q) == 1 else -1.0
        h = np.empty((q + 1, q + 1), dtype=float)
        h[0, 0] = 1.0
        h[0, 1:] = 1.0
        h[1:, 0] = -1.0
        h[1:, 1:] = a
        d = np.diag(np.concatenate(([1.0], -np.ones(q))))
        return (d @ h) / math.sqrt(q + 1)

    @classmethod
    def create(cls, q: int) -> "PaleyHadamardBasis":
        qmat = cls._build_hadamard_paley(q)
        return cls(n=q + 1, Q=qmat)
