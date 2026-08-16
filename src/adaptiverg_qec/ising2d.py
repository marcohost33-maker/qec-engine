"""Phase-3b: vectorized 2D Ising sampler and majority-rule blocking.

The module provides the homogeneous 2D Ising reference model used by the MCRG
experiments.  The blocking rule is deliberately deterministic for reproducible
runs, but its tie rule must also preserve the exact global spin-flip symmetry.
In particular, for every configuration ``s`` and fixed tie-breaking metadata,

    majority_block_b2(-s) == -majority_block_b2(s)

must hold.  This is stronger than merely obtaining a 50/50 tie distribution in
aggregate and prevents artificial coupling of even and odd RG sectors.

Conventions
-----------
Spins ``s_ij`` are in ``{+1, -1}`` on an ``L x L`` square lattice with periodic
boundary conditions.  The Hamiltonian is ``H = -J sum_<ij> s_i s_j`` and the
Boltzmann target is proportional to ``exp(K sum_<ij> s_i s_j)`` with ``K=J/T``.
For ``J=1`` the critical coupling is
``K_c = log(1 + sqrt(2))/2``.

References
----------
- L. Onsager, Phys. Rev. 65 (1944) 117.
- L. P. Kadanoff, Physics 2 (1966) 263.
- R. H. Swendsen, Phys. Rev. Lett. 42 (1979) 859.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KC_2D",
    "TC_2D",
    "checkerboard_metropolis",
    "Ising2DChain",
    "energy_per_spin",
    "magnetization_per_spin",
    "majority_block_b2",
    "exact_energy_per_spin_2x2",
]

KC_2D: float = 0.5 * math.log(1.0 + math.sqrt(2.0))
TC_2D: float = 1.0 / KC_2D


@dataclass(frozen=True)
class Ising2DChain:
    """Result of a 2D-Ising Metropolis run.

    ``configs`` has shape ``(n_records, L, L)`` and values in ``{+1, -1}``.
    """

    configs: np.ndarray
    K: float
    L: int
    acceptance: float
    seed: int

    def __post_init__(self) -> None:
        if self.configs.ndim != 3:
            raise ValueError(f"configs must be 3D (n,L,L), got ndim={self.configs.ndim}")
        if self.configs.shape[1:] != (self.L, self.L):
            raise ValueError(
                f"configs trailing shape must be ({self.L},{self.L}), got {self.configs.shape[1:]}"
            )
        if not np.all((self.configs == 1) | (self.configs == -1)):
            raise ValueError("configs must contain only +/-1 spins")
        if not np.isfinite(self.K) or self.K <= 0.0:
            raise ValueError(f"K must be finite and > 0, got {self.K}")
        if not np.isfinite(self.acceptance) or not (0.0 <= self.acceptance <= 1.0):
            raise ValueError(f"acceptance must be in [0,1], got {self.acceptance}")


def _neighbor_field(s: np.ndarray) -> np.ndarray:
    """Return the sum of the four periodic nearest-neighbour spins."""
    return (
        np.roll(s, 1, axis=0)
        + np.roll(s, -1, axis=0)
        + np.roll(s, 1, axis=1)
        + np.roll(s, -1, axis=1)
    )


def _checkerboard_masks(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Return even/odd checkerboard sub-lattice masks."""
    ii, jj = np.indices((L, L))
    even = (ii + jj) % 2 == 0
    return even, ~even


def checkerboard_metropolis(
    K: float,
    L: int,
    *,
    n_sweeps: int,
    burn_in: int,
    seed: int,
    record_every: int = 1,
) -> Ising2DChain:
    """Run vectorized checkerboard single-spin Metropolis dynamics.

    A sweep updates both independent checkerboard sub-lattices once.  The flip
    log-probability ratio is ``-2*K*s*sum_neighbours``.
    """
    if not np.isfinite(K):
        raise ValueError(f"K must be finite, got {K}")
    if K <= 0.0:
        raise ValueError(f"K must be > 0 (ferromagnetic), got {K}")
    if not isinstance(L, (int, np.integer)):
        raise TypeError(f"L must be an integer, got {type(L).__name__}")
    if L < 4:
        raise ValueError(f"L must be >= 4, got {L}")
    if L % 2 != 0:
        raise ValueError(f"L must be even for b=2 majority blocking, got {L}")
    if not isinstance(n_sweeps, (int, np.integer)) or n_sweeps < 1:
        raise ValueError(f"n_sweeps must be an integer >= 1, got {n_sweeps}")
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError(f"burn_in must be an integer >= 0, got {burn_in}")
    if not isinstance(record_every, (int, np.integer)) or record_every < 1:
        raise ValueError(f"record_every must be an integer >= 1, got {record_every}")

    rng = np.random.default_rng(seed)
    s = np.where(rng.random((L, L)) < 0.5, 1.0, -1.0)
    even, odd = _checkerboard_masks(L)

    n_accept = 0
    n_attempt = 0
    records: list[np.ndarray] = []
    total = burn_in + n_sweeps

    for sweep in range(total):
        for mask in (even, odd):
            nb = _neighbor_field(s)
            dlogpi = -2.0 * K * s * nb
            accept_prob = np.exp(np.minimum(dlogpi, 0.0))
            flip = mask & (rng.random((L, L)) < accept_prob)
            if sweep >= burn_in:
                n_accept += int(flip.sum())
                n_attempt += int(mask.sum())
            s = np.where(flip, -s, s)
        if sweep >= burn_in and ((sweep - burn_in) % record_every == 0):
            records.append(s.copy())

    configs = np.asarray(records, dtype=np.int8)
    acceptance = n_accept / n_attempt if n_attempt else 0.0
    return Ising2DChain(
        configs=configs,
        K=float(K),
        L=int(L),
        acceptance=float(acceptance),
        seed=int(seed),
    )


def energy_per_spin(s: np.ndarray) -> np.ndarray:
    """Return periodic nearest-neighbour Ising energy per spin, with ``J=1``."""
    s = np.asarray(s, dtype=np.float64)
    if s.ndim < 2:
        raise ValueError(f"s must have at least two dimensions, got ndim={s.ndim}")
    if s.shape[-1] < 2 or s.shape[-2] < 2:
        raise ValueError(f"need L>=2 in both dims, got {s.shape[-2:]}")
    right = s * np.roll(s, -1, axis=-1)
    down = s * np.roll(s, -1, axis=-2)
    bonds = right.sum(axis=(-1, -2)) + down.sum(axis=(-1, -2))
    n = s.shape[-1] * s.shape[-2]
    return -bonds / n


def magnetization_per_spin(s: np.ndarray) -> np.ndarray:
    """Return signed magnetization per spin."""
    s = np.asarray(s, dtype=np.float64)
    if s.ndim < 2:
        raise ValueError(f"s must have at least two dimensions, got ndim={s.ndim}")
    n = s.shape[-1] * s.shape[-2]
    return s.sum(axis=(-1, -2)) / n


def exact_energy_per_spin_2x2(K: float) -> float:
    """Return exact ``<E/N>`` for the periodic 2x2 lattice by enumeration."""
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be finite and > 0, got {K}")
    states = np.arange(16, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(4)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(16, 2, 2).astype(np.float64)
    e = energy_per_spin(s)
    logw = -K * 4.0 * e
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    return float((w * e).sum())


def _splitmix64_grid(config_index: int, br: np.ndarray, bc: np.ndarray, seed: int) -> np.ndarray:
    """Return a deterministic uint64 hash for each block coordinate."""
    mask = (1 << 64) - 1
    ci_u = np.uint64(int(config_index) & mask)
    seed_u = np.uint64(int(seed) & mask)
    with np.errstate(over="ignore"):
        h = (
            (ci_u * np.uint64(0x9E3779B97F4A7C15))
            ^ (br.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9))
            ^ (bc.astype(np.uint64) * np.uint64(0x94D049BB133111EB))
            ^ seed_u
        )
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return h


def majority_block_b2(s: np.ndarray, *, config_index: int = 0, seed: int = 0) -> np.ndarray:
    """Apply a ``2x2 -> 1`` majority-rule block-spin transformation.

    Non-tied blocks use the sign of the four-spin sum.  A two-vs-two tie is
    resolved by selecting one of the *four input spins* using a deterministic
    SplitMix64-style block hash.  Selecting an input spin, instead of generating
    an independent ``+/-1`` hash bit, enforces exact global-spin-flip
    equivariance while retaining reproducibility and an unbiased selector:

    ``majority_block_b2(-s, ...) == -majority_block_b2(s, ...)``.
    """
    s = np.asarray(s, dtype=np.int64)
    if s.ndim != 2:
        raise ValueError(f"s must be 2D (L,L), got ndim={s.ndim}")
    L = s.shape[0]
    if s.shape[1] != L:
        raise ValueError(f"s must be square, got {s.shape}")
    if L < 2 or L % 2 != 0:
        raise ValueError(f"L must be even and >= 2, got {L}")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError("s must contain only +/-1 spins")

    lb = L // 2
    block_view = s.reshape(lb, 2, lb, 2)
    block_sum = block_view.sum(axis=(1, 3))

    br, bc = np.indices((lb, lb))
    h = _splitmix64_grid(config_index, br, bc, seed)
    selector = (h & np.uint64(3)).astype(np.intp)

    # Convert (block-row, intra-row, block-col, intra-col) to
    # (block-row, block-col, flat-intra-block-index) and select the same
    # physical input position for s and -s.  The selected spin therefore flips
    # exactly under the global Z2 transformation.
    blocks = block_view.transpose(0, 2, 1, 3).reshape(lb, lb, 4)
    tie_spin = np.take_along_axis(blocks, selector[..., None], axis=2)[..., 0]

    out = np.sign(block_sum).astype(np.int64)
    out = np.where(block_sum == 0, tie_spin, out)
    return out.astype(np.int8)
