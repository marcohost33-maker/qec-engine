"""Reproducible RBIM scan orchestration with explicit random-stream policy.

The historical :mod:`rbim_nishimori` baseline used arithmetic seeds of the form
``base_seed + d`` at every p point.  Reusing those seeds can be a deliberate
common-random-numbers (CRN) design, but it must not happen implicitly because it
correlates scan points.

This module keeps the historical baseline untouched and provides a versionable
scan path with two explicit policies:

``independent`` (default)
    Mix ``p``, lattice size, replicate id and the root seed through NumPy
    ``SeedSequence``.  Different p points therefore receive independent (with
    very high probability) disorder and thermal streams.

``common_random_numbers``
    Deliberately omit ``p`` from the stream identity.  The same underlying
    streams are then reused across p values.  This is useful only when CRN is an
    intentional variance-reduction design and the induced covariance is handled
    in the analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .rbim_nishimori import (
    DisorderResult,
    nishimori_beta,
    rbim_wolff_sample,
    sample_bonds,
)

SeedPolicy = Literal["independent", "common_random_numbers"]


@dataclass(frozen=True)
class StreamSeeds:
    """Concrete child seeds used for one disorder replicate."""

    bond_seed: int
    thermal_seed: int
    policy: SeedPolicy
    p: float
    L: int
    replicate: int


def _float64_words(value: float) -> tuple[int, int]:
    """Encode a finite float64 exactly as two uint32 words for SeedSequence."""
    if not np.isfinite(value):
        raise ValueError(f"value must be finite, got {value}")
    bits = int(np.asarray(value, dtype=np.float64).view(np.uint64))
    return bits & 0xFFFFFFFF, bits >> 32


def derive_stream_seeds(
    *,
    base_seed: int,
    p: float,
    L: int,
    replicate: int,
    policy: SeedPolicy = "independent",
) -> StreamSeeds:
    """Derive reproducible disorder/thermal child streams for one scan cell.

    ``SeedSequence`` hashes the complete stream identity instead of relying on
    neighbouring integer seeds.  Child streams are created with ``spawn(2)`` so
    bond generation and thermal sampling never share a generator state.
    """
    if not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
        raise ValueError(f"base_seed must be a non-negative integer, got {base_seed}")
    if not np.isfinite(p) or not (0.0 < p < 0.5):
        raise ValueError(f"p must be in (0, 0.5), got {p}")
    if not isinstance(L, (int, np.integer)) or int(L) < 4:
        raise ValueError(f"L must be an integer >= 4, got {L}")
    if not isinstance(replicate, (int, np.integer)) or int(replicate) < 0:
        raise ValueError(f"replicate must be a non-negative integer, got {replicate}")
    if policy not in ("independent", "common_random_numbers"):
        raise ValueError(f"unknown seed policy: {policy}")

    p_lo, p_hi = _float64_words(p)
    policy_tag = 0x494E4450 if policy == "independent" else 0x43524E00
    entropy = [int(base_seed), int(L), int(replicate), policy_tag]
    if policy == "independent":
        entropy.extend((p_lo, p_hi))

    root = np.random.SeedSequence(entropy)
    bond_ss, thermal_ss = root.spawn(2)
    bond_seed = int(bond_ss.generate_state(1, dtype=np.uint64)[0])
    thermal_seed = int(thermal_ss.generate_state(1, dtype=np.uint64)[0])
    return StreamSeeds(
        bond_seed=bond_seed,
        thermal_seed=thermal_seed,
        policy=policy,
        p=float(p),
        L=int(L),
        replicate=int(replicate),
    )


def nishimori_scan_seeded(
    p: float,
    L: int,
    *,
    n_disorder: int,
    n_records: int,
    burn_in: int,
    base_seed: int,
    seed_policy: SeedPolicy = "independent",
    n_skip: int = 1,
    sweeps_per_step: int = 2,
    aligned_start: bool = True,
) -> DisorderResult:
    """Run one Nishimori scan point using an explicit random-stream policy."""
    if not isinstance(n_disorder, (int, np.integer)) or n_disorder < 1:
        raise ValueError(f"n_disorder must be an integer >= 1, got {n_disorder}")
    beta = nishimori_beta(p)
    n = L * L
    per_real_absm: list[float] = []
    per_real_m2: list[float] = []
    per_real_m4: list[float] = []
    cfracs: list[float] = []

    for replicate in range(n_disorder):
        seeds = derive_stream_seeds(
            base_seed=base_seed,
            p=p,
            L=L,
            replicate=replicate,
            policy=seed_policy,
        )
        bonds = sample_bonds(p, L, seed=seeds.bond_seed)
        configs, cf = rbim_wolff_sample(
            bonds,
            beta,
            n_records=n_records,
            burn_in=burn_in,
            seed=seeds.thermal_seed,
            n_skip=n_skip,
            sweeps_per_step=sweeps_per_step,
            aligned_start=aligned_start,
        )
        m = configs.reshape(configs.shape[0], -1).sum(axis=1) / n
        per_real_absm.append(float(np.mean(np.abs(m))))
        per_real_m2.append(float(np.mean(m * m)))
        per_real_m4.append(float(np.mean(m**4)))
        cfracs.append(cf)

    absm_arr = np.asarray(per_real_absm, dtype=np.float64)
    m2 = float(np.mean(per_real_m2))
    m4 = float(np.mean(per_real_m4))
    binder = 1.0 - m4 / (3.0 * m2 * m2) if m2 > 0.0 else float("nan")
    sem = float(np.std(absm_arr, ddof=1) / math.sqrt(len(absm_arr))) if len(absm_arr) > 1 else 0.0
    return DisorderResult(
        p=float(p),
        beta=float(beta),
        L=int(L),
        abs_m=float(np.mean(absm_arr)),
        abs_m_err=sem,
        m2=m2,
        m4=m4,
        binder=binder,
        mean_cluster_frac=float(np.mean(cfracs)),
        n_disorder=int(n_disorder),
        n_records=int(n_records),
    )


def nishimori_scan_grid(
    ps: tuple[float, ...] | list[float],
    L: int,
    **kwargs,
) -> list[DisorderResult]:
    """Run an ordered p-grid with the same explicit seed policy at every point."""
    ps_arr = np.asarray(ps, dtype=np.float64)
    if ps_arr.ndim != 1 or len(ps_arr) < 2:
        raise ValueError("ps must be a one-dimensional sequence with at least two points")
    if not np.all(np.isfinite(ps_arr)) or np.any((ps_arr <= 0.0) | (ps_arr >= 0.5)):
        raise ValueError("all p values must be finite and in (0, 0.5)")
    if np.any(np.diff(ps_arr) <= 0.0):
        raise ValueError("ps must be strictly increasing")
    return [nishimori_scan_seeded(float(p), L, **kwargs) for p in ps_arr]
