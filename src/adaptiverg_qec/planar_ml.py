"""Exact maximum-likelihood decoding of the planar surface code by transfer matrix (Issue #43).

Second, independent route to the Nishimori point -- the one the QEC side of the
repo actually cares about. For code-capacity bit-flip noise the probability of
an error coset is the partition function of the 2D +/-J RBIM on the Nishimori
line (Dennis, Kitaev, Landahl, Preskill, J. Math. Phys. 43, 4452 (2002)):

    P(E . <stabilisers>) = (1 - p)^n e^{-beta n} Z_beta(J),   J_e = (-1)^{E_e},
    p / (1 - p) = e^{-2 beta}.

The maximum-likelihood (ML) decoder compares the two cosets ``E`` and
``E . Xbar``; ``Z(J with the Xbar column negated) / Z(J)`` is the exponentiated
domain-wall free energy. Its failure probability ``P_fail(d, p)`` is therefore
a dimensionless disorder average whose curves for different distances cross at
the Nishimori point ``p_c`` -- with the crucial difference to Monte Carlo that
every thermal average here is EXACT. There is no equilibration problem (the
limitation that stopped :mod:`rbim_fss` at L=12), only disorder sampling.

Code (unrotated planar code, distance ``d``, ``n = d^2 + (d-1)^2`` qubits):

* spins (X-star generators) on a grid of ``R = d`` rows x ``C = d - 1`` columns,
* horizontal qubits ``h[r, c]``, ``c = 0..C``: ``c = 0`` joins spin (r, 0) to the
  LEFT rough boundary, ``c = C`` joins (r, C-1) to the RIGHT one, the others
  join (r, c-1)-(r, c); vertical qubits ``v[r, c]`` join (r, c)-(r+1, c),
* Z checks (plaquettes) ``P[r, c]``, ``r = 0..R-2``, ``c = 0..C``: ``h[r, c]``,
  ``h[r+1, c]``, ``v[r, c-1]`` (c >= 1), ``v[r, c]`` (c <= C-1),
* ``Xbar`` = column ``h[:, 0]`` (weight d, commutes with every plaquette),
* ``Zbar`` = row ``h[0, :]`` (weight d, commutes with every star, meets Xbar once).

Transfer matrix: spins are summed site by site in row-major order with a
frontier of ``C`` spins (state vector of size ``2^C``), vectorised over a batch
of disorder samples and both cosets; the vector is renormalised after every
row (log scale accumulated). Cost ``O(d^2 2^(d-1))`` per partition function,
so this exact decoder is practical for ``d <= 13`` -- the regime where
Bravyi, Suchara & Vargo (PRA 90, 032326 (2014)) switch to an approximate MPS
contraction for larger codes.

Paired baseline: PyMatching MWPM on the SAME parity-check matrix and the SAME
error samples (optional ``[surface]`` extra), so ML <= MWPM becomes a paired,
non-vacuous check. Expected thresholds: ML -> Nishimori ``p_c = 0.1094(2)``
(Honecker/Picco/Pujol 2001) / ``0.10919(7)`` (Hasenbusch et al. 2008); MWPM ->
zero-temperature RBIM ``~0.103``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rbim_fss import LITERATURE_NU, LITERATURE_PC, _environment, crossing_point, fit_collapse
from .rbim_scan import _float64_words

__all__ = [
    "PlanarCode",
    "planar_code",
    "log_partition",
    "ml_decode_fails",
    "mwpm_decode_fails",
    "exact_coset_log_probs",
    "simulate_decoders",
    "run_threshold",
]

_TAG = 0x504D4C44  # "PMLD"


@dataclass(frozen=True)
class PlanarCode:
    """Index bookkeeping of the distance-d unrotated planar code (X sector)."""

    d: int
    R: int
    C: int
    n: int
    h_index: np.ndarray
    """(R, C+1) qubit index of horizontal edges."""
    v_index: np.ndarray
    """(R-1, C) qubit index of vertical edges."""
    checks: np.ndarray
    """(n_checks, n) uint8 plaquette parity-check matrix (Z checks, detect X)."""
    stars: np.ndarray
    """(R*C, n) uint8 X-stabiliser generators (star of each spin)."""
    xbar: np.ndarray
    """(n,) uint8 logical X (column h[:, 0])."""
    zbar: np.ndarray
    """(n,) uint8 logical Z (row h[0, :]), used to score MWPM corrections."""


def planar_code(d: int) -> PlanarCode:
    """Build the distance-``d`` planar code (odd or even ``d >= 2``)."""
    if not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError(f"d must be an integer >= 2, got {d!r}")
    R, C = int(d), int(d) - 1
    nh = R * (C + 1)
    h_index = np.arange(nh).reshape(R, C + 1)
    v_index = nh + np.arange((R - 1) * C).reshape(R - 1, C)
    n = nh + (R - 1) * C
    checks = np.zeros(((R - 1) * (C + 1), n), dtype=np.uint8)
    for r in range(R - 1):
        for c in range(C + 1):
            row = checks[r * (C + 1) + c]
            row[h_index[r, c]] = row[h_index[r + 1, c]] = 1
            if c >= 1:
                row[v_index[r, c - 1]] = 1
            if c <= C - 1:
                row[v_index[r, c]] = 1
    stars = np.zeros((R * C, n), dtype=np.uint8)
    for r in range(R):
        for c in range(C):
            row = stars[r * C + c]
            row[h_index[r, c]] = row[h_index[r, c + 1]] = 1
            if r >= 1:
                row[v_index[r - 1, c]] = 1
            if r <= R - 2:
                row[v_index[r, c]] = 1
    xbar = np.zeros(n, dtype=np.uint8)
    xbar[h_index[:, 0]] = 1
    zbar = np.zeros(n, dtype=np.uint8)
    zbar[h_index[0, :]] = 1
    return PlanarCode(
        d=int(d),
        R=R,
        C=C,
        n=n,
        h_index=h_index,
        v_index=v_index,
        checks=checks,
        stars=stars,
        xbar=xbar,
        zbar=zbar,
    )


def log_partition(code: PlanarCode, errors: np.ndarray, beta: float) -> np.ndarray:
    """Exact ``log Z_beta(J)`` with ``J = (-1)^errors`` for a batch of error patterns.

    ``Z = sum_sigma exp(beta sum_e J_e sigma_u sigma_v)``, boundary edges couple to
    a fixed ``+1`` boundary spin. ``errors`` has shape (B, n) with entries in {0, 1}.
    """
    E = np.asarray(errors)
    if E.ndim != 2 or E.shape[1] != code.n:
        raise ValueError(f"errors must have shape (B, {code.n}), got {E.shape}")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be finite and > 0, got {beta}")
    R, C = code.R, code.C
    B = E.shape[0]
    J = 1 - 2 * E.astype(np.int8)
    Jh = J[:, code.h_index]  # (B, R, C+1)
    Jv = J[:, code.v_index]  # (B, R-1, C)
    ep, em = math.exp(beta), math.exp(-beta)
    idx = np.arange(1 << C)
    sig = 1 - 2 * ((idx[None, :] >> np.arange(C)[:, None]) & 1)  # (C, 2^C) in {+1,-1}
    pair = [None] + [sig[c - 1] * sig[c] for c in range(1, C)]
    psi = np.ones((B, 1 << C))
    log_scale = np.zeros(B)
    for r in range(R):
        for c in range(C):
            if r > 0:
                # replace frontier bit c (spin (r-1, c)) by spin (r, c) over bond v[r-1, c]
                a = np.where(Jv[:, r - 1, c] > 0, ep, em)[:, None, None]
                b = np.where(Jv[:, r - 1, c] > 0, em, ep)[:, None, None]
                x = psi.reshape(B, 1 << (C - 1 - c), 2, 1 << c)
                x0, x1 = x[:, :, 0, :], x[:, :, 1, :]
                psi = np.stack([a * x0 + b * x1, b * x0 + a * x1], axis=2).reshape(B, -1)
            jh = Jh[:, r, c][:, None]
            prod = sig[0][None, :] if c == 0 else pair[c][None, :]
            psi = psi * np.where(jh * prod > 0, ep, em)
        psi = psi * np.where(Jh[:, r, C][:, None] * sig[C - 1][None, :] > 0, ep, em)
        m = psi.max(axis=1)
        psi /= m[:, None]
        log_scale += np.log(m)
    return np.log(psi.sum(axis=1)) + log_scale


def ml_decode_fails(
    code: PlanarCode, errors: np.ndarray, p: float, *, rtol: float = 1e-10
) -> np.ndarray:
    """Per-sample ML failure indicator in {0, 0.5, 1} (0.5 = exact tie, coin flip).

    The ML decoder picks the more likely of the cosets ``E`` and ``E Xbar``
    (both consistent with the syndrome of ``E``); it fails when ``E Xbar`` is
    strictly more likely than the true coset.
    """
    if not (0.0 < p < 0.5):
        raise ValueError(f"p must be in (0, 0.5), got {p}")
    beta = 0.5 * math.log((1.0 - p) / p)
    E = np.asarray(errors, dtype=np.uint8)
    both = np.concatenate([E, E ^ code.xbar[None, :]])
    lz = log_partition(code, both, beta)
    lz_true, lz_flip = lz[: E.shape[0]], lz[E.shape[0] :]
    tol = rtol * np.maximum(np.abs(lz_true), 1.0)
    diff = lz_flip - lz_true
    return np.where(diff > tol, 1.0, np.where(diff < -tol, 0.0, 0.5))


def mwpm_decode_fails(code: PlanarCode, errors: np.ndarray) -> np.ndarray:
    """Per-sample MWPM (PyMatching) failure indicator on the SAME samples.

    Requires the optional ``[surface]`` extra. A residual ``E + correction`` has
    trivial syndrome; it is a logical error iff it anticommutes with ``Zbar``.
    """
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError('mwpm_decode_fails needs pip install ".[surface]"') from exc
    E = np.asarray(errors, dtype=np.uint8)
    matching = pymatching.Matching(code.checks)
    synd = (E.astype(np.int64) @ code.checks.T.astype(np.int64)) % 2
    corr = matching.decode_batch(synd.astype(np.uint8))
    resid = (E ^ corr.astype(np.uint8)).astype(np.int64)
    if np.any((resid @ code.checks.T.astype(np.int64)) % 2):
        raise RuntimeError("MWPM correction does not reproduce the syndrome")
    return ((resid @ code.zbar.astype(np.int64)) % 2).astype(np.float64)


def exact_coset_log_probs(code: PlanarCode, error: np.ndarray, p: float) -> tuple[float, float]:
    """Brute-force ``log P(E <S>)`` and ``log P(E Xbar <S>)`` by enumerating the stabiliser group.

    Independent oracle for small codes (``2^(R C)`` group elements): pure GF(2)
    combinatorics on qubits, no spin model and no transfer matrix involved.
    """
    k = code.stars.shape[0]
    if k > 20:
        raise ValueError(f"stabiliser group too large for enumeration (2^{k})")
    coeff = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(np.int64)
    group = (coeff @ code.stars.astype(np.int64)) % 2  # (2^k, n)
    E = np.asarray(error, dtype=np.int64)
    out = []
    for base in (E, E ^ code.xbar.astype(np.int64)):
        w = (group ^ base[None, :]).sum(axis=1)
        logw = w * math.log(p) + (code.n - w) * math.log1p(-p)
        mx = logw.max()
        out.append(float(mx + math.log(np.exp(logw - mx).sum())))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Sampling + threshold study
# ---------------------------------------------------------------------------
def _cell_rng(base_seed: int, d: int, p: float, chunk: int) -> np.random.Generator:
    p_lo, p_hi = _float64_words(p)
    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(d), int(chunk), _TAG, p_lo, p_hi])
    )


@dataclass(frozen=True)
class DecoderCell:
    d: int
    p: float
    shots: int
    ml_fail: float
    """Number of ML failures (ties count 1/2)."""
    ml_ties: int
    mwpm_fail: float
    """Number of MWPM failures (nan if MWPM was not run)."""
    both_fail: float
    """Samples on which both decoders fail (paired); nan without MWPM."""
    wall_seconds: float


def simulate_decoders(
    d: int,
    p: float,
    *,
    shots: int,
    base_seed: int,
    chunk: int = 1000,
    with_mwpm: bool = True,
) -> DecoderCell:
    """Sample iid bit-flip errors and decode each sample with ML (and MWPM)."""
    if not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError(f"shots must be an integer >= 1, got {shots!r}")
    if not (0.0 < p < 0.5):
        raise ValueError(f"p must be in (0, 0.5), got {p}")
    if not isinstance(base_seed, (int, np.integer)) or base_seed < 0:
        raise ValueError(f"base_seed must be a non-negative integer, got {base_seed!r}")
    code = planar_code(d)
    t0 = time.perf_counter()
    ml = mw = both = 0.0
    ties = 0
    for k, lo in enumerate(range(0, shots, chunk)):
        m = min(chunk, shots - lo)
        E = (_cell_rng(base_seed, d, p, k).random((m, code.n)) < p).astype(np.uint8)
        f_ml = ml_decode_fails(code, E, p)
        ml += float(f_ml.sum())
        ties += int(np.sum(f_ml == 0.5))
        if with_mwpm:
            f_mw = mwpm_decode_fails(code, E)
            mw += float(f_mw.sum())
            both += float(np.sum(np.minimum(f_ml, f_mw)))
    return DecoderCell(
        d=int(d),
        p=float(p),
        shots=int(shots),
        ml_fail=ml,
        ml_ties=ties,
        mwpm_fail=mw if with_mwpm else float("nan"),
        both_fail=both if with_mwpm else float("nan"),
        wall_seconds=time.perf_counter() - t0,
    )


def _jeffreys_se(k: float, n: int) -> float:
    pt = (k + 0.5) / (n + 1.0)
    return math.sqrt(pt * (1.0 - pt) / n)


def _crossings_from_rates(
    ds: list[int], ps: list[float], rate: dict[tuple[int, float], float]
) -> dict[str, float]:
    out = {}
    for i, da in enumerate(ds):
        for db in ds[i + 1 :]:
            out[f"{da}-{db}"] = crossing_point(
                ps, [rate[(da, p)] for p in ps], [rate[(db, p)] for p in ps]
            )
    return out


def threshold_analysis(
    cells: Sequence[DecoderCell], *, which: str, n_boot: int, seed: int, deg: int = 3
) -> dict:
    """Pairwise crossings + collapse (p_c, nu) with a parametric bootstrap over shots."""
    if which not in ("ml", "mwpm"):
        raise ValueError(f"which must be 'ml' or 'mwpm', got {which!r}")
    ds = sorted({c.d for c in cells})
    ps = sorted({c.p for c in cells})
    by = {(c.d, c.p): c for c in cells}
    if len(by) != len(ds) * len(ps):
        raise ValueError("cells must form a complete (d x p) grid")
    k_of = {key: (c.ml_fail if which == "ml" else c.mwpm_fail) for key, c in by.items()}
    rate = {key: k_of[key] / by[key].shots for key in by}
    central = _crossings_from_rates(ds, ps, rate)
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {k: [] for k in central}
    boot_rates = []
    for _ in range(n_boot):
        # Ties (weight 1/2) are rare; resampling the rounded-down count keeps the
        # binomial bootstrap exact for the tie-free part and conservative otherwise.
        br = {
            key: rng.binomial(by[key].shots, min(max(rate[key], 0.0), 1.0)) / by[key].shots
            for key in by
        }
        boot_rates.append(br)
        for k, v in _crossings_from_rates(ds, ps, br).items():
            boots[k].append(v)
    crossings = {}
    for k, v in central.items():
        b = np.asarray(boots[k])
        ok = b[np.isfinite(b)]
        crossings[k] = {
            "p_cross": v,
            "ci_lo": float(np.percentile(ok, 2.5)) if ok.size > 1 else float("nan"),
            "ci_hi": float(np.percentile(ok, 97.5)) if ok.size > 1 else float("nan"),
            "boot_se": float(np.std(ok, ddof=1)) if ok.size > 1 else float("nan"),
            "boot_fail_frac": float(np.mean(~np.isfinite(b))),
        }
    keys = [(d, p) for d in ds for p in ps]
    P = np.array([p for _, p in keys])
    D = np.array([d for d, _ in keys], dtype=np.float64)
    Y = np.array([rate[k] for k in keys])
    S = np.array([_jeffreys_se(k_of[k], by[k].shots) for k in keys])
    fit = fit_collapse(P, D, Y, S, pc0=float(np.mean(ps)), nu0=1.5, deg=deg)
    pcs, nus = [], []
    for br in boot_rates:
        f = fit_collapse(P, D, np.array([br[k] for k in keys]), S, pc0=fit.pc, nu0=fit.nu, deg=deg)
        if f.success:
            pcs.append(f.pc)
            nus.append(f.nu)
    return {
        "crossings": crossings,
        "collapse": {
            "pc": fit.pc,
            "pc_ci_lo": float(np.percentile(pcs, 2.5)),
            "pc_ci_hi": float(np.percentile(pcs, 97.5)),
            "pc_boot_se": float(np.std(pcs, ddof=1)),
            "nu": fit.nu,
            "nu_ci_lo": float(np.percentile(nus, 2.5)),
            "nu_ci_hi": float(np.percentile(nus, 97.5)),
            "nu_boot_se": float(np.std(nus, ddof=1)),
            "chi2_dof": fit.chi2_dof,
            "n_points": fit.n_points,
            "poly_deg": deg,
            "ds": ds,
        },
    }


def _job(args: tuple) -> DecoderCell:
    d, p, kw = args
    return simulate_decoders(d, p, **kw)


def run_threshold(
    *,
    ds: Sequence[int],
    ps: Sequence[float],
    shots: dict[int, int],
    base_seed: int,
    n_boot: int = 400,
    workers: int = 1,
    with_mwpm: bool = True,
    verbose: bool = True,
) -> dict:
    """ML (+ paired MWPM) decoding over a (d x p) grid -> evidence-pack dict."""
    missing = [d for d in ds if d not in shots]
    if missing:
        raise ValueError(f"shots has no entry for d={missing}")
    jobs = [
        (d, p, dict(shots=shots[d], base_seed=base_seed, with_mwpm=with_mwpm))
        for d in ds
        for p in ps
    ]
    jobs.sort(key=lambda j: -(j[2]["shots"] * j[0] ** 2 * 2 ** (j[0] - 1)))
    cells: list[DecoderCell] = []
    runner = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        it = runner.map(_job, jobs) if runner else map(_job, jobs)
        for c in it:
            cells.append(c)
            if verbose:
                print(
                    f"d={c.d:2d} p={c.p:.4f} ML={c.ml_fail / c.shots:.5f} "
                    f"MWPM={c.mwpm_fail / c.shots:.5f} ({c.wall_seconds:.0f}s)",
                    flush=True,
                )
    finally:
        if runner:
            runner.shutdown()
    cells.sort(key=lambda c: (c.d, c.p))
    rows = []
    for c in cells:
        n = c.shots
        row = {
            "d": c.d,
            "p": c.p,
            "shots": n,
            "ml_fail": c.ml_fail,
            "ml_rate": c.ml_fail / n,
            "ml_se": _jeffreys_se(c.ml_fail, n),
            "ml_ties": c.ml_ties,
            "wall_seconds": c.wall_seconds,
        }
        if with_mwpm:
            # Paired difference MWPM - ML per sample: values in {-1, -1/2, 0, 1/2, 1}.
            # Its variance is bounded by the discordant fraction; use that bound.
            discord = (c.ml_fail - c.both_fail) + (c.mwpm_fail - c.both_fail)
            diff = (c.mwpm_fail - c.ml_fail) / n
            se = math.sqrt(max(discord / n - diff**2, 0.0) / n) if n > 1 else float("nan")
            row.update(
                {
                    "mwpm_fail": c.mwpm_fail,
                    "mwpm_rate": c.mwpm_fail / n,
                    "mwpm_se": _jeffreys_se(c.mwpm_fail, n),
                    "paired_mwpm_minus_ml": diff,
                    "paired_se": se,
                    "paired_z": diff / se if se > 0 else float("inf"),
                }
            )
        rows.append(row)
    report = {
        "tool": "adaptiverg_qec.planar_ml (Issue #43: exact ML decoder <-> Nishimori point)",
        "claim_tier": (
            "exact per-sample thermal averages (transfer matrix); FSS over small d, "
            "no correction-to-scaling control"
        ),
        "model": (
            "unrotated planar surface code, code-capacity iid bit flips p; ML = compare "
            "Z_RBIM(E) vs Z_RBIM(E Xbar) at beta_N; MWPM = PyMatching on the same checks"
        ),
        "rng_policy": "SeedSequence([base_seed, d, chunk, 'PMLD', p_words]) per cell chunk",
        "parameters": {
            "ds": sorted(set(ds)),
            "ps": sorted(set(ps)),
            "shots": {str(k): v for k, v in sorted(shots.items())},
            "base_seed": base_seed,
            "n_boot": n_boot,
        },
        "environment": _environment(),
        "literature": {
            "ml_pc_nishimori": LITERATURE_PC,
            "ml_pc_honecker": 0.1094,
            "nu_nishimori": LITERATURE_NU,
            "mwpm_pc": 0.103,
            "refs": [
                "Dennis, Kitaev, Landahl, Preskill, J. Math. Phys. 43, 4452 (2002)",
                "Honecker, Picco, Pujol, PRL 87, 047201 (2001)",
                "Hasenbusch, Parisen Toldin, Pelissetto, Vicari, PRE 77, 051115 (2008)",
                "Bravyi, Suchara, Vargo, PRA 90, 032326 (2014)",
            ],
        },
        "cells": rows,
        "ml": threshold_analysis(cells, which="ml", n_boot=n_boot, seed=base_seed + 1),
    }
    if with_mwpm:
        report["mwpm"] = threshold_analysis(cells, which="mwpm", n_boot=n_boot, seed=base_seed + 2)
    return report


def _parse_map(items: Sequence[str]) -> dict[int, int]:
    out = {}
    for it in items:
        k, _, v = it.partition(":")
        if not v:
            raise argparse.ArgumentTypeError(f"entry must be d:shots, got {it!r}")
        out[int(k)] = int(v)
    return out


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    ap.add_argument(
        "--ps", type=float, nargs="+", default=[0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125]
    )
    ap.add_argument(
        "--shots",
        nargs="+",
        default=["5:100000", "7:100000", "9:60000", "11:40000", "13:20000"],
        help="d:shots pairs",
    )
    ap.add_argument("--no-mwpm", action="store_true")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--json", type=Path, default=Path("results/planar-ml-threshold.json"))
    raw = list(argv) if argv is not None else sys.argv[1:]
    a = ap.parse_args(raw)
    t0 = time.perf_counter()
    report = run_threshold(
        ds=a.ds,
        ps=a.ps,
        shots=_parse_map(a.shots),
        base_seed=a.seed,
        n_boot=a.n_boot,
        workers=a.workers,
        with_mwpm=not a.no_mwpm,
    )
    report["wall_seconds_total"] = time.perf_counter() - t0
    report["command"] = "python -m adaptiverg_qec.planar_ml " + " ".join(raw)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k in ("ml", "mwpm")}, indent=2))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
