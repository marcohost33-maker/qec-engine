"""Multi-L finite-size scaling of the 2D +/-J RBIM on the Nishimori line (Issue #43).

Research gate for an OWN numerical estimate of the Nishimori point ``p_c``. The
small single-L ``|m|`` scan of Inkr.4 was statistically underpowered (it moved
from ``p*~0.11`` to ``0.155`` once the RNG streams were separated correctly).
This module replaces "steepest |m| drop at one L" by the standard FSS toolkit:

* several lattice sizes ``L`` and a dense ``p`` grid around the Nishimori point,
* dimensionless observables whose curves CROSS at ``p_c`` (Binder ratio
  ``U4 = [<m^4>]/[<m^2>]^2`` and second-moment ``xi/L``),
* a scaling-collapse fit ``U(p, L) = f((p - p_c) L^{1/nu})`` for ``(p_c, nu)``,
* bootstrap over disorder realisations (cells are independent) for every CI.

Exact Nishimori-line oracles (independent of any fit, valid at EVERY finite L):

* **Internal energy** (Nishimori 1981): ``[<E>]/N = -2 tanh(beta_N) = -2 (1 - 2p)``
  on the square lattice (two bonds per spin). Checks bond convention + sampler.
* **Correlation identity** ``[<s_i s_j>] = [<s_i s_j>^2]``; summed over ``i, j`` it
  gives ``[<m^2>] = [<q^2>]`` with the two-replica overlap ``q``. It holds only
  in equilibrium; a hot start with too little burn-in violates it.

Equilibration is additionally checked by BRACKETING (:func:`equilibration_bracket`):
the same disorder realisations are run from a hot and from an aligned start and
the paired difference of ``[<m^2>]`` must vanish. On the Nishimori line the
aligned start is the gauge image of the *planted* configuration (Nishimori
gauge argument; "quiet planting", Krzakala & Zdeborova, PRL 102, 238701 (2009)):
the chain is then stationary for every gauge-invariant observable (e.g. the
energy is exact from t=0), and ``m^2`` relaxes from ABOVE as the equilibrium
auto-overlap decays, whereas the hot start coarsens from BELOW. Agreement of
both is a two-sided thermalisation certificate. Measured (L=16, p=0.11, 96
realisations): after 1000 sweeps aligned 0.631 vs hot 0.544 (not equilibrated;
the identity gate fires for the hot start, z=3.2), after >=3000 sweeps both
0.59 +- 0.02. The aligned start therefore is the production default.

Sampler: vectorised checkerboard Metropolis over a batch of disorder
realisations x 2 thermal replicas. The RBIM Wolff single-cluster update of
:mod:`rbim_nishimori` is deliberately NOT used here: at the Nishimori point
(``beta_N ~ 1.05``, ``P_add ~ 0.88``) the cluster covered 99.4% of the lattice
(measured), i.e. it degenerates to a global spin flip at the cost of an O(L)
BFS -- no decorrelation gain.

Random streams: bonds of replicate ``r`` come from
:func:`rbim_scan.derive_stream_seeds` (``policy="independent"``, the same bonds as
the seeded scan path); the batched thermal stream is a separate ``SeedSequence``
keyed by ``(base_seed, L, p, chunk, start)`` with its own tag, so no two cells and
no bond/thermal pair share a generator, while a hot/aligned bracket reuses the
BONDS (paired comparison) with different thermal streams.

Literature anchor (NOT a unit test for a small MC run): Hasenbusch, Parisen
Toldin, Pelissetto, Vicari, Phys. Rev. E 77, 051115 (2008): ``p_c = 0.10919(7)``
(their ``p* = 0.89081(7)`` counts +J bonds), ``y_1 = 0.655(15)`` ->
``nu = 1/y_1 = 1.53(4)``; Honecker, Picco, Pujol, PRL 87, 047201 (2001):
``p_c = 0.1094(2)``.

Claim ceiling: small L (Python/numpy) -> corrections to scaling are not
controlled; the result is "FSS-supported simulation, small L", never a frontier
value. The drift of pairwise crossings with L is reported instead of hidden.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize

from .autocorr import integrated_autocorr_time
from .rbim_nishimori import nishimori_beta, sample_bonds
from .rbim_scan import _float64_words, derive_stream_seeds

__all__ = [
    "LITERATURE_PC",
    "LITERATURE_NU",
    "CellSamples",
    "CellStats",
    "NishimoriCheck",
    "CollapseFit",
    "simulate_cell",
    "cell_stats",
    "nishimori_energy",
    "nishimori_checks",
    "equilibration_bracket",
    "equilibration_doubling",
    "crossing_point",
    "fit_collapse",
    "bootstrap_crossings",
    "bootstrap_collapse",
    "run_fss",
]

LITERATURE_PC: float = 0.10919
"""Hasenbusch et al. PRE 77, 051115 (2008): p_c = 0.10919(7) (= 1 - p*, p* = 0.89081)."""
LITERATURE_PC_ERR: float = 0.00007
LITERATURE_NU: float = 1.0 / 0.655
"""nu = 1/y_1 with y_1 = 0.655(15) (same reference) -> 1.53(4)."""
LITERATURE_NU_ERR: float = LITERATURE_NU * 0.015 / 0.655

_THERMAL_TAG = 0x46535354  # "FSST": separates the batched thermal stream from bond streams
_START_CODE = {"aligned": 0, "hot": 1}


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
def thermal_seed_sequence(
    base_seed: int, p: float, L: int, chunk: int, start: str = "aligned"
) -> np.random.SeedSequence:
    """Independent thermal stream of one (p, L, chunk, start) cell."""
    if not isinstance(base_seed, (int, np.integer)) or int(base_seed) < 0:
        raise ValueError(f"base_seed must be a non-negative integer, got {base_seed!r}")
    if not isinstance(chunk, (int, np.integer)) or int(chunk) < 0:
        raise ValueError(f"chunk must be a non-negative integer, got {chunk!r}")
    if start not in _START_CODE:
        raise ValueError(f"start must be 'aligned' or 'hot', got {start!r}")
    p_lo, p_hi = _float64_words(p)
    return np.random.SeedSequence(
        [int(base_seed), int(L), int(chunk), _THERMAL_TAG, _START_CODE[start], p_lo, p_hi]
    )


# ---------------------------------------------------------------------------
# Batched Metropolis sampler
# ---------------------------------------------------------------------------
def _metropolis_sweep_batch(
    s: np.ndarray,
    jx: np.ndarray,
    jy: np.ndarray,
    jx_left: np.ndarray,
    jy_up: np.ndarray,
    accept_table: np.ndarray,
    even: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One checkerboard Metropolis sweep for a batch ``s`` of shape (B, L, L).

    Axis 2 is x (``jx[i, j]`` couples (i, j)-(i, j+1)), axis 1 is y (``jy[i, j]``
    couples (i, j)-(i+1, j)) -- the :class:`rbim_nishimori.RBIMBonds` convention.
    With the integer local field ``h`` a flip changes ``log pi`` by ``-2 beta s h``;
    ``accept_table[s*h + 4] = min(1, exp(-2 beta s h))``. Sites of one colour have
    no common bond, so updating a colour in parallel is exact.
    """
    for mask in (even, ~even):
        h = (
            jx * np.roll(s, -1, axis=2)
            + jx_left * np.roll(s, 1, axis=2)
            + jy * np.roll(s, -1, axis=1)
            + jy_up * np.roll(s, 1, axis=1)
        )
        prob = accept_table[(s * h).astype(np.intp) + 4]
        flip = mask & (rng.random(s.shape, dtype=np.float32) < prob)
        s = np.where(flip, -s, s).astype(np.int8)
    return s


@dataclass(frozen=True)
class CellSamples:
    """Per-realisation thermal averages of one (p, L) cell (both replicas pooled).

    Every observable array has shape ``(n_disorder,)``; the bootstrap resamples rows.
    """

    p: float
    L: int
    beta: float
    n_disorder: int
    n_records: int
    burn_in: int
    n_skip: int
    start: str
    m2: np.ndarray
    m4: np.ndarray
    q2: np.ndarray
    q4: np.ndarray
    fk: np.ndarray
    """<|S(k_min)|^2>/N, averaged over the x and y minimal momenta."""
    energy: np.ndarray
    """<E>/N per realisation."""
    tau_int_records: float
    """tau_int (in records) of the disorder-mean m^2 series (first chunk)."""
    wall_seconds: float = field(default=0.0, compare=False)


def simulate_cell(
    p: float,
    L: int,
    *,
    n_disorder: int,
    n_records: int,
    burn_in: int,
    base_seed: int,
    n_skip: int = 1,
    start: str = "aligned",
    chunk_size: int = 256,
) -> CellSamples:
    """Simulate ``n_disorder`` realisations x 2 replicas at (p, beta_N(p), L).

    ``burn_in`` and ``n_skip`` count Metropolis sweeps. ``start`` is ``"aligned"``
    (planted-gauge start, production default) or ``"hot"`` (uniform random).
    """
    if not isinstance(L, (int, np.integer)) or L < 4 or L % 2:
        raise ValueError(f"L must be an even integer >= 4, got {L}")
    for name, v, lo in (
        ("n_disorder", n_disorder, 2),
        ("n_records", n_records, 1),
        ("n_skip", n_skip, 1),
        ("chunk_size", chunk_size, 1),
    ):
        if not isinstance(v, (int, np.integer)) or v < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError(f"burn_in must be an integer >= 0, got {burn_in!r}")
    if start not in _START_CODE:
        raise ValueError(f"start must be 'aligned' or 'hot', got {start!r}")
    beta = nishimori_beta(p)
    n = L * L
    t0 = time.perf_counter()
    accept_table = np.minimum(1.0, np.exp(-2.0 * beta * np.arange(-4, 5))).astype(np.float32)
    ii, jj = np.indices((L, L))
    even = ((ii + jj) % 2 == 0)[None, :, :]
    phase = np.exp(2j * np.pi * np.arange(L) / L)

    out = {k: np.empty(n_disorder) for k in ("m2", "m4", "q2", "q4", "fk", "energy")}
    tau_records = float("nan")
    for chunk, lo in enumerate(range(0, n_disorder, chunk_size)):
        hi = min(lo + chunk_size, n_disorder)
        R = hi - lo
        jx = np.empty((R, L, L), dtype=np.int8)
        jy = np.empty((R, L, L), dtype=np.int8)
        for k, rep in enumerate(range(lo, hi)):
            seeds = derive_stream_seeds(base_seed=base_seed, p=p, L=L, replicate=rep)
            b = sample_bonds(p, L, seed=seeds.bond_seed)
            jx[k], jy[k] = b.jx, b.jy
        jx2 = np.concatenate([jx, jx])  # replica a = rows [0, R), replica b = [R, 2R)
        jy2 = np.concatenate([jy, jy])
        jx_left = np.roll(jx2, 1, axis=2)
        jy_up = np.roll(jy2, 1, axis=1)
        rng = np.random.default_rng(thermal_seed_sequence(base_seed, p, L, chunk, start))
        if start == "aligned":
            s = np.ones((2 * R, L, L), dtype=np.int8)
        else:
            s = np.where(rng.random((2 * R, L, L)) < 0.5, np.int8(1), np.int8(-1))
        args = (jx2, jy2, jx_left, jy_up, accept_table, even, rng)

        for _ in range(burn_in):
            s = _metropolis_sweep_batch(s, *args)
        acc = {k: np.zeros(R) for k in out}
        trace = np.empty(n_records)
        for t in range(n_records):
            for _ in range(n_skip):
                s = _metropolis_sweep_batch(s, *args)
            sf = s.astype(np.float64)
            m = sf.sum(axis=(1, 2)) / n
            m_sq = m * m
            q = (sf[:R] * sf[R:]).sum(axis=(1, 2)) / n
            sx = sf.sum(axis=1) @ phase  # sum over y, Fourier transform in x
            sy = sf.sum(axis=2) @ phase
            fk = 0.5 * (np.abs(sx) ** 2 + np.abs(sy) ** 2) / n
            bond = jx2 * sf * np.roll(sf, -1, axis=2) + jy2 * sf * np.roll(sf, -1, axis=1)
            e = -bond.sum(axis=(1, 2)) / n
            m2_pair = 0.5 * (m_sq[:R] + m_sq[R:])
            acc["m2"] += m2_pair
            acc["m4"] += 0.5 * (m_sq[:R] ** 2 + m_sq[R:] ** 2)
            acc["q2"] += q * q
            acc["q4"] += q**4
            acc["fk"] += 0.5 * (fk[:R] + fk[R:])
            acc["energy"] += 0.5 * (e[:R] + e[R:])
            trace[t] = m2_pair.mean()
        for k in out:
            out[k][lo:hi] = acc[k] / n_records
        if chunk == 0 and n_records >= 8:
            tau_records = integrated_autocorr_time(trace).tau_int

    return CellSamples(
        p=float(p),
        L=int(L),
        beta=float(beta),
        n_disorder=int(n_disorder),
        n_records=int(n_records),
        burn_in=int(burn_in),
        n_skip=int(n_skip),
        start=start,
        tau_int_records=float(tau_records),
        wall_seconds=time.perf_counter() - t0,
        **out,
    )


# ---------------------------------------------------------------------------
# Cell statistics + exact Nishimori oracles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellStats:
    """Disorder-averaged dimensionless observables of one cell."""

    p: float
    L: int
    u4: float
    """Binder ratio [<m^4>]/[<m^2>]^2 (1 deep in FM, 3 for a Gaussian PM)."""
    xi_over_l: float
    """Second-moment xi/L from chi = N[<m^2>] and F = [<|S(k_min)|^2>]/N."""
    m2: float
    q2: float
    energy: float


def _u4(m2: np.ndarray, m4: np.ndarray) -> float:
    a = float(np.mean(m2))
    return float(np.mean(m4)) / (a * a) if a > 0 else float("nan")


def _xi_over_l(m2: np.ndarray, fk: np.ndarray, L: int) -> float:
    chi = float(np.mean(m2)) * L * L
    f = float(np.mean(fk))
    if f <= 0.0 or chi <= f:
        return float("nan")
    return math.sqrt(chi / f - 1.0) / (2.0 * math.sin(math.pi / L)) / L


def cell_stats(c: CellSamples, idx: np.ndarray | None = None) -> CellStats:
    """Observables of a cell, optionally on a bootstrap row selection ``idx``."""
    if idx is None:
        idx = np.arange(c.n_disorder)
    return CellStats(
        p=c.p,
        L=c.L,
        u4=_u4(c.m2[idx], c.m4[idx]),
        xi_over_l=_xi_over_l(c.m2[idx], c.fk[idx], c.L),
        m2=float(np.mean(c.m2[idx])),
        q2=float(np.mean(c.q2[idx])),
        energy=float(np.mean(c.energy[idx])),
    )


def nishimori_energy(p: float) -> float:
    """Exact disorder-averaged energy per spin on the Nishimori line (square lattice)."""
    return -2.0 * math.tanh(nishimori_beta(p))


def _z(x: np.ndarray, target: float = 0.0) -> float:
    """(mean - target) / SEM over realisations."""
    sem = float(np.std(x, ddof=1) / math.sqrt(x.size))
    mean = float(np.mean(x))
    if sem == 0.0:
        return 0.0 if mean == target else math.copysign(math.inf, mean - target)
    return (mean - target) / sem


@dataclass(frozen=True)
class NishimoriCheck:
    """Exact-oracle residuals of one cell, in units of their disorder SEM."""

    p: float
    L: int
    energy: float
    energy_exact: float
    energy_z: float
    m2_minus_q2: float
    identity_z: float
    """([<m^2>] - [<q^2>]) / SEM of the per-realisation difference (paired)."""

    def passes(self, z_max: float = 4.0) -> bool:
        return abs(self.energy_z) < z_max and abs(self.identity_z) < z_max


def nishimori_checks(c: CellSamples) -> NishimoriCheck:
    """Both exact Nishimori-line oracles for one cell."""
    e_exact = nishimori_energy(c.p)
    d = c.m2 - c.q2
    return NishimoriCheck(
        p=c.p,
        L=c.L,
        energy=float(np.mean(c.energy)),
        energy_exact=e_exact,
        energy_z=_z(c.energy, e_exact),
        m2_minus_q2=float(np.mean(d)),
        identity_z=_z(d),
    )


def equilibration_bracket(aligned: CellSamples, hot: CellSamples) -> dict[str, float]:
    """Paired hot-vs-aligned comparison on IDENTICAL disorder realisations.

    Both cells must share (p, L, n_disorder) and the bond streams (same
    ``base_seed``); the thermal streams differ. Returns the paired difference of
    the per-realisation ``<m^2>`` and its z-score; ``|z| < 3`` certifies that
    neither start still remembers its initial condition at the resolution of the
    disorder statistics.
    """
    if (aligned.p, aligned.L, aligned.n_disorder) != (hot.p, hot.L, hot.n_disorder):
        raise ValueError("bracket needs the same (p, L, n_disorder) for both starts")
    if aligned.start != "aligned" or hot.start != "hot":
        raise ValueError("first argument must be the aligned run, second the hot run")
    d = aligned.m2 - hot.m2
    return {
        "p": aligned.p,
        "L": aligned.L,
        "m2_aligned": float(np.mean(aligned.m2)),
        "m2_hot": float(np.mean(hot.m2)),
        "diff": float(np.mean(d)),
        "z": _z(d),
        "u4_aligned": _u4(aligned.m2, aligned.m4),
        "u4_hot": _u4(hot.m2, hot.m4),
    }


def equilibration_doubling(long: CellSamples, short: CellSamples) -> dict[str, float]:
    """Paired aligned-start comparison of a long vs a shorter burn-in (same bonds).

    From the aligned (planted-gauge) start ``[<m^2>](t)`` is an equilibrium
    auto-overlap and relaxes monotonically from above, so a burn-in that is long
    enough must agree with a fraction of itself. Measured limitation: at L>=16
    the difference between 4000 and 16000 sweeps (L=16) and between 10000 and
    40000 sweeps (L=24) was still systematic (U4 +0.02 per 4x), i.e. single-spin
    Metropolis could not certify equilibrium there within this budget.
    """
    if (long.p, long.L, long.n_disorder) != (short.p, short.L, short.n_disorder):
        raise ValueError("doubling needs the same (p, L, n_disorder) for both runs")
    if long.start != "aligned" or short.start != "aligned":
        raise ValueError("doubling compares two aligned-start runs")
    if long.burn_in <= short.burn_in:
        raise ValueError("first argument must have the longer burn-in")
    d = short.m2 - long.m2
    return {
        "p": long.p,
        "L": long.L,
        "burn_in_long": long.burn_in,
        "burn_in_short": short.burn_in,
        "m2_long": float(np.mean(long.m2)),
        "m2_short": float(np.mean(short.m2)),
        "diff_short_minus_long": float(np.mean(d)),
        "z": _z(d),
        "u4_long": _u4(long.m2, long.m4),
        "u4_short": _u4(short.m2, short.m4),
    }


# ---------------------------------------------------------------------------
# FSS estimators
# ---------------------------------------------------------------------------
def crossing_point(
    ps: Sequence[float],
    y_small: Sequence[float],
    y_large: Sequence[float],
    *,
    deg: int = 2,
    window: int = 5,
) -> float:
    """p where two curves (small L, large L) cross, from a LOCAL polynomial fit of their gap.

    The gap ``y_large - y_small`` is fitted by a degree-``deg`` polynomial on the
    ``window`` grid points closest to the linearly interpolated first sampled
    sign change (a global low-order fit over a wide window is biased by the
    curvature of the scaling function -- measured on planted data); the returned
    root is the in-window root closest to that sign change. ``nan`` if the gap
    does not change sign on the grid (no crossing inside the window -> no claim).
    """
    x = np.asarray(ps, dtype=np.float64)
    g = np.asarray(y_large, dtype=np.float64) - np.asarray(y_small, dtype=np.float64)
    if x.ndim != 1 or x.size != g.size or x.size < deg + 1:
        raise ValueError("need matching 1-D arrays with at least deg+1 points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("ps must be strictly increasing")
    if not np.all(np.isfinite(g)):
        return float("nan")
    sign_change = np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
    if sign_change.size == 0:
        return float("nan")
    k = int(sign_change[0])
    guess = x[k] - g[k] * (x[k + 1] - x[k]) / (g[k + 1] - g[k])
    sel = np.sort(np.argsort(np.abs(x - guess))[: max(window, deg + 1)])
    xw, gw = x[sel], g[sel]
    xm = float(xw.mean())
    xs = float(xw.std())
    coef = np.polyfit((xw - xm) / xs, gw, deg)
    roots = np.roots(coef)
    roots = roots[np.abs(roots.imag) < 1e-9].real * xs + xm
    roots = roots[(roots >= xw[0]) & (roots <= xw[-1])]
    if roots.size == 0:
        return float("nan")
    return float(roots[np.argmin(np.abs(roots - guess))])


@dataclass(frozen=True)
class CollapseFit:
    pc: float
    nu: float
    chi2_dof: float
    coef: tuple[float, ...]
    n_points: int
    success: bool


def fit_collapse(
    ps: np.ndarray,
    Ls: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    pc0: float,
    nu0: float,
    deg: int = 3,
) -> CollapseFit:
    """Weighted least-squares scaling collapse ``y = sum_k a_k (c (p-pc) L^{1/nu})^k``.

    Leading-order FSS (no correction-to-scaling term): with small L the fitted
    (pc, nu) are effective values; the crossing drift quantifies what is left.
    ``c`` is a fixed normalisation that keeps the polynomial well conditioned.
    """
    ps, Ls, y, sigma = (np.asarray(a, dtype=np.float64) for a in (ps, Ls, y, sigma))
    if not (ps.shape == Ls.shape == y.shape == sigma.shape) or ps.ndim != 1:
        raise ValueError("ps, Ls, y, sigma must be 1-D arrays of equal length")
    if not np.all(np.isfinite(np.concatenate([ps, Ls, y, sigma]))) or np.any(sigma <= 0):
        raise ValueError("need finite data and sigma > 0")
    n_par = deg + 3
    if ps.size <= n_par:
        raise ValueError(f"need more than {n_par} points for a degree-{deg} collapse")
    if np.unique(Ls).size < 2:
        raise ValueError("a collapse needs at least two lattice sizes")
    norm = 1.0 / (float(np.ptp(ps)) * float(np.median(Ls)) ** (1.0 / nu0))

    def xvar(pc: float, nu: float) -> np.ndarray:
        return (ps - pc) * Ls ** (1.0 / nu) * norm

    lo = [float(ps.min()), 0.3] + [-np.inf] * (deg + 1)
    hi = [float(ps.max()), 5.0] + [np.inf] * (deg + 1)
    pc_start = min(max(pc0, lo[0]), hi[0])
    a0 = np.polyfit(xvar(pc_start, nu0), y, deg, w=1.0 / sigma)[::-1]

    def resid(theta: np.ndarray) -> np.ndarray:
        return (np.polynomial.polynomial.polyval(xvar(theta[0], theta[1]), theta[2:]) - y) / sigma

    res = optimize.least_squares(
        resid, np.concatenate([[pc_start, nu0], a0]), bounds=(lo, hi), x_scale="jac"
    )
    dof = ps.size - n_par
    return CollapseFit(
        pc=float(res.x[0]),
        nu=float(res.x[1]),
        chi2_dof=float(np.sum(res.fun**2) / dof),
        coef=tuple(float(a) for a in res.x[2:]),
        n_points=int(ps.size),
        success=bool(res.success),
    )


def _grid(cells: Sequence[CellSamples]) -> tuple[list[int], list[float]]:
    Ls = sorted({c.L for c in cells})
    ps = sorted({c.p for c in cells})
    if len({(c.L, c.p) for c in cells}) != len(cells) or len(cells) != len(Ls) * len(ps):
        raise ValueError("cells must form a complete (L x p) grid without duplicates")
    return Ls, ps


def _stats_grid(
    cells: Sequence[CellSamples], rng: np.random.Generator | None
) -> dict[tuple[int, float], CellStats]:
    out = {}
    for c in cells:
        idx = None if rng is None else rng.integers(c.n_disorder, size=c.n_disorder)
        out[(c.L, c.p)] = cell_stats(c, idx)
    return out


_OBS: dict[str, Callable[[CellStats], float]] = {
    "u4": lambda s: s.u4,
    "xi_over_l": lambda s: s.xi_over_l,
}


def _crossings(
    stats: dict[tuple[int, float], CellStats], Ls: list[int], ps: list[float], obs: str
) -> dict[str, float]:
    get = _OBS[obs]
    out = {}
    for i, La in enumerate(Ls):
        for Lb in Ls[i + 1 :]:
            ya = [get(stats[(La, p)]) for p in ps]
            yb = [get(stats[(Lb, p)]) for p in ps]
            out[f"{La}-{Lb}"] = crossing_point(ps, ya, yb)
    return out


def _percentiles(b: np.ndarray) -> tuple[float, float, float]:
    ok = b[np.isfinite(b)]
    if ok.size < 2:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(ok, 2.5)),
        float(np.percentile(ok, 97.5)),
        float(np.std(ok, ddof=1)),
    )


def bootstrap_crossings(
    cells: Sequence[CellSamples], *, obs: str, n_boot: int, seed: int
) -> dict[str, dict[str, float]]:
    """Pairwise crossings with percentile CIs (resampling realisations per cell)."""
    if obs not in _OBS:
        raise ValueError(f"unknown observable {obs!r}; choose from {sorted(_OBS)}")
    Ls, ps = _grid(cells)
    central = _crossings(_stats_grid(cells, None), Ls, ps, obs)
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {k: [] for k in central}
    for _ in range(n_boot):
        for k, v in _crossings(_stats_grid(cells, rng), Ls, ps, obs).items():
            boots[k].append(v)
    out = {}
    for k, v in central.items():
        b = np.asarray(boots[k])
        lo, hi, se = _percentiles(b)
        out[k] = {
            "p_cross": v,
            "ci_lo": lo,
            "ci_hi": hi,
            "boot_se": se,
            "boot_fail_frac": float(np.mean(~np.isfinite(b))) if b.size else float("nan"),
        }
    return out


def _collapse_arrays(
    stats: dict[tuple[int, float], CellStats], Ls: list[int], ps: list[float], obs: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    get = _OBS[obs]
    pairs = [(L, p) for L in Ls for p in ps]
    return (
        np.array([p for _, p in pairs]),
        np.array([L for L, _ in pairs], dtype=np.float64),
        np.array([get(stats[k]) for k in pairs]),
    )


def bootstrap_collapse(
    cells: Sequence[CellSamples],
    *,
    obs: str,
    n_boot: int,
    seed: int,
    deg: int = 3,
    pc0: float = 0.11,
    nu0: float = 1.5,
) -> dict[str, float]:
    """Scaling-collapse (pc, nu) with bootstrap CIs; sigma per cell from the bootstrap."""
    if obs not in _OBS:
        raise ValueError(f"unknown observable {obs!r}; choose from {sorted(_OBS)}")
    if n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {n_boot}")
    Ls, ps = _grid(cells)
    rng = np.random.default_rng(seed)
    P, LL, Y = _collapse_arrays(_stats_grid(cells, None), Ls, ps, obs)
    Yb = np.array(
        [_collapse_arrays(_stats_grid(cells, rng), Ls, ps, obs)[2] for _ in range(n_boot)]
    )
    sigma = np.nanstd(Yb, axis=0, ddof=1)
    fit = fit_collapse(P, LL, Y, sigma, pc0=pc0, nu0=nu0, deg=deg)
    pcs, nus = [], []
    for yb in Yb:
        if np.all(np.isfinite(yb)):
            f = fit_collapse(P, LL, yb, sigma, pc0=fit.pc, nu0=fit.nu, deg=deg)
            if f.success:
                pcs.append(f.pc)
                nus.append(f.nu)
    pc_lo, pc_hi, pc_se = _percentiles(np.asarray(pcs))
    nu_lo, nu_hi, nu_se = _percentiles(np.asarray(nus))
    return {
        "pc": fit.pc,
        "pc_ci_lo": pc_lo,
        "pc_ci_hi": pc_hi,
        "pc_boot_se": pc_se,
        "nu": fit.nu,
        "nu_ci_lo": nu_lo,
        "nu_ci_hi": nu_hi,
        "nu_boot_se": nu_se,
        "chi2_dof": fit.chi2_dof,
        "n_points": fit.n_points,
        "poly_deg": deg,
        "n_boot_ok": len(pcs),
        "Ls": Ls,
    }


# ---------------------------------------------------------------------------
# Research run (evidence pack)
# ---------------------------------------------------------------------------
def _environment() -> dict[str, str]:
    import scipy

    from . import __version__
    from .manifest import _git_sha

    return {
        "adaptiverg_qec": __version__,
        "git_sha": _git_sha(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _simulate_job(job: tuple) -> CellSamples:
    p, L, kw = job
    return simulate_cell(p, L, **kw)


def run_fss(
    *,
    Ls: Sequence[int],
    ps: Sequence[float],
    n_disorder: int,
    n_records: int,
    burn_in: dict[int, int],
    base_seed: int,
    n_skip: int = 1,
    n_boot: int = 400,
    workers: int = 1,
    bracket_ps: Sequence[float] = (),
    verbose: bool = True,
) -> dict:
    """Full grid + exact oracles + bracket + crossings + collapse -> evidence-pack dict.

    ``burn_in`` maps L -> sweeps (critical slowing down: it must grow with L).
    ``bracket_ps`` adds, for the LARGEST-L cells at those p values, a hot-start
    twin (:func:`equilibration_bracket`) and an aligned twin with a quarter of the
    burn-in (:func:`equilibration_doubling`). Both reuse the production bonds.
    """
    missing = [L for L in Ls if L not in burn_in]
    if missing:
        raise ValueError(f"burn_in has no entry for L={missing}")
    L_max = max(Ls)
    jobs = []
    for L in Ls:
        for p in ps:
            kw = dict(
                n_disorder=n_disorder,
                n_records=n_records,
                burn_in=burn_in[L],
                base_seed=base_seed,
                n_skip=n_skip,
                start="aligned",
            )
            jobs.append((p, L, kw))
    for p in bracket_ps:
        if p not in ps:
            raise ValueError(f"bracket p={p} is not on the grid")
        kw = dict(jobs[0][2], burn_in=burn_in[L_max], start="hot")
        jobs.append((p, L_max, kw))
        jobs.append((p, L_max, dict(kw, burn_in=burn_in[L_max] // 4, start="aligned")))
    # Largest cells first so the pool is not left waiting on one long job.
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i][1])
    results: list[CellSamples | None] = [None] * len(jobs)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, c in zip(order, ex.map(_simulate_job, [jobs[i] for i in order]), strict=True):
                results[i] = c
                if verbose:
                    _report_cell(c)
    else:
        for i in order:
            results[i] = _simulate_job(jobs[i])
            if verbose:
                _report_cell(results[i])
    done = [c for c in results if c is not None]
    cells = [c for c in done if c.start == "aligned" and c.burn_in == burn_in[c.L]]
    hots = {(c.p, c.L): c for c in done if c.start == "hot"}
    shorts = {(c.p, c.L): c for c in done if c.start == "aligned" and c.burn_in != burn_in[c.L]}

    checks = [nishimori_checks(c) for c in cells]
    stats = _stats_grid(cells, None)
    Ls_sorted, ps_sorted = _grid(cells)
    rows = []
    for c, chk in zip(cells, checks, strict=True):
        s = stats[(c.L, c.p)]
        rows.append(
            {
                "L": c.L,
                "p": c.p,
                "beta": c.beta,
                "burn_in": c.burn_in,
                "u4": s.u4,
                "xi_over_l": s.xi_over_l,
                "m2": s.m2,
                "q2": s.q2,
                "energy": chk.energy,
                "energy_exact": chk.energy_exact,
                "energy_z": chk.energy_z,
                "identity_m2_minus_q2": chk.m2_minus_q2,
                "identity_z": chk.identity_z,
                "tau_int_records": c.tau_int_records,
                "wall_seconds": c.wall_seconds,
            }
        )
    aligned_by_key = {(c.p, c.L): c for c in cells}
    brackets = [equilibration_bracket(aligned_by_key[k], h) for k, h in sorted(hots.items())]
    doublings = [equilibration_doubling(aligned_by_key[k], c) for k, c in sorted(shorts.items())]
    crossings = {
        obs: bootstrap_crossings(cells, obs=obs, n_boot=n_boot, seed=base_seed + i + 1)
        for i, obs in enumerate(_OBS)
    }
    collapse = {}
    for i, obs in enumerate(_OBS):
        collapse[obs] = {
            "all_L": bootstrap_collapse(cells, obs=obs, n_boot=n_boot, seed=base_seed + 100 + i)
        }
        if len(Ls_sorted) >= 3:
            big = [c for c in cells if Ls_sorted[0] < c.L]
            collapse[obs]["drop_smallest_L"] = bootstrap_collapse(
                big, obs=obs, n_boot=n_boot, seed=base_seed + 200 + i
            )
    z_all = [max(abs(c.energy_z), abs(c.identity_z)) for c in checks]
    tau_ratio = [c.tau_int_records / c.n_records for c in cells if math.isfinite(c.tau_int_records)]
    return {
        "tool": "adaptiverg_qec.rbim_fss (Issue #43 research gate)",
        "claim_tier": "FSS-supported simulation, small L (no correction-to-scaling control)",
        "model": "2D +/-J RBIM, periodic, Nishimori line beta = 0.5 ln((1-p)/p)",
        "sampler": (
            "vectorised checkerboard Metropolis, aligned (planted-gauge) start, "
            "2 independent thermal replicas per disorder realisation"
        ),
        "rng_policy": (
            "bonds: rbim_scan.derive_stream_seeds(policy='independent') per replicate; "
            "thermal: SeedSequence([base_seed, L, chunk, 'FSST', start, p_words]) per cell"
        ),
        "parameters": {
            "Ls": list(Ls_sorted),
            "ps": list(ps_sorted),
            "n_disorder": n_disorder,
            "n_records": n_records,
            "burn_in": {str(k): v for k, v in sorted(burn_in.items())},
            "n_skip": n_skip,
            "base_seed": base_seed,
            "n_boot": n_boot,
            "bracket_ps": list(bracket_ps),
        },
        "environment": _environment(),
        "literature": {
            "pc": LITERATURE_PC,
            "pc_err": LITERATURE_PC_ERR,
            "nu": LITERATURE_NU,
            "nu_err": LITERATURE_NU_ERR,
            "refs": [
                "Hasenbusch, Parisen Toldin, Pelissetto, Vicari, PRE 77, 051115 (2008)",
                "Honecker, Picco, Pujol, PRL 87, 047201 (2001)",
                "Nishimori, Prog. Theor. Phys. 66, 1169 (1981) (exact NL identities)",
            ],
        },
        "nishimori_oracles": {
            "all_pass_4sigma": all(c.passes() for c in checks),
            "max_abs_z": max(z_all),
            "n_cells": len(checks),
            "max_tau_int_over_n_records": max(tau_ratio) if tau_ratio else float("nan"),
        },
        "equilibration_brackets": brackets,
        "equilibration_doubling": doublings,
        "cells": rows,
        "crossings": crossings,
        "collapse": collapse,
    }


def _report_cell(c: CellSamples) -> None:
    s = cell_stats(c)
    k = nishimori_checks(c)
    print(
        f"L={c.L:3d} p={c.p:.4f} {c.start:7s} U4={s.u4:.4f} xi/L={s.xi_over_l:.4f} "
        f"m2={s.m2:.4f} zE={k.energy_z:+.2f} zI={k.identity_z:+.2f} "
        f"tau={c.tau_int_records:.1f} ({c.wall_seconds:.0f}s)",
        flush=True,
    )


def _parse_burn_in(items: Sequence[str]) -> dict[int, int]:
    out = {}
    for it in items:
        L, _, b = it.partition(":")
        if not b:
            raise argparse.ArgumentTypeError(f"burn-in entry must be L:sweeps, got {it!r}")
        out[int(L)] = int(b)
    return out


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--Ls", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument(
        "--ps",
        type=float,
        nargs="+",
        default=[0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125],
    )
    ap.add_argument("--n-disorder", type=int, default=1000)
    ap.add_argument("--n-records", type=int, default=1000)
    ap.add_argument("--n-skip", type=int, default=2)
    ap.add_argument(
        "--burn-in",
        nargs="+",
        default=["6:2000", "8:4000", "10:8000", "12:16000"],
        help="L:sweeps pairs",
    )
    ap.add_argument("--bracket-ps", type=float, nargs="*", default=[0.095, 0.110])
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--json", type=Path, default=Path("results/rbim-nishimori-fss-mc.json"))
    raw = list(argv) if argv is not None else sys.argv[1:]
    a = ap.parse_args(raw)
    t0 = time.perf_counter()
    report = run_fss(
        Ls=a.Ls,
        ps=a.ps,
        n_disorder=a.n_disorder,
        n_records=a.n_records,
        burn_in=_parse_burn_in(a.burn_in),
        base_seed=a.seed,
        n_skip=a.n_skip,
        n_boot=a.n_boot,
        workers=a.workers,
        bracket_ps=a.bracket_ps,
    )
    report["wall_seconds_total"] = time.perf_counter() - t0
    report["command"] = "python -m adaptiverg_qec.rbim_fss " + " ".join(raw)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    keys = ("nishimori_oracles", "equilibration_brackets", "equilibration_doubling", "collapse")
    summary = {k: report[k] for k in keys}
    print(json.dumps(summary, indent=2))
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
