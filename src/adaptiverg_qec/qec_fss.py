"""Inkrement 3.4 (Statistik-Kern): Binomial-Intervalle, exakte Coverage, Bootstrap-FSS.

Reines numpy/scipy -- KEIN [surface]-Extra noetig. Der Stim-Sweep in ``qec_sweep.py``
liefert nur Zellzaehler (k, n) je (d, p); alles Inferentielle passiert hier und ist gegen
geschlossene Orakel getestet.

1. INTERVALLE (Brown, Cai & DasGupta 2001, Statist. Sci. 16(2):101-133,
   doi:10.1214/ss/1009213286):
   - Wilson-Score: Zentrum (k + z^2/2)/(n + z^2), Halbbreite z/(n+z^2)*sqrt(k(n-k)/n + z^2/4).
   - Jeffreys: Beta(k+1/2, n-k+1/2)-Quantile, Randkonvention lo=0 bei k=0, hi=1 bei k=n.
   - Clopper-Pearson (aus ``qec_decoder_ab``), Wald nur als NEGATIV-Kontrolle.
   ``exact_coverage`` summiert die Binomial-pmf ueber alle k -- keine Monte-Carlo-Schaetzung.
   Damit ist der Wald-Zusammenbruch bei seltenen Fehlern (k=0 -> Breite 0) exakt belegbar.

2. FINITE-SIZE-SCALING (Wang, Harrington & Preskill 2003, Ann. Phys. 303:31,
   doi:10.1016/S0003-4916(02)00019-2): nahe der Schwelle
       p_L(d, p) ~ A + B x + C x^2,   x = (p - p_th) d^{1/nu}.
   Gefittet per gewichteter kleinster Quadrate (Gewicht 1/sigma_i mit Jeffreys-geglaettetem
   p~ = (k+1/2)/(n+1), damit k=0-Zellen nicht unendliches Gewicht bekommen). A, B, C gehen
   LINEAR ein und werden fuer jedes (p_th, nu) exakt eliminiert (Variable Projection,
   Golub & Pereyra 1973); nichtlinear optimiert werden nur (p_th, nu), mit Multi-Start.
   Guete: chi^2/dof und chi^2-p-Wert -- ein Fit, der die Daten nicht traegt, wird
   ausgewiesen statt still verwendet.

3. UNSICHERHEIT: parametrischer Bootstrap ueber ZELLEN, k*_i ~ Binomial(n_i, k_i/n_i).
   Gueltig, weil jede Zelle eine eigene RNG-Identitaet hat (Zellen unabhaengig, s.
   ``qec_sweep``). Perzentil-CIs fuer p_th und nu, dazu paarweise Kurven-Kreuzungen je
   benachbartem Distanzpaar auf DENSELBEN Resamples (Finite-Size-Drift sichtbar).

4. ORAKEL (``repetition_fss_dataset``): Repetition-Code unter Code-Capacity-Bitflip mit
   Majority-Vote. Geschlossen gilt p_L(d, 1/2) = 1/2 fuer jedes ungerade d, also kreuzen
   sich ALLE Kurven exakt bei p_th = 1/2 (keine Finite-Size-Drift der Schwelle). Die
   Steigung bei p=1/2 ist exakt d*C(d-1,(d-1)/2)/2^(d-1) ~ sqrt(2d/pi); daraus folgt ein
   EXAKTES effektives nu ueber die deklarierten Distanzen (``repetition_nu_oracle``),
   asymptotisch nu = 2 (CLT). Beides ist unabhaengig vom Fit hergeleitet.

CLAIM CEILING: Statistik-Werkzeug. Ein FSS-Fit auf endlichen Distanzen ist ein
Schaetzer mit ausgewiesener Unsicherheit, kein L->oo-Beweis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

from .qec_decoder_ab import _alpha, _count, clopper_pearson
from .qec_diagnostics import logical_error_rate_exact

__all__ = [
    "Cell",
    "wilson_interval",
    "jeffreys_interval",
    "wald_interval",
    "exact_coverage",
    "fss_fit",
    "fss_bootstrap",
    "pairwise_crossings",
    "fss_systematics",
    "drifted_repetition_dataset",
    "repetition_fss_dataset",
    "repetition_nu_oracle",
]


# =============================================================================
# 1. Binomial-Intervalle + exakte Coverage
# =============================================================================


def _check_kn(k, n) -> tuple[int, int]:
    k = _count("k", k)
    n = _count("n", n)
    if n < 1 or k > n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    return k, n


def _wilson_arrays(k: np.ndarray, n: int, z: float) -> tuple[np.ndarray, np.ndarray]:
    k = k.astype(float)
    denom = n + z * z
    centre = (k + z * z / 2.0) / denom
    half = z / denom * np.sqrt(k * (n - k) / n + z * z / 4.0)
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def _jeffreys_arrays(k: np.ndarray, n: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    kf = k.astype(float)
    lo = stats.beta.ppf(a / 2.0, kf + 0.5, n - kf + 0.5)
    hi = stats.beta.ppf(1.0 - a / 2.0, kf + 0.5, n - kf + 0.5)
    lo = np.where(k == 0, 0.0, lo)
    hi = np.where(k == n, 1.0, hi)
    return lo, hi


def _wald_arrays(k: np.ndarray, n: int, z: float) -> tuple[np.ndarray, np.ndarray]:
    ph = k.astype(float) / n
    half = z * np.sqrt(ph * (1.0 - ph) / n)
    return np.clip(ph - half, 0.0, 1.0), np.clip(ph + half, 0.0, 1.0)


def _cp_arrays(k: np.ndarray, n: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    kf = k.astype(float)
    with np.errstate(invalid="ignore"):
        lo = stats.beta.ppf(a / 2.0, kf, n - kf + 1.0)
        hi = stats.beta.ppf(1.0 - a / 2.0, kf + 1.0, n - kf)
    lo = np.where(k == 0, 0.0, lo)
    hi = np.where(k == n, 1.0, hi)
    return lo, hi


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson-Score-Intervall (Brown/Cai/DasGupta 2001, Gl. 4)."""
    k, n = _check_kn(k, n)
    z = float(stats.norm.ppf(1.0 - _alpha(alpha) / 2.0))
    lo, hi = _wilson_arrays(np.array([k]), n, z)
    return float(lo[0]), float(hi[0])


def jeffreys_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Equal-tailed Jeffreys-Intervall mit der Randkonvention von Brown et al. 2001."""
    k, n = _check_kn(k, n)
    lo, hi = _jeffreys_arrays(np.array([k]), n, _alpha(alpha))
    return float(lo[0]), float(hi[0])


def wald_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wald-Intervall -- NUR als Negativkontrolle (k=0 -> Breite 0)."""
    k, n = _check_kn(k, n)
    z = float(stats.norm.ppf(1.0 - _alpha(alpha) / 2.0))
    lo, hi = _wald_arrays(np.array([k]), n, z)
    return float(lo[0]), float(hi[0])


_METHODS = ("wilson", "jeffreys", "clopper_pearson", "wald")


def exact_coverage(method: str, n: int, p: float, alpha: float = 0.05) -> float:
    """EXAKTE Coverage P_p[lo(K) <= p <= hi(K)], K ~ Binomial(n, p) (pmf-Summe ueber k)."""
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; known: {_METHODS}")
    n = _count("n", n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.floating)):
        raise TypeError("p must be a real number")
    p = float(p)
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = _alpha(alpha)
    z = float(stats.norm.ppf(1.0 - a / 2.0))
    k = np.arange(n + 1)
    if method == "wilson":
        lo, hi = _wilson_arrays(k, n, z)
    elif method == "jeffreys":
        lo, hi = _jeffreys_arrays(k, n, a)
    elif method == "clopper_pearson":
        lo, hi = _cp_arrays(k, n, a)
    else:
        lo, hi = _wald_arrays(k, n, z)
    pmf = stats.binom.pmf(k, n, p)
    return float(np.sum(pmf[(lo <= p) & (p <= hi)]))


def intervals(k: int, n: int, alpha: float = 0.05) -> dict:
    """Alle drei gueltigen Intervalle einer Zelle (Wald bewusst nicht)."""
    return {
        "wilson": list(wilson_interval(k, n, alpha)),
        "jeffreys": list(jeffreys_interval(k, n, alpha)),
        "clopper_pearson": list(clopper_pearson(k, n, alpha)),
    }


# =============================================================================
# 2. FSS-Fit (Variable Projection ueber A, B, C)
# =============================================================================


@dataclass(frozen=True)
class Cell:
    """Eine Sweep-Zelle: Distanz d, physikalische Rate p, k Fehler aus n Shots."""

    d: int
    p: float
    k: int
    n: int

    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, (int, np.integer)) or self.d < 1:
            raise ValueError(f"d must be an int >= 1, got {self.d!r}")
        if not (isinstance(self.p, (int, float, np.floating)) and 0.0 < float(self.p) < 1.0):
            raise ValueError(f"p must be in (0, 1), got {self.p!r}")
        _check_kn(self.k, self.n)
        object.__setattr__(self, "d", int(self.d))
        object.__setattr__(self, "p", float(self.p))
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "n", int(self.n))


def _arrays(cells: Sequence[Cell]):
    if len(cells) < 6:
        raise ValueError("FSS fit needs at least 6 cells (5 parameters + 1 dof)")
    if not all(isinstance(c, Cell) for c in cells):
        raise TypeError("cells must be Cell instances")
    d = np.array([c.d for c in cells], dtype=float)
    p = np.array([c.p for c in cells], dtype=float)
    k = np.array([c.k for c in cells], dtype=float)
    n = np.array([c.n for c in cells], dtype=float)
    if len(set(d.tolist())) < 2:
        raise ValueError("FSS fit needs at least two distinct distances")
    if len(set(p.tolist())) < 3:
        raise ValueError("FSS fit needs at least three distinct p values")
    return d, p, k, n


def _sigma(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    pt = (k + 0.5) / (n + 1.0)
    return np.sqrt(pt * (1.0 - pt) / n)


def _design(d, p, p_th, nu):
    x = (p - p_th) * d ** (1.0 / nu)
    return np.column_stack([np.ones_like(x), x, x * x])


def _project(d, p, y, w, theta):
    """Gewichtete LS fuer (A,B,C) bei festem (p_th, nu); gibt Residuen + Koeffizienten."""
    X = _design(d, p, theta[0], theta[1])
    Xw = X * w[:, None]
    coef, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
    return (y - X @ coef) * w, coef


def _fit_core(d, p, y, w, starts, bounds):
    best = None
    for s in starts:
        try:
            res = optimize.least_squares(
                lambda th: _project(d, p, y, w, th)[0],
                x0=np.asarray(s, dtype=float),
                bounds=bounds,
                method="trf",
                x_scale=[max(bounds[1][0] - bounds[0][0], 1e-12) / 10.0, 0.5],
            )
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - numerisch
            continue
        if best is None or res.cost < best.cost:
            best = res
    return best


def _bounds_and_starts(p: np.ndarray):
    p_min, p_max = float(p.min()), float(p.max())
    span = p_max - p_min
    lo = max(p_min - 0.5 * span, 1e-12)
    hi = min(p_max + 0.5 * span, 1.0 - 1e-12)
    bounds = ([lo, 0.2], [hi, 10.0])
    starts = [(p_min + f * span, nu) for f in (0.25, 0.5, 0.75) for nu in (0.8, 1.5, 3.0)]
    return bounds, starts


def fss_fit(cells: Sequence[Cell], *, starts=None) -> dict:
    """Gewichteter Fit p_L = A + Bx + Cx^2, x=(p-p_th)d^{1/nu}; Guete via chi^2."""
    d, p, k, n = _arrays(cells)
    y = k / n
    w = 1.0 / _sigma(k, n)
    bounds, default_starts = _bounds_and_starts(p)
    res = _fit_core(d, p, y, w, starts if starts is not None else default_starts, bounds)
    if res is None:  # pragma: no cover - numerisch
        raise RuntimeError("FSS fit failed for every start")
    resid, coef = _project(d, p, y, w, res.x)
    chi2 = float(np.sum(resid**2))
    dof = len(y) - 5
    p_th, nu = float(res.x[0]), float(res.x[1])
    at_bound = bool(
        np.isclose(p_th, bounds[0][0])
        or np.isclose(p_th, bounds[1][0])
        or np.isclose(nu, bounds[0][1])
        or np.isclose(nu, bounds[1][1])
    )
    return {
        "p_th": p_th,
        "nu": nu,
        "A": float(coef[0]),
        "B": float(coef[1]),
        "C": float(coef[2]),
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "chi2_p_value": float(stats.chi2.sf(chi2, dof)),
        "p_th_in_window": bool(p.min() < p_th < p.max()),
        "at_parameter_bound": at_bound,
        "model": "p_L = A + B x + C x^2, x = (p - p_th) d^(1/nu)  [Wang/Harrington/Preskill 2003]",
        "weights": "1/sigma, sigma^2 = p~(1-p~)/n, p~ = (k+1/2)/(n+1)",
    }


def pairwise_crossings(cells: Sequence[Cell]) -> list[dict]:
    """Kreuzung je benachbartem Distanzpaar: erster Vorzeichenwechsel (- -> +) von
    p_L(d_b) - p_L(d_a) auf dem p-Gitter, linear interpoliert. None = keine Kreuzung."""
    table: dict[int, dict[float, float]] = {}
    for c in cells:
        table.setdefault(c.d, {})[c.p] = c.k / c.n
    ds = sorted(table)
    out = []
    for da, db in zip(ds, ds[1:], strict=False):
        grid = sorted(set(table[da]) & set(table[db]))
        diff = [table[db][q] - table[da][q] for q in grid]
        crossing = None
        for i in range(len(grid) - 1):
            if diff[i] < 0.0 <= diff[i + 1]:
                t = -diff[i] / (diff[i + 1] - diff[i])
                crossing = grid[i] + t * (grid[i + 1] - grid[i])
                break
        out.append({"d_pair": [da, db], "crossing": crossing})
    return out


def _percentile_ci(v: np.ndarray, alpha: float) -> list[float] | None:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.quantile(v, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(lo), float(hi)]


def fss_bootstrap(
    cells: Sequence[Cell],
    *,
    n_boot: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Parametrischer Zell-Bootstrap fuer p_th, nu und die paarweisen Kreuzungen."""
    n_boot = _count("n_boot", n_boot)
    if n_boot < 50:
        raise ValueError(f"n_boot must be >= 50, got {n_boot}")
    seed = _count("seed", seed)
    a = _alpha(alpha)
    point = fss_fit(cells)
    cross0 = pairwise_crossings(cells)
    d, p, k, n = _arrays(cells)
    bounds, _ = _bounds_and_starts(p)
    rng = np.random.Generator(np.random.Philox(seed))
    ph = k / n
    p_th_s = np.full(n_boot, np.nan)
    nu_s = np.full(n_boot, np.nan)
    cross_s = np.full((n_boot, len(cross0)), np.nan)
    warm = [(point["p_th"], point["nu"])]
    for b in range(n_boot):
        kb = rng.binomial(n.astype(np.int64), ph).astype(float)
        yb = kb / n
        wb = 1.0 / _sigma(kb, n)
        res = _fit_core(d, p, yb, wb, warm, bounds)
        if res is not None:
            p_th_s[b], nu_s[b] = res.x
        cb = pairwise_crossings(
            [Cell(c.d, c.p, int(kk), c.n) for c, kk in zip(cells, kb, strict=True)]
        )
        for j, row in enumerate(cb):
            if row["crossing"] is not None:
                cross_s[b, j] = row["crossing"]
    crossings = []
    for j, row in enumerate(cross0):
        col = cross_s[:, j]
        crossings.append(
            {
                **row,
                "ci": _percentile_ci(col, a),
                "found_fraction": float(np.mean(np.isfinite(col))),
            }
        )
    in_window = (p_th_s > p.min()) & (p_th_s < p.max())
    return {
        "point": point,
        "p_th_ci": _percentile_ci(p_th_s, a),
        "nu_ci": _percentile_ci(nu_s, a),
        "p_th_boot_sd": float(np.nanstd(p_th_s, ddof=1)),
        "fit_success_fraction": float(np.mean(np.isfinite(p_th_s))),
        "p_th_in_window_fraction": float(np.mean(in_window)),
        "crossings": crossings,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": a,
        "method": (
            "parametric bootstrap over independent cells, k* ~ Binomial(n, k/n); "
            "percentile CIs; refit warm-started at the point estimate"
        ),
    }


_VARIANT_KEYS = (
    "p_th",
    "nu",
    "chi2_per_dof",
    "chi2_p_value",
    "p_th_in_window",
    "at_parameter_bound",
)


def _variant_fit(cells: Sequence[Cell]) -> dict | None:
    try:
        fit = fss_fit(cells)
    except ValueError:
        return None
    return {k: fit[k] for k in _VARIANT_KEYS}


def _variant_selectors(cells: Sequence[Cell]) -> dict[str, np.ndarray]:
    ds = sorted({c.d for c in cells})
    ps = sorted({c.p for c in cells})
    d = np.array([c.d for c in cells])
    p = np.array([c.p for c in cells])
    sel: dict[str, np.ndarray] = {}
    if len(ds) >= 3:
        sel["drop_smallest_d"] = d != ds[0]
        sel["drop_largest_d"] = d != ds[-1]
    if len(ps) >= 5:
        sel["inner_window"] = (p > ps[0]) & (p < ps[-1])
    return sel


def fss_systematics(
    cells: Sequence[Cell],
    *,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
    stat_ci: Sequence[float] | None = None,
) -> dict:
    """Systematik, die der parametrische Bootstrap NICHT sieht (Finite-Size, Fenster, Auswahl).

    Varianten (jeweils voller Refit auf einer Teilmenge der Zellen):
      * ``drop_smallest_d`` / ``drop_largest_d``: Distanz-Bins. Verschiebt sich p_th, wenn die
        kleinste Distanz wegfaellt, ist der Ansatz dort noch nicht asymptotisch (Stephens 2014,
        arXiv:1311.5003: verlaesslich ab d ~ 7; Distanz-Bins wie arXiv:2401.04008).
      * ``inner_window``: aeusserste p-Punkte weg (der Ansatz gilt nur nahe p_th).

    Eine Teilmenge hat weniger Daten, ihr p_th streut also auch OHNE jede Drift. Deshalb wird
    jeder Shift Delta = p_th(Variante) - p_th(primaer) GEPAART gebootstrapt (beide Fits auf
    denselben parametrischen Resamples k* ~ Bin(n, k/n)); nur ein Shift, dessen CI die 0
    ausschliesst, ist eine belegte Drift (``significant``).

    Dazu ``case_resampling``: nichtparametrischer Bootstrap UEBER ZELLEN (welche (d, p)-Paare
    in den Fit gehen), als zweite, methodisch andere Unsicherheit.

    ``combined_half_width`` = sqrt(stat^2 + max|Delta|^2) ist bewusst KONSERVATIV (der
    Shift enthaelt selbst Rauschen), eine Groessenordnung, kein exaktes Niveau.
    """
    n_boot = _count("n_boot", n_boot)
    if n_boot < 50:
        raise ValueError(f"n_boot must be >= 50, got {n_boot}")
    seed = _count("seed", seed)
    a = _alpha(alpha)
    cells = list(cells)
    primary = fss_fit(cells)
    sel = _variant_selectors(cells)
    variants = {
        name: _variant_fit([c for c, m in zip(cells, mask, strict=True) if m])
        for name, mask in sel.items()
    }

    d, p, k, n = _arrays(cells)
    bounds, _ = _bounds_and_starts(p)
    rng = np.random.Generator(np.random.Philox(seed))
    ph = k / n
    deltas = {name: np.full(n_boot, np.nan) for name in sel}
    warm_p = [(primary["p_th"], primary["nu"])]
    for b in range(n_boot):
        kb = rng.binomial(n.astype(np.int64), ph).astype(float)
        yb, wb = kb / n, 1.0 / _sigma(kb, n)
        res0 = _fit_core(d, p, yb, wb, warm_p, bounds)
        if res0 is None:  # pragma: no cover - numerisch
            continue
        for name, mask in sel.items():
            v = variants[name]
            if v is None:
                continue
            bnd_v, _ = _bounds_and_starts(p[mask])
            res = _fit_core(d[mask], p[mask], yb[mask], wb[mask], [(v["p_th"], v["nu"])], bnd_v)
            if res is not None:
                deltas[name][b] = res.x[0] - res0.x[0]

    shifts = {}
    for name, v in variants.items():
        if v is None:
            shifts[name] = None
            continue
        ci = _percentile_ci(deltas[name], a)
        delta = v["p_th"] - primary["p_th"]
        shifts[name] = {
            "delta": delta,
            "ci": ci,
            "significant": bool(ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)),
        }
    valid = [s["delta"] for s in shifts.values() if s is not None]
    max_shift = float(max((abs(x) for x in valid), default=0.0))

    case_rng = np.random.Generator(np.random.Philox([seed, 1]))
    boot, n_fail = [], 0
    for _ in range(n_boot):
        idx = case_rng.integers(0, len(cells), size=len(cells))
        v = _variant_fit([cells[i] for i in idx])
        if v is None:
            n_fail += 1
        else:
            boot.append(v["p_th"])

    out = {
        "primary_p_th": primary["p_th"],
        "variants": variants,
        "shifts": shifts,
        "max_abs_shift": max_shift,
        "any_significant_shift": any(s is not None and s["significant"] for s in shifts.values()),
        "case_resampling": {
            "ci": _percentile_ci(np.asarray(boot, dtype=float), a),
            "n_boot": n_boot,
            "degenerate_resamples": n_fail,
        },
        "n_boot": n_boot,
        "seed": seed,
        "alpha": a,
        "note": (
            "post-hoc robustness analysis (declared after the primary estimate); shift CIs are "
            "paired parametric bootstraps; combined_half_width is conservative"
        ),
    }
    if stat_ci is not None:
        stat_hw = (float(stat_ci[1]) - float(stat_ci[0])) / 2.0
        out["stat_half_width"] = stat_hw
        out["combined_half_width"] = math.hypot(stat_hw, max_shift)
    return out


def drifted_repetition_dataset(
    distances: Sequence[int],
    p_grid: Sequence[float],
    shots: int,
    seed: int,
    *,
    c: float,
) -> list[Cell]:
    """Gegenkontrolle mit BEKANNTER Finite-Size-Drift: p_L(d, p - c/d). Jede Kurve ist um
    c/d verschoben, die Kreuzung benachbarter Distanzen wandert also mit d; ein Fit ohne
    kleinste Distanz MUSS einen anderen p_th liefern."""
    shots = _count("shots", shots)
    seed = _count("seed", seed)
    children = np.random.SeedSequence(seed).spawn(len(distances) * len(p_grid))
    cells, i = [], 0
    for dd in distances:
        for pp in p_grid:
            rng = np.random.Generator(np.random.Philox(children[i]))
            i += 1
            q = min(max(float(pp) - c / int(dd), 0.0), 1.0)
            cells.append(
                Cell(
                    int(dd),
                    float(pp),
                    int(rng.binomial(shots, logical_error_rate_exact(int(dd), q))),
                    shots,
                )
            )
    return cells


# =============================================================================
# 3. Geschlossenes Orakel: Repetition-Code
# =============================================================================


def _rep_slope_at_half(d: int) -> float:
    # d * C(d-1, (d-1)/2) / 2^(d-1), log-stabil.
    m = (d - 1) // 2
    log_c = gammaln(d) - 2.0 * gammaln(m + 1)
    return float(d * math.exp(log_c - (d - 1) * math.log(2.0)))


def repetition_nu_oracle(distances: Sequence[int]) -> dict:
    """Exaktes effektives nu ueber die Distanzen: Steigung von log(dp_L/dp|_{1/2}) gegen
    log d ist 1/nu. Asymptotisch sqrt(2d/pi) -> nu = 2."""
    ds = [int(x) for x in distances]
    if len(ds) < 2 or any(x < 1 or x % 2 == 0 for x in ds):
        raise ValueError("need >= 2 odd distances")
    slopes = [_rep_slope_at_half(x) for x in ds]
    inv_nu = float(np.polyfit(np.log(ds), np.log(slopes), 1)[0])
    return {"nu_effective": 1.0 / inv_nu, "nu_asymptotic": 2.0, "slopes_at_half": slopes}


def repetition_fss_dataset(
    distances: Sequence[int],
    p_grid: Sequence[float],
    shots: int,
    seed: int,
) -> list[Cell]:
    """Binomial-Ziehungen aus dem EXAKTEN Majority-Vote-p_L; je Zelle eigener Stream."""
    shots = _count("shots", shots)
    seed = _count("seed", seed)
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(distances) * len(p_grid))
    cells = []
    i = 0
    for d in distances:
        for p in p_grid:
            rng = np.random.Generator(np.random.Philox(children[i]))
            i += 1
            p_exact = logical_error_rate_exact(int(d), float(p))
            cells.append(Cell(int(d), float(p), int(rng.binomial(shots, p_exact)), shots))
    return cells
