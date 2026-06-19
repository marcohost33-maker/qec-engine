"""QEC-Fit-Diagnostik (Inkrement 2): Code-Distanz-Exponent, Pseudo-Threshold-Bisektion,
Lambda-Suppressionsfaktor -- alle gegen ein geschlossenes/analytisches Orakel.

ZWECK / POSITIONIERUNG (ehrlich):
Inkrement 1 (`qec_diagnostics.py`) lieferte die exakte logische Fehlerrate p_L(n, p)
des n-Bit-Repetition-Codes (Binomial-Orakel) + Monte-Carlo-Validierung. Inkrement 2
baut DARAUF auf und extrahiert die drei diagnostischen Standard-Kennzahlen, mit denen
in der QEC-Literatur Codes/Decoder verglichen werden -- aber HIER vollstaendig
orakelbar (keine Decoder-Erfindung, keine neuen Dependencies):

  1. **Code-Distanz-Exponent** (sub-threshold scaling): fuer p << p* gilt
         p_L(d, p) ~ A_d * p^((d+1)/2),   A_d = C(d, (d+1)/2),
     d.h. die log-log-Steigung d(log p_L)/d(log p) -> (d+1)/2. Das ist der
     empirische Fehler-Unterdrueckungs-Exponent; fuer den Repetition-Code exakt
     gleich der halben (Code-Distanz + 1) = Anzahl korrigierbarer Fehler + 1.

  2. **Pseudo-Threshold** durch Kurven-Kreuzungs-Bisektion: der numerische
     Schnittpunkt von p_L(d1, .) und p_L(d2, .) muss den analytisch bekannten
     Wert p* = 1/2 treffen (Symmetrie p_L(d, 1/2) = 1/2 fuer jedes ungerade d).

  3. **Lambda-Suppressionsfaktor** Lambda_d = p_L(d, p) / p_L(d+2, p): der
     Faktor, um den ein Distanz-Sprung d -> d+2 die logische Fehlerrate drueckt
     (vgl. Google "below threshold", Nature 2024). Fuer p -> 0 divergiert er wie
     Lambda_d ~ (A_d / A_{d+2}) / p  (geschlossenes Orakel).

KORREKTHEITS-DISZIPLIN (Codie): jede Kennzahl wird gegen ihr UNABHAENGIGES,
geschlossen hergeleitetes Orakel geprueft (Binomialkoeffizient bzw. p* = 1/2),
nicht gegen sich selbst. Die p_L-Werte stammen aus `qec_diagnostics.logical_error_rate_exact`,
das in Inkrement 1 bereits gegen Brute-Force-Enumeration validiert wurde.

NICHT-Ziel (ehrlich abgegrenzt): KEIN Surface-Code, KEIN MWPM-/Tensor-Network-
Decoder, KEIN phenomenological-noise-Spacetime-Decoder. Diese erfordern eine
etablierte Matching-Bibliothek (PyMatching) hinter optional-dependency-Gate und
sind Inkrement 3 der Roadmap (s. docs/QEC_DIAGNOSTICS_ROADMAP.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .qec_diagnostics import logical_error_rate_exact


def _check_odd_distance(d: int) -> None:
    if not isinstance(d, int):
        raise TypeError(f"distance d must be an int, got {type(d).__name__}")
    if d < 1 or d % 2 == 0:
        raise ValueError(f"distance d must be odd and >= 1, got {d}")


def leading_coefficient_exact(d: int) -> int:
    """Analytisches Orakel fuer den fuehrenden Koeffizienten A_d = C(d, (d+1)/2).

    Fuer p -> 0 dominiert der kleinste Term der Binomialsumme (k = (d+1)/2):
        p_L(d, p) = C(d, (d+1)/2) p^((d+1)/2) (1-p)^((d-1)/2) + O(p^((d+3)/2))
                  ~ A_d p^((d+1)/2).

    Returns:
        A_d als exakter Binomialkoeffizient (int).
    """
    _check_odd_distance(d)
    return math.comb(d, d // 2 + 1)


def distance_exponent_exact(d: int) -> float:
    """Analytisches Orakel fuer den sub-threshold-Exponenten = (d+1)/2.

    = Anzahl korrigierbarer Fehler t+1 = ceil(d/2) fuer den Repetition-Code.
    """
    _check_odd_distance(d)
    return (d + 1) / 2.0


@dataclass(frozen=True)
class ExponentFit:
    """Empirischer Code-Distanz-Exponent via log-log-Regression vs. Orakel.

    Attributes:
        d: ungerade Code-Distanz.
        slope_empirical: geschaetzte log-log-Steigung d(log p_L)/d(log p).
        slope_exact: analytisches Orakel (d+1)/2.
        abs_error: |slope_empirical - slope_exact|.
        p_probe: verwendete (kleinen) Sondier-p-Werte.
    """

    d: int
    slope_empirical: float
    slope_exact: float
    abs_error: float
    p_probe: tuple[float, ...]


def fit_distance_exponent(
    d: int,
    p_probe: tuple[float, ...] = (1e-4, 2e-4, 4e-4),
) -> ExponentFit:
    """Schaetze den sub-threshold-Exponenten via log-log-Regression von p_L vs p.

    Lineare Regression von log p_L gegen log p bei kleinen p; die Steigung muss
    den analytischen Wert (d+1)/2 treffen. KEIN Monte-Carlo -- die p_L-Werte sind
    das exakte Binomial-Orakel, die Methode prueft das Skalierungs-Verhalten.

    Args:
        d: ungerade Code-Distanz >= 3 (d=1 hat trivialen Exponent 1, erlaubt).
        p_probe: streng aufsteigende kleine Wahrscheinlichkeiten in (0, 0.1).

    Returns:
        ExponentFit mit empirischer und exakter Steigung.

    Raises:
        ValueError/TypeError: invalide d oder p_probe (Silent-Failure-Gate).
    """
    _check_odd_distance(d)
    if len(p_probe) < 2:
        raise ValueError("p_probe must contain at least 2 points for a regression")
    for pp in p_probe:
        if not math.isfinite(pp) or pp <= 0.0 or pp >= 0.1:
            raise ValueError(f"p_probe values must be in (0, 0.1), got {pp}")
    if list(p_probe) != sorted(p_probe) or len(set(p_probe)) != len(p_probe):
        raise ValueError("p_probe must be strictly increasing (distinct, ascending)")

    xs = [math.log(pp) for pp in p_probe]
    ys = [math.log(logical_error_rate_exact(d, pp)) for pp in p_probe]
    n = len(xs)
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    exact = distance_exponent_exact(d)
    return ExponentFit(
        d=d,
        slope_empirical=slope,
        slope_exact=exact,
        abs_error=abs(slope - exact),
        p_probe=tuple(p_probe),
    )


@dataclass(frozen=True)
class ThresholdCrossing:
    """Pseudo-Threshold via Kurven-Kreuzungs-Bisektion vs. analytisches p* = 1/2.

    Attributes:
        d1, d2: zwei verschiedene ungerade Distanzen.
        p_star_numeric: Bisektions-Schnittpunkt von p_L(d1,.) und p_L(d2,.).
        p_star_exact: analytisches Orakel 0.5.
        abs_error: |p_star_numeric - 0.5|.
        iterations: Bisektions-Schritte.
    """

    d1: int
    d2: int
    p_star_numeric: float
    p_star_exact: float
    abs_error: float
    iterations: int


def pseudo_threshold_bisection(
    d1: int,
    d2: int,
    lo: float = 0.01,
    hi: float = 0.99,
    iterations: int = 80,
) -> ThresholdCrossing:
    """Finde den p_L(d1,.) == p_L(d2,.)-Schnittpunkt per Bisektion; Orakel = 0.5.

    f(p) = p_L(d1, p) - p_L(d2, p) hat genau eine Nullstelle bei p* = 1/2 im
    offenen Intervall (0, 1): unterhalb hilft mehr Distanz (f hat festes
    Vorzeichen), oberhalb schadet sie. Bisektion muss 1/2 maschinengenau treffen.

    Args:
        d1, d2: verschiedene ungerade Distanzen >= 1.
        lo, hi: Startklammer mit 0 < lo < 0.5 < hi < 1.
        iterations: Bisektions-Iterationen (80 -> < 2^-80 Restfehler).

    Returns:
        ThresholdCrossing mit numerischem p* und Orakel-Vergleich.

    Raises:
        ValueError/TypeError: invalide Distanzen oder Klammer (Silent-Failure-Gate).
    """
    _check_odd_distance(d1)
    _check_odd_distance(d2)
    if d1 == d2:
        raise ValueError("d1 and d2 must differ")
    if not (0.0 < lo < 0.5 < hi < 1.0):
        raise ValueError(f"bracket must satisfy 0 < lo < 0.5 < hi < 1, got lo={lo}, hi={hi}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    def f(p: float) -> float:
        return logical_error_rate_exact(d1, p) - logical_error_rate_exact(d2, p)

    f_lo = f(lo)
    if f_lo == 0.0:
        return ThresholdCrossing(d1, d2, lo, 0.5, abs(lo - 0.5), 0)
    a, b = lo, hi
    for _ in range(iterations):
        mid = 0.5 * (a + b)
        if f_lo * f(mid) <= 0.0:
            b = mid
        else:
            a = mid
    p_star = 0.5 * (a + b)
    return ThresholdCrossing(
        d1=d1,
        d2=d2,
        p_star_numeric=p_star,
        p_star_exact=0.5,
        abs_error=abs(p_star - 0.5),
        iterations=iterations,
    )


@dataclass(frozen=True)
class LambdaFactor:
    """Lambda-Suppressionsfaktor p_L(d) / p_L(d+2) + small-p-Orakel.

    Attributes:
        d: ungerade Basis-Distanz.
        p: physikalische Fehlerrate (< p* = 0.5, sonst kein Suppression-Regime).
        lambda_value: p_L(d, p) / p_L(d+2, p).
        lambda_leading_small_p: geschlossenes small-p-Orakel (A_d / A_{d+2}) / p.
    """

    d: int
    p: float
    lambda_value: float
    lambda_leading_small_p: float


def lambda_suppression(d: int, p: float) -> LambdaFactor:
    """Fehler-Unterdrueckungs-Faktor bei Distanz-Sprung d -> d+2.

    Lambda_d(p) = p_L(d, p) / p_L(d+2, p). Im Limes p -> 0:
        Lambda_d ~ (A_d / A_{d+2}) / p,   A_k = C(k, (k+1)/2),
    weil der fuehrende Exponent von p_L um genau 1 steigt (kmin: (d+1)/2 -> (d+3)/2).
    Der Faktor MUSS > 1 sein fuer p < p* (mehr Distanz hilft); das ist die
    Mindest-Bedingung fuer "below threshold".

    Args:
        d: ungerade Basis-Distanz >= 1.
        p: physikalische Fehlerrate in (0, 0.5) (sub-threshold-Regime).

    Returns:
        LambdaFactor mit Ist-Wert und small-p-Orakel.

    Raises:
        ValueError/TypeError: invalide d oder p (Silent-Failure-Gate).
    """
    _check_odd_distance(d)
    if not math.isfinite(p) or p <= 0.0 or p >= 0.5:
        raise ValueError(f"p must be in the sub-threshold range (0, 0.5), got {p}")
    pl_d = logical_error_rate_exact(d, p)
    pl_d2 = logical_error_rate_exact(d + 2, p)
    if pl_d2 <= 0.0:
        raise ValueError(f"p_L(d+2, p) underflowed to 0 at p={p}; choose larger p")
    a_d = leading_coefficient_exact(d)
    a_d2 = leading_coefficient_exact(d + 2)
    return LambdaFactor(
        d=d,
        p=p,
        lambda_value=pl_d / pl_d2,
        lambda_leading_small_p=(a_d / a_d2) / p,
    )


def run_fit_diagnostics(
    distances: tuple[int, ...] = (3, 5, 7, 9),
    lambda_p: float = 0.1,
) -> dict:
    """Voller Fit-Diagnostik-Lauf: Exponent-Fit + Threshold-Crossing + Lambda.

    Gibt ein JSON-serialisierbares dict zurueck (der `__main__`-Block persistiert
    es nach results/). Jede Zeile traegt Ist-Wert und unabhaengiges Orakel.
    """
    exponent_rows = []
    worst_slope_err = 0.0
    for d in distances:
        fit = fit_distance_exponent(d)
        worst_slope_err = max(worst_slope_err, fit.abs_error)
        exponent_rows.append(
            {
                "d": fit.d,
                "slope_empirical": fit.slope_empirical,
                "slope_exact": fit.slope_exact,
                "abs_error": fit.abs_error,
            }
        )

    crossing_rows = []
    worst_thr_err = 0.0
    for i in range(len(distances) - 1):
        cross = pseudo_threshold_bisection(distances[i], distances[i + 1])
        worst_thr_err = max(worst_thr_err, cross.abs_error)
        crossing_rows.append(
            {
                "d1": cross.d1,
                "d2": cross.d2,
                "p_star_numeric": cross.p_star_numeric,
                "p_star_exact": cross.p_star_exact,
                "abs_error": cross.abs_error,
            }
        )

    lambda_rows = []
    for d in distances[:-1]:
        lam = lambda_suppression(d, lambda_p)
        lambda_rows.append(
            {
                "d": lam.d,
                "p": lam.p,
                "lambda_value": lam.lambda_value,
                "lambda_leading_small_p": lam.lambda_leading_small_p,
                "above_one": lam.lambda_value > 1.0,
            }
        )

    return {
        "tool": "adaptiverg_qec.qec_fit_diagnostics",
        "model": "n-bit repetition code, independent bit-flip, majority-vote (exact oracle)",
        "diagnostics": {
            "distance_exponent": {
                "oracle": "slope d(log p_L)/d(log p) -> (d+1)/2 as p->0",
                "rows": exponent_rows,
                "worst_abs_error": worst_slope_err,
            },
            "pseudo_threshold": {
                "oracle": "curve crossing p* = 1/2 (symmetry p_L(d,1/2)=1/2)",
                "rows": crossing_rows,
                "worst_abs_error": worst_thr_err,
            },
            "lambda_suppression": {
                "oracle": "Lambda_d ~ (A_d/A_{d+2})/p as p->0, A_k=C(k,(k+1)/2)",
                "p": lambda_p,
                "rows": lambda_rows,
            },
        },
        "distances": list(distances),
    }


def _main() -> int:
    """Schreibe results/qec-fit-diagnostics-rep-code.json + Konsolen-Tabelle (Evidenz)."""
    import json
    from pathlib import Path

    payload = run_fit_diagnostics()
    out = Path("results") / "qec-fit-diagnostics-rep-code.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    diag = payload["diagnostics"]
    print("QEC-Fit-Diagnostik (Inkrement 2): Exponent / Pseudo-Threshold / Lambda")
    print("Repetition-Code, exaktes Binomial-Orakel (Inkr.1), keine Decoder-Erfindung")
    print("-" * 78)
    print("Code-Distanz-Exponent (sub-threshold log-log slope -> (d+1)/2):")
    print(f"  {'d':>3} {'slope_emp':>12} {'slope_exact':>12} {'abs_err':>10}")
    for r in diag["distance_exponent"]["rows"]:
        print(
            f"  {r['d']:>3} {r['slope_empirical']:>12.6f} "
            f"{r['slope_exact']:>12.6f} {r['abs_error']:>10.2e}"
        )
    print(f"  worst_abs_error = {diag['distance_exponent']['worst_abs_error']:.2e}")
    print("Pseudo-Threshold (curve-crossing bisection -> p* = 0.5):")
    print(f"  {'d1':>3} {'d2':>3} {'p*_numeric':>16} {'abs_err':>10}")
    for r in diag["pseudo_threshold"]["rows"]:
        print(f"  {r['d1']:>3} {r['d2']:>3} {r['p_star_numeric']:>16.12f} {r['abs_error']:>10.2e}")
    print(f"  worst_abs_error = {diag['pseudo_threshold']['worst_abs_error']:.2e}")
    print(f"Lambda-Suppression (d -> d+2) at p = {diag['lambda_suppression']['p']}:")
    print(f"  {'d':>3} {'Lambda':>10} {'small-p oracle':>16} {'>1?':>5}")
    for r in diag["lambda_suppression"]["rows"]:
        print(
            f"  {r['d']:>3} {r['lambda_value']:>10.4f} "
            f"{r['lambda_leading_small_p']:>16.4f} {str(r['above_one']):>5}"
        )
    print("-" * 78)
    print(f"evidence -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
