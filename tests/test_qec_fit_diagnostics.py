"""QEC-Fit-Diagnostik (Inkrement 2): jede Kennzahl gegen ihr UNABHAENGIGES Orakel.

Codie-Disziplin: Code-Distanz-Exponent vs. analytisches (d+1)/2; Pseudo-Threshold
vs. analytisches p* = 1/2; Lambda vs. geschlossenes small-p-Orakel (A_d/A_{d+2})/p.
Plus Negativ-/Edge-Input-Gates (Silent-Failure-Gate: invalide Eingaben werfen LAUT).
"""

from __future__ import annotations

import math

import pytest

from adaptiverg_qec import qec_fit_diagnostics as fd
from adaptiverg_qec.qec_diagnostics import logical_error_rate_exact

# ---------------------------------------------------------------------------
# Analytische Orakel: geschlossene Werte.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "expected_A"),
    [(1, 1), (3, 3), (5, 10), (7, 35), (9, 126), (11, 462)],
)
def test_leading_coefficient_is_central_binomial(d: int, expected_A: int) -> None:
    """A_d = C(d, (d+1)/2): geschlossener fuehrender Koeffizient von p_L."""
    assert fd.leading_coefficient_exact(d) == expected_A


@pytest.mark.parametrize("d", [1, 3, 5, 7, 9, 13])
def test_leading_coefficient_matches_numeric_limit(d: int) -> None:
    """p_L(d, p) / p^((d+1)/2) -> A_d als p -> 0 (numerischer Konvergenz-Check)."""
    kmin = d // 2 + 1
    p = 1e-6
    ratio = logical_error_rate_exact(d, p) / p**kmin
    assert abs(ratio - fd.leading_coefficient_exact(d)) < 1e-3 * fd.leading_coefficient_exact(d)


@pytest.mark.parametrize(("d", "expected"), [(1, 1.0), (3, 2.0), (5, 3.0), (7, 4.0), (9, 5.0)])
def test_distance_exponent_exact(d: int, expected: float) -> None:
    assert fd.distance_exponent_exact(d) == expected


# ---------------------------------------------------------------------------
# 1) Code-Distanz-Exponent: empirische log-log-Steigung == (d+1)/2.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_fit_distance_exponent_matches_oracle(d: int) -> None:
    """Empirische sub-threshold-Steigung trifft (d+1)/2 (kleine Sondier-p)."""
    fit = fd.fit_distance_exponent(d)
    assert fit.slope_exact == (d + 1) / 2.0
    # bei p_probe ~ 1e-4 dominiert der Leitterm -> Steigung sehr nah am Exponenten.
    assert fit.abs_error < 1e-2


def test_fit_distance_exponent_d1_is_identity() -> None:
    """d=1 (keine Redundanz): p_L = p -> Steigung exakt 1."""
    fit = fd.fit_distance_exponent(1)
    assert abs(fit.slope_empirical - 1.0) < 1e-9
    assert fit.slope_exact == 1.0


# ---------------------------------------------------------------------------
# 2) Pseudo-Threshold: Bisektions-Kreuzung == 1/2 (maschinengenau).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("d1", "d2"), [(3, 5), (3, 7), (5, 9), (1, 3), (7, 11)])
def test_pseudo_threshold_crossing_is_one_half(d1: int, d2: int) -> None:
    cross = fd.pseudo_threshold_bisection(d1, d2)
    assert cross.p_star_exact == 0.5
    assert cross.abs_error < 1e-12


def test_pseudo_threshold_symmetry_value() -> None:
    """Bestaetige das Orakel-Fundament: p_L(d, 1/2) = 1/2 fuer jedes ungerade d."""
    for d in (1, 3, 5, 7, 9, 11):
        assert abs(logical_error_rate_exact(d, 0.5) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 3) Lambda-Suppressionsfaktor: > 1 sub-threshold + small-p-Orakel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.3])
def test_lambda_above_one_below_threshold(d: int, p: float) -> None:
    """Fuer p < p* = 0.5 senkt ein Distanz-Sprung p_L -> Lambda > 1."""
    lam = fd.lambda_suppression(d, p)
    assert lam.lambda_value > 1.0


@pytest.mark.parametrize("d", [3, 5, 7])
def test_lambda_small_p_oracle(d: int) -> None:
    """Lambda(d) -> (A_d/A_{d+2})/p als p -> 0 (geschlossenes Orakel)."""
    p = 1e-4
    lam = fd.lambda_suppression(d, p)
    rel = abs(lam.lambda_value - lam.lambda_leading_small_p) / lam.lambda_leading_small_p
    assert rel < 1e-2


def test_lambda_d3_value_p_0_1() -> None:
    """Scharfer Einzelwert d=3->5 @ p=0.1: Lambda = p_L(3)/p_L(5)."""
    pl3 = logical_error_rate_exact(3, 0.1)
    pl5 = logical_error_rate_exact(5, 0.1)
    lam = fd.lambda_suppression(3, 0.1)
    assert abs(lam.lambda_value - pl3 / pl5) < 1e-12


# ---------------------------------------------------------------------------
# Aggregierter Sweep: alle worst-case-Fehler unter Toleranz.
# ---------------------------------------------------------------------------


def test_run_fit_diagnostics_all_within_tolerance() -> None:
    payload = fd.run_fit_diagnostics()
    diag = payload["diagnostics"]
    assert diag["distance_exponent"]["worst_abs_error"] < 1e-2
    assert diag["pseudo_threshold"]["worst_abs_error"] < 1e-12
    assert all(r["above_one"] for r in diag["lambda_suppression"]["rows"])


# ---------------------------------------------------------------------------
# Silent-Failure-Gate: invalide Eingaben werfen LAUT (kein stiller rc=0/NaN).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_d", [0, -1, 2, 4, 100])
def test_even_or_nonpositive_distance_rejected(bad_d: int) -> None:
    with pytest.raises(ValueError):
        fd.leading_coefficient_exact(bad_d)
    with pytest.raises(ValueError):
        fd.distance_exponent_exact(bad_d)
    with pytest.raises(ValueError):
        fd.fit_distance_exponent(bad_d)


def test_non_integer_distance_rejected() -> None:
    with pytest.raises(TypeError):
        fd.leading_coefficient_exact(3.0)  # type: ignore[arg-type]


def test_fit_rejects_bad_probe() -> None:
    with pytest.raises(ValueError):
        fd.fit_distance_exponent(3, p_probe=(1e-4,))  # too few points
    with pytest.raises(ValueError):
        fd.fit_distance_exponent(3, p_probe=(0.2, 0.3))  # not sub-threshold (>=0.1)
    with pytest.raises(ValueError):
        fd.fit_distance_exponent(3, p_probe=(0.0, 1e-4))  # p <= 0
    with pytest.raises(ValueError):
        fd.fit_distance_exponent(3, p_probe=(4e-4, 2e-4))  # not strictly increasing
    with pytest.raises(ValueError):
        fd.fit_distance_exponent(3, p_probe=(1e-4, 1e-4))  # duplicate


def test_pseudo_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fd.pseudo_threshold_bisection(4, 5)  # even
    with pytest.raises(ValueError):
        fd.pseudo_threshold_bisection(3, 3)  # equal
    with pytest.raises(ValueError):
        fd.pseudo_threshold_bisection(3, 5, lo=0.6, hi=0.9)  # lo not < 0.5
    with pytest.raises(ValueError):
        fd.pseudo_threshold_bisection(3, 5, lo=0.01, hi=0.4)  # hi not > 0.5
    with pytest.raises(ValueError):
        fd.pseudo_threshold_bisection(3, 5, iterations=0)


def test_lambda_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError):
        fd.lambda_suppression(3, 0.0)  # p <= 0
    with pytest.raises(ValueError):
        fd.lambda_suppression(3, 0.5)  # at threshold (not sub-threshold)
    with pytest.raises(ValueError):
        fd.lambda_suppression(3, 0.6)  # above threshold
    with pytest.raises(ValueError):
        fd.lambda_suppression(3, math.nan)
    with pytest.raises(ValueError):
        fd.lambda_suppression(2, 0.1)  # even distance
