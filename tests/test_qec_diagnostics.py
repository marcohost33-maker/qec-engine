"""QEC-Diagnostik-Harness: MC-logische-Fehlerrate vs. exaktes Binomial-Orakel.

Codie-Disziplin: jede MC-Zahl gegen ein UNABHAENGIGES Orakel (geschlossene
Binomialsumme), Reproduzierbarkeit (Seed -> bit-identisch), und Negativ-/
Edge-Input-Gates (Silent-Failure-Gate: invalide Eingaben werfen LAUT).
"""

from __future__ import annotations

import math

import pytest

from adaptiverg_qec import qec_diagnostics as qd

# ---------------------------------------------------------------------------
# Exaktes Orakel: geschlossene Werte (unabhaengig per Hand/Brute-Force).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0])
def test_n3_closed_form_matches_oracle(p: float) -> None:
    """n=3: p_L = 3 p^2 (1-p) + p^3 = 3 p^2 - 2 p^3 (KORRIGIERTE Formel)."""
    expected = 3.0 * p**2 - 2.0 * p**3
    assert abs(qd.logical_error_rate_exact(3, p) - expected) < 1e-12


def test_n3_specific_value_p_0_1() -> None:
    """Scharfer Einzelwert: p=0.1 -> 0.028 (NICHT 0.019 der falschen Web-Formel)."""
    val = qd.logical_error_rate_exact(3, 0.1)
    assert abs(val - 0.028) < 1e-12
    # Gegenprobe: die kursierende falsche Formel p^3 + 2 p^2 (1-p) = 0.019 stimmt NICHT.
    assert abs(val - 0.019) > 1e-3


@pytest.mark.parametrize("n", [1, 3, 5, 7, 9])
def test_oracle_matches_brute_force_binomial(n: int) -> None:
    """Orakel == direkte Brute-Force-Enumeration aller 2^n Flip-Muster (klein-n)."""
    for p in (0.0, 0.07, 0.25, 0.5, 0.83, 1.0):
        brute = 0.0
        for mask in range(1 << n):
            k = bin(mask).count("1")  # Anzahl geflippter Bits
            prob = (p**k) * ((1.0 - p) ** (n - k))
            if k > n / 2.0:  # Majority-Vote falsch
                brute += prob
        assert abs(brute - qd.logical_error_rate_exact(n, p)) < 1e-12


def test_n1_is_identity_channel() -> None:
    """n=1 (keine Redundanz): p_L == p fuer alle p."""
    for p in (0.0, 0.1, 0.37, 0.5, 1.0):
        assert abs(qd.logical_error_rate_exact(1, p) - p) < 1e-12


def test_endpoints_exact() -> None:
    """p=0 -> 0; p=1 -> 1 (alle Bits flippen, Majority sicher falsch)."""
    for n in (1, 3, 5, 7):
        assert qd.logical_error_rate_exact(n, 0.0) == 0.0
        assert abs(qd.logical_error_rate_exact(n, 1.0) - 1.0) < 1e-12


def test_more_distance_helps_below_half() -> None:
    """p < 1/2: groessere Code-Distanz senkt p_L (Fehlerkorrektur wirkt)."""
    for p in (0.1, 0.2, 0.4):
        seq = [qd.logical_error_rate_exact(n, p) for n in (1, 3, 5, 7, 9)]
        assert all(b < a for a, b in zip(seq, seq[1:], strict=False))


def test_more_distance_hurts_above_half() -> None:
    """p > 1/2: groessere Distanz ERHOEHT p_L (jenseits des Pseudo-Thresholds)."""
    for p in (0.6, 0.8):
        seq = [qd.logical_error_rate_exact(n, p) for n in (1, 3, 5, 7, 9)]
        assert all(b > a for a, b in zip(seq, seq[1:], strict=False))


def test_pseudo_threshold_is_one_half() -> None:
    """Alle ungeraden Distanzen schneiden sich exakt bei p* = 1/2."""
    half = 0.5
    base = qd.logical_error_rate_exact(3, half)
    assert abs(base - half) < 1e-12  # bei p=1/2 ist p_L = 1/2 fuer jedes n
    for n in (1, 5, 7, 9):
        assert abs(qd.logical_error_rate_exact(n, half) - half) < 1e-12
    assert qd.pseudo_threshold(3, 5) == 0.5


# ---------------------------------------------------------------------------
# Monte Carlo vs. Orakel + Reproduzierbarkeit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("p", [0.05, 0.15, 0.3])
def test_mc_matches_oracle_within_3sigma(n: int, p: float) -> None:
    """MC-p_L trifft das exakte Orakel innerhalb 3 sigma (grosse shots)."""
    est = qd.simulate_logical_error_rate(n, p, shots=200_000, seed=2026)
    assert abs(est.p_L_mc - est.p_L_exact) < 4.0 * est.std_err + 1e-9
    assert est.n_sigma < 4.0


def test_mc_reproducible_same_seed() -> None:
    """Gleicher Seed -> bit-identischer MC-Schaetzer; anderer Seed -> verschieden."""
    a = qd.simulate_logical_error_rate(5, 0.2, shots=50_000, seed=7)
    b = qd.simulate_logical_error_rate(5, 0.2, shots=50_000, seed=7)
    c = qd.simulate_logical_error_rate(5, 0.2, shots=50_000, seed=8)
    assert a.n_logical_errors == b.n_logical_errors
    assert a.p_L_mc == b.p_L_mc
    assert a.n_logical_errors != c.n_logical_errors


def test_mc_error_shrinks_with_shots() -> None:
    """Standardfehler faellt ~1/sqrt(shots) (10x shots -> ~0.32x sem)."""
    small = qd.simulate_logical_error_rate(5, 0.2, shots=10_000, seed=3)
    large = qd.simulate_logical_error_rate(5, 0.2, shots=1_000_000, seed=3)
    # Erwartet sem_large/sem_small ~ sqrt(10k/1M) = 0.1; tolerant gegen Schaetz-Streuung.
    ratio = large.std_err / small.std_err
    assert ratio < 0.25


# ---------------------------------------------------------------------------
# Silent-Failure-Gate: invalide Eingaben werfen LAUT (kein stiller rc=0/NaN).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_n", [0, -1, 2, 4, 100])
def test_even_or_nonpositive_n_rejected(bad_n: int) -> None:
    with pytest.raises(ValueError):
        qd.logical_error_rate_exact(bad_n, 0.1)


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, 2.0, -1.0, math.inf, math.nan])
def test_p_outside_unit_interval_rejected(bad_p: float) -> None:
    with pytest.raises(ValueError):
        qd.logical_error_rate_exact(3, bad_p)


def test_non_integer_n_rejected() -> None:
    with pytest.raises(TypeError):
        qd.logical_error_rate_exact(3.0, 0.1)  # type: ignore[arg-type]


def test_simulate_rejects_bad_shots() -> None:
    with pytest.raises(ValueError):
        qd.simulate_logical_error_rate(3, 0.1, shots=0, seed=1)
    with pytest.raises(TypeError):
        qd.simulate_logical_error_rate(3, 0.1, shots=1.5, seed=1)  # type: ignore[arg-type]


def test_simulate_rejects_bad_n_and_p_before_sampling() -> None:
    """fail-closed-early: invalide n/p werfen, BEVOR teuer gesampelt wird."""
    with pytest.raises(ValueError):
        qd.simulate_logical_error_rate(4, 0.1, shots=10, seed=1)
    with pytest.raises(ValueError):
        qd.simulate_logical_error_rate(3, 1.5, shots=10, seed=1)


def test_pseudo_threshold_rejects_bad_distances() -> None:
    with pytest.raises(ValueError):
        qd.pseudo_threshold(4, 5)  # even
    with pytest.raises(ValueError):
        qd.pseudo_threshold(3, 3)  # equal
