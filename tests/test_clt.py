"""Tests fuer die MCMC-CLT-Varianz sigma^2_g (Phase-5).

Orakel: AR(1)-Prozess mit GESCHLOSSENER long-run-Varianz
sigma^2_g = sigma_eps^2/(1-phi)^2 = Var(x)*(1+phi)/(1-phi). Unabhaengig von
der Implementierung. Coverage-Test beidseitig: korrektes sigma^2_g deckt ~95 %,
die iid-Annahme UNTERdeckt.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import clt


def ar1(n: int, phi: float, seed: int, sigma_eps: float = 1.0) -> np.ndarray:
    """AR(1)-Prozess mit stationaerer Anfangsbedingung (Mittel 0)."""
    rng = np.random.default_rng(seed)
    e = sigma_eps * rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_ar1_oracle_closed_form() -> None:
    """ar1_clt_variance liefert die geschlossene Form exakt (Selbst-Konsistenz der Formel)."""
    for phi in (0.5, 0.8, 0.9):
        o = clt.ar1_clt_variance(phi, sigma_eps=2.0)
        # sigma2_g == sigma_eps^2/(1-phi)^2
        assert o["sigma2_g"] == pytest.approx(4.0 / (1 - phi) ** 2)
        # == Var * (1+phi)/(1-phi)
        assert o["sigma2_g"] == pytest.approx(o["var_marginal"] * (1 + phi) / (1 - phi))
        # tau_int konsistent: sigma2_g == 2 tau_int Var
        assert o["sigma2_g"] == pytest.approx(2.0 * o["tau_int"] * o["var_marginal"])


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_gamma_estimator_matches_ar1_oracle(phi: float) -> None:
    """Gamma-CLT-Varianz trifft das AR(1)-Orakel (Mittel ueber Replikate)."""
    o = clt.ar1_clt_variance(phi)
    ests = [clt.clt_variance(ar1(40000, phi, sd)).sigma2_g_gamma for sd in range(20)]
    rel = abs(float(np.mean(ests)) - o["sigma2_g"]) / o["sigma2_g"]
    assert rel < 0.06, (np.mean(ests), o["sigma2_g"], rel)


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_obm_estimator_matches_ar1_oracle(phi: float) -> None:
    """OBM-CLT-Varianz (UNABHAENGIGE Methode) trifft dasselbe Orakel."""
    o = clt.ar1_clt_variance(phi)
    ests = [clt.clt_variance(ar1(40000, phi, sd)).sigma2_g_obm for sd in range(20)]
    rel = abs(float(np.mean(ests)) - o["sigma2_g"]) / o["sigma2_g"]
    assert rel < 0.08, (np.mean(ests), o["sigma2_g"], rel)


def test_obm_and_gamma_agree() -> None:
    """Die zwei unabhaengigen Schaetzer stimmen im Mittel ueberein (Cross-Check)."""
    phi = 0.8
    g = [clt.clt_variance(ar1(40000, phi, sd)).sigma2_g_gamma for sd in range(15)]
    o = [clt.clt_variance(ar1(40000, phi, sd)).sigma2_g_obm for sd in range(15)]
    assert abs(np.mean(g) - np.mean(o)) / np.mean(g) < 0.06


def test_coverage_correct_sigma2_covers_95() -> None:
    """NON-VAKUOES (a): korrektes sigma^2_g deckt den wahren Mittelwert mit ~95 %."""
    phi, n, reps, true_mean = 0.8, 4000, 400, 0.0
    cov = 0
    for sd in range(reps):
        r = clt.clt_variance(ar1(n, phi, sd))
        lo, hi = clt.confidence_interval(r.mean, r.sigma2_g_gamma, n, alpha=0.05)
        cov += int(lo <= true_mean <= hi)
    rate = cov / reps
    # Binomial-Streuung bei reps=400: ~95% +/- ~2% -> erlaube [0.90, 0.985].
    assert 0.90 <= rate <= 0.985, rate


def test_coverage_iid_assumption_undercovers() -> None:
    """NON-VAKUOES (b): ABSICHTLICH falsches sigma^2_g (iid Var) VERFEHLT die 95 %."""
    phi, n, reps, true_mean = 0.8, 4000, 400, 0.0
    cov = 0
    for sd in range(reps):
        r = clt.clt_variance(ar1(n, phi, sd))
        # iid-Annahme: sigma^2_g = Var (statt 2 tau_int Var) -> viel zu klein.
        lo, hi = clt.confidence_interval(r.mean, r.var_marginal, n, alpha=0.05)
        cov += int(lo <= true_mean <= hi)
    rate = cov / reps
    # Muss deutlich unter 95 % liegen (tau_int=4.5 -> CI ~3x zu schmal).
    assert rate < 0.75, rate


def test_obm_vectorized_matches_naive() -> None:
    """OBM-Cumsum-Vektorisierung == naive Schleife (Implementierungs-Korrektheit)."""
    x = ar1(2000, 0.7, 3)
    b = 40
    n = x.size
    # naiv
    means = np.array([x[k : k + b].mean() for k in range(n - b + 1)])
    gm = x.mean()
    ssq = float(np.sum((means - gm) ** 2))
    naive = (n * b) / ((n - b) * (n - b + 1)) * ssq
    assert clt.obm_variance(x, batch_size=b) == pytest.approx(naive, rel=1e-12)


def test_clt_edge_inputs_raise() -> None:
    """Silent-Failure-Gate: invalide Eingaben werfen sauber (vor den Tests)."""
    with pytest.raises(ValueError):
        clt.clt_variance(np.array([1.0, 2.0]))  # N<4
    with pytest.raises(ValueError):
        clt.obm_variance(np.arange(100.0), batch_size=0)  # b<1
    with pytest.raises(ValueError):
        clt.obm_variance(np.arange(100.0), batch_size=100)  # b>=N
    with pytest.raises(ValueError):
        clt.confidence_interval(0.0, -1.0, 10)  # negative sigma2_g
    with pytest.raises(ValueError):
        clt.confidence_interval(0.0, 1.0, 10, alpha=1.5)  # alpha out of (0,1)
    with pytest.raises(ValueError):
        clt.ar1_clt_variance(1.0)  # |phi|>=1
    with pytest.raises(ValueError):
        clt.ar1_clt_variance(0.5, sigma_eps=0.0)  # sigma_eps<=0
