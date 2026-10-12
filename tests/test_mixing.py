"""Phase-7: exakte TV-Mischung (mixing.py) gegen unabhaengige Orakel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptiverg_qec import a_kernel, ising1d
from adaptiverg_qec import mixing as mx
from adaptiverg_qec.mvp_instance import MVPConfig

L = 6
# Absoluter Rundungsboden einer TV-Summe ueber 2^L=64 Eintraege nach vielen
# Matrixprodukten (~ 2^L * eps * Iterationen); die Schranken selbst fallen weit
# darunter (lambda_*^40 ~ 1e-24), der exakte Verlauf kann das nicht.
_ROUNDOFF = 1e-13


# --- Kern und Zielverteilung gegen unabhaengige Orakel -----------------------


@pytest.mark.parametrize("beta", [0.3, 0.8, 1.7])
def test_stationary_matches_ising1d_oracle(beta: float) -> None:
    """ising1d.exact_distribution ist eine separat geschriebene Enumeration."""
    pi = mx.stationary_distribution(L, beta)
    np.testing.assert_allclose(pi, ising1d.exact_distribution(beta, L), rtol=0, atol=1e-15)
    mean_h = float(pi @ mx.hamiltonians(L))
    assert abs(mean_h - ising1d.mean_energy(beta, L)) < 1e-12


def test_hamiltonians_match_a_kernel_hamiltonian() -> None:
    H = mx.hamiltonians(L)
    for k, x in enumerate(mx.all_states(L)):
        assert H[k] == a_kernel.hamiltonian(x)
    assert np.array_equal(mx._encode(mx.all_states(L)), np.arange(2**L))


@pytest.mark.parametrize("beta", [0.0, 0.8, 2.0])
def test_single_flip_kernel_is_stochastic_stationary_reversible(beta: float) -> None:
    P = mx.single_flip_kernel(L, beta)
    pi = mx.stationary_distribution(L, beta)
    assert np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pi @ P, pi, rtol=0, atol=1e-15)
    flow = pi[:, None] * P
    assert np.max(np.abs(flow - flow.T)) < 1e-15


def test_sweep_spectrum_is_single_step_spectrum_to_the_L() -> None:
    beta = 0.8
    pi = mx.stationary_distribution(L, beta)
    s1 = mx.spectral_summary(mx.single_flip_kernel(L, beta), pi)
    sL = mx.spectral_summary(mx.sweep_kernel(L, beta), pi)
    np.testing.assert_allclose(np.sort(sL.eigenvalues), np.sort(s1.eigenvalues**L), atol=1e-12)


# --- Phase-1-Akzeptanz: TV faellt geometrisch, im Spektral-Sandwich -----------


@pytest.mark.parametrize("beta", [0.3, 0.8, 1.5])
def test_exact_tv_lies_in_spectral_sandwich(beta: float) -> None:
    P = mx.sweep_kernel(L, beta)
    pi = mx.stationary_distribution(L, beta)
    lower, d, upper, spec = mx.tv_sandwich(P, pi, 40)
    assert spec.geometrically_ergodic
    assert np.all(lower <= d + _ROUNDOFF)
    assert np.all(d <= upper + _ROUNDOFF)
    for x0 in (0, 21, 2**L - 1):
        tv = mx.exact_tv_curve(P, pi, x0, 40)
        start_bound = 0.5 * math.sqrt((1 - pi[x0]) / pi[x0]) * spec.lambda_star ** np.arange(41)
        assert np.all(tv <= start_bound + _ROUNDOFF)
        assert np.all(tv <= d + _ROUNDOFF)


def test_exact_tv_decays_at_rate_lambda_star() -> None:
    P = mx.sweep_kernel(L, 0.8)
    pi = mx.stationary_distribution(L, 0.8)
    spec = mx.spectral_summary(P, pi)
    rate, used = mx.fit_geometric_rate(mx.exact_tv_curve(P, pi, 0, 30), floor=1e-12)
    assert used >= 20
    assert abs(rate - spec.lambda_star) / spec.lambda_star < 0.01


def test_beta_zero_is_flagged_not_geometrically_ergodic() -> None:
    """Negativ-Kontrolle: bei beta=0 wird jeder Flip akzeptiert; L gerade ->
    ein Sweep erhaelt die Paritaet, der Kern ist reduzibel und TV faellt nie."""
    P = mx.sweep_kernel(L, 0.0)
    pi = mx.stationary_distribution(L, 0.0)
    spec = mx.spectral_summary(P, pi)
    assert not spec.geometrically_ergodic
    assert spec.lambda_star == pytest.approx(1.0, abs=1e-12)
    assert mx.exact_tv_curve(P, pi, 0, 50)[-1] >= 0.49


# --- Der ECHTE Sampler traegt den analysierten Kern ----------------------------


def test_empirical_marginals_consistent_with_exact_kernel() -> None:
    emp = mx.empirical_tv_curve(L, 0.8, x0=0, n_chains=4000, n_sweeps=12, seed=7)
    assert mx.marginal_band_ratio(emp, mx.sweep_kernel(L, 0.8), delta=1e-6) <= 1.0
    tv_exact = mx.exact_tv_curve(mx.sweep_kernel(L, 0.8), mx.stationary_distribution(L, 0.8), 0, 12)
    band = mx.empirical_tv_band(mx.sweep_kernel(L, 0.8), 0, 12, 4000, 1e-6)
    assert np.all(np.abs(emp.tv - tv_exact) <= band)


def test_empirical_marginals_reject_wrong_kernel() -> None:
    """Trennschaerfe: dieselben Ketten gegen den Kern bei beta=1.2 -> ausserhalb."""
    emp = mx.empirical_tv_curve(L, 0.8, x0=0, n_chains=4000, n_sweeps=12, seed=7)
    assert mx.marginal_band_ratio(emp, mx.sweep_kernel(L, 1.2), delta=1e-6) > 2.0


def test_one_sweep_parity_detects_off_by_one_step_count() -> None:
    """Codex P2 (test_mixing.py:102): ein _sweep mit L-1 oder L+1 Einzelschritten liess das
    12-Sweep-Band gruen (0.76 / 0.81). Der Ein-Sweep-Paritaetstest vom alternierenden Start
    trifft den echten Sampler (<= 1) und verwirft BEIDE Nachbarkerne P_1^{L+-1} (> 3)."""
    one = mx.empirical_tv_curve(L, 0.3, x0=21, n_chains=4000, n_sweeps=1, seed=7)
    P1 = mx.single_flip_kernel(L, 0.3)
    ratio = {
        k: mx.parity_band_ratio(one, np.linalg.matrix_power(P1, k), delta=1e-6)
        for k in (L - 1, L, L + 1)
    }
    assert ratio[L] <= 1.0
    assert ratio[L - 1] > 3.0
    assert ratio[L + 1] > 3.0
    # Konsistenz: P_1^L IST der Sweep-Kern, gegen den alle anderen Gates pruefen.
    np.testing.assert_allclose(np.linalg.matrix_power(P1, L), mx.sweep_kernel(L, 0.3), atol=1e-15)


def test_parity_band_ratio_rejects_bad_arguments() -> None:
    one = mx.empirical_tv_curve(L, 0.3, x0=21, n_chains=10, n_sweeps=1, seed=1)
    P = mx.sweep_kernel(L, 0.3)
    for bad_t in (-1, 2, True, 1.0):
        with pytest.raises(ValueError, match="t must"):
            mx.parity_band_ratio(one, P, delta=1e-6, t=bad_t)
    for bad_delta in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="delta"):
            mx.parity_band_ratio(one, P, delta=bad_delta)
    with pytest.raises(ValueError, match="shape"):
        mx.parity_band_ratio(one, mx.sweep_kernel(4, 0.3), delta=1e-6)


def test_empirical_tv_curve_reproducible_and_seed_sensitive() -> None:
    a = mx.empirical_tv_curve(L, 0.8, x0=5, n_chains=200, n_sweeps=4, seed=11)
    b = mx.empirical_tv_curve(L, 0.8, x0=5, n_chains=200, n_sweeps=4, seed=11)
    c = mx.empirical_tv_curve(L, 0.8, x0=5, n_chains=200, n_sweeps=4, seed=12)
    assert np.array_equal(a.histograms, b.histograms)
    assert not np.array_equal(a.histograms, c.histograms)


# --- Phase-2-Akzeptanz: adaptive Kette + Containment --------------------------

_CFG = MVPConfig(L=L, beta_min=0.1, beta_max=2.0)


@pytest.mark.parametrize("warm", [0, 1, 30, 79])
def test_adaptive_schedule_is_bit_identical_to_a_kernel(warm) -> None:
    """Schedule == run_adaptive_mcmc(warmup_steps=W).beta_traj unter dem Freeze-Vertrag."""
    a = a_kernel.diminishing_step_sizes(80, 0.5, 1.0)
    kw = {"beta_start": 0.1} if warm else {}
    run = a_kernel.run_adaptive_mcmc(
        _CFG,
        beta_target=0.8,
        n_steps=80,
        burn_in=0,
        seed=1,
        adapt_c=0.5,
        adapt_T0=1.0,
        warmup_steps=warm,
        **kw,
    )
    start = 0.1 if warm else 0.8
    sched = mx.adaptive_beta_schedule(
        _CFG, beta_start=start, beta_target=0.8, a_t=a, freeze_at=warm
    )
    assert np.array_equal(run.beta_traj, sched)
    assert np.all(sched[warm:] == 0.8)  # bit-gleich, nicht nur nahe


def test_schedule_requires_valid_freeze_at() -> None:
    a = np.full(5, 0.1)
    for bad in (-1, 6, 1.5, True):
        with pytest.raises(ValueError):
            mx.adaptive_beta_schedule(_CFG, beta_start=0.1, beta_target=0.8, a_t=a, freeze_at=bad)


def test_freeze_contract_converges_to_target_even_with_summable_warmup() -> None:
    """Freeze-Vertrag (#51): der schlechteste Schedule (T0=1) erreicht pi_target exakt."""
    a = a_kernel.diminishing_step_sizes(400, 0.5, 1.0)
    r = mx.adaptive_exact_tv(_CFG, beta_start=0.1, beta_target=0.8, a_t=a, x0=0, freeze_at=30)
    assert r.beta_limit == 0.8
    assert r.frozen_floor == 0.0
    assert r.tv_to_target[-1] < 1e-10


def test_default_schedule_adaptive_chain_converges_to_target() -> None:
    a = a_kernel.diminishing_step_sizes(300, 0.5, 100.0)
    r = mx.adaptive_exact_tv(_CFG, beta_start=0.1, beta_target=0.8, a_t=a, x0=0, freeze_at=300)
    assert r.frozen_floor < 1e-12
    assert r.tv_to_target[-1] < 1e-10
    assert r.tv_to_target[0] > 0.8


def test_summable_schedule_freezes_adaptation_measurable_floor() -> None:
    """[LUECKE quantifiziert] OHNE Freeze (Defektklasse vor #51): beta_t -> beta_inf != target.

    Die Kette konvergiert exakt nach pi_{beta_inf}; der Abstand zu pi_target
    bleibt >= ||pi_{beta_inf} - pi_target||_TV.
    """
    c, T0 = 0.5, 1.0
    a = a_kernel.diminishing_step_sizes(400, c, T0)
    r = mx.adaptive_exact_tv(_CFG, beta_start=0.1, beta_target=0.8, a_t=a, x0=0, freeze_at=a.size)
    closed = 0.8 + (0.1 - 0.8) * float(np.prod(1 - a_kernel.diminishing_step_sizes(10**6, c, T0)))
    assert abs(r.beta_limit - closed) < 1e-3
    assert abs(r.beta_limit - 0.8) > 0.2
    assert r.frozen_floor > 0.05
    assert r.tv_to_limit[-1] < 1e-6
    assert r.tv_to_target[-1] >= r.frozen_floor - r.tv_to_limit[-1] - 1e-12


def test_containment_relaxation_time_bounded_on_theta_unbounded_toward_beta_c() -> None:
    """Codex P2 (cli.py:1557): die Schranke gilt fuer das GANZE Theta, nicht fuer 12 Punkte."""
    sup = mx.relaxation_time_sup_bound(L, _CFG.beta_min, _CFG.beta_max, step=0.01)
    assert math.isfinite(sup.upper_bound)
    assert sup.grid_max <= sup.upper_bound < 25.0
    far = mx.relaxation_time_profile(L, np.array([4.0, 6.0]))
    assert far[1] > 1e4
    assert far[1] > 30 * far[0]


@pytest.mark.parametrize("ring", [3, 4, 5, 6, 7, 8])
def test_divergence_lower_bound_holds_step_by_step_against_the_exact_kernel(ring: int) -> None:
    """Codex P2 (cli.py:1651): die Divergenz ist bewiesen, nicht aus t_rel(6) > 1e4 gefolgert.

    Unabhaengiges Orakel fuer JEDEN Beweisschritt aus dem exakten Kern: pi(S) <= 1/2,
    Phi(S) exakt aus Q(S,S^c) <= Phi_bar, Cheeger 1 - lambda_2 <= 2 Phi (Eigenwerte der
    symmetrisierten Matrix), und am Ende t_rel des Sweep-Kerns >= Schranke."""
    in_s = mx.all_states(ring).sum(axis=1) > ring / 2
    m = ring // 2 + 1
    for beta in (0.25, 1.0, 2.0, 3.0, 4.0, 6.0):
        P1 = mx.single_flip_kernel(ring, beta)
        pi = mx.stationary_distribution(ring, beta)
        assert pi[in_s].sum() <= 0.5 + 1e-15
        phi = float((pi[in_s, None] * P1[np.ix_(in_s, ~in_s)]).sum() / pi[in_s].sum())
        assert phi <= math.comb(ring, m) * (m / ring) * math.exp(-2 * beta) * (1 + 1e-12)
        A = np.sqrt(pi)[:, None] * P1 / np.sqrt(pi)[None, :]
        lam2 = np.sort(np.linalg.eigvalsh((A + A.T) / 2))[-2]
        assert 1 - lam2 <= 2 * phi + 1e-12
        t_rel = float(mx.relaxation_time_profile(ring, np.array([beta]))[0])
        assert t_rel >= mx.relaxation_time_divergence_lower_bound(ring, beta)


def test_divergence_lower_bound_is_unbounded_and_rejects_bad_input() -> None:
    lb = [mx.relaxation_time_divergence_lower_bound(L, b) for b in (0.0, 5.0, 10.0, 50.0)]
    assert all(a < b for a, b in zip(lb, lb[1:], strict=False))
    assert lb[0] == pytest.approx(1 / 120)  # L=6: 2 * C(6,4) * 4 = 120
    assert lb[-1] > 1e40
    assert mx.relaxation_time_divergence_lower_bound(L, 400.0) == math.inf
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            mx.relaxation_time_divergence_lower_bound(L, bad)
    with pytest.raises(ValueError):
        mx.relaxation_time_divergence_lower_bound(2, 1.0)


def test_t_rel_is_not_monotone_on_theta() -> None:
    """Warum ein Endpunkt-/Monotonie-Argument NICHT reicht: fuer kleine beta dominiert der
    Paritaets-Eigenwert nahe -1, t_rel faellt zuerst und steigt dann."""
    t = mx.relaxation_time_profile(L, np.array([0.1, 0.3, 2.0]))
    assert t[0] > t[1] < t[2]


def test_sup_bound_dominates_t_rel_between_grid_points() -> None:
    """Orakel: t_rel an Punkten ZWISCHEN den Gitterknoten (direkt aus dem Sweep-Kern)
    muss unter der Intervallschranke liegen. Auf [1.9, 2.0] steigt t_rel; eine Schranke,
    die nur die Knoten nimmt (delta=0), laege unter den Zwischenwerten."""
    sup = mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.05)
    assert sup.n_points == 3
    between = mx.relaxation_time_profile(L, np.array([1.91, 1.93, 1.96, 1.99, 2.0]))
    assert np.all(between <= sup.upper_bound)
    assert sup.grid_max == pytest.approx(float(between[-1]), rel=1e-12)  # Knoten 2.0
    # Konvergenz: feineres Gitter -> engere Schranke, nie unter dem Gittermaximum.
    fine = mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.005)
    assert sup.grid_max <= fine.upper_bound < sup.upper_bound


def _scaled_eigensolver(monkeypatch, factor: float, *, vectors=None) -> None:
    """Eigenloeser, der die Eigenwerte um `factor` skaliert (und ggf. falsche Vektoren gibt).

    Beide LAPACK-Einstiege werden ersetzt (eigh: neuer Pfad, eigvalsh: alter Pfad), damit
    derselbe Test den Vor-Fix-Stand an einer ZUSICHERUNG scheitern laesst.
    """
    eigh, eigvalsh = np.linalg.eigh, np.linalg.eigvalsh

    def fake_eigh(a):
        w, V = eigh(a)
        return w * factor, (V if vectors is None else vectors(V))

    monkeypatch.setattr(np.linalg, "eigh", fake_eigh)
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda a: eigvalsh(a) * factor)


def test_sup_bound_stays_an_upper_bound_with_an_inaccurate_eigensolver(monkeypatch) -> None:
    """Codex P2 (mixing.py:686): mit fester 1e-12-Polsterung macht ein Eigenloeser, der
    um 1e-3 relativ unterschaetzt, die 'rigorose' Schranke still FALSCH (sie faellt unter
    t_rel(2.0)). Mit a-posteriori-Einschliessung waechst der Radius, die Schranke bleibt
    eine Schranke. Orakel: t_rel am Knoten 2.0 aus spectral_summary VOR dem Patch."""
    true_t = float(mx.relaxation_time_profile(L, np.array([2.0]))[0])
    honest = mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.005)
    _scaled_eigensolver(monkeypatch, 1.0 - 1e-3)
    bad = mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.005)
    assert bad.upper_bound >= true_t
    assert bad.upper_bound >= honest.upper_bound
    assert bad.lower_bound <= true_t
    assert bad.eigen_radius_max > 1e-3 * 0.9  # der Defekt ist im Radius sichtbar


def test_sup_bound_fails_closed_for_a_garbage_eigensolver(monkeypatch) -> None:
    """Halbierte Eigenwerte bzw. Einheitsvektoren statt Eigenvektoren: Radius >= 1/2,
    mu_* + Radius >= 1 -> Schranke inf (fail-closed), statt einer erfundenen Zahl."""
    _scaled_eigensolver(monkeypatch, 0.5)
    assert mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.05).upper_bound == math.inf
    monkeypatch.undo()
    _scaled_eigensolver(monkeypatch, 1.0, vectors=lambda V: np.eye(V.shape[0]))
    assert mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.05).upper_bound == math.inf


def test_honest_sup_bound_reports_a_tiny_certified_radius() -> None:
    sup = mx.relaxation_time_sup_bound(L, 0.1, 2.0, step=0.01)
    assert 0.0 < sup.eigen_radius_max < 1e-10
    assert 0.0 <= sup.exp_rel_err_max < 1e-15  # np.exp hier auf ~1 ulp gemessen
    assert sup.lower_bound <= sup.grid_max <= sup.upper_bound
    assert sup.grid_max - sup.lower_bound < 1e-6


def test_eigen_enclosure_covers_known_eigenvalue_errors() -> None:
    """Einheitsorakel fuer _sym_eigen_enclosure: verschiebt man ehrliche Eigenwerte um
    bekannte delta_i, muss der Radius max|delta_i| decken (Restfehler von eigh ~1e-15)."""
    rng = np.random.default_rng(3)
    b = 1.3
    S = mx.sweep_kernel(L, b)  # nur fuer die Groesse; Struktur aus dem Einzelschritt:
    pi = mx.stationary_distribution(L, b)
    s = np.sqrt(pi)
    A = s[:, None] * mx.single_flip_kernel(L, b) / s[None, :]
    A = 0.5 * (A + A.T)
    n = S.shape[0]
    idx = np.arange(n)
    flips = [idx ^ (1 << i) for i in range(L)]
    diag = np.diagonal(A).copy()
    offs = [A[idx, j] for j in flips]
    w, V = np.linalg.eigh(A)
    honest = mx._sym_eigen_enclosure(w, V, diag, offs, flips)
    assert 0.0 < honest < 1e-11
    for scale in (1e-9, 1e-6, 1e-3):
        delta = scale * rng.uniform(-1.0, 1.0, size=n)
        r = mx._sym_eigen_enclosure(w + delta, V, diag, offs, flips)
        assert r >= float(np.max(np.abs(delta)))
    bad_V = V.copy()
    bad_V[:, 0] *= 3.0  # Orthogonalitaetsdefekt alpha >= 1/2 -> fail-closed
    assert mx._sym_eigen_enclosure(w, bad_V, diag, offs, flips) == math.inf
    assert mx._sym_eigen_enclosure(w * np.nan, V, diag, offs, flips) == math.inf


def test_certified_exp_measures_the_exp_error(monkeypatch) -> None:
    ks = np.array([0, 2, 4])
    vals, rho = mx._certified_exp(1.7, ks, 0.5)
    np.testing.assert_array_equal(vals, np.exp(-1.7 * (ks * 0.5)))
    assert rho < 1e-15
    monkeypatch.setattr(mx, "_exp", lambda x: np.exp(x) * (1.0 + 1e-9))
    _, rho_bad = mx._certified_exp(1.7, ks, 0.5)
    assert rho_bad >= 0.99e-9
    # Unterlauf (exp -> 0) ist kein Wert mit kleinem Fehler, sondern fail-closed.
    monkeypatch.undo()
    assert mx._certified_exp(1e6, np.array([4]), 1.0)[1] == math.inf
    with pytest.raises(ValueError, match="scale"):
        mx._certified_exp(1.0, ks, 2.0)


def test_sup_bound_rejects_a_construction_that_drifts_from_the_kernel(monkeypatch) -> None:
    """Ein exp, das um 1e-9 abweicht, laesst A_1 (geschlossen) und single_flip_kernel
    auseinanderlaufen -> lauter Fehler statt einer Schranke fuer die falsche Matrix."""
    monkeypatch.setattr(mx, "_exp", lambda x: np.exp(x) * (1.0 + 1e-9))
    with pytest.raises(ValueError, match="disagrees"):
        mx.relaxation_time_sup_bound(L, 1.9, 2.0, step=0.05)


@pytest.mark.parametrize(
    ("lo", "hi", "step"),
    [
        (0.5, 0.5, 0.1),
        (1.0, 0.5, 0.1),
        (0.1, 2.0, 0.0),
        (0.1, 2.0, -0.1),
        (0.1, 2.0, float("nan")),
        (0.1, float("nan"), 0.1),
        (0.1, 2.0, 1e-7),
    ],
)
def test_sup_bound_rejects_bad_interval(lo: float, hi: float, step: float) -> None:
    with pytest.raises(ValueError):
        mx.relaxation_time_sup_bound(L, lo, hi, step=step)


# --- Silent-Failure-Gate ------------------------------------------------------


@pytest.mark.parametrize("bad_L", [2, 13, True, 6.0])
def test_bad_L_rejected(bad_L) -> None:
    with pytest.raises(ValueError):
        mx.single_flip_kernel(bad_L, 0.5)


@pytest.mark.parametrize("bad_beta", [float("nan"), float("inf"), -0.1])
def test_bad_beta_rejected(bad_beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        mx.stationary_distribution(L, bad_beta)


def test_non_reversible_kernel_rejected() -> None:
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="reversible"):
        mx.spectral_summary(P, np.full(3, 1 / 3))


def test_non_stochastic_or_nan_kernel_rejected() -> None:
    pi = np.full(2, 0.5)
    with pytest.raises(ValueError, match="stochastic"):
        mx.spectral_summary(np.array([[0.5, 0.6], [0.5, 0.5]]), pi)
    with pytest.raises(ValueError, match="finite"):
        mx.spectral_summary(np.array([[np.nan, 0.5], [0.5, 0.5]]), pi)


def test_fit_rate_needs_points_above_floor() -> None:
    with pytest.raises(ValueError, match="fewer than 3"):
        mx.fit_geometric_rate(np.array([1.0, 1e-3, 1e-20, 1e-30]), floor=1e-2)
    with pytest.raises(ValueError, match="finite"):
        mx.fit_geometric_rate(np.array([1.0, np.nan, 0.1, 0.01]), floor=1e-3)


def test_empirical_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        mx.empirical_tv_curve(L, 0.0, x0=0, n_chains=10, n_sweeps=2, seed=1)
    with pytest.raises(ValueError):
        mx.empirical_tv_curve(L, 0.5, x0=2**L, n_chains=10, n_sweeps=2, seed=1)
    with pytest.raises(ValueError):
        mx.empirical_tv_curve(L, 0.5, x0=0, n_chains=10, n_sweeps=2, seed=-1)


def test_adaptive_rejects_target_outside_theta() -> None:
    with pytest.raises(ValueError, match="Containment"):
        mx.adaptive_exact_tv(
            _CFG, beta_start=0.1, beta_target=3.0, a_t=np.ones(3), x0=0, freeze_at=3
        )


# --- finite-family error budget for G47 ---------------------------------------


def test_family_band_is_pointwise_band_with_allocated_error() -> None:
    P = mx.sweep_kernel(L, 0.8)
    family = mx.empirical_tv_family_band(P, 0, 12, 4000, 1e-6)
    allocated = mx.empirical_tv_band(P, 0, 12, 4000, 1e-6 / 13)
    np.testing.assert_array_equal(family, allocated)


@pytest.mark.parametrize("bad", [0.0, 1.0, -1e-6, float("nan")])
def test_family_band_rejects_invalid_family_delta(bad: float) -> None:
    with pytest.raises(ValueError, match="family_delta"):
        mx.empirical_tv_family_band(mx.sweep_kernel(L, 0.8), 0, 12, 4000, bad)


def test_familywise_sampler_gate_discriminates_wrong_kernel() -> None:
    emp = mx.empirical_tv_curve(L, 0.8, x0=0, n_chains=4000, n_sweeps=12, seed=7)
    assert mx.marginal_family_band_ratio(emp, mx.sweep_kernel(L, 0.8), family_delta=1e-6) <= 1.0
    assert mx.marginal_family_band_ratio(emp, mx.sweep_kernel(L, 1.2), family_delta=1e-6) > 2.0


def test_family_band_is_wider_than_pointwise_band() -> None:
    """Bonferroni darf das Band nur VERBREITERN (delta/K < delta), nie verengen."""
    P = mx.sweep_kernel(L, 0.8)
    family = mx.empirical_tv_family_band(P, 0, 12, 4000, 1e-6)
    pointwise = mx.empirical_tv_band(P, 0, 12, 4000, 1e-6)
    assert np.all(family > pointwise)


@pytest.mark.parametrize("tiny", [1e-300, 1e-309, 1e-320, 5e-324])
def test_band_stays_finite_for_subnormal_delta(tiny: float) -> None:
    """ln(2/delta) lief fuer delta < ~1.1e-308 nach inf -> Band inf -> jeder Kern 'vertraeglich'."""
    band = mx.empirical_tv_band(mx.sweep_kernel(L, 0.8), 0, 3, 4000, tiny)
    assert np.all(np.isfinite(band))
    expected_mcd = math.sqrt((math.log(2.0) - math.log(tiny)) / (2.0 * 4000))
    assert band[0] == pytest.approx(expected_mcd, rel=1e-12)  # t=0: Punktmasse, Jensen-Term 0


# --- Codex-P2-Runde PR #49 @b2e2341 -------------------------------------------


def test_reversibility_check_is_scale_aware_for_rare_states() -> None:
    """Codex P2 (mixing.py:236): bei beta=6 ist pi_min ~ 1e-16. Ein O(1)-Fehler der
    BEDINGTEN Uebergangswkt eines seltenen Zustands (Haelfte von P[r, j] auf die Diagonale)
    bleibt im absoluten Fluss unter 1e-12 und wurde als reversibel akzeptiert."""
    beta = 6.0
    pi = mx.stationary_distribution(L, beta)
    P = mx.sweep_kernel(L, beta).copy()
    r = int(np.argmin(pi))
    off = P[r].copy()
    off[r] = 0.0
    j = int(np.argmax(off))
    assert pi[r] < 1e-15 and P[r, j] > 0.05  # wirklich selten, wirklich O(1)
    P[r, r] += 0.5 * P[r, j]
    P[r, j] *= 0.5
    flow = pi[:, None] * P
    assert np.max(np.abs(flow - flow.T)) < 1e-12  # die alte Fluss-Schwelle sieht es nicht
    with pytest.raises(ValueError, match="reversible"):
        mx.spectral_summary(P, pi)


@pytest.mark.parametrize("beta", [6.0, 10.0, 20.0])
def test_scale_aware_check_accepts_genuine_high_beta_kernels(beta: float) -> None:
    """Gegenprobe: die strengere Pruefung darf echte Kerne mit pi_min bis ~1e-53 nicht verwerfen."""
    pi = mx.stationary_distribution(L, beta)
    for P in (mx.single_flip_kernel(L, beta), mx.sweep_kernel(L, beta)):
        spec = mx.spectral_summary(P, pi)
        assert spec.similarity_asymmetry <= 1e-15
        assert 0.0 < spec.lambda_star < 1.0


_BAD_BETAS = [float("nan"), float("inf"), -float("inf"), 2.0 + 1e-9, 0.1 - 1e-9, -0.5, True]


@pytest.mark.parametrize("bad", _BAD_BETAS)
@pytest.mark.parametrize("which", ["beta_start", "beta_target"])
def test_schedule_helper_rejects_beta_outside_theta(which: str, bad) -> None:
    """Codex P2 (mixing.py:466): der exportierte Helfer lief mit Zielen, die
    a_kernel.advance_chain vor jedem Sweep ablehnt (NaN-Trajektorie, Zuweisung ausserhalb Theta)."""
    kw = {"beta_start": 0.1, "beta_target": 0.8, which: bad}
    for freeze_at in (0, 3, 5):
        with pytest.raises(ValueError, match="Theta"):
            mx.adaptive_beta_schedule(_CFG, a_t=np.full(5, 0.1), freeze_at=freeze_at, **kw)


def test_schedule_helper_accepts_theta_endpoints() -> None:
    a = np.full(4, 0.5)
    lo = mx.adaptive_beta_schedule(_CFG, beta_start=0.1, beta_target=2.0, a_t=a, freeze_at=2)
    assert lo[-1] == 2.0 and np.all(np.isfinite(lo))


@pytest.mark.parametrize("bad_x0", [-1, -64, 64, 1.5, 0.0, True, np.float64(3.0)])
def test_x0_rejected_at_every_public_entry_point(bad_x0) -> None:
    """Codex P2 (mixing.py:524): x0=-1 analysierte still den Zustand 2^L-1, 1.5 wurde zu 1."""
    P = mx.sweep_kernel(L, 0.8)
    pi = mx.stationary_distribution(L, 0.8)
    with pytest.raises(ValueError, match="x0"):
        mx.adaptive_exact_tv(
            _CFG, beta_start=0.1, beta_target=0.8, a_t=np.full(3, 0.1), x0=bad_x0, freeze_at=1
        )
    with pytest.raises(ValueError, match="x0"):
        mx.exact_tv_curve(P, pi, bad_x0, 3)
    with pytest.raises(ValueError, match="x0"):
        mx.empirical_tv_band(P, bad_x0, 3, 100, 1e-6)
    with pytest.raises(ValueError, match="x0"):
        mx.empirical_tv_curve(L, 0.8, x0=bad_x0, n_chains=2, n_sweeps=1, seed=1)


def test_x0_numpy_integer_still_accepted() -> None:
    r = mx.adaptive_exact_tv(
        _CFG, beta_start=0.1, beta_target=0.8, a_t=np.full(3, 0.1), x0=np.int64(63), freeze_at=1
    )
    ref = mx.adaptive_exact_tv(
        _CFG, beta_start=0.1, beta_target=0.8, a_t=np.full(3, 0.1), x0=63, freeze_at=1
    )
    np.testing.assert_array_equal(r.tv_to_target, ref.tv_to_target)
