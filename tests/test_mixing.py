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


def test_empirical_tv_curve_reproducible_and_seed_sensitive() -> None:
    a = mx.empirical_tv_curve(L, 0.8, x0=5, n_chains=200, n_sweeps=4, seed=11)
    b = mx.empirical_tv_curve(L, 0.8, x0=5, n_chains=200, n_sweeps=4, seed=11)
    c = mx.empirical_tv_curve(L, 0.8, x0=5, n_chains=200, n_sweeps=4, seed=12)
    assert np.array_equal(a.histograms, b.histograms)
    assert not np.array_equal(a.histograms, c.histograms)


# --- Phase-2-Akzeptanz: adaptive Kette + Containment --------------------------

_CFG = MVPConfig(L=L, beta_min=0.1, beta_max=2.0)


def test_adaptive_schedule_is_bit_identical_to_a_kernel() -> None:
    a = a_kernel.diminishing_step_sizes(80, 0.5, 1.0)
    run = a_kernel.run_adaptive_mcmc(
        _CFG, beta_target=0.8, n_steps=80, burn_in=0, seed=1, adapt_c=0.5, adapt_T0=1.0
    )
    sched = mx.adaptive_beta_schedule(_CFG, beta_start=0.1, beta_target=0.8, a_t=a)
    assert np.array_equal(run.beta_traj, sched)


def test_default_schedule_adaptive_chain_converges_to_target() -> None:
    a = a_kernel.diminishing_step_sizes(300, 0.5, 100.0)
    r = mx.adaptive_exact_tv(_CFG, beta_start=0.1, beta_target=0.8, a_t=a, x0=0)
    assert r.frozen_floor < 1e-12
    assert r.tv_to_target[-1] < 1e-10
    assert r.tv_to_target[0] > 0.8


def test_summable_schedule_freezes_adaptation_measurable_floor() -> None:
    """[LUECKE quantifiziert] sum a_t < inf mit kleinem T0: beta_t -> beta_inf != target.

    Die Kette konvergiert exakt nach pi_{beta_inf}; der Abstand zu pi_target
    bleibt >= ||pi_{beta_inf} - pi_target||_TV.
    """
    c, T0 = 0.5, 1.0
    a = a_kernel.diminishing_step_sizes(400, c, T0)
    r = mx.adaptive_exact_tv(_CFG, beta_start=0.1, beta_target=0.8, a_t=a, x0=0)
    closed = 0.8 + (0.1 - 0.8) * float(np.prod(1 - a_kernel.diminishing_step_sizes(10**6, c, T0)))
    assert abs(r.beta_limit - closed) < 1e-3
    assert abs(r.beta_limit - 0.8) > 0.2
    assert r.frozen_floor > 0.05
    assert r.tv_to_limit[-1] < 1e-6
    assert r.tv_to_target[-1] >= r.frozen_floor - r.tv_to_limit[-1] - 1e-12


def test_containment_relaxation_time_bounded_on_theta_unbounded_toward_beta_c() -> None:
    grid = np.linspace(_CFG.beta_min, _CFG.beta_max, 12)
    t_rel = mx.relaxation_time_profile(L, grid)
    assert np.all(np.isfinite(t_rel))
    assert float(np.max(t_rel)) < 25.0
    far = mx.relaxation_time_profile(L, np.array([4.0, 6.0]))
    assert far[1] > 1e4
    assert far[1] > 30 * far[0]


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
        mx.adaptive_exact_tv(_CFG, beta_start=0.1, beta_target=3.0, a_t=np.ones(3), x0=0)
