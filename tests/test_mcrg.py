"""Phase-2 Swendsen-MCRG-Schaetzer: Korrektheit gegen unabhaengige Orakel.

Orakel 1 (exakt): R'(K) = tanh(2K) (Decimation-RG, rg_map.py).
Orakel 2 (exakt): <s_i s_{i+1}> = tanh K (Transfer-Matrix, offene Kette).
Die Swendsen-Identitaet T-hat = <S'S>_c/<S'S'>_c gilt im exakten Ensemble bis
Maschinengenauigkeit (Enumeration) und im MCMC-Schaetzer bis O(1/sqrt(N)).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptiverg_qec import mcrg, rg_map


def _enumerate_T_exact(K: float, L: int, *, periodic: bool) -> float:
    """T-hat im EXAKTEN Ensemble via Voll-Enumeration (unabh. Orakel, klein-L)."""
    states = np.arange(1 << L, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(L)[None, :]) & 1).astype(np.int8)
    s = mcrg.spins_from_bits(bits)
    Sfull = mcrg.nn_operator(s, periodic=periodic)
    Sp = mcrg.nn_operator(mcrg.decimate_b2(s), periodic=periodic)
    w = np.exp(K * Sfull)
    w /= w.sum()
    e_s, e_sp = float((w * Sfull).sum()), float((w * Sp).sum())
    cov_sps = float((w * Sp * Sfull).sum()) - e_sp * e_s
    cov_spsp = float((w * Sp * Sp).sum()) - e_sp * e_sp
    return cov_sps / cov_spsp


@pytest.mark.parametrize("K", [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("periodic", [False, True])
def test_swendsen_identity_exact_ensemble(K: float, periodic: bool) -> None:
    """Im exakten Ensemble: T-hat == tanh(2K) bis Maschinengenauigkeit (L=10)."""
    T_exact = _enumerate_T_exact(K, 10, periodic=periodic)
    assert math.isclose(T_exact, math.tanh(2.0 * K), abs_tol=1e-12)


def test_swendsen_matches_analytic_derivative() -> None:
    """Konsistenz mit rg_map.rg_derivative_analytic (gleiches Orakel)."""
    for K in (0.4, 0.8):
        T_exact = _enumerate_T_exact(K, 8, periodic=False)
        assert math.isclose(T_exact, rg_map.rg_derivative_analytic(K), abs_tol=1e-12)


def test_sampler_reproduces_nn_correlation() -> None:
    """Exakter Sampler: <s_i s_{i+1}> -> tanh K (unabhaengiges Orakel)."""
    K = 0.7
    s = mcrg.sample_ising_open_chain(K, 64, 50000, seed=0)
    nn = float(np.mean(s[:, :-1] * s[:, 1:]))
    assert abs(nn - math.tanh(K)) < 0.01


def test_mc_swendsen_within_3sigma() -> None:
    """MC-Schaetzer trifft tanh(2K) innerhalb 3 Multi-Seed-sigma fuer alle K."""
    rows = mcrg.validate_swendsen(
        K_values=(0.3, 0.5, 0.7, 0.9), L=64, n_samples=3000, seeds=tuple(range(8))
    )
    for r in rows:
        assert r.n_sigma <= 3.0, f"K={r.K}: {r.n_sigma:.2f} sigma off oracle"


def test_bias_decreases_with_N() -> None:
    """Multi-Seed-Streuung faellt mit N (Richtung 1/sqrt(N))."""
    K, L, seeds = 0.6, 64, tuple(range(8))

    def spread(n: int) -> float:
        vals = np.array(
            [
                mcrg.swendsen_T_scalar(
                    mcrg.sample_ising_open_chain(K, L, n, seed=mcrg.K_seed(K, sd))
                ).T_hat
                for sd in seeds
            ]
        )
        return float(vals.std(ddof=1))

    assert spread(8000) < spread(1000)


def test_reproducible_sample_and_estimate() -> None:
    """Gleicher Seed -> bit-identische Stichprobe + identischer T-hat."""
    a = mcrg.sample_ising_open_chain(0.5, 32, 2000, seed=7)
    b = mcrg.sample_ising_open_chain(0.5, 32, 2000, seed=7)
    assert np.array_equal(a, b)
    assert mcrg.swendsen_T_scalar(a).T_hat == mcrg.swendsen_T_scalar(b).T_hat


def test_connected_correlation_against_numpy_cov() -> None:
    """connected_correlation == np.cov (Plug-in, ddof=0) bis eps."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    b = 0.5 * a + rng.normal(size=500)
    ours = mcrg.connected_correlation(a, b)
    ref = float(np.cov(a, b, ddof=0)[0, 1])
    assert math.isclose(ours, ref, abs_tol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: mcrg.sample_ising_open_chain(0.0, 16, 100, seed=1),  # K<=0
        lambda: mcrg.sample_ising_open_chain(float("inf"), 16, 100, seed=1),  # K not finite
        lambda: mcrg.sample_ising_open_chain(0.5, 1, 100, seed=1),  # L<2
        lambda: mcrg.sample_ising_open_chain(0.5, 16, 1, seed=1),  # n_samples<2
        lambda: mcrg.swendsen_T_scalar(np.ones((10, 2))),  # L<4 after decimation
        lambda: mcrg.swendsen_T_scalar(np.ones(10)),  # not 2D
        lambda: mcrg.spins_from_bits(np.array([0, 2, 1])),  # x not in {0,1}
        lambda: mcrg.connected_correlation(np.array([1.0]), np.array([1.0])),  # <2 samples
        lambda: mcrg.validate_swendsen(seeds=(0,)),  # <2 seeds -> no error bar
        lambda: mcrg.nn_operator(np.ones((5, 1))),  # L<2 operator
    ],
)
def test_edge_inputs_rejected(bad) -> None:
    """Silent-Failure-Gate: invalide Eingaben werfen laut (kein stiller Garbage)."""
    with pytest.raises((ValueError, TypeError)):
        bad()


def test_degenerate_zero_fluctuation_rejected() -> None:
    """Alle Spins gleich -> <S'S'>_c == 0 -> sauberer Fehler statt 0/0."""
    s = np.ones((20, 8))  # konstante Konfiguration, keine Fluktuation
    with pytest.raises(ValueError):
        mcrg.swendsen_T_scalar(s)
