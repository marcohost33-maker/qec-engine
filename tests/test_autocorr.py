"""Tests fuer die autokorrelations-bewusste Fehler-Statistik (Phase-3a).

Orakel: AR(1)-Prozess mit GESCHLOSSENER Autokorrelation rho(t)=phi^t,
tau_int = 1/2 + phi/(1-phi). Unabhaengig von der Implementierung.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import autocorr, mcrg
from adaptiverg_qec.a_kernel import run_adaptive_mcmc
from adaptiverg_qec.mvp_instance import MVPConfig


def ar1(n: int, phi: float, seed: int) -> np.ndarray:
    """AR(1)-Prozess mit stationaerer Anfangsbedingung (Var=1)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8, 0.9])
def test_tau_int_matches_ar1_oracle(phi: float) -> None:
    tau_true = 0.5 + phi / (1.0 - phi)
    taus = [autocorr.integrated_autocorr_time(ar1(40000, phi, sd)).tau_int for sd in range(10)]
    tau_hat = float(np.mean(taus))
    assert abs(tau_hat - tau_true) / tau_true < 0.08, (tau_hat, tau_true)


def test_rho_zero_is_one_and_decays() -> None:
    rho = autocorr.autocorr_function_fft(ar1(20000, 0.7, 1))
    assert abs(rho[0] - 1.0) < 1e-12
    # AR(1): rho(1) ~ phi = 0.7 (Stichproben-Streuung erlaubt).
    assert 0.6 < rho[1] < 0.8


def test_iid_limit_no_inflation() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30000)
    r = autocorr.integrated_autocorr_time(x)
    assert abs(r.tau_int - 0.5) < 0.05
    assert abs(r.inflation - 1.0) < 0.05
    assert r.n_eff > 0.95 * r.n


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_gamma_method_equals_binning_plateau(phi: float) -> None:
    """Rigorose Validierung: Gamma-sem == Binning-Plateau-sem (Wolff)."""
    ratios = []
    for sd in range(6):
        x = ar1(40000, phi, sd)
        g = autocorr.integrated_autocorr_time(x)
        b = autocorr.binning_error(x)
        ratios.append(b.sem_plateau / g.sem)
    assert 0.85 < float(np.mean(ratios)) < 1.15


def test_jackknife_ratio_recovers_known_ratio() -> None:
    """Jackknife auf ein einfaches Verhaeltnis <Y>/<Z> (i.i.d.) -> korrektes Mittel + Fehler."""
    rng = np.random.default_rng(42)
    n = 4000
    y = 2.0 + rng.standard_normal(n) * 0.5
    z = 1.0 + rng.standard_normal(n) * 0.1
    jk = autocorr.jackknife_ratio(
        y[:, None], z[:, None], block_size=20, combine=lambda a, b: float(a[0] / b[0])
    )
    true_ratio = float(y.mean() / z.mean())
    assert abs(jk.ratio - true_ratio) < 1e-9
    # 2.0/1.0 ~ 2.0; Fehler sollte positiv + klein sein.
    assert jk.error > 0
    assert abs(jk.ratio - 2.0) < 5 * jk.error


def test_jackknife_block_error_grows_with_correlation() -> None:
    """Bei korreliertem Zaehler waechst der Block-Jackknife-Fehler ggue. b=1."""
    phi = 0.85
    yz = ar1(20000, phi, 3) + 5.0  # positiver, korrelierter Zaehler
    z = np.ones_like(yz)
    num = (yz)[:, None]
    den = z[:, None]
    e1 = autocorr.jackknife_ratio(
        num, den, block_size=1, combine=lambda a, b: float(a[0] / b[0])
    ).error
    eb = autocorr.jackknife_ratio(
        num, den, block_size=40, combine=lambda a, b: float(a[0] / b[0])
    ).error
    assert eb > e1  # Block-Fehler groesser als naiv-i.i.d.


def test_akernel_swendsen_hits_oracle_within_correlated_error() -> None:
    """T_hat aus dem korrelierten A-Kernel trifft tanh(2K) (K<=0.7, finite-L sauber)."""
    for K in (0.3, 0.5, 0.7):
        est = mcrg.validate_swendsen_akernel(
            K_values=(K,), L=64, n_steps=10000, burn_in=1500, seed=0
        )[0]
        assert est.n_sigma <= 3.5, (K, est.T_hat, est.oracle, est.n_sigma)


def test_akernel_neff_less_than_n() -> None:
    """Korrelierte Kette: N_eff < N und tau_int > 0.5 (Phase-3a-Kernpunkt)."""
    est = mcrg.validate_swendsen_akernel(
        K_values=(0.9,), L=64, n_steps=10000, burn_in=1500, seed=2
    )[0]
    tau_max = max(est.tau_int_S, est.tau_int_Sp, est.tau_int_product)
    assert tau_max > 0.5
    assert est.n_eff < est.n_samples
    assert est.error_correlated >= est.error_iid_naive


def test_akernel_error_inflation_at_high_K() -> None:
    """Bei groesserem K (langsameres Mischen) ist err_correlated > err_iid."""
    est = mcrg.validate_swendsen_akernel(
        K_values=(0.9,), L=64, n_steps=12000, burn_in=2000, seed=4
    )[0]
    assert est.error_inflation > 1.0


def test_operator_timeseries_shape_and_bc() -> None:
    cfg = MVPConfig(L=64, beta_min=0.1, beta_max=2.0)
    res = run_adaptive_mcmc(
        cfg, beta_target=1.0, n_steps=500, burn_in=0, seed=1, beta_start=1.0, record_configs=True
    )
    assert res.configs is not None
    assert res.configs.shape == (500, 64)
    S, Sp = mcrg.operator_timeseries_from_configs(res.configs, periodic=True)
    assert S.shape == (500,)
    assert Sp.shape == (500,)
    # periodischer NN-Operator in [-L, L].
    assert np.all(np.abs(S) <= 64)
    assert np.all(np.abs(Sp) <= 32)


def test_reproducibility_seed() -> None:
    a = mcrg.validate_swendsen_akernel(K_values=(0.6,), L=64, n_steps=6000, burn_in=1000, seed=9)
    b = mcrg.validate_swendsen_akernel(K_values=(0.6,), L=64, n_steps=6000, burn_in=1000, seed=9)
    assert a[0].T_hat == b[0].T_hat
    assert a[0].error_correlated == b[0].error_correlated


def test_record_configs_default_off() -> None:
    """Default record_configs=False -> configs None (Speicher-Ersparnis)."""
    cfg = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)
    res = run_adaptive_mcmc(cfg, beta_target=0.8, n_steps=200, burn_in=0, seed=1, beta_start=0.8)
    assert res.configs is None


@pytest.mark.parametrize(
    "bad",
    [
        lambda: autocorr.autocorr_function_fft(np.array([1.0])),
        lambda: autocorr.autocorr_function_fft(np.full(50, 2.0)),
        lambda: autocorr.integrated_autocorr_time(np.arange(50.0), c_window=-1.0),
        lambda: autocorr.binning_error(np.arange(10.0)),
        lambda: autocorr.jackknife_ratio(
            np.ones((50, 1)), np.ones((50, 1)), block_size=0, combine=lambda a, b: 1.0
        ),
        lambda: autocorr.jackknife_ratio(
            np.ones((50, 1)), np.ones((50, 1)), block_size=50, combine=lambda a, b: 1.0
        ),
        lambda: mcrg.swendsen_T_from_chain(np.arange(10.0), np.arange(10.0), K=0.5),
        lambda: mcrg.operator_timeseries_from_configs(np.zeros((50, 2), dtype=np.int8)),
    ],
)
def test_invalid_inputs_raise(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        bad()
