"""Tests fuer snis.py: SNIS gegen geschlossene chi^2/ESS/Bias-Orakel (Phase-6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptiverg_qec import snis
from adaptiverg_qec.mcrg import sample_ising_open_chain


def _enumerate_open_chain(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Alle 2^L Spin-Konfigurationen der offenen Kette + ihre S-Werte (Orakel)."""
    states = np.arange(1 << L, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(L)[None, :]) & 1).astype(np.int8)
    s = 1.0 - 2.0 * bits.astype(np.float64)
    S = (s[:, :-1] * s[:, 1:]).sum(axis=1)
    return s, S


class TestClosedFormOracles:
    def test_log_partition_vs_enumeration(self):
        L = 8
        _, S = _enumerate_open_chain(L)
        for K in (0.2, 0.5, 0.9):
            exact = math.log(float(np.sum(np.exp(K * S))))
            assert snis.log_partition_open_chain(K, L) == pytest.approx(exact, rel=1e-12)

    def test_mean_bond_correlation_vs_enumeration(self):
        L = 8
        _, S = _enumerate_open_chain(L)
        K = 0.6
        w = np.exp(K * S)
        w /= w.sum()
        g_exact = float(np.sum(w * S)) / (L - 1)
        assert snis.mean_bond_correlation(K) == pytest.approx(g_exact, rel=1e-12)

    def test_chi2_vs_enumeration(self):
        """chi^2-Formel gegen Brute-Force-Enumeration (unabhaengiges Orakel)."""
        L = 8
        _, S = _enumerate_open_chain(L)
        Kp, Kt = 0.3, 0.7
        wp = np.exp(Kp * S)
        wp /= wp.sum()
        wt = np.exp(Kt * S)
        wt /= wt.sum()
        chi2_enum = float(np.sum(wt**2 / wp)) - 1.0
        assert snis.chi2_divergence_open_chain(Kt, Kp, L) == pytest.approx(chi2_enum, rel=1e-10)

    def test_chi2_overflow_returns_inf_not_raise(self):
        """Codex-Fix: extremer Mismatch -> chi2=inf (ESS-Kollaps), kein OverflowError."""
        chi2 = snis.chi2_divergence_open_chain(1.2, 0.1, 2048)
        assert math.isinf(chi2) and chi2 > 0

    def test_chi2_zero_iff_equal_couplings(self):
        assert snis.chi2_divergence_open_chain(0.5, 0.5, 16) == pytest.approx(0.0, abs=1e-14)
        assert snis.chi2_divergence_open_chain(0.6, 0.5, 16) > 0.0
        assert snis.chi2_divergence_open_chain(0.4, 0.5, 16) > 0.0

    def test_bias_coefficient_sign(self):
        # K_t > K_p: 2K_t-K_p > K_t -> tanh waechst -> Koeffizient negativ.
        assert snis.snis_bias_coefficient(0.6, 0.3, 16) < 0.0
        # K_t < K_p: umgekehrtes Vorzeichen.
        assert snis.snis_bias_coefficient(0.3, 0.6, 16) > 0.0


class TestSnisReweight:
    def test_identity_reweighting_is_plain_mean(self):
        s = sample_ising_open_chain(0.5, 16, 4000, seed=1)
        e = snis.snis_reweight(s, K_proposal=0.5, K_target=0.5)
        g_plain = float(np.mean((s[:, :-1] * s[:, 1:]).sum(axis=1)) / 15)
        assert e.g_hat == pytest.approx(g_plain, rel=1e-12)
        assert e.ess == pytest.approx(4000.0, rel=1e-12)
        assert e.chi2_oracle == pytest.approx(0.0, abs=1e-14)

    def test_hits_oracle_within_4_sigma(self):
        e = snis.snis_from_couplings(K_proposal=0.3, K_target=0.5, L=16, n_samples=20000, seed=42)
        assert e.ess_adequate
        assert e.n_sigma <= 4.0
        assert abs(e.ess_rel - e.ess_rel_oracle) < 0.05

    def test_ess_collapse_flagged(self):
        e = snis.snis_from_couplings(K_proposal=0.1, K_target=1.2, L=32, n_samples=2000, seed=5)
        assert not e.ess_adequate
        assert e.ess < e.min_ess

    def test_overflow_regime_flags_collapse_instead_of_raising(self):
        """Codex-Fix: chi2=inf-Regime laeuft durch und meldet ess_adequate=False."""
        s = sample_ising_open_chain(0.1, 2048, 64, seed=3)
        e = snis.snis_reweight(s, K_proposal=0.1, K_target=1.2)
        assert math.isinf(e.chi2_oracle)
        assert e.ess_rel_oracle == 0.0
        assert not e.ess_adequate

    def test_degenerate_sample_miss_is_infinite_sigma(self):
        """Codex-Fix: error==0 bei abs_error>0 -> n_sigma=inf (kein 'Treffer')."""
        e = snis.snis_reweight(np.ones((10, 8)), K_proposal=0.3, K_target=0.4)
        assert e.error == 0.0 and e.abs_error > 0.0
        assert math.isinf(e.n_sigma)

    def test_rejects_non_spin_alphabet(self):
        """Codex-Fix: {0,1}-Bits (A-Kernel-Konvention) werden LAUT abgelehnt."""
        with pytest.raises(ValueError):
            snis.snis_reweight(np.zeros((10, 8)), K_proposal=0.3, K_target=0.4)
        bits01 = np.random.default_rng(0).integers(0, 2, size=(10, 8)).astype(float)
        with pytest.raises(ValueError):
            snis.snis_reweight(bits01, K_proposal=0.3, K_target=0.4)

    def test_reproducible(self):
        a = snis.snis_from_couplings(K_proposal=0.4, K_target=0.5, L=16, n_samples=3000, seed=7)
        b = snis.snis_from_couplings(K_proposal=0.4, K_target=0.5, L=16, n_samples=3000, seed=7)
        c = snis.snis_from_couplings(K_proposal=0.4, K_target=0.5, L=16, n_samples=3000, seed=8)
        assert a.g_hat == b.g_hat and a.error == b.error
        assert a.g_hat != c.g_hat


class TestBiasScaling:
    def test_bias_sign_and_decrease(self):
        rep = snis.measure_bias_scaling(n_values=(100, 400), n_replicates=(3000, 3000), seed=11)
        assert all(np.sign(b) == np.sign(rep.bias_coefficient_oracle) for b in rep.bias_hat)
        assert abs(rep.bias_hat[1]) < abs(rep.bias_hat[0])
        # MSE-Bound (Agapiou) haelt.
        assert all(m <= b for m, b in zip(rep.mse_hat, rep.mse_bound, strict=True))

    def test_scaled_bias_near_one(self):
        rep = snis.measure_bias_scaling(n_values=(100,), n_replicates=(6000,), seed=3)
        assert 0.5 <= rep.scaled_bias[0] <= 1.5

    def test_chunk_independent_results(self):
        """Codex-Fix: chunk ist reines Speicher-Tuning -- Ergebnisse byte-identisch."""
        kw = dict(n_values=(50,), n_replicates=(200,), seed=17)
        a = snis.measure_bias_scaling(chunk=200, **kw)
        b = snis.measure_bias_scaling(chunk=37, **kw)
        c = snis.measure_bias_scaling(chunk=1, **kw)
        assert a.bias_hat == b.bias_hat == c.bias_hat
        assert a.mse_hat == b.mse_hat == c.mse_hat
        assert a.var_deltamethod_mean == b.var_deltamethod_mean == c.var_deltamethod_mean


class TestEdgeInputs:
    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            snis.snis_reweight(np.ones(10), K_proposal=0.3, K_target=0.4)
        with pytest.raises(ValueError):
            snis.snis_reweight(np.ones((1, 8)), K_proposal=0.3, K_target=0.4)
        with pytest.raises(ValueError):
            snis.snis_reweight(np.ones((10, 1)), K_proposal=0.3, K_target=0.4)
        with pytest.raises(ValueError):
            snis.snis_reweight(np.ones((10, 8)), K_proposal=float("nan"), K_target=0.4)
        with pytest.raises(ValueError):
            snis.snis_reweight(np.ones((10, 8)), K_proposal=0.3, K_target=0.4, min_ess=-1.0)
        with pytest.raises(ValueError):
            snis.chi2_divergence_open_chain(0.4, 0.3, 1)
        with pytest.raises(ValueError):
            snis.measure_bias_scaling(n_values=(100,), n_replicates=(10, 10))
        with pytest.raises(ValueError):
            snis.measure_bias_scaling(n_values=(), n_replicates=())
        with pytest.raises(ValueError):
            snis.measure_bias_scaling(n_values=(1,), n_replicates=(10,))
        with pytest.raises(ValueError):
            snis.measure_bias_scaling(chunk=0)
