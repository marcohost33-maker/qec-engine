"""Tests fuer surrogate.py: Delayed-Acceptance + Surrogate-Drift-Guard (Phase-6)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import a_kernel, ising1d, surrogate
from adaptiverg_qec.mvp_instance import MVPConfig

CFG = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)


class TestExactness:
    def test_gamma_zero_bit_identical_to_metropolis(self):
        """Perfektes Surrogat: DA-Kernel == Metropolis-Kernel, bit-genau."""
        beta = 0.9
        a = a_kernel.run_adaptive_mcmc(
            CFG, beta_target=beta, n_steps=1200, burn_in=200, seed=31, beta_start=beta
        )
        da = surrogate.run_da_mcmc(CFG, beta=beta, n_steps=1200, burn_in=200, seed=31, gamma=0.0)
        assert np.array_equal(a.H_traj, da.H_traj)
        assert np.array_equal(a.final_state, da.final_state)
        assert a.mean_H == da.mean_H

    def test_miscalibrated_surrogate_still_exact(self):
        """Christen-Fox: auch ein SCHLECHTES Surrogat aendert die Stationaritaet nicht."""
        beta = 0.8
        exact = ising1d.mean_energy(beta, CFG.L)
        for gamma in (-0.25, 0.3):
            da = surrogate.run_da_mcmc(
                CFG, beta=beta, n_steps=6000, burn_in=1500, seed=23, gamma=gamma
            )
            assert abs(da.mean_H - exact) < 0.25, f"gamma={gamma}: {da.mean_H} vs {exact}"


class TestEfficiencyAccounting:
    def test_exact_evals_equal_stage1_accepts(self):
        da = surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=500, burn_in=0, seed=3, gamma=0.2)
        assert da.n_attempts == 500 * CFG.L
        assert da.n_exact_evals == round(da.stage1_accept_rate * da.n_attempts)
        assert 0 < da.n_exact_evals < da.n_attempts
        assert da.exact_eval_savings == pytest.approx(1.0 - da.n_exact_evals / da.n_attempts)


class TestDriftGuard:
    def test_guard_holds_for_perfect_surrogate(self):
        da = surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=800, burn_in=0, seed=5, gamma=0.0)
        assert not da.drift_guard.fired
        assert da.drift_guard.mean_discrepancy == 0.0
        assert da.stage2_reject_rate == 0.0

    def test_guard_fires_for_bad_surrogate(self):
        da = surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=800, burn_in=0, seed=5, gamma=0.5)
        assert da.drift_guard.fired
        assert da.drift_guard.mean_discrepancy > da.drift_guard.threshold
        assert da.stage2_reject_rate > 0.0

    def test_guard_threshold_configurable(self):
        # Sehr hohe Schwelle -> feuert nicht, selbst bei schlechtem Surrogat.
        da = surrogate.run_da_mcmc(
            CFG, beta=0.8, n_steps=400, burn_in=0, seed=5, gamma=0.5, drift_threshold=10.0
        )
        assert not da.drift_guard.fired


class TestReproducibility:
    def test_same_seed_bit_identical(self):
        kw = dict(beta=0.8, n_steps=600, burn_in=100, gamma=0.2)
        a = surrogate.run_da_mcmc(CFG, seed=77, **kw)
        b = surrogate.run_da_mcmc(CFG, seed=77, **kw)
        c = surrogate.run_da_mcmc(CFG, seed=78, **kw)
        assert np.array_equal(a.H_traj, b.H_traj)
        assert not np.array_equal(a.H_traj, c.H_traj)


class TestEdgeInputs:
    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=0, burn_in=0, seed=1)
        with pytest.raises(ValueError):
            surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=10, burn_in=10, seed=1)
        with pytest.raises(ValueError):
            surrogate.run_da_mcmc(CFG, beta=99.0, n_steps=10, burn_in=0, seed=1)
        with pytest.raises(ValueError):
            surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=10, burn_in=0, seed=1, gamma=-1.0)
        with pytest.raises(ValueError):
            surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=10, burn_in=0, seed=1, gamma=float("inf"))
        with pytest.raises(ValueError):
            surrogate.run_da_mcmc(CFG, beta=0.8, n_steps=10, burn_in=0, seed=1, drift_threshold=0.0)
