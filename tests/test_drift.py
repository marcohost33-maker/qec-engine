"""Foster-Lyapunov-Drift-Guard: holds bei Kontraktion, feuert sonst."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import a_kernel, drift, ising1d
from adaptiverg_qec.mvp_instance import MVPConfig

CFG = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)


def test_lyapunov_V_ge_one() -> None:
    H = np.array([0.0, 1.0, 5.0, 16.0])
    V = drift.lyapunov_V(H)
    assert np.all(V >= 1.0)
    assert np.array_equal(V, 1.0 + H)


def test_negative_H_rejected() -> None:
    with pytest.raises(ValueError):
        drift.lyapunov_V(np.array([-1.0, 0.0]))


def test_drift_holds_on_equilibrium_chain() -> None:
    r = a_kernel.run_adaptive_mcmc(
        CFG, beta_target=1.5, n_steps=8000, burn_in=0, seed=7, beta_start=1.5
    )
    d = max(1.0 + ising1d.mean_energy(1.5, CFG.L), 2.0)
    rep = drift.estimate_drift(r.H_traj, d=d)
    assert rep.holds
    assert rep.lambda_hat < 1.0


def test_drift_fires_on_non_contracting_walk() -> None:
    rng = np.random.default_rng(0)
    walk = np.abs(np.cumsum(rng.integers(-1, 2, size=6000))).astype(float) + 1.0
    rep = drift.estimate_drift(walk, d=5.0)
    assert not rep.holds


def test_drift_fail_closed_on_insufficient_outside() -> None:
    # Trajectory entirely inside C -> no outside transitions -> fail-closed.
    H = np.zeros(50)
    rep = drift.estimate_drift(H, d=10.0)
    assert not rep.holds
    assert np.isnan(rep.lambda_hat)


def test_drift_d_below_one_rejected() -> None:
    with pytest.raises(ValueError):
        drift.estimate_drift(np.array([1.0, 2.0, 3.0]), d=0.5)


def test_drift_too_short_traj_rejected() -> None:
    with pytest.raises(ValueError):
        drift.estimate_drift(np.array([1.0]), d=2.0)
