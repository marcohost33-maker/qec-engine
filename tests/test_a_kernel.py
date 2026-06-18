"""A-Kernel (adaptive MCMC) vs analytisches Orakel + Reproduzierbarkeit."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import a_kernel, ising1d
from adaptiverg_qec.mvp_instance import MVPConfig

CFG = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)


def test_hamiltonian_known_configs() -> None:
    assert a_kernel.hamiltonian(np.zeros(8, dtype=np.int8)) == 0
    # alternating ring 0101... has a domain wall on every edge (L even)
    alt = np.array([0, 1] * 4, dtype=np.int8)
    assert a_kernel.hamiltonian(alt) == 8


def test_delta_H_consistent_with_full_recompute() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=CFG.L, dtype=np.int8)
    for i in range(CFG.L):
        before = a_kernel.hamiltonian(x)
        dH = a_kernel._delta_H_flip(x, i)
        x[i] = 1 - x[i]
        after = a_kernel.hamiltonian(x)
        assert after - before == dH
        x[i] = 1 - x[i]  # revert


@pytest.mark.parametrize("beta", [0.4, 0.8, 1.2])
def test_mcmc_reproduces_analytic_mean_energy(beta: float) -> None:
    r = a_kernel.run_adaptive_mcmc(
        CFG,
        beta_target=beta,
        n_steps=6000,
        burn_in=1500,
        seed=2024,
        beta_start=beta,
    )
    exact = ising1d.mean_energy(beta, CFG.L)
    assert abs(r.mean_H - exact) < 0.15


def test_reproducibility_bit_exact() -> None:
    kw = dict(beta_target=0.9, n_steps=1500, burn_in=300, seed=99, beta_start=0.3)
    a = a_kernel.run_adaptive_mcmc(CFG, **kw)
    b = a_kernel.run_adaptive_mcmc(CFG, **kw)
    assert np.array_equal(a.H_traj, b.H_traj)
    assert np.array_equal(a.final_state, b.final_state)
    assert a.mean_H == b.mean_H


def test_different_seed_differs() -> None:
    kw = dict(beta_target=0.9, n_steps=1500, burn_in=300, beta_start=0.3)
    a = a_kernel.run_adaptive_mcmc(CFG, seed=1, **kw)
    b = a_kernel.run_adaptive_mcmc(CFG, seed=2, **kw)
    assert not np.array_equal(a.H_traj, b.H_traj)


def test_diminishing_adaptation_summable() -> None:
    s1 = float(np.sum(a_kernel.diminishing_step_sizes(100_000, 0.5, 100.0)))
    s2 = float(np.sum(a_kernel.diminishing_step_sizes(200_000, 0.5, 100.0)))
    assert np.isfinite(s1)
    assert abs(s2 - s1) < 0.01 * s1  # tail negligible -> summable


def test_beta_target_outside_compact_theta_rejected() -> None:
    with pytest.raises(ValueError):
        a_kernel.run_adaptive_mcmc(CFG, beta_target=99.0, n_steps=10, burn_in=0, seed=1)


def test_burn_in_too_large_rejected() -> None:
    with pytest.raises(ValueError):
        a_kernel.run_adaptive_mcmc(CFG, beta_target=0.5, n_steps=10, burn_in=10, seed=1)


def test_acceptance_in_unit_interval() -> None:
    r = a_kernel.run_adaptive_mcmc(CFG, beta_target=1.0, n_steps=500, burn_in=100, seed=5)
    assert 0.0 <= r.acceptance <= 1.0
