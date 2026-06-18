"""C-Kernel: RG-Map + Jacobian (Complex-Step==FD==analytisch) + Exponenten."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import rg_map


def test_rg_map_trivial_fixpoint() -> None:
    assert abs(rg_map.rg_map_scalar(0.0)) < 1e-15


def test_rg_iteration_contracts_to_zero() -> None:
    K = 0.3
    for _ in range(60):
        K = float(rg_map.rg_map_scalar(K))
    assert K < 1e-6


@pytest.mark.parametrize("K", [0.0, 0.2, 0.5, 1.0])
def test_complex_step_matches_analytic_derivative(K: float) -> None:
    g = np.array([K])
    cs = rg_map.jacobian_complex_step(rg_map.rg_map, g)[0, 0]
    assert abs(cs - rg_map.rg_derivative_analytic(K)) < 1e-12


@pytest.mark.parametrize("K", [0.2, 0.5, 1.0])
def test_complex_step_matches_finite_difference(K: float) -> None:
    g = np.array([K])
    cs = rg_map.jacobian_complex_step(rg_map.rg_map, g)[0, 0]
    fd = rg_map.jacobian_finite_difference(rg_map.rg_map, g)[0, 0]
    assert abs(cs - fd) < 1e-7


def test_multidim_jacobian_against_known_oracle() -> None:
    def f(v: np.ndarray) -> np.ndarray:
        x, y = v[0], v[1]
        return np.array([np.sin(x) + y**2, x * y])

    p = np.array([0.7, 1.3])
    j_an = np.array([[np.cos(0.7), 2 * 1.3], [1.3, 0.7]])
    j_cs = rg_map.jacobian_complex_step(f, p)
    j_fd = rg_map.jacobian_finite_difference(f, p)
    assert np.max(np.abs(j_cs - j_an)) < 1e-12
    assert np.max(np.abs(j_cs - j_fd)) < 1e-6


def test_exponents_and_hyperbolicity() -> None:
    M = rg_map.jacobian_complex_step(rg_map.rg_map, np.array([0.5]))
    rep = rg_map.exponents_from_jacobian(M)
    # lambda = tanh(1) ~ 0.7616, |lambda| != 1 -> hyperbolic, y = log_2|lambda|
    assert rep.hyperbolic
    assert abs(rep.eigenvalues[0] - np.tanh(1.0)) < 1e-10
    assert abs(rep.exponents[0] - np.log2(np.tanh(1.0))) < 1e-10


def test_marginal_eigenvalue_flagged_not_hyperbolic() -> None:
    rep = rg_map.exponents_from_jacobian(np.array([[1.0]]))
    assert not rep.hyperbolic
    assert bool(rep.marginal_mask[0])


def test_fd_eta_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        rg_map.jacobian_finite_difference(rg_map.rg_map, np.array([0.5]), eta=1.0)


def test_exponents_non_square_rejected() -> None:
    with pytest.raises(ValueError):
        rg_map.exponents_from_jacobian(np.array([[1.0, 2.0]]))


def test_exponents_bad_scale_factor_rejected() -> None:
    with pytest.raises(ValueError):
        rg_map.exponents_from_jacobian(np.array([[0.5]]), b=1)
