"""Analytisches Ising-Orakel vs Brute-Force-Enumeration (unabhaengig)."""

from __future__ import annotations

import math

import pytest

from adaptiverg_qec import ising1d


@pytest.mark.parametrize("L", [4, 6, 8])
@pytest.mark.parametrize("beta", [0.3, 0.8, 1.5])
def test_mean_energy_matches_brute_force(L: int, beta: float) -> None:
    probs = ising1d.exact_distribution(beta, L)
    bf = sum(p * ising1d.domain_walls(s, L) for s, p in enumerate(probs))
    assert abs(bf - ising1d.mean_energy(beta, L)) < 1e-9


@pytest.mark.parametrize("L", [4, 6, 8])
@pytest.mark.parametrize("beta", [0.3, 0.8, 1.5])
def test_log_partition_matches_brute_force(L: int, beta: float) -> None:
    lz_bf = math.log(sum(math.exp(-beta * ising1d.domain_walls(s, L)) for s in range(1 << L)))
    assert abs(lz_bf - ising1d.log_partition(beta, L)) < 1e-9


def test_distribution_normalized() -> None:
    probs = ising1d.exact_distribution(0.7, 6)
    assert abs(math.fsum(probs) - 1.0) < 1e-12


@pytest.mark.parametrize("bad_beta", [0.0, -1.0])
def test_negative_beta_rejected(bad_beta: float) -> None:
    with pytest.raises(ValueError):
        ising1d.mean_energy(bad_beta, 8)


def test_too_small_L_rejected() -> None:
    with pytest.raises(ValueError):
        ising1d.mean_energy(0.5, 1)


def test_exact_distribution_guards_exponential_blowup() -> None:
    with pytest.raises(ValueError):
        ising1d.exact_distribution(0.5, 30)
