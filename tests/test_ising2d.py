"""Tests fuer den 2D-Ising-Sampler + Majority-Rule-Blocking (Phase-3b)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import ising2d as i2


def test_kc_tc_constants() -> None:
    """K_c = ln(1+sqrt(2))/2, T_c = 1/K_c (Onsager)."""
    assert pytest.approx(0.5 * np.log(1 + np.sqrt(2))) == i2.KC_2D
    assert pytest.approx(1.0 / i2.KC_2D) == i2.TC_2D
    assert pytest.approx(2.2691853, abs=1e-5) == i2.TC_2D


def test_energy_per_spin_ground_state() -> None:
    """Voll ausgerichtetes Gitter -> E/N = -2 (alle 2N Bonds = +1)."""
    s = np.ones((8, 8))
    assert float(i2.energy_per_spin(s)) == pytest.approx(-2.0)
    assert float(i2.energy_per_spin(-s)) == pytest.approx(-2.0)


def test_magnetization_per_spin() -> None:
    s = np.ones((4, 4))
    assert float(i2.magnetization_per_spin(s)) == pytest.approx(1.0)
    assert float(i2.magnetization_per_spin(-s)) == pytest.approx(-1.0)


def test_metropolis_vs_exact_L4() -> None:
    """Sampler reproduziert die exakte L=4-Enumeration der Energie (Orakel)."""
    L, n = 4, 16
    states = np.arange(1 << n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(-1, L, L).astype(np.float64)
    e_all = i2.energy_per_spin(s)
    for K in (0.25, i2.KC_2D, 0.55):
        w = np.exp(-K * n * e_all)
        w /= w.sum()
        e_exact = float((w * e_all).sum())
        ch = i2.checkerboard_metropolis(K, L, n_sweeps=20000, burn_in=3000, seed=7, record_every=2)
        e_mc = float(i2.energy_per_spin(ch.configs).mean())
        assert abs(e_mc - e_exact) < 0.02, f"K={K}: mc={e_mc} exact={e_exact}"


def test_metropolis_reproducible() -> None:
    """Gleicher Seed -> bit-identische Trajektorie; anderer Seed -> verschieden."""
    kw = dict(n_sweeps=500, burn_in=100, record_every=1)
    a = i2.checkerboard_metropolis(0.4, 16, seed=5, **kw)
    b = i2.checkerboard_metropolis(0.4, 16, seed=5, **kw)
    c = i2.checkerboard_metropolis(0.4, 16, seed=6, **kw)
    assert np.array_equal(a.configs, b.configs)
    assert not np.array_equal(a.configs, c.configs)


def test_metropolis_ordered_above_kc() -> None:
    """Bei K > K_c (kalt) -> grosse Magnetisierung; bei K < K_c -> klein."""
    cold = i2.checkerboard_metropolis(0.6, 16, n_sweeps=3000, burn_in=1500, seed=1, record_every=5)
    hot = i2.checkerboard_metropolis(0.25, 16, n_sweeps=3000, burn_in=1500, seed=1, record_every=5)
    m_cold = float(np.abs(i2.magnetization_per_spin(cold.configs)).mean())
    m_hot = float(np.abs(i2.magnetization_per_spin(hot.configs)).mean())
    assert m_cold > 0.85
    assert m_hot < 0.4
    assert m_cold > m_hot


def test_majority_block_shape_and_rule() -> None:
    """b=2-Blocking halbiert jede Dimension; Mehrheit korrekt."""
    s = np.ones((8, 8), dtype=np.int64)
    b = i2.majority_block_b2(s)
    assert b.shape == (4, 4)
    assert np.all(b == 1)
    assert i2.majority_block_b2(np.array([[1, 1], [1, -1]]))[0, 0] == 1
    assert i2.majority_block_b2(np.array([[-1, -1], [-1, 1]]))[0, 0] == -1


def test_majority_tie_break_unbiased() -> None:
    """Gleichstand-Bloecke (Summe 0) werden im Mittel symmetrisch +/- gebrochen."""
    L = 32
    ii, jj = np.indices((L, L))
    s = np.where((ii + jj) % 2 == 0, 1, -1)  # jeder 2x2-Block summiert zu 0
    plus = total = 0
    for ci in range(200):
        b = i2.majority_block_b2(s, config_index=ci, seed=42)
        plus += int((b == 1).sum())
        total += b.size
    assert abs(plus / total - 0.5) < 0.03


def test_majority_tie_break_reproducible() -> None:
    s = np.where((np.indices((8, 8)).sum(axis=0)) % 2 == 0, 1, -1)
    a = i2.majority_block_b2(s, config_index=3, seed=1)
    b = i2.majority_block_b2(s, config_index=3, seed=1)
    c = i2.majority_block_b2(s, config_index=3, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: i2.checkerboard_metropolis(0.0, 16, n_sweeps=5, burn_in=0, seed=1),
        lambda: i2.checkerboard_metropolis(float("inf"), 16, n_sweeps=5, burn_in=0, seed=1),
        lambda: i2.checkerboard_metropolis(0.4, 3, n_sweeps=5, burn_in=0, seed=1),
        lambda: i2.checkerboard_metropolis(0.4, 15, n_sweeps=5, burn_in=0, seed=1),
        lambda: i2.checkerboard_metropolis(0.4, 16, n_sweeps=0, burn_in=0, seed=1),
        lambda: i2.checkerboard_metropolis(0.4, 16, n_sweeps=5, burn_in=-1, seed=1),
        lambda: i2.exact_energy_per_spin_2x2(-1.0),
        lambda: i2.majority_block_b2(np.ones((3, 3))),
        lambda: i2.majority_block_b2(np.ones((4, 6))),
    ],
)
def test_edge_inputs_raise(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        bad()
