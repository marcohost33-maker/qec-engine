"""Tests fuer den Wolff-Single-Cluster-Algorithmus (Phase-4)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import ising2d, wolff2d


def test_p_add_formula() -> None:
    """P_add = 1 - exp(-2K) (web-verifizierte Wolff-Bond-Wahrscheinlichkeit)."""
    K = 0.4406868
    assert wolff2d.p_add(K) == pytest.approx(1.0 - np.exp(-2.0 * K))
    assert 0.0 < wolff2d.p_add(0.1) < 1.0
    assert wolff2d.p_add(5.0) == pytest.approx(1.0, abs=1e-4)


def test_p_add_rejects_invalid() -> None:
    for bad in (-0.1, 0.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            wolff2d.p_add(bad)


def test_cluster_update_flips_at_least_seed() -> None:
    """Ein Cluster-Update flippt mindestens den Seed-Spin (cluster_size >= 1)."""
    rng = np.random.default_rng(0)
    s = np.where(rng.random((8, 8)) < 0.5, np.int8(1), np.int8(-1))
    s2, size = wolff2d.wolff_cluster_update(s, 0.5, rng)
    assert size >= 1
    # Genau die Cluster-Spins haben sich umgedreht.
    flipped = np.sum(s2 != s)
    assert flipped == size


def test_cluster_update_does_not_mutate_input() -> None:
    rng = np.random.default_rng(1)
    s = np.where(rng.random((8, 8)) < 0.5, np.int8(1), np.int8(-1))
    s_orig = s.copy()
    wolff2d.wolff_cluster_update(s, 0.6, rng)
    assert np.array_equal(s, s_orig)


def test_cluster_only_same_orientation() -> None:
    """Der Cluster enthaelt NUR Spins gleicher Orientierung wie der Seed.

    Konstruktion: alle umgedrehten Spins muessen vor dem Flip dieselbe
    Orientierung gehabt haben (Wolff fuegt nur gleich-orientierte Nachbarn hinzu).
    """
    rng = np.random.default_rng(2)
    s = np.where(rng.random((12, 12)) < 0.5, np.int8(1), np.int8(-1))
    s2, _ = wolff2d.wolff_cluster_update(s, 0.7, rng)
    changed = s2 != s
    vals = s[changed]
    # Alle geflippten Spins hatten denselben Wert (= Seed-Spin-Wert).
    assert len(np.unique(vals)) == 1


def test_wolff_energy_matches_exact_l4() -> None:
    """Wolff-Energie reproduziert die exakte L=4-Enumeration (unabhaengiges Orakel)."""
    L = 4
    n = L * L
    states = np.arange(1 << n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(-1, L, L).astype(np.float64)
    e_all = ising2d.energy_per_spin(s)
    for K in (0.25, ising2d.KC_2D, 0.55):
        w = np.exp(-K * n * e_all)
        w /= w.sum()
        e_exact = float((w * e_all).sum())
        ch = wolff2d.wolff_sample(K, L, n_records=8000, burn_in=500, seed=7, n_skip=1)
        e_mc = float(ising2d.energy_per_spin(ch.configs).mean())
        assert abs(e_mc - e_exact) < 0.02, f"K={K}: {e_mc} vs {e_exact}"


def test_cluster_fraction_grows_with_K() -> None:
    """Mittlere Cluster-Groesse waechst monoton mit K (physikalisch korrekt)."""
    fracs = []
    for K in (0.20, ising2d.KC_2D, 0.70):
        ch = wolff2d.wolff_sample(K, 16, n_records=800, burn_in=200, seed=3, n_skip=1)
        fracs.append(wolff2d.mean_cluster_fraction(ch))
    assert fracs[0] < fracs[1] < fracs[2]


def test_reproducible() -> None:
    """Gleicher Seed -> bit-identische Trajektorie; anderer Seed -> verschieden."""
    a = wolff2d.wolff_sample(ising2d.KC_2D, 8, n_records=50, burn_in=20, seed=5)
    b = wolff2d.wolff_sample(ising2d.KC_2D, 8, n_records=50, burn_in=20, seed=5)
    c = wolff2d.wolff_sample(ising2d.KC_2D, 8, n_records=50, burn_in=20, seed=6)
    assert np.array_equal(a.configs, b.configs)
    assert not np.array_equal(a.configs, c.configs)


def test_wolff_sample_rejects_invalid() -> None:
    bad_calls = [
        dict(K=-0.4, L=8, n_records=1, burn_in=0, seed=1),
        dict(K=0.4, L=2, n_records=1, burn_in=0, seed=1),
        dict(K=0.4, L=7, n_records=1, burn_in=0, seed=1),
        dict(K=0.4, L=8, n_records=0, burn_in=0, seed=1),
        dict(K=0.4, L=8, n_records=1, burn_in=-1, seed=1),
        dict(K=0.4, L=8, n_records=1, burn_in=0, seed=1, n_skip=0),
    ]
    for kw in bad_calls:
        with pytest.raises(ValueError):
            wolff2d.wolff_sample(**kw)


def test_tau_wolff_below_metropolis() -> None:
    """tau_int(Wolff) < tau_int(Metropolis) bei K_c (Slowing-Down geschlagen)."""
    from adaptiverg_qec import autocorr

    L = 16
    chw = wolff2d.wolff_sample(ising2d.KC_2D, L, n_records=2000, burn_in=300, seed=0)
    absm_w = np.abs(ising2d.magnetization_per_spin(chw.configs))
    tau_w = autocorr.integrated_autocorr_time(absm_w).tau_int
    chm = ising2d.checkerboard_metropolis(
        ising2d.KC_2D, L, n_sweeps=2000, burn_in=500, seed=0, record_every=1
    )
    absm_m = np.abs(ising2d.magnetization_per_spin(chm.configs))
    tau_m = autocorr.integrated_autocorr_time(absm_m).tau_int
    assert tau_w < tau_m, f"tau_wolff={tau_w} not < tau_metro={tau_m}"
