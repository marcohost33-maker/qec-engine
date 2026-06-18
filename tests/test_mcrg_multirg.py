"""Tests fuer Multi-RG-Iteration + ungerader Sektor y_h (Phase-4)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import ising2d, mcrg_multirg, wolff2d


def _wolff_chain(L: int, n: int, seed: int = 0) -> ising2d.Ising2DChain:
    ch = wolff2d.wolff_sample(ising2d.KC_2D, L, n_records=n, burn_in=300, seed=seed)
    return ising2d.Ising2DChain(configs=ch.configs, K=ising2d.KC_2D, L=L, acceptance=1.0, seed=seed)


def test_oracle_constants() -> None:
    """y_t = 1, y_h = 15/8 = 1.875 (Onsager, web-verifiziert)."""
    assert mcrg_multirg.Y_T_ORACLE == 1.0
    assert pytest.approx(1.875) == mcrg_multirg.Y_H_ORACLE
    assert mcrg_multirg.Y_H_ORACLE == 15.0 / 8.0


def test_odd_operators_antisymmetric() -> None:
    """Ungerade Operatoren wechseln das Vorzeichen unter s -> -s."""
    rng = np.random.default_rng(0)
    s = np.where(rng.random((8, 8)) < 0.5, 1.0, -1.0)
    o = mcrg_multirg.odd_operators(s)
    o_neg = mcrg_multirg.odd_operators(-s)
    assert np.allclose(o, -o_neg)


def test_even_operators_symmetric_baseline() -> None:
    """Gerade Operatoren sind invariant unter s -> -s (Kontrast zu ungeraden)."""
    from adaptiverg_qec.mcrg_matrix import even_operators

    rng = np.random.default_rng(1)
    s = np.where(rng.random((8, 8)) < 0.5, 1.0, -1.0)
    assert np.allclose(even_operators(s), even_operators(-s))


def test_block_chain_levels() -> None:
    """block_chain liefert iterierte L -> L/2 -> L/4 Stufen, stoppt bei L<4."""
    configs = np.where(np.random.default_rng(0).random((10, 16, 16)) < 0.5, 1, -1).astype(np.int8)
    levels = mcrg_multirg.block_chain(configs, n_levels=5)
    sizes = [lv.shape[1] for lv in levels]
    assert sizes == [16, 8, 4]  # 4 -> 2 waere degeneriert, gestoppt (kleinste Kante 4)


def test_swendsen_matrix_raw_shape_and_solve() -> None:
    """T_h loest T B = A (rohe Momente); Residuum ~ Maschinen-eps."""
    rng = np.random.default_rng(2)
    od = rng.normal(size=(500, 2))
    od_block = od * 1.7 + rng.normal(size=(500, 2)) * 0.1
    T = mcrg_multirg.swendsen_matrix_raw(od, od_block)
    A = (od_block.T @ od) / od.shape[0]
    B = (od_block.T @ od_block) / od.shape[0]
    assert np.max(np.abs(T @ B - A)) < 1e-9 * np.max(np.abs(A))


def test_multi_rg_y_t_converges_toward_one() -> None:
    """Multi-RG y_t: bester Iterationswert klar besser als Phase-3b (~0.035)."""
    chain = _wolff_chain(L=32, n=3000, seed=0)
    res = mcrg_multirg.multi_rg_y_t(chain, n_op=2, n_levels=3)
    assert res.n_iters >= 2
    best = float(res.abs_err_per_iter.min())
    assert best < 0.035, f"best |y_t-1|={best} not better than Phase-3b 0.035"


def test_multi_rg_y_h_converges_toward_15_8() -> None:
    """Multi-RG y_h: tiefste Iteration konvergiert zu 15/8 (Swendsen)."""
    chain = _wolff_chain(L=32, n=3000, seed=0)
    res = mcrg_multirg.multi_rg_y_h(chain, n_op=2, n_levels=3)
    assert res.n_iters >= 2
    best = float(res.abs_err_per_iter.min())
    assert best < 0.05, f"best |y_h-15/8|={best} not within honest band"


def test_estimate_y_h_single_step() -> None:
    """estimate_y_h liefert ein y_h ~ O(2) mit endlichem Fehlerbalken."""
    chain = _wolff_chain(L=16, n=2000, seed=0)
    e = mcrg_multirg.estimate_y_h(chain, n_op=2)
    assert 1.5 < e.y_h < 2.2  # nahe 15/8, 1-Stufe-Bias erlaubt
    assert e.y_h_error >= 0.0
    assert np.isfinite(e.cond_B)


def test_rejects_invalid() -> None:
    chain = _wolff_chain(L=16, n=200, seed=0)
    with pytest.raises(ValueError):
        mcrg_multirg.multi_rg_y_t(chain, n_op=1)
    with pytest.raises(ValueError):
        mcrg_multirg.estimate_y_h(chain, n_op=0)
    with pytest.raises(ValueError):
        mcrg_multirg.swendsen_matrix_raw(np.ones((100, 2)), np.ones((80, 2)))
    small = ising2d.Ising2DChain(
        configs=np.ones((10, 8, 8), dtype=np.int8), K=0.4, L=8, acceptance=1.0, seed=0
    )
    with pytest.raises(ValueError):
        mcrg_multirg.multi_rg_y_t(small, n_op=2)


def test_reproducible() -> None:
    """Gleicher Seed -> identische y_t/y_h-Reihen."""
    a = mcrg_multirg.multi_rg_y_t(_wolff_chain(16, 800, seed=4), n_levels=2)
    b = mcrg_multirg.multi_rg_y_t(_wolff_chain(16, 800, seed=4), n_levels=2)
    assert np.array_equal(a.y_t_per_iter, b.y_t_per_iter)
