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


# --------------------------------------------------------------------------
# Codex-Fix 2: fail-closed gegen still-Truncation bei n_op > Operator-Basis.
# --------------------------------------------------------------------------
def test_n_op_overflow_raises_even() -> None:
    """multi_rg_y_t: n_op > 3 (verfuegbare gerade Operatoren) MUSS laut brechen."""
    chain = _wolff_chain(L=16, n=200, seed=0)
    # n_op=3 ist gueltig (genau 3 gerade Operatoren), n_op=4 nicht.
    mcrg_multirg.multi_rg_y_t(chain, n_op=3, n_levels=2)  # darf NICHT werfen
    with pytest.raises(ValueError, match="exceeds available even operators"):
        mcrg_multirg.multi_rg_y_t(chain, n_op=4, n_levels=2)


def test_n_op_overflow_raises_odd() -> None:
    """estimate_y_h + multi_rg_y_h: n_op > 2 (ungerade Basis) MUSS laut brechen.

    Vor dem Fix wurde [:, :n_op] still auf 2 Spalten gekuerzt, das Resultat aber
    mit n_op=3 gemeldet -> Mislabel. Jetzt fail-closed in BEIDEN Pfaden.
    """
    chain = _wolff_chain(L=16, n=200, seed=0)
    mcrg_multirg.estimate_y_h(chain, n_op=2)  # gueltig
    mcrg_multirg.multi_rg_y_h(chain, n_op=2, n_levels=2)  # gueltig
    with pytest.raises(ValueError, match="exceeds available odd operators"):
        mcrg_multirg.estimate_y_h(chain, n_op=3)
    with pytest.raises(ValueError, match="exceeds available odd operators"):
        mcrg_multirg.multi_rg_y_h(chain, n_op=3, n_levels=2)


# --------------------------------------------------------------------------
# Codex-Fix 3: Jackknife-Blockgroesse PRO ITERATION (nicht global aus Level 0).
# --------------------------------------------------------------------------
def test_block_size_per_iter_exposed() -> None:
    """Pro-Iteration-Blockgroesse wird gemeldet und ist autokorr-bewusst (>=1)."""
    chain = _wolff_chain(L=32, n=3000, seed=0)
    res = mcrg_multirg.multi_rg_y_t(chain, n_op=2, n_levels=3)
    bsz = np.asarray(res.block_size_per_iter)
    assert bsz.shape[0] == res.n_iters
    assert np.all(bsz >= 1)
    res_odd = mcrg_multirg.multi_rg_y_h(chain, n_op=2, n_levels=3)
    assert np.asarray(res_odd.block_size_per_iter).shape[0] == res_odd.n_iters


def test_fix3_central_values_unchanged() -> None:
    """Codex-Fix 3 aendert NUR Fehlerbalken, NICHT die y_t/y_h-Zentralwerte.

    Der Punktschaetzer nutzt weiter das gemeinsame Level-0-Fenster `keep`; nur
    die Jackknife-Partition wird pro Iteration gewaehlt. Verankert die EXAKTEN
    G27/G28-Zentralwerte (validate_multirg_2d, seed=0, L=32, n_op=2, burn_in=400)
    gegen einen Regress. Die Baseline ist tool-gemessen VOR Fix 3 und wird nach
    Fix 3 byte-identisch reproduziert (nur die Fehlerbalken duerfen sich aendern).
    """
    v = mcrg_multirg.validate_multirg_2d(
        L=32, n_op_even=2, n_op_odd=2, n_levels=3, n_records=3000, burn_in=400, seed=0
    )
    np.testing.assert_allclose(
        v.multirg.y_t_per_iter, [0.93001085, 0.99566981, 1.0219697], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        v.multirg_odd.y_h_per_iter, [1.88098809, 1.87282836, 1.87101431], rtol=0, atol=1e-7
    )
    # Alle Fehlerbalken endlich + nicht-negativ.
    assert np.all(np.isfinite(v.multirg.y_t_err_per_iter)) and np.all(
        v.multirg.y_t_err_per_iter >= 0
    )
    assert np.all(np.isfinite(v.multirg_odd.y_h_err_per_iter)) and np.all(
        v.multirg_odd.y_h_err_per_iter >= 0
    )


def test_explicit_block_size_respected() -> None:
    """Explizites block_size wird respektiert (kein per-iter-Override)."""
    chain = _wolff_chain(L=32, n=3000, seed=0)
    res = mcrg_multirg.multi_rg_y_t(chain, n_op=2, n_levels=3, block_size=10)
    assert np.all(np.asarray(res.block_size_per_iter) == 10)
