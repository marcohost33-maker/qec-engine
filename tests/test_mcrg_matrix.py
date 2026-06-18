"""Tests fuer die Multi-Operator-Swendsen-MATRIX (Phase-3b)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import ising2d as i2
from adaptiverg_qec import mcrg_matrix as mm


def _chain(seed: int = 0, n_sweeps: int = 10000):
    return i2.checkerboard_metropolis(
        i2.KC_2D, 16, n_sweeps=n_sweeps, burn_in=3000, seed=seed, record_every=2
    )


def test_even_operators_shape_and_symmetry() -> None:
    """3 gerade Operatoren je Konfiguration; invariant unter s -> -s."""
    s = np.where(np.random.default_rng(0).random((5, 16, 16)) < 0.5, 1, -1).astype(np.int8)
    ops = mm.even_operators(s)
    assert ops.shape == (5, 3)
    # Gerade Operatoren sind invariant unter globalem Spin-Flip.
    ops_flip = mm.even_operators(-s)
    assert np.allclose(ops, ops_flip)


def test_even_operators_ground_state() -> None:
    """Voll ausgerichtet: S1=2N (alle NN-Bonds +1), S2=2N, S3=N."""
    L = 8
    n = L * L
    s = np.ones((L, L))
    s1, s2, s3 = mm.even_operators(s)
    assert s1 == pytest.approx(2 * n)  # right + down bonds
    assert s2 == pytest.approx(2 * n)  # both diagonals
    assert s3 == pytest.approx(n)  # plaquettes all +1


def test_connected_matrix_symmetry() -> None:
    """B = <S'S'>_c ist symmetrisch + PSD (Kovarianzmatrix)."""
    ts = mm.operator_timeseries(_chain(0), seed=0)
    Sp = ts.Sp[:, :3]
    B = mm._connected_matrix(Sp, Sp)
    assert np.max(np.abs(B - B.T)) < 1e-9
    eig = np.linalg.eigvalsh(B)
    assert np.all(eig > -1e-9 * abs(eig).max())


def test_swendsen_matrix_linear_solve_consistency() -> None:
    """T = A.B^-1 loest T B = A (Residuum ~ Maschinen-eps)."""
    ts = mm.operator_timeseries(_chain(3), seed=3)
    S = ts.S[:, :2]
    Sp = ts.Sp[:, :2]
    A = mm._connected_matrix(Sp, S)
    B = mm._connected_matrix(Sp, Sp)
    T = mm.swendsen_matrix(S, Sp)
    resid = np.max(np.abs(T @ B - A)) / max(np.max(np.abs(A)), 1.0)
    assert resid < 1e-9


def test_exponents_sorted_descending() -> None:
    """Eigenwerte + Exponenten absteigend nach |lambda| sortiert."""
    T = np.array([[2.0, 0.0], [0.0, 0.5]])
    eig, y = mm.exponents_from_T(T, b=2)
    assert abs(eig[0]) >= abs(eig[1])
    # y = log_2 |lambda|: log2(2)=1, log2(0.5)=-1.
    assert y[0] == pytest.approx(1.0)
    assert y[1] == pytest.approx(-1.0)


def test_y_t_near_oracle_coarse() -> None:
    """y_t (Matrix) trifft das Onsager-Orakel y_t=1 innerhalb der ehrlichen Bande.

    EHRLICH: single-spin + 1 RG-Stufe + kleines L -> grob (|y_t-1| < 0.25). KEIN
    Hochpraezisions-Claim. Multi-Seed entschaerft Einzel-Seed-Streuung.
    """
    y_ts = [mm.validate_y_t_2d(L=16, n_op=2, seed=sd).y_t for sd in range(3)]
    y_mean = float(np.mean(y_ts))
    assert abs(y_mean - 1.0) < 0.25, f"y_t mean={y_mean}, per-seed={y_ts}"


def test_y_t_has_real_error_bar() -> None:
    """Der Jackknife-Fehler von y_t ist endlich + positiv (echter Fehlerbalken)."""
    est = mm.validate_y_t_2d(L=16, n_op=2, seed=1)
    assert np.isfinite(est.y_t_error)
    assert est.y_t_error > 0.0
    assert est.n_blocks >= 16
    assert est.lambda_max > 1.0  # relevanter (wachsender) Eigenwert


def test_matrix_reproducible() -> None:
    a = mm.validate_y_t_2d(L=16, n_op=2, n_sweeps=6000, burn_in=2000, seed=11)
    b = mm.validate_y_t_2d(L=16, n_op=2, n_sweeps=6000, burn_in=2000, seed=11)
    c = mm.validate_y_t_2d(L=16, n_op=2, n_sweeps=6000, burn_in=2000, seed=12)
    assert a.y_t == b.y_t
    assert np.array_equal(a.T, b.T)
    assert a.y_t != c.y_t


def test_three_operators_runs() -> None:
    """Auch mit 3 Operatoren laeuft die Pipeline (groessere Matrix)."""
    ts = mm.operator_timeseries(_chain(2), seed=2)
    est = mm.estimate_y_t(ts, n_op=3)
    assert est.n_op == 3
    assert est.T.shape == (3, 3)
    assert np.isfinite(est.y_t)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: mm.estimate_y_t(
            mm.OperatorTimeseries(S=np.ones((100, 3)), Sp=np.ones((100, 3)), K=0.4, L=16),
            n_op=1,
        ),
        lambda: mm.estimate_y_t(
            mm.OperatorTimeseries(S=np.ones((10, 3)), Sp=np.ones((10, 3)), K=0.4, L=16),
            n_op=2,
        ),
        lambda: mm.swendsen_matrix(np.ones((100, 2)), np.ones((80, 2))),
        lambda: mm.exponents_from_T(np.ones((2, 3))),
        lambda: mm.exponents_from_T(np.eye(2), b=1),
    ],
)
def test_edge_inputs_raise(bad) -> None:
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        bad()
