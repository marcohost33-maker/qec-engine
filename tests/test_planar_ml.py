"""Tests fuer planar_ml (exakter ML-Decoder <-> Nishimori-Punkt, Issue #43).

Orakel (unabhaengig):
- M1: Code-Algebra: Sterne kommutieren mit Plaketten, Xbar/Zbar sind Logicals,
      Gewichte = Distanz d, n = d^2 + (d-1)^2.
- M2: Transfer-Matrix-log Z == Brute-Force-Coset-Wahrscheinlichkeit (Enumeration der
      Stabilisatorgruppe, reine GF(2)-Kombinatorik) bis auf Rundung, d=2..4.
- M3: exakte ML-Fehlerrate fuer d=3 aus VOLLER Enumeration aller 2^13 Fehler ==
      Monte-Carlo-ML-Rate innerhalb Binomialfehler.
- M4: ML ist optimal: gepaart auf identischen Samples nie schlechter als MWPM
      (mit [surface]-Extra), und fuer d=1-Grenzfaelle trivial.
- M5: Threshold-Analyse findet die Kreuzung synthetischer Kurven; Eingaben fail-closed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptiverg_qec import planar_ml as M

try:
    import pymatching  # noqa: F401

    HAVE_MWPM = True
except ImportError:  # pragma: no cover - default CI has no [surface] extra
    HAVE_MWPM = False

requires_mwpm = pytest.mark.skipif(not HAVE_MWPM, reason='needs pip install ".[surface]"')


# --------------------------------------------------------------------------- M1
@pytest.mark.parametrize("d", [2, 3, 5, 7])
def test_code_algebra(d: int) -> None:
    code = M.planar_code(d)
    H = code.checks.astype(np.int64)
    S = code.stars.astype(np.int64)
    assert code.n == d * d + (d - 1) * (d - 1)
    assert not ((S @ H.T) % 2).any(), "X stars must commute with Z plaquettes"
    assert not ((H @ code.xbar) % 2).any(), "Xbar must commute with all plaquettes"
    assert not ((S @ code.zbar) % 2).any(), "Zbar must commute with all stars"
    assert int(code.xbar @ code.zbar) % 2 == 1
    assert code.xbar.sum() == d and code.zbar.sum() == d
    # Stars independent over GF(2) -> 2^(RC) distinct group elements (d small).
    if d <= 3:
        k = S.shape[0]
        coeff = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1) @ S % 2
        assert len({row.tobytes() for row in coeff.astype(np.uint8)}) == 1 << k


# --------------------------------------------------------------------------- M2
@pytest.mark.parametrize("d", [2, 3, 4])
def test_transfer_matrix_equals_bruteforce_coset(d: int) -> None:
    code = M.planar_code(d)
    p = 0.13
    beta = 0.5 * math.log((1 - p) / p)
    E = (np.random.default_rng(d).random((40, code.n)) < p).astype(np.uint8)
    lz = M.log_partition(code, E, beta)
    lzf = M.log_partition(code, E ^ code.xbar, beta)
    const = code.n * math.log1p(-p) - beta * code.n
    for i in range(E.shape[0]):
        a, b = M.exact_coset_log_probs(code, E[i], p)
        assert lz[i] + const == pytest.approx(a, abs=1e-10)
        assert lzf[i] + const == pytest.approx(b, abs=1e-10)


# --------------------------------------------------------------------------- M3
def _exact_ml_rate_d3(p: float) -> float:
    code = M.planar_code(3)
    n = code.n
    allE = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    fails = M.ml_decode_fails(code, allE, p)
    w = allE.sum(axis=1)
    prob = np.exp(w * math.log(p) + (n - w) * math.log1p(-p))
    assert prob.sum() == pytest.approx(1.0, abs=1e-12)
    return float((prob * fails).sum())


def test_exact_ml_rate_matches_monte_carlo_d3() -> None:
    p = 0.11
    exact = _exact_ml_rate_d3(p)
    cell = M.simulate_decoders(3, p, shots=40_000, base_seed=5, with_mwpm=False)
    rate = cell.ml_fail / cell.shots
    se = math.sqrt(exact * (1 - exact) / cell.shots)
    assert abs(rate - exact) < 4 * se, (rate, exact, se)
    # d=3 corrects every single flip: the small-p rate is A p^2 + O(p^3), so halving p
    # divides it by ~4 (a decoder that missed single flips would scale like p).
    ratio = _exact_ml_rate_d3(0.004) / _exact_ml_rate_d3(0.002)
    assert 3.8 < ratio < 4.2, ratio


# --------------------------------------------------------------------------- M4
@requires_mwpm
@pytest.mark.parametrize("d,p", [(3, 0.08), (5, 0.10), (7, 0.11)])
def test_ml_never_worse_than_mwpm_paired(d: int, p: float) -> None:
    cell = M.simulate_decoders(d, p, shots=3000, base_seed=17)
    assert cell.ml_fail <= cell.mwpm_fail + 3 * math.sqrt(cell.mwpm_fail + 1)
    # MWPM corrections reproduce the syndrome (checked inside) and ML beats it strictly
    # once d is large enough to matter.
    if d >= 5:
        assert cell.ml_fail < cell.mwpm_fail


@requires_mwpm
def test_mwpm_on_zero_error_is_perfect() -> None:
    code = M.planar_code(5)
    assert M.mwpm_decode_fails(code, np.zeros((4, code.n), dtype=np.uint8)).sum() == 0


def test_ml_zero_error_and_logical_error() -> None:
    code = M.planar_code(5)
    E = np.zeros((2, code.n), dtype=np.uint8)
    E[1] = code.xbar  # a bare logical: ML must (correctly) fail on it
    f = M.ml_decode_fails(code, E, 0.05)
    assert f.tolist() == [0.0, 1.0]


# --------------------------------------------------------------------------- M5
def test_threshold_analysis_finds_planted_crossing() -> None:
    pc, nu = 0.109, 1.5
    ps = [0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125]
    cells = []
    for d in (5, 7, 9, 11):
        for p in ps:
            x = (p - pc) * d ** (1 / nu)
            rate = 0.25 + 0.2 * math.tanh(8 * x)
            shots = 10**7  # (almost) noiseless synthetic rates
            cells.append(
                M.DecoderCell(
                    d=d, p=p, shots=shots, ml_fail=rate * shots, ml_ties=0,
                    mwpm_fail=float("nan"), both_fail=float("nan"), wall_seconds=0.0,
                )
            )  # fmt: skip
    res = M.threshold_analysis(cells, which="ml", n_boot=30, seed=1, deg=5)
    for pair, r in res["crossings"].items():
        assert abs(r["p_cross"] - pc) < 5e-4, (pair, r)
    assert abs(res["collapse"]["pc"] - pc) < 5e-4
    assert abs(res["collapse"]["nu"] - nu) < 0.1


@pytest.mark.parametrize(
    "call",
    [
        lambda: M.planar_code(1),
        lambda: M.log_partition(M.planar_code(3), np.zeros((1, 5), dtype=np.uint8), 1.0),
        lambda: M.log_partition(M.planar_code(3), np.zeros((1, 13), dtype=np.uint8), -1.0),
        lambda: M.ml_decode_fails(M.planar_code(3), np.zeros((1, 13), dtype=np.uint8), 0.5),
        lambda: M.simulate_decoders(3, 0.1, shots=0, base_seed=0),
        lambda: M.simulate_decoders(3, 0.1, shots=10, base_seed=-1),
        lambda: M.exact_coset_log_probs(M.planar_code(6), np.zeros(61, dtype=np.uint8), 0.1),
    ],
)
def test_invalid_inputs_fail_closed(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_simulation_reproducible_and_cell_seeded() -> None:
    a = M.simulate_decoders(3, 0.1, shots=500, base_seed=3, with_mwpm=False)
    b = M.simulate_decoders(3, 0.1, shots=500, base_seed=3, with_mwpm=False)
    c = M.simulate_decoders(3, 0.1, shots=500, base_seed=4, with_mwpm=False)
    assert a.ml_fail == b.ml_fail
    assert a.ml_fail != c.ml_fail  # different cell seed -> different samples
