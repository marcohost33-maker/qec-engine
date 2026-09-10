"""Tests for the 2D-Ising sampler and majority-rule blocking."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import ising2d as i2


def test_kc_tc_constants() -> None:
    assert pytest.approx(0.5 * np.log(1 + np.sqrt(2))) == i2.KC_2D
    assert pytest.approx(1.0 / i2.KC_2D) == i2.TC_2D
    assert pytest.approx(2.2691853, abs=1e-5) == i2.TC_2D


def test_energy_per_spin_ground_state() -> None:
    s = np.ones((8, 8))
    assert float(i2.energy_per_spin(s)) == pytest.approx(-2.0)
    assert float(i2.energy_per_spin(-s)) == pytest.approx(-2.0)


def test_magnetization_per_spin() -> None:
    s = np.ones((4, 4))
    assert float(i2.magnetization_per_spin(s)) == pytest.approx(1.0)
    assert float(i2.magnetization_per_spin(-s)) == pytest.approx(-1.0)


def test_metropolis_vs_exact_L4() -> None:
    """Sampler reproduces the exact L=4 energy enumeration."""
    L, n = 4, 16
    states = np.arange(1 << n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(-1, L, L).astype(np.float64)
    e_all = i2.energy_per_spin(s)
    for K in (0.25, i2.KC_2D, 0.55):
        logw = -K * n * e_all
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        e_exact = float((w * e_all).sum())
        ch = i2.checkerboard_metropolis(K, L, n_sweeps=20000, burn_in=3000, seed=7, record_every=2)
        e_mc = float(i2.energy_per_spin(ch.configs).mean())
        assert abs(e_mc - e_exact) < 0.02, f"K={K}: mc={e_mc} exact={e_exact}"


def test_metropolis_reproducible() -> None:
    kw = dict(n_sweeps=500, burn_in=100, record_every=1)
    a = i2.checkerboard_metropolis(0.4, 16, seed=5, **kw)
    b = i2.checkerboard_metropolis(0.4, 16, seed=5, **kw)
    c = i2.checkerboard_metropolis(0.4, 16, seed=6, **kw)
    assert np.array_equal(a.configs, b.configs)
    assert not np.array_equal(a.configs, c.configs)


def test_metropolis_ordered_above_kc() -> None:
    cold = i2.checkerboard_metropolis(0.6, 16, n_sweeps=3000, burn_in=1500, seed=1, record_every=5)
    hot = i2.checkerboard_metropolis(0.25, 16, n_sweeps=3000, burn_in=1500, seed=1, record_every=5)
    m_cold = float(np.abs(i2.magnetization_per_spin(cold.configs)).mean())
    m_hot = float(np.abs(i2.magnetization_per_spin(hot.configs)).mean())
    assert m_cold > 0.85
    assert m_hot < 0.4
    assert m_cold > m_hot


def test_majority_block_shape_and_rule() -> None:
    s = np.ones((8, 8), dtype=np.int64)
    b = i2.majority_block_b2(s)
    assert b.shape == (4, 4)
    assert np.all(b == 1)
    assert i2.majority_block_b2(np.array([[1, 1], [1, -1]]))[0, 0] == 1
    assert i2.majority_block_b2(np.array([[-1, -1], [-1, 1]]))[0, 0] == -1


def test_majority_tie_break_unbiased() -> None:
    """Tie selector is balanced in aggregate across block/config hashes."""
    L = 32
    ii, jj = np.indices((L, L))
    s = np.where((ii + jj) % 2 == 0, 1, -1)
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


def test_majority_block_exact_z2_equivariance() -> None:
    """Global spin flip must commute with blocking, including every tie block."""
    rng = np.random.default_rng(2026)

    # Explicit all-tie checkerboard stresses the formerly defective path.
    ii, jj = np.indices((16, 16))
    tied = np.where((ii + jj) % 2 == 0, 1, -1).astype(np.int8)
    for seed in (0, 1, 7, 2**63 + 5):
        for config_index in (0, 3, 99):
            b = i2.majority_block_b2(tied, config_index=config_index, seed=seed)
            b_flip = i2.majority_block_b2(-tied, config_index=config_index, seed=seed)
            assert np.array_equal(b_flip, -b)

    # Also exercise mixed majority/tie configurations.
    for config_index in range(20):
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(16, 16))
        b = i2.majority_block_b2(s, config_index=config_index, seed=42)
        b_flip = i2.majority_block_b2(-s, config_index=config_index, seed=42)
        assert np.array_equal(b_flip, -b)


def test_majority_tie_output_is_an_input_spin() -> None:
    """On ties the selected coarse spin must come from the corresponding 2x2 block."""
    rng = np.random.default_rng(11)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(12, 12))
    b = i2.majority_block_b2(s, config_index=5, seed=17)
    lb = s.shape[0] // 2
    blocks = s.reshape(lb, 2, lb, 2).transpose(0, 2, 1, 3).reshape(lb, lb, 4)
    sums = blocks.sum(axis=2)
    for r, c in zip(*np.where(sums == 0), strict=True):
        assert b[r, c] in blocks[r, c]


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
        lambda: i2.majority_block_b2(np.array([[1, 0], [-1, 1]])),
    ],
)
def test_edge_inputs_raise(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        bad()


def _all_2x2_configs() -> np.ndarray:
    """Alle 2^4 = 16 moeglichen 2x2-Bloecke in {+1,-1}."""
    states = np.arange(16, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(4)[None, :]) & 1).astype(np.int8)
    return (1 - 2 * bits).reshape(16, 2, 2)


def test_majority_block_z2_equivariance_exhaustive_2x2() -> None:
    """Z2-Aequivarianz erschoepfend auf der Block-Ebene.

    ``majority_block_b2`` wirkt blockweise und unabhaengig je 2x2-Block; die
    einzige Kopplung an die Gittergroesse ist der Hash ueber (block_row,
    block_col).  Daher ist ein Sweep ueber ALLE 16 moeglichen Blockinhalte,
    gekreuzt mit allen vier erreichbaren Selektorwerten, ein vollstaendiger
    Nachweis der Eigenschaft auf Blockebene -- keine Stichprobe.

    Der Tie-Pfad ist dabei nicht Beiwerk: 6 der 16 Blockinhalte (C(4,2)) sind
    2+2-Ties, also genau der Pfad, auf dem die alte Hash-Bit-Regel
    ``B(-s) != -B(s)`` lieferte.
    """
    blocks = _all_2x2_configs()
    seen_selectors: set[int] = set()
    n_tie_checked = 0
    n_checked = 0

    for seed in (0, 1, 17, 2**63 + 5):
        for config_index in range(64):
            selector = int(
                i2._splitmix64_grid(
                    config_index,
                    np.zeros((1, 1), dtype=np.int64),
                    np.zeros((1, 1), dtype=np.int64),
                    seed,
                )[0, 0]
                & np.uint64(3)
            )
            seen_selectors.add(selector)
            for s in blocks:
                b = i2.majority_block_b2(s, config_index=config_index, seed=seed)
                b_flip = i2.majority_block_b2(-s, config_index=config_index, seed=seed)
                assert np.array_equal(b_flip, -b), (s, config_index, seed)
                n_checked += 1
                if s.sum() == 0:
                    n_tie_checked += 1

    # Nicht-Vakuitaet: alle vier Selektor-Slots und alle 6 Tie-Muster wurden
    # wirklich durchlaufen -- sonst wuerde ein Sweep, der den Tie-Pfad nie
    # trifft, ebenfalls bestehen und nichts beweisen.
    assert seen_selectors == {0, 1, 2, 3}, seen_selectors
    assert n_checked == 16 * 64 * 4
    assert n_tie_checked == 6 * 64 * 4


def test_majority_block_z2_equivariance_randomized_large() -> None:
    """Dieselbe Invariante auf grossen Gittern, deterministisch geseedet.

    Deckt die Hash-Indizierung ueber (block_row, block_col) ab, die der
    erschoepfende 1x1-Block-Sweep nicht beruehrt.  Fester ``Generator``, damit
    ein Fehlschlag reproduzierbar ist.
    """
    rng = np.random.default_rng(20260910)
    tie_blocks_seen = 0
    for config_index in range(40):
        for L in (8, 16, 32):
            s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
            b = i2.majority_block_b2(s, config_index=config_index, seed=4711)
            b_flip = i2.majority_block_b2(-s, config_index=config_index, seed=4711)
            assert np.array_equal(b_flip, -b)
            lb = L // 2
            tie_blocks_seen += int((s.reshape(lb, 2, lb, 2).sum(axis=(1, 3)) == 0).sum())
    # Zufallskonfigurationen erzeugen reichlich Ties; ohne sie waere der Sweep
    # blind fuer genau den reparierten Pfad.
    assert tie_blocks_seen > 1000, tie_blocks_seen


def test_majority_block_non_tie_blocks_ignore_the_hash() -> None:
    """Positiv-Kontrolle: der Tie-Pfad darf Nicht-Tie-Bloecke NICHT anfassen.

    Eine Tie-Regel, die alles ueberschreibt (oder eine Implementierung, die
    jeden Block als Tie behandelt), wuerde jeden Aequivarianz-Test bestehen und
    waere trotzdem falsch: Z2-Aequivarianz allein ist von ``B(s) = s[0,0]``
    ebenfalls erfuellt.  Hier wird daher festgehalten, dass Bloecke mit echter
    Mehrheit unabhaengig von config_index/seed das Vorzeichen der Blocksumme
    liefern.
    """
    blocks = _all_2x2_configs()
    non_tie = np.array([blk for blk in blocks if blk.sum() != 0])
    assert non_tie.shape[0] == 10  # 16 - 6 Ties

    for blk in non_tie:
        expected = np.sign(blk.sum())
        for config_index in (0, 5, 12345):
            for seed in (0, 99, 2**63 + 5):
                out = i2.majority_block_b2(blk, config_index=config_index, seed=seed)
                assert out.shape == (1, 1)
                assert out[0, 0] == expected, (blk, config_index, seed)

    # Und auf einem grossen Gitter ohne jeden Tie: Ergebnis == reines Mehrheits-
    # Vorzeichen, hash-unabhaengig.
    rng = np.random.default_rng(7)
    # Jeder 2x2-Block einheitlich +1 oder -1 -> Blocksumme immer +/-4, nie ein Tie,
    # aber das Gitter ist echt variiert (kein triviales Eins-Gitter).
    coarse = rng.choice(np.array([-1, 1], dtype=np.int8), size=(8, 8))
    s = np.kron(coarse, np.ones((2, 2), dtype=np.int8))
    block_sum = s.reshape(8, 2, 8, 2).sum(axis=(1, 3))
    assert np.array_equal(np.sign(block_sum).astype(np.int8), coarse)
    assert np.all(block_sum != 0)
    a = i2.majority_block_b2(s, config_index=1, seed=1)
    b = i2.majority_block_b2(s, config_index=2, seed=2)
    assert np.array_equal(a, b)
    assert np.array_equal(a, np.sign(block_sum).astype(np.int8))


@pytest.mark.parametrize(
    "bad, warum",
    [
        (np.array([[1.5, -1.5], [1.5, -1.5]]), "1.5/-1.5 wuerde ein Cast zu 1/-1 abschneiden"),
        (np.array([[0.5, 0.5], [0.5, 0.5]]), "0.5 -> 0"),
        (np.array([[0.0, 0.0], [0.0, 0.0]]), "0 ist kein Spin"),
        (np.array([[2.0, -2.0], [2.0, -2.0]]), "Betrag != 1"),
        (np.array([[0.9999999999, -1.0], [1.0, -1.0]]), "knapp neben +1 -> 0"),
        (np.array([[np.nan, 1.0], [1.0, -1.0]]), "NaN"),
        (np.array([[np.inf, 1.0], [1.0, -1.0]]), "inf"),
    ],
)
def test_majority_block_rejects_non_spin_values_before_casting(bad, warum) -> None:
    """Die ±1-Pruefung muss VOR dem int-Cast greifen.

    Laeuft der Cast zuerst, kann die Pruefung nicht mehr sehen, wogegen sie
    schuetzt: ``1.5`` und ``-1.5`` werden zu ``1`` und ``-1`` abgeschnitten und
    danach als gueltige Spins akzeptiert -- die Funktion blockt dann still
    veraenderte Daten, statt den dokumentierten ±1-Fehler zu werfen.

    Breit gefangen und der TYP geprueft: stirbt der Aufruf an einer anderen
    Ausnahme (etwa einem Cast-Fehler), waere der Test sonst nicht einzuordnen.
    """
    with pytest.raises(Exception) as exc:
        i2.majority_block_b2(bad)
    assert isinstance(exc.value, ValueError), (
        f"erwartet ValueError ({warum}), kam {type(exc.value).__name__}: {exc.value}"
    )
    assert "+/-1" in str(exc.value), f"erwartet die dokumentierte ±1-Meldung, kam: {exc.value}"


def test_majority_block_still_accepts_valid_spins_in_any_container() -> None:
    """Positiv-Kontrolle: gueltige ±1-Daten muessen weiter akzeptiert werden.

    Ein Validator, der jede float-Eingabe ablehnt, bestuende jeden Negativtest
    und waere trotzdem falsch. Float-Spins (1.0/-1.0), int8-Arrays und rohe
    Listen sind gueltige Eingaben und muessen dasselbe Ergebnis liefern.
    """
    ref = np.array([[1, -1, -1, 1], [1, -1, 1, 1], [-1, -1, 1, -1], [1, 1, -1, -1]])
    erwartet = i2.majority_block_b2(ref, config_index=3, seed=17)
    for variante in (
        ref.astype(np.float64),
        ref.astype(np.float32),
        ref.astype(np.int8),
        ref.tolist(),
    ):
        wie = getattr(variante, "dtype", type(variante).__name__)
        # Eine Ablehnung ist hier ein FEHLSCHLAG, keine Ausnahme, die den Test
        # abstuerzen laesst: ein abgestuerzter Test ist nicht einzuordnen und
        # taugt nicht als Beleg. Darum in eine Zusicherung uebersetzen.
        try:
            out = i2.majority_block_b2(variante, config_index=3, seed=17)
        except Exception as exc:
            raise AssertionError(
                f"gueltige +/-1-Daten als {wie} wurden abgelehnt: {type(exc).__name__}: {exc}"
            ) from exc
        assert np.array_equal(out, erwartet), f"abweichendes Ergebnis fuer {wie}"
        assert out.dtype == np.int8, f"erwartet int8 fuer {wie}, kam {out.dtype}"
