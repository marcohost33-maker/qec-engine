"""Tests fuer rank-normalized split-R-hat + bulk/tail-ESS (Phase-5).

Orakel: bekannte Konvergenz-/Divergenz-Szenarien (Vehtari et al. 2021,
Schwellwert R-hat < 1.01). Beidseitig non-vakuoes: gut gemischte unabhaengige
Ketten -> R-hat < 1.01; absichtlich nicht-konvergierte (Mittel-/Skalen-Drift)
-> R-hat >> 1.01 wird geflaggt.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec import rhat
from adaptiverg_qec.a_kernel import run_adaptive_mcmc
from adaptiverg_qec.mvp_instance import MVPConfig


def test_well_mixed_chains_converge() -> None:
    """NON-VAKUOES (a): M unabhaengige iid-N(0,1)-Ketten -> R-hat < 1.01."""
    chains = np.vstack([np.random.default_rng(s).standard_normal(3000) for s in range(8)])
    r = rhat.split_rhat(chains)
    assert r.rhat < rhat.RHAT_THRESHOLD, r.rhat
    assert r.converged
    # ESS plausibel: bei iid ~ M*n; >= halbe Gesamtzahl ist eine sichere Schranke.
    assert r.ess_bulk > 0.5 * chains.size
    assert r.ess_tail > 0.0


def test_mean_drift_chains_flagged() -> None:
    """NON-VAKUOES (b1): Ketten mit verschiedenen Mitteln -> R-hat >> 1.01 (bulk faengt)."""
    offs = [0, 0, 0, 0, 3, 3, 3, 3]
    chains = np.vstack(
        [np.random.default_rng(s).standard_normal(3000) + o for s, o in enumerate(offs)]
    )
    r = rhat.split_rhat(chains)
    assert r.rhat > 1.1, r.rhat
    assert not r.converged
    assert r.bulk_rhat > 1.1  # Mittel-Drift wird im bulk gefangen


def test_scale_drift_chains_flagged_by_folding() -> None:
    """NON-VAKUOES (b2): verschiedene SKALEN -> folded-R-hat faengt, was bulk verfehlt."""
    scales = [1, 1, 1, 1, 5, 5, 5, 5]
    chains = np.vstack(
        [np.random.default_rng(s).standard_normal(3000) * sc for s, sc in enumerate(scales)]
    )
    r = rhat.split_rhat(chains)
    assert r.rhat > 1.05, r.rhat
    assert not r.converged
    # Genau der Vehtari-Punkt: Skalen-Drift entgeht dem bulk, folded faengt sie.
    assert r.folded_rhat > r.bulk_rhat


def test_too_short_chains_flagged() -> None:
    """NON-VAKUOES (b3): kurze AR(1)-Ketten weit auseinander gestartet -> R-hat hoch."""
    # Sehr persistente AR(1), kurz, mit weit auseinanderliegenden, NICHT relaxierten Starts.
    phi = 0.98
    rows = []
    for s, start in zip(range(6), [-20, -20, -20, 20, 20, 20], strict=True):
        rng = np.random.default_rng(s)
        x = np.empty(300)
        x[0] = start
        for t in range(1, 300):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        rows.append(x)
    r = rhat.split_rhat(np.vstack(rows))
    assert r.rhat > rhat.RHAT_THRESHOLD, r.rhat
    assert not r.converged


def test_rank_normalize_is_standard_normal_scores() -> None:
    """Blom-Rang-Normalisierung ergibt ungefaehr standard-normale Scores."""
    x = np.random.default_rng(0).exponential(size=5000)  # schief -> rank-norm symmetriert
    z = rhat.rank_normalize(x)
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05
    # Monoton: groesster Eingang -> groesster Score.
    assert np.argmax(z) == np.argmax(x)


def test_akernel_multichain_converges() -> None:
    """R-hat auf realem A-Kernel-Multichain-MCMC (M=4) -> konvergiert."""
    cfg = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)
    rows = []
    for c in range(4):
        res = run_adaptive_mcmc(
            cfg, beta_target=1.0, n_steps=2500, burn_in=500, seed=1000 + c, beta_start=0.2
        )
        rows.append(res.H_traj[500:])
    r = rhat.split_rhat(np.vstack(rows))
    assert r.rhat < 1.05, r.rhat  # gut gemischt nach Burn-in (etwas lockerer als 1.01)
    assert r.ess_bulk > 4 * 100  # Vehtari-Faustregel >100/Kette


def test_rhat_edge_inputs_raise() -> None:
    """Silent-Failure-Gate: invalide Eingaben werfen sauber."""
    with pytest.raises(ValueError):
        rhat.split_rhat(np.zeros((1, 100)))  # M<2
    with pytest.raises(ValueError):
        rhat.split_rhat(np.zeros((4, 3)))  # n<4
    with pytest.raises(ValueError):
        rhat.split_rhat(np.zeros(100))  # not 2D
    with pytest.raises(ValueError):
        bad = np.zeros((4, 100))
        bad[0, 0] = np.nan
        rhat.split_rhat(bad)  # non-finite


def test_identical_chains_rhat_one() -> None:
    """Entarteter Fall: identische konstante Ketten -> R-hat == 1 (kein Crash/NaN)."""
    chains = np.ones((4, 100))
    r = rhat.split_rhat(chains)
    assert np.isfinite(r.rhat)
    assert r.rhat == pytest.approx(1.0, abs=1e-9)


def _ar1_chains(m: int, n: int, phi: float, seed: int) -> np.ndarray:
    """M autokorrelierte AR(1)-Ketten -- gesunde, aber NICHT iid Referenz."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((m, n))
    x = np.empty((m, n))
    x[:, 0] = eps[:, 0] / np.sqrt(1.0 - phi**2)  # stationaerer Start
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x


def test_frozen_chains_have_zero_ess() -> None:
    """Entartet: eingefrorene Ketten (Varianz 0) -> ESS == 0, nicht volle Stichprobe.

    REGRESSION. _ess_on gab im Zweig w<=0 AND b<=0 frueher 2M*N zurueck, also die
    MAXIMAL moegliche wirksame Stichprobe fuer eine Kette, die sich nie bewegt hat.
    Eine Kette mit Varianz null hat die Zielverteilung nicht abgetastet und traegt
    null Information; jede positive ESS waere eine Aussage ueber eine Stichprobe,
    die es nicht gibt.
    """
    chains = np.full((4, 2000), 7.0)
    r = rhat.split_rhat(chains)
    assert r.ess_bulk == 0.0, r.ess_bulk
    # tail faellt mangels nicht-entarteter Quantil-Indikatoren auf ess_bulk zurueck.
    assert r.ess_tail == 0.0, r.ess_tail


def test_frozen_chains_score_worse_than_healthy_chains() -> None:
    """Die eigentliche Defekt-Aussage: eingefroren darf nicht BESSER dastehen als gesund.

    Vor dem Fix bekam der eingefrorene Fall ess_bulk = 2M*N (Maximum), waehrend die
    gesunde, autokorrelierte Kette durch ihre Autokorrelation zurecht darunter lag --
    die Diagnostik bewertete den informationslosen Lauf als den besseren.
    """
    frozen = np.full((4, 2000), 7.0)
    healthy = _ar1_chains(4, 2000, phi=0.8, seed=20260828)

    r_frozen = rhat.split_rhat(frozen)
    r_healthy = rhat.split_rhat(healthy)

    # Non-vakuoes: die gesunde Referenz muss ueberhaupt eine nennenswerte ESS haben.
    assert r_healthy.ess_bulk > 100.0, r_healthy.ess_bulk
    assert r_frozen.ess_bulk < r_healthy.ess_bulk, (r_frozen.ess_bulk, r_healthy.ess_bulk)
    assert r_frozen.ess_tail < r_healthy.ess_tail, (r_frozen.ess_tail, r_healthy.ess_tail)


def test_ess_on_degenerate_branches_return_zero() -> None:
    """Unit-Guard auf BEIDE Entartungs-Zweige von _ess_on (w <= 0).

    Direkt auf _ess_on, weil der Zweig w<=0 AND b>0 ueber split_rhat praktisch
    nicht erreichbar ist (siehe test_constant_chains_different_means_flagged):
    rank_normalize erzeugt dort irrationale, aber bit-identische Werte, deren
    np.var-Residuum ~1e-32 > 0 ist. Der Zweig existiert trotzdem und wird hier
    an seiner eigenen Naht geprueft.
    """
    # (a) w == 0, b > 0: konstante Split-Ketten auf VERSCHIEDENEN Werten.
    diff = np.vstack([np.full(1000, float(c)) for c in range(8)])
    assert float(np.mean(diff.var(axis=1, ddof=1))) == 0.0  # Vorbedingung des Zweigs
    assert rhat._ess_on(diff) == 0.0

    # (b) w == 0, b == 0: alle Split-Ketten auf DEMSELBEN Wert (der Fix).
    same = np.full((8, 1000), 3.0)
    assert float(np.mean(same.var(axis=1, ddof=1))) == 0.0
    assert rhat._ess_on(same) == 0.0
    assert rhat._rhat_on(same) == 1.0  # R-hat bleibt bewusst beim Grenzwert


def test_constant_chains_different_means_flagged() -> None:
    """Konstante Ketten mit VERSCHIEDENEN Mitteln werden als nicht-konvergiert geflaggt.

    Charakterisierung des Ist-Verhaltens auf dem oeffentlichen Pfad: hier greift
    NICHT der Entartungs-Zweig (w ~ 1e-32 > 0, s.o.), sondern die regulaere Formel.
    Sie liefert ein astronomisches R-hat -- die Nicht-Konvergenz wird also erkannt.
    """
    chains = np.vstack([np.full(2000, float(c)) for c in range(4)])
    r = rhat.split_rhat(chains)
    assert r.rhat > 1e6, r.rhat
    assert not r.converged
    assert r.ess_bulk < 10.0, r.ess_bulk  # verschwindende wirksame Stichprobe


def test_rhat_alone_does_not_catch_frozen_chains() -> None:
    """Dokumentiert die Grenze von R-hat -- und begruendet, warum ESS mitgeprueft wird.

    R-hat ist ein Verhaeltnis Between/Within und misst nur, ob die Ketten
    untereinander streuen; identische Ketten sind in diesem Sinn perfekt gemischt.
    R-hat bleibt hier bewusst beim analytischen Grenzwert 1.0 und meldet
    converged=True. Wer NUR R-hat prueft, laesst eine eingefrorene Kette durch --
    genau deshalb gehoert ESS > 0 ins Akzeptanzkriterium.
    """
    r = rhat.split_rhat(np.full((4, 2000), 7.0))
    assert r.rhat == pytest.approx(1.0, abs=1e-9)
    assert r.converged  # <- R-hat allein ist hier NICHT diskriminierend ...
    assert r.ess_bulk == 0.0  # <- ... die ESS ist es.
