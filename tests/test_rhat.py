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
