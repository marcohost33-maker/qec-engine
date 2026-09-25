"""Tests fuer rbim_fss (Issue #43: Multi-L-FSS am Nishimori-Punkt).

Orakel-Strategie (unabhaengig, KEIN Self-Check, alles CI-schnell):
- F1: Batched-Metropolis trifft die EXAKTE L=4-Boltzmann-Enumeration (frustrierte
      Bonds, 2^16 Zustaende) fuer E/N und <m^2> -- unabhaengig vom Nishimori-Satz.
- F2: Nishimori-Energie [<E>]/N = -2(1-2p) (exakt, jedes L) und die Identitaet
      [<m^2>] = [<q^2>] halten nach Thermalisierung.
- F3: NON-VAKUOES: Hot-Start ohne Burn-in verletzt die Identitaet (Gate feuert);
      der Bracket aligned-vs-hot feuert bei zu kurzem Burn-in und haelt bei langem.
- F4: Crossing- und Collapse-Schaetzer finden ein GEPFLANZTES (p_c, nu) in
      synthetischen FSS-Daten; ohne Kreuzung im Fenster -> nan (kein Claim).
- F5: Streams: reproduzierbar, zell-/start-getrennt, Bonds identisch zu rbim_scan.
- F6: Silent-Failure-Gate fuer ungueltige Eingaben.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from adaptiverg_qec import rbim_fss as F
from adaptiverg_qec.rbim_nishimori import RBIMBonds, nishimori_beta, sample_bonds
from adaptiverg_qec.rbim_scan import derive_stream_seeds


def _exact_e_m2(bonds: RBIMBonds, beta: float) -> tuple[float, float]:
    L = bonds.L
    n = L * L
    states = np.arange(2**n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(-1, L, L).astype(np.float64)
    bond_sum = (bonds.jx * s * np.roll(s, -1, axis=2)).sum(axis=(1, 2)) + (
        bonds.jy * s * np.roll(s, -1, axis=1)
    ).sum(axis=(1, 2))
    logw = beta * bond_sum
    w = np.exp(logw - logw.max())
    w /= w.sum()
    m = s.sum(axis=(1, 2)) / n
    return float((w * -bond_sum / n).sum()), float((w * m * m).sum())


# --------------------------------------------------------------------------- F1
@pytest.mark.parametrize("p", [0.11, 0.2])
def test_batched_metropolis_matches_exact_enumeration_L4(p: float) -> None:
    """The vectorised sampler targets exactly pi_J(s) ~ exp(beta sum J s s)."""
    L, beta = 4, nishimori_beta(p)
    c = F.simulate_cell(p, L, n_disorder=4, n_records=6000, burn_in=200, base_seed=3)
    for rep in range(4):
        seeds = derive_stream_seeds(base_seed=3, p=p, L=L, replicate=rep)
        b = sample_bonds(p, L, seed=seeds.bond_seed)
        e_ex, m2_ex = _exact_e_m2(b, beta)
        assert abs(c.energy[rep] - e_ex) < 0.03, (rep, c.energy[rep], e_ex)
        assert abs(c.m2[rep] - m2_ex) < 0.03, (rep, c.m2[rep], m2_ex)
        # two independent replicas: <q^2> = sum_ij <s_i s_j>^2 / N^2 <= <m^2>-scale, finite
        assert 0.0 <= c.q2[rep] <= 1.0


# --------------------------------------------------------------------------- F2
def test_nishimori_energy_formula() -> None:
    for p in (0.05, 0.1094, 0.3):
        assert math.isclose(F.nishimori_energy(p), -2.0 * (1.0 - 2.0 * p), rel_tol=1e-12)


def test_nishimori_identities_hold_after_thermalisation() -> None:
    c = F.simulate_cell(0.13, 8, n_disorder=96, n_records=300, burn_in=600, base_seed=11)
    chk = F.nishimori_checks(c)
    assert chk.passes(4.0), chk
    assert abs(chk.energy - chk.energy_exact) < 0.02


# --------------------------------------------------------------------------- F3
def test_identity_gate_fires_for_unthermalised_hot_start() -> None:
    """Non-vacuous: hot start, no burn-in, few records -> q^2 lags m^2 and the gate fires."""
    c = F.simulate_cell(0.10, 12, n_disorder=64, n_records=20, burn_in=0, base_seed=5, start="hot")
    chk = F.nishimori_checks(c)
    assert not chk.passes(4.0), chk
    assert chk.m2_minus_q2 > 0.0


def test_bracket_fires_short_and_holds_long() -> None:
    kw = dict(n_disorder=64, n_records=100, base_seed=9)
    short = F.equilibration_bracket(
        F.simulate_cell(0.11, 8, burn_in=0, start="aligned", **kw),
        F.simulate_cell(0.11, 8, burn_in=0, start="hot", **kw),
    )
    assert short["z"] > 4.0, short  # aligned relaxes from above, hot from below
    long = F.equilibration_bracket(
        F.simulate_cell(0.11, 8, burn_in=1500, start="aligned", **kw),
        F.simulate_cell(0.11, 8, burn_in=1500, start="hot", **kw),
    )
    assert abs(long["z"]) < 3.0, long


def test_doubling_detects_short_burn_in() -> None:
    kw = dict(n_disorder=64, n_records=50, base_seed=13, start="aligned")
    long = F.simulate_cell(0.11, 10, burn_in=2000, **kw)
    short = F.simulate_cell(0.11, 10, burn_in=0, **kw)
    r = F.equilibration_doubling(long, short)
    assert r["diff_short_minus_long"] > 0.0 and r["z"] > 3.0, r  # relaxes from above
    with pytest.raises(ValueError):
        F.equilibration_doubling(short, long)


def test_bracket_requires_matching_cells() -> None:
    kw = dict(n_disorder=4, n_records=2, burn_in=0, base_seed=1)
    a = F.simulate_cell(0.11, 4, start="aligned", **kw)
    h = F.simulate_cell(0.12, 4, start="hot", **kw)
    with pytest.raises(ValueError):
        F.equilibration_bracket(a, h)
    with pytest.raises(ValueError):
        F.equilibration_bracket(a, a)


# --------------------------------------------------------------------------- F4
PC_TRUE, NU_TRUE = 0.109, 1.5


def _synthetic_cells(noise: float, seed: int, ps=None, Ls=(8, 12, 16, 24)) -> list[F.CellSamples]:
    """Cells whose U4 follows an exact scaling form f((p-pc) L^{1/nu}), plus noise.

    Constructed through the per-realisation arrays so that the SAME bootstrap
    machinery as in production runs: m2 = 1 per realisation, m4 = U4 + eps.
    """
    rng = np.random.default_rng(seed)
    ps = ps if ps is not None else np.linspace(0.095, 0.125, 7)
    cells = []
    for L in Ls:
        for p in ps:
            x = (p - PC_TRUE) * L ** (1.0 / NU_TRUE)
            u4 = 1.0 + 2.0 / (1.0 + math.exp(-8.0 * x))
            n = 200
            m4 = u4 + noise * rng.standard_normal(n)
            ones = np.ones(n)
            # xi/L also scaling: choose fk so that xi/L = g(x) exactly on average
            r = 0.8 - 0.4 * math.tanh(6.0 * x)
            chi = L * L
            fk = chi / (1.0 + (r * L * 2.0 * math.sin(math.pi / L)) ** 2) * ones
            cells.append(
                F.CellSamples(
                    p=float(p), L=L, beta=nishimori_beta(p), n_disorder=n, n_records=1,
                    burn_in=0, n_skip=1, start="aligned", m2=ones, m4=m4, q2=ones,
                    q4=m4, fk=fk, energy=ones * F.nishimori_energy(p), tau_int_records=0.5,
                )
            )  # fmt: skip
    return cells


def test_crossing_point_recovers_planted_pc() -> None:
    cells = _synthetic_cells(noise=0.02, seed=1)
    res = F.bootstrap_crossings(cells, obs="u4", n_boot=100, seed=2)
    for pair, r in res.items():
        assert abs(r["p_cross"] - PC_TRUE) < 0.003, (pair, r)
        assert abs(r["p_cross"] - PC_TRUE) < 3.0 * r["boot_se"], (pair, r)
    res_xi = F.bootstrap_crossings(cells, obs="xi_over_l", n_boot=20, seed=3)
    for pair, r in res_xi.items():
        assert abs(r["p_cross"] - PC_TRUE) < 1e-3, (pair, r)


def test_collapse_recovers_planted_pc_and_nu() -> None:
    cells = _synthetic_cells(noise=0.02, seed=4)
    res = F.bootstrap_collapse(cells, obs="u4", n_boot=60, seed=5, deg=5)
    assert abs(res["pc"] - PC_TRUE) < 0.002, res
    assert abs(res["nu"] - NU_TRUE) < 0.25, res
    assert res["pc_ci_lo"] <= PC_TRUE <= res["pc_ci_hi"], res


def test_crossing_is_nan_without_sign_change() -> None:
    ps = [0.12, 0.13, 0.14, 0.15]
    assert math.isnan(F.crossing_point(ps, [1.0, 1.1, 1.2, 1.3], [1.5, 1.6, 1.7, 1.8]))
    # sanity: exact linear crossing
    assert F.crossing_point(ps, [1.0, 1.0, 1.0, 1.0], [0.9, 0.95, 1.05, 1.1]) == pytest.approx(
        0.135, abs=1e-9
    )


def test_crossing_rejects_bad_grid() -> None:
    with pytest.raises(ValueError):
        F.crossing_point([0.1, 0.1, 0.2], [1, 2, 3], [3, 2, 1])
    with pytest.raises(ValueError):
        F.crossing_point([0.1, 0.2], [1, 2], [2, 1])  # fewer than deg+1 points


def test_collapse_input_validation() -> None:
    ps = np.linspace(0.1, 0.12, 10)
    ones = np.ones(10)
    with pytest.raises(ValueError, match="two lattice sizes"):
        F.fit_collapse(ps, 8 * ones, ones, ones, pc0=0.11, nu0=1.5)
    with pytest.raises(ValueError, match="sigma"):
        F.fit_collapse(ps, np.repeat([8, 16], 5), ones, 0 * ones, pc0=0.11, nu0=1.5)
    with pytest.raises(ValueError, match="more than"):
        F.fit_collapse(ps[:5], np.repeat([8, 16], 5)[:5], ones[:5], ones[:5], pc0=0.11, nu0=1.5)


def test_grid_must_be_complete() -> None:
    cells = _synthetic_cells(noise=0.0, seed=0, Ls=(8, 12))
    with pytest.raises(ValueError, match="complete"):
        F.bootstrap_crossings(cells[:-1], obs="u4", n_boot=10, seed=0)
    with pytest.raises(ValueError, match="unknown observable"):
        F.bootstrap_crossings(cells, obs="energy", n_boot=10, seed=0)


# --------------------------------------------------------------------------- F5
def test_simulation_is_reproducible_and_streams_are_separated() -> None:
    kw = dict(n_disorder=6, n_records=10, burn_in=5, base_seed=21)
    a = F.simulate_cell(0.11, 6, **kw)
    b = F.simulate_cell(0.11, 6, **kw)
    np.testing.assert_array_equal(a.m2, b.m2)
    np.testing.assert_array_equal(a.energy, b.energy)
    c = F.simulate_cell(0.11, 6, start="hot", **kw)
    assert not np.array_equal(a.m2, c.m2)
    seqs = {
        tuple(F.thermal_seed_sequence(21, p, L, k, st).generate_state(4))
        for p in (0.1, 0.11)
        for L in (8, 12)
        for k in (0, 1)
        for st in ("aligned", "hot")
    }
    assert len(seqs) == 16


def test_chunking_keeps_bonds_per_replicate() -> None:
    """Bonds depend on the replicate index only; chunk size changes thermal streams only."""
    kw = dict(n_disorder=6, n_records=400, burn_in=200, base_seed=4)
    whole = F.simulate_cell(0.12, 4, chunk_size=6, **kw)
    split = F.simulate_cell(0.12, 4, chunk_size=2, **kw)
    # Same bonds -> exact NL energy per realisation agrees within thermal noise.
    assert np.max(np.abs(whole.energy - split.energy)) < 0.1


# --------------------------------------------------------------------------- F6
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(p=0.11, L=5, n_disorder=4, n_records=2, burn_in=0, base_seed=0),
        dict(p=0.11, L=2, n_disorder=4, n_records=2, burn_in=0, base_seed=0),
        dict(p=0.11, L=4, n_disorder=1, n_records=2, burn_in=0, base_seed=0),
        dict(p=0.11, L=4, n_disorder=4, n_records=0, burn_in=0, base_seed=0),
        dict(p=0.11, L=4, n_disorder=4, n_records=2, burn_in=-1, base_seed=0),
        dict(p=0.11, L=4, n_disorder=4, n_records=2, burn_in=0, base_seed=-1),
        dict(p=0.6, L=4, n_disorder=4, n_records=2, burn_in=0, base_seed=0),
        dict(p=0.11, L=4, n_disorder=4, n_records=2, burn_in=0, base_seed=0, start="cold"),
        dict(p=0.11, L=4, n_disorder=4, n_records=2, burn_in=0, base_seed=0, n_skip=0),
    ],
)
def test_invalid_inputs_fail_closed(kwargs: dict) -> None:
    p = kwargs.pop("p")
    L = kwargs.pop("L")
    with pytest.raises(ValueError):
        F.simulate_cell(p, L, **kwargs)


def test_run_fss_requires_burn_in_per_L_and_on_grid_bracket() -> None:
    with pytest.raises(ValueError, match="burn_in"):
        F.run_fss(Ls=[4, 6], ps=[0.1, 0.12, 0.14], n_disorder=2, n_records=1,
                  burn_in={4: 1}, base_seed=0, verbose=False)  # fmt: skip
    with pytest.raises(ValueError, match="bracket"):
        F.run_fss(Ls=[4, 6], ps=[0.1, 0.12, 0.14], n_disorder=2, n_records=1,
                  burn_in={4: 1, 6: 1}, base_seed=0, bracket_ps=[0.13], verbose=False)  # fmt: skip


def test_run_fss_end_to_end_tiny() -> None:
    """Smoke: complete evidence pack structure on a tiny grid (no physics claim)."""
    rep = F.run_fss(
        Ls=[4, 6, 8],
        ps=[0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        n_disorder=8,
        n_records=20,
        burn_in={4: 20, 6: 20, 8: 20},
        base_seed=1,
        n_boot=12,
        bracket_ps=[0.11],
        verbose=False,
    )
    for key in ("claim_tier", "rng_policy", "environment", "parameters", "nishimori_oracles"):
        assert key in rep
    assert len(rep["cells"]) == 21
    assert rep["equilibration_brackets"][0]["L"] == 8
    dbl = rep["equilibration_doubling"][0]
    assert (dbl["burn_in_long"], dbl["burn_in_short"]) == (20, 5)
    assert set(rep["crossings"]["u4"]) == {"4-6", "4-8", "6-8"}
    assert "drop_smallest_L" in rep["collapse"]["u4"]
    assert all(b >= a for a, b in pairwise(rep["parameters"]["ps"]))
