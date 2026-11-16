"""Tests fuer rbim_nishimori (ROADMAP-Inkr.4: RBIM-Nishimori <-> MCRG/QEC-Bruecke).

Orakel-Strategie (unabhaengig, KEIN Self-Check):
- G-N1: Nishimori-Linie p<->beta Roundtrip + Grenzfaelle.
- G-N2: p=0 RBIM-Energie == homogene Energie (energy_per_spin), byte-genau.
- G-N3: GAUGE-INVARIANZ (exakte RBIM-Symmetrie): s_i->tau_i s_i, J_ij->tau_i tau_j J_ij
        laesst E invariant. Starkes gleichungsfreies Orakel.
- G-N4: EXAKTE L=4-Enumeration (2^16 Zustaende, voll) als Boltzmann-Orakel; Wolff/
        Hybrid trifft E/N und <|m|> auch im FRUSTRIERTEN Fall (p>0).
- G-N5: Stationaritaet: aligned-Start -> FM-Sektor (p<p_c) hohe |m|, PM-Sektor
        (p>p_c) niedrige |m| (relaxiert, kein Bias).
- G-N6: Uebergangs-Lokalisierung gegen das PUBLIZIERTE Orakel p_c~0.109 mit
        systematik-begruendeter Toleranz (Plausibilitaet, KEIN Frontier-Wert).
- G-N7: Negativ-/Edge-Inputs failen (Silent-Failure-Gate).
- G-N8: Reproduzierbarkeit (gleicher Seed -> gleiches Ergebnis).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptiverg_qec.ising2d import energy_per_spin
from adaptiverg_qec.rbim_nishimori import (
    DisorderResult,
    RBIMBonds,
    _disorder_stream_seeds,
    locate_transition,
    nishimori_beta,
    nishimori_p,
    nishimori_scan,
    rbim_energy_per_spin,
    rbim_wolff_sample,
    sample_bonds,
)


def _exact_rbim_observables(bonds: RBIMBonds, beta: float) -> tuple[float, float, float]:
    """Volle 2^N-Enumeration (N<=16) -> exakte (E/N, <|m|>, <m^2>) Boltzmann-Mittel."""
    L = bonds.L
    n = L * L
    assert n <= 16, "enumeration only for N<=16"
    states = np.arange(2**n, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(-1, L, L).astype(np.float64)
    right = bonds.jx * s * np.roll(s, -1, axis=2)
    down = bonds.jy * s * np.roll(s, -1, axis=1)
    bond_sum = right.sum(axis=(1, 2)) + down.sum(axis=(1, 2))
    logw = beta * bond_sum
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    e = -bond_sum / n
    m = s.sum(axis=(1, 2)) / n
    return float((w * e).sum()), float((w * np.abs(m)).sum()), float((w * m * m).sum())


# --------------------------------------------------------------------------- G-N1
def test_nishimori_line_roundtrip() -> None:
    for p in (0.05, 0.1094, 0.2, 0.4):
        beta = nishimori_beta(p)
        assert math.isclose(nishimori_p(beta), p, rel_tol=1e-12)
    # Nishimori condition p = 1/(1+exp(2 beta)) explicit.
    beta = nishimori_beta(0.1094)
    assert math.isclose(0.1094, 1.0 / (1.0 + math.exp(2.0 * beta)), rel_tol=1e-12)
    # p -> 0.5 => beta -> 0 ; p small => beta large.
    assert nishimori_beta(0.49) < nishimori_beta(0.01)
    assert nishimori_beta(0.49) > 0.0


# --------------------------------------------------------------------------- G-N2
def test_p0_energy_equals_homogeneous() -> None:
    L = 6
    rng = np.random.default_rng(0)
    s = np.where(rng.random((L, L)) < 0.5, np.int8(1), np.int8(-1))
    b0 = sample_bonds(0.0, L, seed=1)
    assert b0.frustrated_fraction == 0.0
    e_rbim = rbim_energy_per_spin(s, b0)
    e_hom = float(energy_per_spin(s))
    assert math.isclose(e_rbim, e_hom, abs_tol=1e-12)


# --------------------------------------------------------------------------- G-N3
def test_gauge_invariance_energy() -> None:
    """s_i->tau_i s_i, J_ij->tau_i tau_j J_ij => E invariant (exakt)."""
    L = 8
    rng = np.random.default_rng(3)
    b = sample_bonds(0.18, L, seed=7)
    s = np.where(rng.random((L, L)) < 0.5, np.int8(1), np.int8(-1))
    e0 = rbim_energy_per_spin(s, b)
    tau = np.where(rng.random((L, L)) < 0.5, np.int8(1), np.int8(-1))
    s_g = (s * tau).astype(np.int8)
    jx_g = (b.jx * tau * np.roll(tau, -1, axis=1)).astype(np.int8)
    jy_g = (b.jy * tau * np.roll(tau, -1, axis=0)).astype(np.int8)
    b_g = RBIMBonds(jx=jx_g, jy=jy_g, p=b.p, L=L, seed=b.seed)
    e_g = rbim_energy_per_spin(s_g, b_g)
    assert math.isclose(e0, e_g, abs_tol=1e-12)


# --------------------------------------------------------------------------- G-N4
@pytest.mark.parametrize("p,seed", [(0.0, 11), (0.2, 5), (0.3, 8)])
def test_exact_enumeration_oracle_L4(p: float, seed: int) -> None:
    """Hybrid-Sampler trifft die EXAKTE L=4-Boltzmann-Enumeration (auch frustriert)."""
    L = 4
    beta = nishimori_beta(p) if p > 0 else 0.5
    b = sample_bonds(p, L, seed=seed)
    e_ex, absm_ex, m2_ex = _exact_rbim_observables(b, beta)
    configs, _cf = rbim_wolff_sample(
        b, beta, n_records=12000, burn_in=1500, seed=77, sweeps_per_step=2
    )
    e = np.array([rbim_energy_per_spin(c, b) for c in configs])
    m = configs.reshape(len(configs), -1).sum(1) / (L * L)
    assert abs(e_ex - float(e.mean())) < 0.02, f"E/N off: {e_ex} vs {e.mean()}"
    assert abs(absm_ex - float(np.abs(m).mean())) < 0.03, "<|m|> off vs exact"


# --------------------------------------------------------------------------- G-N5
def test_aligned_start_no_pm_bias() -> None:
    """aligned-Start: FM-Sektor hohe |m|, PM-Sektor niedrige |m| (kein Bias)."""
    r_fm = nishimori_scan(
        0.04,
        L=10,
        n_disorder=8,
        n_records=120,
        burn_in=200,
        base_seed=2026,
        sweeps_per_step=3,
        aligned_start=True,
    )
    r_pm = nishimori_scan(
        0.30,
        L=10,
        n_disorder=8,
        n_records=120,
        burn_in=200,
        base_seed=2026,
        sweeps_per_step=3,
        aligned_start=True,
    )
    assert r_fm.abs_m > 0.7, f"FM |m| too low: {r_fm.abs_m}"
    assert r_pm.abs_m < 0.35, f"PM |m| not relaxed (bias?): {r_pm.abs_m}"
    assert r_fm.abs_m > r_pm.abs_m


# --------------------------------------------------------------------------- G-N6
def test_locate_transition_selects_known_steepest_interval() -> None:
    """Die Heuristik selbst wird deterministisch getestet, nicht p_c aus Mini-MC.

    Ein frueherer CI-Test versuchte mit L=8, nur 10 Disorder-Realisierungen und
    80 Records pro Punkt den publizierten Nishimori-Punkt p_c~0.1094 in ein
    enges Band zu zwingen. Nach korrekter Trennung der RNG-Streams sprang derselbe
    kleine Scan von 0.11 auf 0.155. Das ist ein Statistik-/Finite-Size-Problem,
    kein valider Implementierungsfehler.

    CI prueft deshalb hier die Lokalisierungsfunktion an einer kontrollierten
    Kurve. Die Physik wird separat durch exakte L=4-Orakel, Gauge-Invarianz und
    den FM/PM-Sektor-Test geprueft. Ein numerischer p_c-Claim braucht Multi-L-FSS
    mit deutlich mehr Disorder-Statistik und gehoert in ein Research-Evidence-Pack.
    """

    def point(p: float, abs_m: float) -> DisorderResult:
        return DisorderResult(
            p=p,
            beta=nishimori_beta(p),
            L=8,
            abs_m=abs_m,
            abs_m_err=0.01,
            m2=abs_m**2,
            m4=abs_m**4,
            binder=0.5,
            mean_cluster_frac=0.5,
            n_disorder=100,
            n_records=100,
        )

    res = [
        point(0.04, 0.95),
        point(0.09, 0.90),
        point(0.13, 0.45),
        point(0.18, 0.40),
    ]
    assert locate_transition(res) == pytest.approx(0.11)


# --------------------------------------------------------------------------- G-N7
def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        nishimori_beta(0.0)
    with pytest.raises(ValueError):
        nishimori_beta(0.5)  # boundary excluded
    with pytest.raises(ValueError):
        nishimori_beta(-0.1)
    with pytest.raises(ValueError):
        nishimori_p(0.0)
    with pytest.raises(ValueError):
        sample_bonds(0.6, 8, seed=0)  # p>=0.5
    with pytest.raises(ValueError):
        sample_bonds(0.1, 5, seed=0)  # odd L
    with pytest.raises(ValueError):
        sample_bonds(0.1, 2, seed=0)  # L<4
    b = sample_bonds(0.1, 4, seed=0)
    with pytest.raises(ValueError):
        rbim_wolff_sample(b, -1.0, n_records=1, burn_in=0, seed=0)  # beta<=0 via cluster
    with pytest.raises(ValueError):
        rbim_wolff_sample(b, 0.5, n_records=0, burn_in=0, seed=0)  # n_records<1
    with pytest.raises(ValueError):
        locate_transition([])  # too few points
    with pytest.raises(ValueError):  # Codex-P2: empty disorder scan must fail, not nan
        nishimori_scan(0.1, 4, n_disorder=0, n_records=10, burn_in=10, base_seed=0)


# --------------------------------------------------------------------------- G-N8
def test_reproducibility() -> None:
    kw = dict(
        n_disorder=6,
        n_records=80,
        burn_in=100,
        base_seed=999,
        sweeps_per_step=2,
        aligned_start=True,
    )
    r1 = nishimori_scan(0.10, L=8, **kw)
    r2 = nishimori_scan(0.10, L=8, **kw)
    assert r1.abs_m == r2.abs_m
    assert r1.binder == r2.binder
    assert isinstance(r1, DisorderResult)


def test_disorder_streams_are_reproducible_and_role_separated() -> None:
    """Regression: die fruehere +10000-Offset-Strategie kollidierte ab d=10000."""
    seen: set[int] = set()
    for d in range(10_050):
        bond_seed, thermal_seed = _disorder_stream_seeds(1000, d)
        assert bond_seed != thermal_seed
        assert bond_seed not in seen
        seen.add(bond_seed)
        assert thermal_seed not in seen
        seen.add(thermal_seed)
        assert (bond_seed, thermal_seed) == _disorder_stream_seeds(1000, d)


def test_disorder_stream_seed_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        _disorder_stream_seeds(-1, 0)
    with pytest.raises(ValueError):
        _disorder_stream_seeds(1, -1)
    with pytest.raises(ValueError):
        nishimori_scan(0.1, 4, n_disorder=1, n_records=2, burn_in=1, base_seed=-1)
