"""Inkrement 3.4: Binomial-Intervalle, exakte Coverage, FSS-Fit + Bootstrap (ohne Extras)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from adaptiverg_qec import qec_fss as f

# --- Intervalle ----------------------------------------------------------------


@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (5, 10), (10, 10), (0, 50_000), (37, 50_000)])
def test_wilson_is_score_test_inversion(k, n):
    z = stats.norm.ppf(0.975)
    lo, hi = f.wilson_interval(k, n)
    for edge in (lo, hi):
        if 0.0 < edge < 1.0:
            # Auf dem Rand ist der Score-Test genau auf dem Niveau: |ph-p| = z*sqrt(p(1-p)/n).
            assert abs(k / n - edge) == pytest.approx(
                z * math.sqrt(edge * (1 - edge) / n), rel=1e-9
            )


@pytest.mark.parametrize("k,n", [(1, 10), (5, 10), (37, 50_000)])
def test_jeffreys_equals_beta_quantiles(k, n):
    lo, hi = f.jeffreys_interval(k, n)
    ref = stats.beta(k + 0.5, n - k + 0.5)
    assert lo == pytest.approx(ref.ppf(0.025), abs=1e-14)
    assert hi == pytest.approx(ref.ppf(0.975), abs=1e-14)


def test_boundary_conventions_and_wald_degeneracy():
    assert f.jeffreys_interval(0, 100)[0] == 0.0
    assert f.jeffreys_interval(100, 100)[1] == 1.0
    assert f.wilson_interval(0, 100)[1] > 0.0  # nicht entartet bei k=0
    assert f.wald_interval(0, 100) == (0.0, 0.0)  # genau der Defekt, den F2 zeigt


@pytest.mark.parametrize("bad", [(-1, 10), (11, 10), (0, 0), (1.5, 10), (True, 10)])
def test_interval_inputs_fail_closed(bad):
    with pytest.raises((TypeError, ValueError)):
        f.wilson_interval(*bad)


def test_exact_coverage_matches_brute_force_sum():
    n, p = 30, 0.07
    lo_hi = [f.wilson_interval(k, n) for k in range(n + 1)]
    ref = sum(stats.binom.pmf(k, n, p) for k, (lo, hi) in enumerate(lo_hi) if lo <= p <= hi)
    assert f.exact_coverage("wilson", n, p) == pytest.approx(ref, abs=1e-14)


def test_exact_coverage_rare_events_two_sided():
    grid = np.geomspace(2e-5, 2e-3, 9)
    wilson = np.mean([f.exact_coverage("wilson", 20_000, float(p)) for p in grid])
    wald = np.mean([f.exact_coverage("wald", 20_000, float(p)) for p in grid])
    cp_min = min(f.exact_coverage("clopper_pearson", 20_000, float(p)) for p in grid)
    assert 0.93 < wilson < 0.97
    assert wald < 0.85
    assert cp_min >= 0.95


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, float("nan")])
def test_exact_coverage_rejects_degenerate_p(p):
    with pytest.raises(ValueError):
        f.exact_coverage("wilson", 100, p)


def test_exact_coverage_unknown_method():
    with pytest.raises(ValueError):
        f.exact_coverage("agresti", 100, 0.1)


# --- FSS ------------------------------------------------------------------------

DS = (9, 15, 21, 27)
PG = tuple(float(x) for x in np.linspace(0.47, 0.53, 7))


def test_nu_oracle_closed_form():
    # d=3: 3*C(2,1)/4 = 1.5 ; asymptotisch sqrt(2d/pi).
    assert f._rep_slope_at_half(3) == pytest.approx(1.5, abs=1e-14)
    assert f._rep_slope_at_half(401) == pytest.approx(math.sqrt(2 * 401 / math.pi), rel=2e-3)
    nu = f.repetition_nu_oracle(DS)["nu_effective"]
    assert 2.0 < nu < 2.15


def test_noiseless_oracle_fit_recovers_exact_threshold():
    # Unendlich viele Shots: k/n = exaktes p_L. Der Fit muss p_th=1/2 praktisch exakt treffen.
    n = 10**9
    cells = [f.Cell(d, p, round(f.logical_error_rate_exact(d, p) * n), n) for d in DS for p in PG]
    fit = f.fss_fit(cells)
    assert fit["p_th"] == pytest.approx(0.5, abs=2e-4)
    assert fit["p_th_in_window"] and not fit["at_parameter_bound"]


def test_bootstrap_ci_covers_oracle_and_is_reproducible():
    cells = f.repetition_fss_dataset(DS, PG, 50_000, 11)
    a = f.fss_bootstrap(cells, n_boot=100, seed=3)
    b = f.fss_bootstrap(cells, n_boot=100, seed=3)
    assert a == b
    assert a["p_th_ci"][0] <= 0.5 <= a["p_th_ci"][1]
    nu = f.repetition_nu_oracle(DS)["nu_effective"]
    assert a["nu_ci"][0] <= nu <= a["nu_ci"][1]
    assert a["fit_success_fraction"] == 1.0
    assert [c["d_pair"] for c in a["crossings"]] == [[9, 15], [15, 21], [21, 27]]


def test_dataset_cells_have_independent_streams():
    cells = f.repetition_fss_dataset((9, 15), (0.5, 0.5 + 1e-9, 0.5 + 2e-9), 10_000, 1)
    ks = [c.k for c in cells]
    assert len(set(ks)) > 1  # gleiche Rate, eigene Streams -> verschiedene Ziehungen


def test_negative_control_no_crossing_is_flagged():
    rng = np.random.Generator(np.random.Philox(5))
    cells = [
        f.Cell(d, p, int(rng.binomial(50_000, p * (1 + 0.05 * d))), 50_000)
        for d in (5, 7, 9)
        for p in (0.1, 0.12, 0.14, 0.16)
    ]
    assert all(c["crossing"] is None for c in f.pairwise_crossings(cells))
    fit = f.fss_fit(cells)
    assert (not fit["p_th_in_window"]) or fit["at_parameter_bound"] or fit["chi2_p_value"] < 1e-3


def test_pairwise_crossing_linear_interpolation():
    cells = [
        f.Cell(3, 0.1, 10, 100),
        f.Cell(3, 0.2, 20, 100),
        f.Cell(5, 0.1, 5, 100),
        f.Cell(5, 0.2, 25, 100),
    ]
    # diff = -0.05 bei 0.1, +0.05 bei 0.2 -> Kreuzung genau 0.15
    assert f.pairwise_crossings(cells)[0]["crossing"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    "cells",
    [
        [f.Cell(3, 0.1, 1, 10)] * 5,  # zu wenige Zellen
        [f.Cell(3, 0.1 + 0.01 * i, 1, 10) for i in range(6)],  # nur eine Distanz
        [f.Cell(3 + 2 * (i % 3), 0.1 + 0.01 * (i % 2), 1, 10) for i in range(6)],  # 2 p-Werte
    ],
)
def test_fss_fit_rejects_underdetermined_grids(cells):
    with pytest.raises(ValueError):
        f.fss_fit(cells)


@pytest.mark.parametrize(
    "kw",
    [
        {"d": 0, "p": 0.1, "k": 0, "n": 1},
        {"d": 3, "p": 0.0, "k": 0, "n": 1},
        {"d": 3, "p": 0.1, "k": 2, "n": 1},
    ],
)
def test_cell_validation(kw):
    with pytest.raises(ValueError):
        f.Cell(**kw)


# --- Systematik ------------------------------------------------------------------


def test_systematics_two_sided():
    clean = f.fss_systematics(f.repetition_fss_dataset(DS, PG, 50_000, 11), n_boot=60, seed=6)
    assert not clean["any_significant_shift"]
    drift = f.fss_systematics(
        f.drifted_repetition_dataset(DS, PG, 50_000, 12, c=0.3), n_boot=60, seed=7
    )
    assert drift["shifts"]["drop_smallest_d"]["significant"]
    assert drift["shifts"]["drop_smallest_d"]["delta"] > 0  # Richtung: ohne d_min hoeher


def test_systematics_combined_width_is_conservative():
    cells = f.repetition_fss_dataset(DS, PG, 50_000, 11)
    out = f.fss_systematics(cells, n_boot=50, seed=1, stat_ci=(0.499, 0.501))
    assert out["stat_half_width"] == pytest.approx(0.001)
    assert out["combined_half_width"] >= max(out["stat_half_width"], out["max_abs_shift"])
    assert set(out["variants"]) == {"drop_smallest_d", "drop_largest_d", "inner_window"}


def test_systematics_skips_variants_the_grid_cannot_support():
    cells = f.repetition_fss_dataset((9, 15), (0.47, 0.5, 0.53), 10_000, 1)
    out = f.fss_systematics(cells, n_boot=50, seed=1)
    assert out["variants"] == {} and out["max_abs_shift"] == 0.0


def test_drifted_dataset_moves_crossings_with_distance():
    n = 10**9
    ds = (9, 27)
    exact = [
        f.Cell(d, p, round(f.logical_error_rate_exact(d, p - 0.2 / d) * n), n)
        for d in ds
        for p in PG
    ]
    cross = f.pairwise_crossings(exact)[0]["crossing"]
    assert cross is not None and cross < 0.5  # Drift ist real, nicht nur Rauschen
