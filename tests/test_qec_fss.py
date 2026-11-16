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


_EDGE_KN = [(0, 1), (1, 1), (0, 10), (1, 10), (9, 10), (10, 10), (0, 50_000), (50_000, 50_000)]


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.32])
@pytest.mark.parametrize("k,n", _EDGE_KN + [(37, 50_000), (2500, 50_000)])
def test_intervals_match_scipy_binomtest_and_beta_quantiles(k, n, alpha):
    """Unabhaengiges Orakel: scipy.stats.binomtest (Wilson ohne Stetigkeitskorrektur,
    'exact' = Clopper-Pearson) und Beta-Quantile fuer Jeffreys, inkl. k=0 / k=n / n=1."""
    from adaptiverg_qec.qec_decoder_ab import clopper_pearson

    ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="wilson")
    w = f.wilson_interval(k, n, alpha)
    assert w[0] == pytest.approx(ci.low, abs=1e-12)
    assert w[1] == pytest.approx(ci.high, abs=1e-12)

    ex = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    cp = clopper_pearson(k, n, alpha)
    assert cp[0] == pytest.approx(ex.low, abs=1e-12)
    assert cp[1] == pytest.approx(ex.high, abs=1e-12)

    j = f.jeffreys_interval(k, n, alpha)
    beta = stats.beta(k + 0.5, n - k + 0.5)
    assert j[0] == (0.0 if k == 0 else pytest.approx(beta.ppf(alpha / 2), abs=1e-14))
    assert j[1] == (1.0 if k == n else pytest.approx(beta.ppf(1 - alpha / 2), abs=1e-14))
    # Ordnung + Einschluss der Punktschaetzung bei allen drei gueltigen Intervallen.
    for lo, hi in (w, cp, j):
        assert 0.0 <= lo <= k / n <= hi <= 1.0


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


def test_no_drift_verdict_fails_closed_when_every_paired_refit_fails(monkeypatch):
    cells = f.repetition_fss_dataset(DS, PG, 50_000, 11)
    clean = f.fss_systematics(cells, n_boot=50, seed=6)
    assert clean["undetermined_shifts"] == [] and f.no_drift_verdict(clean)  # Positivkontrolle

    real = f._fit_core

    def broken(d, p, y, w, starts, bounds):  # Bootstrap-Refits (ein Warmstart) scheitern
        return None if len(starts) == 1 else real(d, p, y, w, starts, bounds)

    monkeypatch.setattr(f, "_fit_core", broken)
    blind = f.fss_systematics(cells, n_boot=50, seed=6)
    assert not blind["any_significant_shift"]  # genau die alte, blinde Lesart
    assert blind["undetermined_shifts"] == sorted(blind["shifts"])
    assert not f.no_drift_verdict(blind)


def test_no_drift_verdict_requires_at_least_one_shift():
    out = f.fss_systematics(
        f.repetition_fss_dataset((9, 15), (0.47, 0.5, 0.53), 10_000, 1), n_boot=50, seed=1
    )
    assert out["shifts"] == {} and not f.no_drift_verdict(out)


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


# --- Codex-P2 (#67): Identifizierbarkeit + Konvergenz -------------------------------


def _all_zero_cells():
    return [f.Cell(d, p, 0, 1000) for d in (3, 5, 7) for p in (0.01, 0.02, 0.03)]


@pytest.mark.parametrize(
    "cells",
    [
        _all_zero_cells(),
        [f.Cell(d, p, 100, 1000) for d in (3, 5, 7) for p in (0.01, 0.02, 0.03)],  # konstant
        list(reversed(_all_zero_cells())),  # Permutation: gleiches Urteil
    ],
    ids=["all_zero", "constant_rate", "all_zero_permuted"],
)
def test_unidentifiable_fit_reports_no_threshold(cells):
    fit = f.fss_fit(cells)
    assert fit["identifiable"] is False
    assert math.isnan(fit["p_th"]) and math.isnan(fit["nu"])
    assert not fit["p_th_in_window"]
    boot = f.fss_bootstrap(cells, n_boot=50, seed=1)
    assert boot["p_th_ci"] is None and boot["nu_ci"] is None
    assert boot["fit_success_fraction"] == 0.0


def test_identifiable_fit_positive_control_survives_permutation():
    cells = f.repetition_fss_dataset(DS, PG, 50_000, 11)
    a, b = f.fss_fit(cells), f.fss_fit(list(reversed(cells)))
    assert a["identifiable"] and b["identifiable"]
    assert a["p_th"] == pytest.approx(b["p_th"], abs=1e-9)


def test_identifiability_gate_is_nan_fail_closed():
    class R:
        jac = np.array([[np.nan, 1.0], [1.0, 1.0], [0.0, 2.0]])

    y, w = np.array([0.1, 0.2, 0.3]), np.ones(3)
    assert f._identifiable(R(), y, w, ([0.0, 0.2], [1.0, 10.0])) is False


def _starved_least_squares(real):
    def starved(*args, **kwargs):  # Budget erschoepft: finite x, success=False (status 0)
        kwargs["max_nfev"] = 1
        return real(*args, **kwargs)

    return starved


def test_non_converged_refits_are_counted_as_failures(monkeypatch):
    cells = f.repetition_fss_dataset(DS, PG, 50_000, 11)
    point = f.fss_fit(cells)
    monkeypatch.setattr(f, "fss_fit", lambda _cells: point)
    monkeypatch.setattr(
        f.optimize, "least_squares", _starved_least_squares(f.optimize.least_squares)
    )
    out = f.fss_bootstrap(cells, n_boot=50, seed=3)
    assert out["fit_success_fraction"] == 0.0
    assert out["p_th_ci"] is None


def test_fit_core_rejects_unsuccessful_optimizer_result(monkeypatch):
    cells = f.repetition_fss_dataset(DS, PG, 50_000, 11)
    d, p, k, n = f._arrays(cells)
    bounds, _ = f._bounds_and_starts(p)
    monkeypatch.setattr(
        f.optimize, "least_squares", _starved_least_squares(f.optimize.least_squares)
    )
    assert f._fit_core(d, p, k / n, 1.0 / f._sigma(k, n), [(0.47, 1.5)], bounds) is None
