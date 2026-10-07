"""Inkrement 3.3: gepaarter Decoder-A/B (Statistik ohne Extras, Decoding mit [surface])."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from adaptiverg_qec import qec_decoder_ab as ab
from adaptiverg_qec import surface_decoder as sd
from adaptiverg_qec.qec_manifest_v2 import QECExperimentManifestV2, StimNoiseProfile

requires_surface = pytest.mark.skipif(
    not sd.HAVE_SURFACE, reason="needs the [surface] extras (stim + pymatching)"
)


# --- McNemar -------------------------------------------------------------------


@pytest.mark.parametrize("b,c", [(1, 0), (0, 7), (5, 5), (12, 3), (187, 107), (40, 41)])
def test_mcnemar_matches_scipy_binomtest(b, c):
    ref = stats.binomtest(b, b + c, 0.5).pvalue
    assert ab.mcnemar_exact(b, c)["p_value"] == pytest.approx(ref, abs=1e-12)


def test_mcnemar_symmetric_and_no_discordance_is_p_one():
    assert ab.mcnemar_exact(9, 2)["p_value"] == ab.mcnemar_exact(2, 9)["p_value"]
    assert ab.mcnemar_exact(0, 0) == {
        "b": 0,
        "c": 0,
        "n_discordant": 0,
        "p_value": 1.0,
        "p_value_mid": 1.0,
    }


def test_mcnemar_mid_p_formula_and_less_conservative():
    b, c = 12, 3
    n, k = b + c, 3
    expected = 2 * stats.binom.cdf(k, n, 0.5) - stats.binom.pmf(k, n, 0.5)
    r = ab.mcnemar_exact(b, c)
    assert r["p_value_mid"] == pytest.approx(expected, abs=1e-15)
    assert r["p_value_mid"] < r["p_value"]


def test_mcnemar_mid_p_is_capped_when_tables_balanced():
    # b == c: 2*cdf > 1; mid-p = 2*cdf - pmf ist exakt 1 (Symmetrie), nie > 1.
    r = ab.mcnemar_exact(6, 6)
    assert r["p_value"] == 1.0
    assert r["p_value_mid"] == pytest.approx(1.0, abs=1e-12)
    assert r["p_value_mid"] <= 1.0


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_mcnemar_rejects_non_counts(bad):
    with pytest.raises((TypeError, ValueError)):
        ab.mcnemar_exact(bad, 2)


# --- Clopper-Pearson -------------------------------------------------------------


def test_clopper_pearson_edges_and_coverage_of_point():
    assert ab.clopper_pearson(0, 100)[0] == 0.0
    assert ab.clopper_pearson(100, 100)[1] == 1.0
    lo, hi = ab.clopper_pearson(30, 100, 0.05)
    assert lo < 0.30 < hi
    # Bekannter Referenzwert (Clopper & Pearson 1934, k=0, n=100, 95%): 1-(0.025)^(1/100)
    assert ab.clopper_pearson(0, 100, 0.05)[1] == pytest.approx(1 - 0.025 ** (1 / 100))


@pytest.mark.parametrize("k,n,alpha", [(5, 4, 0.05), (1, 0, 0.05), (1, 10, 0.0), (1, 10, 1.0)])
def test_clopper_pearson_rejects_invalid(k, n, alpha):
    with pytest.raises(ValueError):
        ab.clopper_pearson(k, n, alpha)


# --- gepaarter Bootstrap ----------------------------------------------------------


def test_paired_bootstrap_matches_index_resampling_distribution():
    """Multinomial-Abkuerzung == Resampling der Shot-Indizes (gleiche Momente)."""
    n00, n01, n10, n11 = 700, 40, 90, 170
    r = ab.paired_bootstrap_delta(n00, n01, n10, n11, n_boot=20_000, seed=5)
    assert r["delta"] == pytest.approx((n01 - n10) / 1000)
    base = np.repeat([0, 0, 1, 1], [n00, n01, n10, n11]).astype(bool)
    cand = np.repeat([0, 1, 0, 1], [n00, n01, n10, n11]).astype(bool)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, base.size, size=(4_000, base.size))
    d_idx = cand[idx].mean(axis=1) - base[idx].mean(axis=1)
    lo, hi = r["ci"]
    assert lo == pytest.approx(np.quantile(d_idx, 0.025), abs=0.004)
    assert hi == pytest.approx(np.quantile(d_idx, 0.975), abs=0.004)


def test_paired_bootstrap_degenerate_and_deterministic():
    r = ab.paired_bootstrap_delta(100, 0, 0, 20, n_boot=500, seed=3)
    assert r["delta"] == 0.0 and r["ci"] == [0.0, 0.0]
    a = ab.paired_bootstrap_delta(500, 10, 30, 60, n_boot=1_000, seed=9)
    b = ab.paired_bootstrap_delta(500, 10, 30, 60, n_boot=1_000, seed=9)
    assert a == b


def test_pairing_narrows_interval_with_positively_correlated_failures():
    r = ab.paired_bootstrap_delta(9_000, 100, 200, 700, n_boot=5_000, seed=2)
    assert r["ci_width"] < r["unpaired_wald_ci_width"]


@pytest.mark.parametrize(
    "kwargs",
    [{"n_boot": 10}, {"alpha": 0.0}, {"seed": -1}, {"n_boot": 1.5}],
)
def test_paired_bootstrap_rejects_invalid(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ab.paired_bootstrap_delta(10, 1, 1, 1, **kwargs)


def test_paired_bootstrap_rejects_empty_table():
    with pytest.raises(ValueError):
        ab.paired_bootstrap_delta(0, 0, 0, 0)


def test_paired_counts_from_failures():
    base = np.array([0, 0, 1, 1, 1], dtype=bool)
    cand = np.array([0, 1, 0, 1, 0], dtype=bool)
    t = ab.PairedCounts.from_failures(base, cand)
    assert (t.n00, t.n01, t.n10, t.n11) == (1, 1, 2, 1)
    assert t.shots == 5 and t.baseline_failures == 3 and t.candidate_failures == 2
    with pytest.raises(ValueError):
        ab.PairedCounts.from_failures(base, cand[:4])


# --- Decoding (Surface-Extras) ------------------------------------------------------


def _circuit_manifest(**kw) -> QECExperimentManifestV2:
    noise = StimNoiseProfile(
        before_round_data_depolarization=0.006,
        before_measure_flip_probability=0.006,
        after_clifford_depolarization=0.006,
        after_reset_flip_probability=0.006,
    )
    return QECExperimentManifestV2(
        distances=(3, 5), shots_per_cell=10_000, base_seed=81, noise=noise, **kw
    )


@requires_surface
def test_ab_rejects_unknown_and_mismatched_decoders():
    m = _circuit_manifest()
    with pytest.raises(ValueError, match="unknown"):
        ab.paired_decoder_ab(m, candidate="bp-osd")
    with pytest.raises(ValueError, match="manifest decoder"):
        ab.paired_decoder_ab(m, baseline=ab.CORRELATED_DECODER)
    with pytest.raises(TypeError):
        ab.paired_decoder_ab(m.to_dict())


@requires_surface
def test_ab_baseline_is_bitwise_the_manifest_run_and_rerun_is_identical():
    m = _circuit_manifest()
    res = ab.paired_decoder_ab(m, n_boot=500)
    direct = sd.run_experiment_manifest(m)["rows"]
    for row, cell in zip(direct, res["cells"], strict=True):
        assert row["seed"] == cell["seed"]
        assert round(row["p_logical"] * row["shots"]) == cell["baseline"]["failures"]
    assert ab.paired_decoder_ab(m, n_boot=500)["cells"] == res["cells"]


@requires_surface
def test_ab_no_hyperedges_means_identical_decisions():
    m = QECExperimentManifestV2(
        distances=(3, 5),
        shots_per_cell=5_000,
        base_seed=61,
        noise=StimNoiseProfile(
            before_measure_flip_probability=0.02, after_reset_flip_probability=0.02
        ),
    )
    for cell in ab.paired_decoder_ab(m, n_boot=500)["cells"]:
        assert cell["hyperedges"] == 0
        assert cell["decisions_identical"]
        assert cell["baseline"]["failures"] > 0
        assert cell["mcnemar"]["n_discordant"] == 0


@requires_surface
def test_ab_correlated_matching_beats_baseline_under_circuit_noise():
    res = ab.paired_decoder_ab(_circuit_manifest(), n_boot=2_000)
    d5 = next(c for c in res["cells"] if c["d"] == 5)
    assert d5["hyperedges"] > 0 and not d5["decisions_identical"]
    assert d5["mcnemar"]["b"] > d5["mcnemar"]["c"]
    assert d5["mcnemar"]["p_value"] < 1e-3
    assert d5["delta"]["ci"][1] < 0.0


@requires_surface
def test_evidence_runner_quick_all_pass(tmp_path):
    from adaptiverg_qec import qec_evidence

    out = tmp_path / "ev.json"
    assert qec_evidence.main(["--quick", "--json", str(out)]) == 0
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["all_pass"] and payload["n_gates"] == 10
    assert [g["id"] for g in payload["gates"]] == [f"Q{i}" for i in range(1, 11)]


# --- Review-Haertung PR #54 ---------------------------------------------------------


def test_alpha_and_counts_validation():
    with pytest.raises(TypeError):
        ab.clopper_pearson(3, 10, "0.05")
    with pytest.raises(ValueError):
        ab.PairedCounts(-5, 1, 1, 1)
    with pytest.raises(ValueError):
        ab.PairedCounts.from_failures(np.array([0.2, 2, 0]), np.array([0, 1, 0]))
    t = ab.PairedCounts.from_failures(np.array([0, 1, 1]), np.array([1, 1, 0]))
    assert (t.n00, t.n01, t.n10, t.n11) == (0, 1, 1, 1)


def test_independent_pairing_table_keeps_margins_and_kills_pairing_gain():
    t = ab.PairedCounts(9_000, 100, 200, 700)
    ind = ab.independent_pairing_table(t)
    assert ind.shots == t.shots
    assert ind.baseline_failures == t.baseline_failures
    assert ind.candidate_failures == t.candidate_failures
    r = ab.paired_bootstrap_delta(ind.n00, ind.n01, ind.n10, ind.n11, n_boot=10_000, seed=1)
    assert 0.9 <= r["ci_width"] / r["unpaired_wald_ci_width"] <= 1.1


def test_failures_digest_is_shot_exact():
    a = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    b = a.copy()
    b[[0, 1]] = b[[1, 0]]  # gleiche Zahl, anderer Shot
    assert a.sum() == b.sum()
    assert ab.failures_digest(a) == ab.failures_digest(a.copy())
    assert ab.failures_digest(a) != ab.failures_digest(b)
    # Laenge zaehlt (packbits polstert auf Byte-Grenzen).
    assert ab.failures_digest(a[:8]) != ab.failures_digest(np.append(a[:8], False))


@pytest.mark.parametrize("version,ok", [("2.4.0", True), ("2.3rc1", True), ("2.2.9", False)])
def test_pymatching_version_gate(monkeypatch, version, ok):
    if not sd.HAVE_SURFACE:
        pytest.skip("needs the [surface] extras")
    import pymatching

    monkeypatch.setattr(pymatching, "__version__", version)
    if ok:
        ab._require_decoders()
    else:
        with pytest.raises(ImportError, match="pymatching>=2.3"):
            ab._require_decoders()


@requires_surface
def test_q4_counter_direction_fails_if_sampler_ignores_seed(monkeypatch):
    """Review-Szenario: ein Sampler, der den Seed ignoriert, darf Q4 nicht bestehen."""
    from adaptiverg_qec import qec_evidence

    real = sd._generated_memory_decode

    def seed_blind(d, **kw):
        kw["seed"] = 12345
        return real(d, **kw)

    monkeypatch.setattr(sd, "_generated_memory_decode", seed_blind)
    gates = {g["id"]: g for g in qec_evidence.build_evidence(quick=True)["gates"]}
    assert gates["Q4"]["status"] == "FAIL"
