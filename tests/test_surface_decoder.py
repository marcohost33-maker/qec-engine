"""QEC-Diagnostik (Inkrement 3): ECHTES MWPM (Stim + PyMatching) vs. zwei Orakel.

OPTIONAL-DEPENDENCY-GATE: ohne `pip install 'adaptiverg-qec[surface]'` (stim +
pymatching) werden ALLE Tests dieses Moduls geskippt (`pytest.mark.skipif`) -- sie
failen NICHT hart. Mit den Extras laufen sie und pruefen:

  A. Repetition-Code-MWPM gegen das exakte Binomial-Orakel (Inkrement 1).
  B. Surface-Code-MWPM-Threshold gegen den publizierten Wert ~10.3% (Dennis 2002),
     bewusst NICHT gegen den optimalen ML/Nishimori-Wert 10.94%.

Plus Silent-Failure-Gate: invalide Eingaben werfen LAUT (auch ohne die Extras, da
die Validierung vor dem Decoder-Aufruf greift -- ausser dem Import-Gate selbst).
"""

from __future__ import annotations

import math

import pytest

from adaptiverg_qec import surface_decoder as sd

requires_surface = pytest.mark.skipif(
    not sd.HAVE_SURFACE,
    reason="needs optional extras: pip install 'adaptiverg-qec[surface]' (stim, pymatching)",
)


# ---------------------------------------------------------------------------
# Optional-dependency-Gate-Verhalten selbst.
# ---------------------------------------------------------------------------


def test_import_gate_is_a_boolean() -> None:
    """HAVE_SURFACE ist deterministisch bool (kein NameError, kein None)."""
    assert isinstance(sd.HAVE_SURFACE, bool)


def test_functions_raise_clear_importerror_without_extras() -> None:
    """Ohne die Extras MUSS ein klarer ImportError kommen (kein stiller None/Crash)."""
    if sd.HAVE_SURFACE:
        pytest.skip("extras installed; ImportError path not reachable")
    with pytest.raises(ImportError, match=r"adaptiverg-qec\[surface\]"):
        sd.repetition_mwpm_vs_oracle(3, 0.1, 10, seed=0)


def test_literature_constants_are_distinct_and_ordered() -> None:
    """MWPM-Threshold 0.103 < optimaler/ML-Threshold 0.1094 (Konventions-Anker)."""
    assert sd.MWPM_THRESHOLD_LITERATURE < sd.ML_THRESHOLD_LITERATURE
    assert abs(sd.MWPM_THRESHOLD_LITERATURE - 0.103) < 1e-9
    assert abs(sd.ML_THRESHOLD_LITERATURE - 0.1094) < 1e-9


# ---------------------------------------------------------------------------
# Orakel A: Repetition-Code-MWPM vs. exaktes Binomial-Orakel.
# ---------------------------------------------------------------------------


@requires_surface
@pytest.mark.parametrize("d", [3, 5, 7])
def test_repetition_mwpm_matches_binomial_oracle(d: int) -> None:
    """MWPM (PyMatching) trifft das exakte Binomial-Orakel innerhalb weniger Sigma.

    Auf dem 1D-Matching-Graphen ist MWPM ML-optimal -> die MC-Abweichung zum Orakel
    muss rein statistisch sein. Toleranz 5 sigma (sehr konservativ gegen Flake).
    """
    est = sd.repetition_mwpm_vs_oracle(d, p=0.10, shots=120_000, seed=2026)
    assert est.p_L_exact > 0.0
    assert est.std_err > 0.0
    assert est.n_sigma < 5.0, (
        f"d={d}: MWPM {est.p_L_mwpm:.5f} vs oracle {est.p_L_exact:.5f}, n_sigma={est.n_sigma:.2f}"
    )


@requires_surface
def test_repetition_mwpm_is_reproducible() -> None:
    """Gleicher Seed -> bit-identischer MWPM-Schaetzer (Determinismus)."""
    a = sd.repetition_mwpm_vs_oracle(5, 0.12, shots=20_000, seed=7)
    b = sd.repetition_mwpm_vs_oracle(5, 0.12, shots=20_000, seed=7)
    assert a.p_L_mwpm == b.p_L_mwpm


@requires_surface
def test_repetition_mwpm_higher_distance_lower_error_below_threshold() -> None:
    """Sub-threshold (p<0.5): groessere Distanz drueckt die logische Fehlerrate."""
    e3 = sd.repetition_mwpm_vs_oracle(3, 0.08, shots=200_000, seed=1)
    e7 = sd.repetition_mwpm_vs_oracle(7, 0.08, shots=200_000, seed=1)
    assert e7.p_L_mwpm < e3.p_L_mwpm


# ---------------------------------------------------------------------------
# Orakel B: Surface-Code-MWPM-Threshold vs. publizierter Wert ~10.3%.
# ---------------------------------------------------------------------------


@requires_surface
def test_surface_threshold_behaviour_curves_cross() -> None:
    """Threshold-Verhalten: unter p_th hilft Distanz, darueber schadet sie.

    Bei p=0.08 (< Threshold) hat d=9 niedrigere, bei p=0.13 (> Threshold) hoehere
    logische Fehlerrate als d=5 -> die Kurven kreuzen dazwischen.
    """
    lo5 = sd.surface_logical_error_rate(5, 0.08, shots=20_000, seed=3)
    lo9 = sd.surface_logical_error_rate(9, 0.08, shots=20_000, seed=3)
    hi5 = sd.surface_logical_error_rate(5, 0.13, shots=20_000, seed=3)
    hi9 = sd.surface_logical_error_rate(9, 0.13, shots=20_000, seed=3)
    assert lo9 < lo5, f"below threshold expected lo9<lo5, got {lo9:.4f} vs {lo5:.4f}"
    assert hi9 > hi5, f"above threshold expected hi9>hi5, got {hi9:.4f} vs {hi5:.4f}"


@requires_surface
def test_zero_plateau_is_not_a_crossing() -> None:
    """Codex-Fix: beide Kurven ohne beobachtete Fehler (diff == 0 flach) ist
    KEINE Threshold-Evidenz -> crossing_found=False, p_threshold NaN."""
    thr = sd.estimate_mwpm_threshold(3, 5, ps=(0.001, 0.002), shots=200, seed=1)
    assert not thr.crossing_found
    assert math.isnan(thr.p_threshold)
    assert thr.abs_error == float("inf")


@requires_surface
def test_surface_mwpm_threshold_near_published_value() -> None:
    """Kreuzungs-Schaetzer (groesste Distanzen) nahe publiziertem MWPM-Threshold 0.103.

    Toleranz 0.012 (absolut): erlaubt MC-Rauschen + finite-size-Drift, schliesst aber
    den optimalen ML/Nishimori-Wert 0.1094 NICHT faelschlich als "getroffen" ein-
    bzw. aus -- entscheidend ist Naehe zu 0.103, nicht zu 0.1094.
    """
    thr = sd.estimate_mwpm_threshold(7, 11, shots=40_000, seed=11)
    assert math.isfinite(thr.p_threshold), "no crossing found in the bracket"
    assert thr.abs_error < 0.012, (
        f"p_threshold={thr.p_threshold:.4f} vs literature {thr.p_literature} "
        f"(abs_error={thr.abs_error:.4f})"
    )


# ---------------------------------------------------------------------------
# Silent-Failure-Gate: invalide Eingaben werfen LAUT.
# ---------------------------------------------------------------------------


@requires_surface
@pytest.mark.parametrize("d", [2, 4, 0, -3])
def test_repetition_rejects_even_or_nonpositive_distance(d: int) -> None:
    with pytest.raises(ValueError, match="odd"):
        sd.repetition_mwpm_vs_oracle(d, 0.1, 10, seed=0)


@requires_surface
@pytest.mark.parametrize("p", [0.0, -0.1, 0.5, 0.9, float("nan"), float("inf")])
def test_repetition_rejects_p_outside_subthreshold(p: float) -> None:
    with pytest.raises(ValueError):
        sd.repetition_mwpm_vs_oracle(5, p, 10, seed=0)


@requires_surface
@pytest.mark.parametrize("shots", [0, -5])
def test_repetition_rejects_nonpositive_shots(shots: int) -> None:
    with pytest.raises(ValueError, match="shots"):
        sd.repetition_mwpm_vs_oracle(5, 0.1, shots, seed=0)


@requires_surface
def test_repetition_rejects_noninteger_distance() -> None:
    with pytest.raises(TypeError, match="integer"):
        sd.repetition_mwpm_vs_oracle(5.0, 0.1, 10, seed=0)  # type: ignore[arg-type]


@requires_surface
@pytest.mark.parametrize("d", [2, 0, -1])
def test_surface_rejects_even_or_nonpositive_distance(d: int) -> None:
    with pytest.raises(ValueError, match="odd"):
        sd.surface_logical_error_rate(d, 0.1, 10, seed=0)


@requires_surface
@pytest.mark.parametrize("p", [0.0, 0.5, -0.2, float("nan")])
def test_surface_rejects_p_outside_range(p: float) -> None:
    with pytest.raises(ValueError):
        sd.surface_logical_error_rate(5, p, 10, seed=0)


@requires_surface
def test_threshold_rejects_dsmall_ge_dlarge() -> None:
    with pytest.raises(ValueError, match="d_small < d_large"):
        sd.estimate_mwpm_threshold(9, 7)


@requires_surface
def test_threshold_rejects_nonascending_ps() -> None:
    with pytest.raises(ValueError, match="ascending|increasing"):
        sd.estimate_mwpm_threshold(7, 9, ps=(0.11, 0.10, 0.105))


@requires_surface
def test_threshold_rejects_ps_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\(0, 0\.5\)"):
        sd.estimate_mwpm_threshold(7, 9, ps=(0.1, 0.6))
