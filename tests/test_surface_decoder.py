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
import warnings

import pytest

from adaptiverg_qec import surface_decoder as sd
from adaptiverg_qec.qec_manifest_v2 import QECExperimentManifestV2, StimNoiseProfile

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


# ---------------------------------------------------------------------------
# Inkrement 3.1: Multi-Round-Phenomenological-Baseline.
# ---------------------------------------------------------------------------


@requires_surface
@pytest.mark.parametrize("basis", ["x", "z"])
def test_phenomenological_zero_noise_is_exactly_zero(basis: str) -> None:
    """Ohne injizierte Fehler darf der annotierte Raum-Zeit-Pfad nie logisch failen."""
    est = sd.surface_phenomenological_logical_error_rate(
        3, rounds=3, p_data=0.0, p_meas=0.0, shots=512, seed=11, memory_basis=basis
    )
    assert est.p_logical == 0.0
    assert est.std_err > 0.0  # Jeffreys-Sentinel bleibt auch bei k=0 falsifizierbar


@requires_surface
def test_phenomenological_run_is_seed_reproducible() -> None:
    """Gleicher Stim-Seed -> identische logisch dekodierte Fehlerrate."""
    kw = dict(d=3, rounds=4, p_data=0.01, p_meas=0.01, shots=2000, seed=17)
    a = sd.surface_phenomenological_logical_error_rate(**kw)
    b = sd.surface_phenomenological_logical_error_rate(**kw)
    assert a.p_logical == b.p_logical
    assert a.std_err == b.std_err


@requires_surface
def test_phenomenological_diagnostics_are_explicitly_non_threshold() -> None:
    payload = sd.run_phenomenological_diagnostics(
        distances=(3,), p_data=0.005, p_meas=0.005, shots=500, seed=3
    )
    assert payload["rows"][0]["rounds"] == 3
    assert "no literature-threshold claim" in payload["claim_ceiling"]


@requires_surface
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d=2, rounds=3, p_data=0.01, p_meas=0.01, shots=10, seed=0),
        dict(d=3, rounds=0, p_data=0.01, p_meas=0.01, shots=10, seed=0),
        dict(d=3, rounds=3, p_data=-0.01, p_meas=0.01, shots=10, seed=0),
        dict(d=3, rounds=3, p_data=0.01, p_meas=0.5, shots=10, seed=0),
        dict(d=3, rounds=3, p_data=0.01, p_meas=0.01, shots=0, seed=0),
        dict(d=3, rounds=3, p_data=0.01, p_meas=0.01, shots=10, seed=-1),
    ],
)
def test_phenomenological_invalid_inputs_fail_closed(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        sd.surface_phenomenological_logical_error_rate(**kwargs)


@requires_surface
def test_phenomenological_rejects_unknown_memory_basis() -> None:
    with pytest.raises(ValueError, match="memory_basis"):
        sd.surface_phenomenological_logical_error_rate(
            3, rounds=3, p_data=0.01, p_meas=0.01, shots=10, seed=0, memory_basis="y"
        )


# ---------------------------------------------------------------------------
# Codex-Review PR #40: Seed-Provenienz, leere Distanzen, Zell-Seed-Identitaet.
# ---------------------------------------------------------------------------


@requires_surface
def test_phenomenological_payload_records_base_and_cell_seeds() -> None:
    """Der Artefakt-Payload muss den Lauf rekonstruierbar machen (Basis- + Zell-Seed)."""
    payload = sd.run_phenomenological_diagnostics(
        distances=(3,), p_data=0.005, p_meas=0.005, shots=200, seed=12345
    )
    # .get/in statt [..]: fehlende Provenienz soll als Assertion scheitern, nicht als KeyError.
    assert payload.get("seed") == 12345
    row = payload["rows"][0]
    assert "seed" in row, sorted(row)
    replay = sd.surface_phenomenological_logical_error_rate(
        3, rounds=row["rounds"], p_data=0.005, p_meas=0.005, shots=200, seed=row["seed"]
    )
    assert replay.p_logical == row["p_logical"]


@requires_surface
def test_phenomenological_rejects_empty_distances() -> None:
    """Keine Distanz = keine Messung; darf kein erfolgreich aussehender Payload werden."""
    with pytest.raises(ValueError, match="distances"):
        sd.run_phenomenological_diagnostics(distances=(), shots=10, seed=0)


@requires_surface
def test_phenomenological_empty_distances_do_not_hide_invalid_settings() -> None:
    with pytest.raises(ValueError):
        sd.run_phenomenological_diagnostics(distances=(), p_data=-1.0, shots=0, seed=-5)


@requires_surface
def test_phenomenological_cell_seed_uses_every_noise_coordinate() -> None:
    """(p_data, p_meas) = (0.01, 0) und (0, 0.01) sind verschiedene Zellen -> verschiedene Seeds."""
    a = sd.run_phenomenological_diagnostics(
        distances=(3,), p_data=0.01, p_meas=0.0, shots=10, seed=7
    )
    b = sd.run_phenomenological_diagnostics(
        distances=(3,), p_data=0.0, p_meas=0.01, shots=10, seed=7
    )
    c = sd.run_phenomenological_diagnostics(
        distances=(3,), p_data=0.01, p_meas=0.0, shots=10, seed=7, memory_basis="x"
    )
    seeds = {a["rows"][0]["seed"], b["rows"][0]["seed"], c["rows"][0]["seed"]}
    assert len(seeds) == 3


# ---------------------------------------------------------------------------
# Equalita-Runde 2026-09-27 (#40/#41): Wrapper-Vertrag am oeffentlichen Eingang.
# ---------------------------------------------------------------------------


@requires_surface
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(p_data=0.6),
        dict(p_meas=0.7),
        dict(p_data=-0.01),
        dict(p_meas=0.5),
        dict(p_data=float("nan")),
    ],
)
def test_phenomenological_wrapper_rejects_out_of_range_noise(kwargs: dict) -> None:
    """Der oeffentliche Wrapper selbst muss p ausserhalb [0, 0.5) ablehnen."""
    with pytest.raises(ValueError):
        sd.run_phenomenological_diagnostics(distances=(3,), shots=10, seed=1, **kwargs)


@requires_surface
def test_phenomenological_wrapper_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        sd.run_phenomenological_diagnostics(distances=(3,), shots=10, seed=-5)


@requires_surface
def test_phenomenological_payload_names_seed_policy() -> None:
    payload = sd.run_phenomenological_diagnostics(distances=(3,), shots=10, seed=1)
    # Ab Manifest v2 leitet der Wrapper die Zell-Seeds aus dem Vertrag ab.
    assert payload.get("seed_policy") == "manifest-sha256-v1"


@requires_surface
def test_phenomenological_base_seed_changes_cell_seeds() -> None:
    """Verschiedene Basis-Seeds muessen verschiedene Zell-Seeds liefern."""
    a = sd.run_phenomenological_diagnostics(distances=(3, 5), shots=10, seed=1)
    b = sd.run_phenomenological_diagnostics(distances=(3, 5), shots=10, seed=2)
    seeds_a = [row["seed"] for row in a["rows"]]
    seeds_b = [row["seed"] for row in b["rows"]]
    assert all(x != y for x, y in zip(seeds_a, seeds_b, strict=True)), (seeds_a, seeds_b)


@requires_surface
def test_manifest_v2_executes_the_declared_noise_contract() -> None:
    manifest = QECExperimentManifestV2(
        distances=(3,),
        shots_per_cell=256,
        base_seed=77,
        noise=StimNoiseProfile(),
    )
    payload = sd.run_experiment_manifest(manifest)
    assert payload["manifest_fingerprint"] == manifest.fingerprint()
    assert payload["rows"][0]["d"] == 3
    assert payload["rows"][0]["rounds"] == 3
    assert payload["rows"][0]["seed"] == manifest.cell_seed(3)
    assert payload["rows"][0]["p_logical"] == 0.0
    assert payload["reproducibility"]["tier"] == manifest.reproducibility_tier
    assert payload["runtime_environment"]["machine"]
    assert payload["stim_version"]


@requires_surface
def test_manifest_v2_noise_change_changes_cell_identity_and_evidence() -> None:
    a = QECExperimentManifestV2(
        distances=(3,),
        shots_per_cell=256,
        base_seed=77,
        noise=StimNoiseProfile(before_measure_flip_probability=0.0),
    )
    b = QECExperimentManifestV2(
        distances=(3,),
        shots_per_cell=256,
        base_seed=77,
        noise=StimNoiseProfile(before_measure_flip_probability=0.01),
    )
    assert a.cell_seed(3) != b.cell_seed(3)
    pa = sd.run_experiment_manifest(a)
    pb = sd.run_experiment_manifest(b)
    assert pa["manifest_fingerprint"] != pb["manifest_fingerprint"]


@requires_surface
def test_phenomenological_rows_keep_p_data_and_p_meas() -> None:
    """Codex #41: das Zeilenschema des oeffentlichen Wrappers bleibt kompatibel."""
    payload = sd.run_phenomenological_diagnostics(
        distances=(3,), p_data=0.004, p_meas=0.006, shots=20, seed=5
    )
    row = payload["rows"][0]
    assert row.get("p_data") == 0.004
    assert row.get("p_meas") == 0.006


@requires_surface
def test_phenomenological_wrapper_accepts_numpy_integers() -> None:
    """Codex #41: np.int64 fuer shots/seed/distances war vor dem Refactor gueltig."""
    import numpy as np

    err = None
    try:
        payload = sd.run_phenomenological_diagnostics(
            distances=(np.int64(3),), shots=np.int64(10), seed=np.int64(1)
        )
    except ValueError as exc:  # Absturz waere kein Beleg -> als Zusicherung melden
        err = exc
    assert err is None, f"NumPy-Ganzzahlen abgewiesen: {err}"
    manifest = payload["manifest"]
    assert manifest["shots_per_cell"] == 10 and type(manifest["shots_per_cell"]) is int
    assert manifest["base_seed"] == 1 and type(manifest["base_seed"]) is int
    assert manifest["distances"] == [3]


@requires_surface
def test_phenomenological_wrapper_still_rejects_numpy_bool_shots() -> None:
    import numpy as np

    with pytest.raises(ValueError):
        sd.run_phenomenological_diagnostics(distances=(3,), shots=np.bool_(True), seed=1)


@requires_surface
def test_manifest_runtime_records_package_version_and_git_sha() -> None:
    """Codex #41: gleiche Manifeste aus verschiedenen Revisionen muessen unterscheidbar sein."""
    import adaptiverg_qec

    payload = sd.run_phenomenological_diagnostics(distances=(3,), shots=10, seed=1)
    env = payload["runtime_environment"]
    assert env.get("package_version") == adaptiverg_qec.__version__
    assert isinstance(env.get("git_sha"), str) and env["git_sha"]


@requires_surface
def test_manifest_runtime_records_installed_distribution_version() -> None:
    """Codex #41 R2: Modul-Konstante und installierte Distribution koennen abweichen."""
    from importlib import metadata

    try:
        expected = metadata.version("adaptiverg-qec")
    except metadata.PackageNotFoundError:
        expected = None
    payload = sd.run_phenomenological_diagnostics(distances=(3,), shots=10, seed=1)
    env = payload["runtime_environment"]
    assert "distribution_version" in env, sorted(env)
    assert env["distribution_version"] == expected


@requires_surface
def test_phenomenological_wrapper_accepts_numpy_float_noise() -> None:
    """Codex #41 R2: np.float32 fuer p_data/p_meas war vor dem Refactor gueltig."""
    import numpy as np

    err = None
    try:
        payload = sd.run_phenomenological_diagnostics(
            distances=(3,), p_data=np.float32(0.004), p_meas=np.float64(0.006), shots=10, seed=1
        )
    except TypeError as exc:
        err = exc
    assert err is None, f"NumPy-Floats abgewiesen: {err}"
    row = payload["rows"][0]
    assert type(row["p_data"]) is float and type(row["p_meas"]) is float


@requires_surface
def test_phenomenological_wrapper_keeps_requested_distance_order() -> None:
    """Codex #41 R2: (5, 3) lief frueher in der angefragten Reihenfolge."""
    err = None
    try:
        payload = sd.run_phenomenological_diagnostics(distances=(5, 3, 3), shots=10, seed=1)
    except ValueError as exc:
        err = exc
    assert err is None, f"Reihenfolge abgewiesen: {err}"
    assert [row["d"] for row in payload["rows"]] == [5, 3, 3]
    assert payload["manifest"]["distances"] == [3, 5]
    assert payload["requested_distances"] == [5, 3, 3]


# ---------------------------------------------------------------------------
# Codex-Review PR #41 (Runde 3): oeffentliche Namen des alten Seed-Schemas.
# Orakel: Golden-Werte aus der Fassung auf main 288f82a (nicht aus dem Shim).
# Ohne [surface]-Extra lauffaehig: das Seed-Schema braucht weder Stim noch MWPM.
# ---------------------------------------------------------------------------

_LEGACY_GOLDEN = (
    ((20260919, 3, 3, 0.005, 0.005, "z"), 6007515778355924656),
    ((20260919, 5, 5, 0.005, 0.005, "x"), 2804414358119849615),
    ((7, 3, 3, 0.01, 0.0, "z"), 651880645769600113),
    ((7, 3, 3, 0.0, 0.01, "z"), 159691680220916230),
)


def _deprecations(record: list, needle: str) -> list:
    return [
        w for w in record if issubclass(w.category, DeprecationWarning) and needle in str(w.message)
    ]


# Rot NUR per Zusicherung (getattr/try statt Import-/Attribut-Absturz), damit der
# Diskriminierungsbeweis "Test faengt den Fehler" belegt und nicht "Test stuerzt ab".
@pytest.mark.parametrize(("args", "expected"), _LEGACY_GOLDEN)
def test_legacy_phenomenological_cell_seed_is_importable_and_bit_identical(
    args: tuple, expected: int
) -> None:
    base, d, rounds, p_data, p_meas, basis = args
    fn = getattr(sd, "phenomenological_cell_seed", None)
    assert fn is not None, "phenomenological_cell_seed fehlt im oeffentlichen Modul"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = fn(base, d=d, rounds=rounds, p_data=p_data, p_meas=p_meas, memory_basis=basis)
    assert got == expected
    assert _deprecations(rec, "phenom-cell-sha256-v1"), [str(w.message) for w in rec]


def test_legacy_seed_policy_constant_is_importable_with_deprecation() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            from adaptiverg_qec.surface_decoder import PHENOMENOLOGICAL_SEED_POLICY as value
        except ImportError as exc:
            value = f"ImportError: {exc}"
    assert value == "phenom-cell-sha256-v1", value
    assert _deprecations(rec, "PHENOMENOLOGICAL_SEED_POLICY"), [str(w.message) for w in rec]


def test_legacy_phenomenological_cell_seed_still_rejects_nan() -> None:
    fn = getattr(sd, "phenomenological_cell_seed", None)
    assert fn is not None, "phenomenological_cell_seed fehlt im oeffentlichen Modul"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError):
            fn(1, d=3, rounds=3, p_data=float("nan"), p_meas=0.0, memory_basis="z")


def test_module_getattr_does_not_swallow_unknown_names() -> None:
    try:
        value = sd.no_such_symbol
    except AttributeError as exc:
        value = exc
    assert isinstance(value, AttributeError), value
    assert "no_such_symbol" in str(value)
    assert not hasattr(sd, "PHENOMENOLOGICAL_SEED_POLICY_TYPO")


@requires_surface
def test_wrapper_does_not_route_through_legacy_seed_scheme(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Der Shim ist nur Kompatibilitaet: der Wrapper darf ihn nicht (wieder) benutzen."""
    payload = sd.run_phenomenological_diagnostics(distances=(3,), shots=10, seed=7)
    assert not _deprecations(list(recwarn), ""), [str(w.message) for w in recwarn]
    assert payload["seed_policy"] == "manifest-sha256-v1"


# ---------------------------------------------------------------------------
# Codex-Review PR #41 (Runde 4): ``from surface_decoder import *`` muss die alte
# Konstante weiter liefern. Orakel: die oeffentlichen Top-Level-Namen der Fassung
# auf main 288f82a (per AST extrahiert; ohne __all__ = Stern-Import-Oberflaeche),
# ohne die drei nur mit [surface]-Extra gebundenen Namen pymatching/stim/csc_matrix.
# ---------------------------------------------------------------------------

_MAIN_288F82A_STAR_NAMES = frozenset(
    {
        "HAVE_SURFACE",
        "ML_THRESHOLD_LITERATURE",
        "MWPM_THRESHOLD_LITERATURE",
        "PHENOMENOLOGICAL_SEED_POLICY",
        "PhenomenologicalEstimate",
        "RepetitionMwpmEstimate",
        "ThresholdEstimate",
        "annotations",
        "cell_seed",
        "dataclass",
        "estimate_mwpm_threshold",
        "hashlib",
        "json",
        "logical_error_rate_exact",
        "math",
        "np",
        "phenomenological_cell_seed",
        "repetition_mwpm_vs_oracle",
        "run_phenomenological_diagnostics",
        "run_surface_diagnostics",
        "surface_logical_error_rate",
        "surface_phenomenological_logical_error_rate",
    }
)


def _star_import() -> tuple[dict, list]:
    namespace: dict = {}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        exec("from adaptiverg_qec.surface_decoder import *", namespace)
    namespace.pop("__builtins__", None)
    return namespace, list(rec)


def test_wildcard_import_keeps_every_public_name_from_main() -> None:
    namespace, rec = _star_import()
    missing = sorted(_MAIN_288F82A_STAR_NAMES - set(namespace))
    assert not missing, f"Stern-Import verliert Namen von main: {missing}"
    assert namespace["PHENOMENOLOGICAL_SEED_POLICY"] == "phenom-cell-sha256-v1"
    assert _deprecations(rec, "PHENOMENOLOGICAL_SEED_POLICY"), [str(w.message) for w in rec]


def test_wildcard_import_is_not_narrowed_by_all() -> None:
    """Ein __all__ darf den Stern-Import nicht auf weniger als alle oeffentlichen Namen kuerzen."""
    namespace, _ = _star_import()
    public = {name for name in vars(sd) if not name.startswith("_")}
    missing = sorted(public - set(namespace))
    assert not missing, f"Stern-Import unterschlaegt oeffentliche Namen: {missing}"


def test_dir_lists_the_deprecated_constant() -> None:
    names = dir(sd)
    assert "PHENOMENOLOGICAL_SEED_POLICY" in names
    assert "phenomenological_cell_seed" in names
