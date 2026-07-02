"""Tests für den ProofBlock-v1.3-Orakel-Reproducer (Defekt 3 & 4).

AGENTS.md WA1: „Keine Komponente als 'validiert' behaupten ohne lauffähigen Code + Test."
Der Reproducer liegt in spec/reproducers/ (nicht auf pythonpath) -> per Datei-Pfad importiert.
Kleine Trial-/Run-Zahlen, damit die Suite schnell bleibt; das voll verankerte Orakel
(seed=2026, 400 Trials / 200 Läufe) läuft via `python spec/reproducers/proofblock_v13_oracles.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPRO = Path(__file__).resolve().parents[1] / "spec" / "reproducers" / "proofblock_v13_oracles.py"


def _load():
    spec = importlib.util.spec_from_file_location("proofblock_v13_oracles", _REPRO)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_reproducer_file_exists():
    assert _REPRO.is_file()


def test_kappa_analytic_matches_sqrt_1_plus_t2():
    """Per-Eigenwert-κ von A=[[1,t],[0,2]] muss analytisch sqrt(1+t^2) sein (beide EW)."""
    for t in (0.0, 5.0, 50.0):
        A = np.array([[1.0, t], [0.0, 2.0]])
        _lam, kappas = mod.eigenvalue_condition_numbers(A)
        expected = np.sqrt(1.0 + t * t)
        assert np.allclose(kappas, expected, rtol=1e-9, atol=1e-9), (t, kappas, expected)


def test_defect3_scales_with_kappa_not_sep():
    """Gemessen skaliert mit κ (≤ κ·(1+tol)) und überschreitet sep⁻¹≡1 für t>0."""
    d3 = mod.defect3_oracle(n_trials=60, seed=2026)
    assert d3["gate_pass"] is True
    for r in d3["rows"]:
        assert r["bound_kappa_holds"] is True
        assert r["measured_dlambda_over_E"] <= r["kappa_lambda_analytic"] * 1.05 + 1e-9
        if r["t"] > 0:
            assert r["measured_dlambda_over_E"] > r["sep_inv_docx_bound"]  # sep-Bound verletzt
    # Verhältnis gemessen/κ ist über t näherungsweise konstant (lineare κ-Skalierung).
    ratios = [r["measured_over_kappa"] for r in d3["rows"]]
    assert max(ratios) - min(ratios) < 0.15


def test_defect3_perturbation_has_exact_spectral_norm():
    """_random_perturbation muss ‖E‖_2 == eps exakt liefern (Bound-Konsistenz)."""
    rng = np.random.default_rng(0)
    E = mod._random_perturbation(rng, 2, 1e-6)
    assert np.isclose(np.linalg.norm(E, ord=2), 1e-6, rtol=1e-12)


def test_defect4_untuned_blows_up_tuned_bounded():
    """Untuned verlässt B(g*,r_lin) und bläst auf; getunt auf W^s bleibt O(σ)."""
    d4 = mod.defect4_oracle(n_runs=40, seed=2026)
    assert d4["gate_pass"] is True
    assert d4["mean_end_norm_untuned"] > 100.0 * d4["r_lin"]  # Blow-up
    assert d4["mean_end_norm_tuned"] <= 10.0 * d4["sigma"]  # O(σ)
    assert d4["mean_exit_step_untuned"] < d4["n_iters"]  # verlässt Kugel vor Horizont-Ende
    # Codex-Befund-2: Austritt geschieht früh (~ein Dutzend Schritte), nicht erst am Ende.
    assert d4["mean_exit_step_untuned"] < 20


def test_run_gate_pass_and_serializable():
    """Voller run() (kleine Defaults via Monkey-Kürzung nicht nötig): gate_pass True, JSON-fähig."""
    import json

    payload = mod.run()
    assert payload["gate_pass"] is True
    assert payload["seed"] == 2026
    json.dumps(payload)  # muss serialisierbar sein (results/-Gate-Log)


@pytest.mark.parametrize("bad_dim", [1, 3])
def test_condition_numbers_shape(bad_dim):
    """Konditionszahl-Helper generisch über Dimension (kein 2x2-Hardcode)."""
    A = np.eye(bad_dim)
    _lam, kappas = mod.eigenvalue_condition_numbers(A)
    assert kappas.shape == (bad_dim,)
    assert np.allclose(kappas, 1.0)  # normale (hier identische) Matrix -> κ=1
