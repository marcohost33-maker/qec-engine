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


def test_defect2_analytic_jacobian_matches_A0_cos_gstar():
    """DR(g*)_ij = A0_ij·cos(g*_j) muss der analytische Ground-Truth der Test-Map sein."""
    A0 = mod._d2_A0()
    g_star = mod._d2_gstar()
    DR = mod.defect2_analytic_jacobian(g_star, A0)
    assert np.allclose(DR, A0 * np.cos(g_star)[None, :])
    # Cross-Check: zentrale Differenz ohne Rauschen konvergiert bei kleinem eps gegen DR.
    rng = np.random.default_rng(0)
    M = mod._d2_estimate_jacobian(A0, g_star, 1e-4, 0.0, rng)
    assert np.allclose(M, DR, atol=1e-7)


def test_defect2_variance_slope_and_eps_opt_scaling():
    """REGRESSIONSGATE der D2-Korrektur (SoT v1.3 Z.87/Z.81-83):
    Gate A — Varianz-Term von ‖M̂−DR‖_F skaliert ~ 1/eps_diff (Steigung ≈ −1; docx liess
    1/eps_diff fallen). Gate B — eps_opt ~ (n·N_rep)^(−1/6). Reduzierte K/Gitter für Tempo;
    das voll verankerte Orakel (seed=2026) läuft via `python spec/reproducers/...`.
    """
    d2 = mod.defect2_oracle(
        gate_a_k=120, gate_b_k=100, gate_b_nn=(3e4, 3e5, 3e6, 3e7), gate_b_eps=(1e-2, 1.0, 10)
    )
    assert d2["gate_pass"] is True
    ga = d2["gate_a_variance_amplification"]
    gb = d2["gate_b_eps_opt_scaling"]
    tb = d2["truncation_bias_check"]
    # Gate A: Varianz-Steigung ≈ −1 (PROPOSED [−1.15,−0.85]) + hoher R².
    assert -1.15 <= ga["measured_slope"] <= -0.85, ga["measured_slope"]
    assert ga["r2"] >= 0.99
    # Trunkierungsbias (rauschfrei) ≈ +2 (belegt O(eps²)-Term).
    assert 1.8 <= tb["measured_slope"] <= 2.2, tb["measured_slope"]
    # Gate B: eps_opt-Exponent ≈ −1/6 (PROPOSED [−0.20,−0.13]) + hoher R².
    assert -0.20 <= gb["measured_exponent"] <= -0.13, gb["measured_exponent"]
    assert gb["r2"] >= 0.95
    # eps_opt fällt monoton mit n·N_rep (mehr Samples -> kleinere optimale Schrittweite).
    eo = gb["eps_opt_measured"]
    assert all(eo[i] > eo[i + 1] for i in range(len(eo) - 1)), eo


def test_defect2_noise_variance_scales_as_expected():
    """Silent-Failure-Gate: der Stochastik-Term wächst bei kleinerem eps und größerem
    Rauschen — kein stiller Kollaps auf 0 (Var(stoch) = noise_var/(2·eps²) pro Eintrag)."""
    A0, g_star = mod._d2_A0(), mod._d2_gstar()

    # halbes eps => ~doppelter Stochastik-RMS; 4x Rauschvarianz => ~2x RMS.
    def stoch(eps, v, seed):
        rng = np.random.default_rng(seed)
        Ms = np.array([mod._d2_estimate_jacobian(A0, g_star, eps, v, rng) for _ in range(300)])
        r = Ms - Ms.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(r**2, axis=(1, 2)))))

    base = stoch(0.05, 1e-6, 1)
    half_eps = stoch(0.025, 1e-6, 1)
    quad_var = stoch(0.05, 4e-6, 1)
    assert 1.85 <= half_eps / base <= 2.15, (base, half_eps)
    assert 1.85 <= quad_var / base <= 2.15, (base, quad_var)


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


def _measured_worst_case_sensitivity(A: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Numerisch gemessene WORST-CASE-Eigenwert-Sensitivität |δλ|/‖E‖ pro Eigenwert.

    Unabhängiges Orakel (löst das GESTÖRTE Eigenproblem numerisch, nicht die
    Störungsformel): Für Eigenwert λᵢ mit linkem/rechtem EV uᵢ,vᵢ maximiert die
    rangeins-Störung  E = ε·uᵢvᵢᴴ/(‖uᵢ‖‖vᵢ‖)  (Spektralnorm ‖E‖₂=ε) die
    Eigenwert-Verschiebung. Der erreichte Wert |δλ|/ε → κ(λ)=1/|uᴴv| (Bauer-Fike).
    """
    lam_r, V = np.linalg.eig(A)
    lam_l, U = np.linalg.eig(A.conj().T)  # Aᴴ u = conj(λ) u  → linke Eigenvektoren
    order_r, order_l = np.argsort(lam_r.real), np.argsort(lam_l.real)
    lam_r, V, U = lam_r[order_r], V[:, order_r], U[:, order_l]
    out = np.empty(len(lam_r))
    for i in range(len(lam_r)):
        u, v = U[:, i], V[:, i]
        E = eps * np.outer(u, v.conj()) / (np.linalg.norm(u) * np.linalg.norm(v))
        lam_p = np.linalg.eig(A + E)[0]
        j = int(np.argmin(np.abs(lam_p - lam_r[i])))  # gestörten EW dem Original zuordnen
        out[i] = abs(lam_p[j] - lam_r[i]) / eps
    return lam_r.real, out


@pytest.mark.parametrize("t", [1.0, 10.0, 50.0, 200.0])
def test_kappa_matches_measured_worst_case_sensitivity(t):
    """REGRESSIONSGATE der validierten D3-Korrektur (Wilkinson/Bauer-Fike):
    κ(λ)=√(1+t²) muss die NUMERISCH gemessene worst-case |δλ|/‖E‖ treffen —
    NICHT sep⁻¹≡1. Sichert dauerhaft, dass die Eigenwert-Sensitivität von der
    Kondition κ(λ)=1/|uᴴv| (nicht-orthogonale links/rechts-EV) regiert wird.
    """
    A = np.array([[1.0, t], [0.0, 2.0]])
    eps = 1e-8  # klein genug, dass Terme 2. Ordnung (~κ·ε) << 1. Ordnung bleiben
    _lam, measured = _measured_worst_case_sensitivity(A, eps)
    kappa = np.sqrt(1.0 + t * t)
    # (a) gemessene worst-case == κ(λ) (validierte Korrektur)
    assert np.allclose(measured, kappa, rtol=1e-4), (t, measured, kappa)
    # (b) sep⁻¹≡1 (docx-Bound) wird für t>0 klar überschritten → falscher Bound
    sep_inv = 1.0
    assert measured.min() > sep_inv, (t, measured.min(), sep_inv)
    # (c) Cross-Check gegen Modul-κ-Helfer (dieselbe Größe, andere Herleitung)
    _lam2, kappas_mod = mod.eigenvalue_condition_numbers(A)
    assert np.allclose(kappas_mod, kappa, rtol=1e-9)


def test_random_perturbation_never_exceeds_kappa_ceiling():
    """κ(λ) ist die OBERE Schranke: zufällige ‖E‖₂=ε-Störungen bleiben ≤ κ·(1+tol).
    Belegt worst-case-Charakter — kein Zufalls-E schlägt die optimal ausgerichtete.
    """
    rng = np.random.default_rng(2026)
    for t in (10.0, 200.0):
        A = np.array([[1.0, t], [0.0, 2.0]])
        eps = 1e-8
        kappa = np.sqrt(1.0 + t * t)
        lam0 = np.sort(np.linalg.eig(A)[0].real)
        worst = 0.0
        for _ in range(50):
            E = mod._random_perturbation(rng, 2, eps)
            lam_p = np.sort(np.linalg.eig(A + E)[0].real)
            worst = max(worst, float(np.max(np.abs(lam_p - lam0)) / eps))
        assert worst <= kappa * (1.0 + 1e-3), (t, worst, kappa)


@pytest.mark.parametrize("bad_dim", [1, 3])
def test_condition_numbers_shape(bad_dim):
    """Konditionszahl-Helper generisch über Dimension (kein 2x2-Hardcode)."""
    A = np.eye(bad_dim)
    _lam, kappas = mod.eigenvalue_condition_numbers(A)
    assert kappas.shape == (bad_dim,)
    assert np.allclose(kappas, 1.0)  # normale (hier identische) Matrix -> κ=1
