"""CLI/Selftest-Gate-Vertrag: alle Gates PASS, exit 0; demo exit 0."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from adaptiverg_qec import cli


def test_selftest_all_pass_exit_zero() -> None:
    assert cli.run_selftest() == 0


def test_demo_exit_zero() -> None:
    assert cli.run_demo() == 0


def test_selftest_writes_json_gate_log(tmp_path) -> None:
    path = tmp_path / "gate.json"
    rc = cli.run_selftest(str(path))
    assert rc == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["all_pass"] is True
    assert payload["n_pass"] == payload["n_total"] == len(cli._GATES)
    assert all(g["pass"] for g in payload["gates"])


def test_main_selftest_flag() -> None:
    assert cli.main(["--selftest"]) == 0


def test_main_default_is_demo() -> None:
    assert cli.main([]) == 0


def test_json_path_bare_name_goes_to_results(tmp_path, monkeypatch) -> None:
    """Aegis-P3: nackter Dateiname landet in cwd/results/ (normalisiert)."""
    monkeypatch.chdir(tmp_path)
    resolved = cli._resolve_json_path("gate.json")
    assert resolved == (tmp_path / "results" / "gate.json").resolve()


def test_json_path_traversal_rejected(tmp_path, monkeypatch) -> None:
    """Aegis-P3: '..'-Ausbruch aus cwd wird fail-closed abgewiesen."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="path traversal"):
        cli._resolve_json_path(os.path.join("..", "escape.json"))


def test_json_path_absolute_accepted_and_normalized(tmp_path, monkeypatch) -> None:
    """Aegis-P3: expliziter absoluter Operator-Pfad wird akzeptiert + resolved."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "explicit" / "gate.json"
    resolved = cli._resolve_json_path(str(target))
    assert resolved == target.resolve()


def test_selftest_json_written_under_results(tmp_path, monkeypatch) -> None:
    """End-to-end: --json gate.json schreibt nach cwd/results/gate.json."""
    monkeypatch.chdir(tmp_path)
    rc = cli.run_selftest("gate.json")
    assert rc == 0
    written = tmp_path / "results" / "gate.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["all_pass"] is True


def test_phase7_writes_artifact(tmp_path) -> None:
    path = tmp_path / "phase7.json"
    assert cli.main(["phase7", "--json", str(path)]) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "7"
    by_beta = {k["beta"]: k for k in payload["kernels"]}
    assert by_beta[0.0]["geometrically_ergodic"] is False
    assert all(by_beta[b]["geometrically_ergodic"] for b in (0.3, 0.8, 1.5, 2.0))
    assert payload["empirical_sampler"]["marginal_band_ratio_true_kernel"] <= 1.0
    t0_1 = [r for r in payload["adaptive_chain"] if r["adapt_T0"] == 1.0]
    no_freeze = [r for r in t0_1 if not r["frozen"]]
    freeze = [r for r in t0_1 if r["frozen"]]
    assert no_freeze and freeze
    # Defektklasse ohne Freeze (vor #51/#53): Boden bleibt.
    assert all(r["frozen_floor_tv"] > 0.05 for r in no_freeze)
    # Freeze-Vertrag: exakt beta_target, TV zu pi_target verschwindet.
    assert all(r["frozen_floor_tv"] == 0.0 and r["tv_to_target_end"] < 1e-10 for r in freeze)
    # Codex P2 (README:281): die publizierte 1e-15-Zahl steht GEMESSEN im Artefakt.
    stat = payload["stationary_oracle"]
    assert stat["pass"] is True
    assert stat["max_abs_err_distribution"] <= stat["tol_distribution"] == 1e-15
    assert {r["beta"] for r in stat["per_beta"]} == {0.3, 0.8, 1.5, 2.0}
    # Codex P2 (cli.py:1557): Supremum eingegrenzt, nicht nur Gittermaximum.
    sup = payload["containment"]["sup_t_rel_on_theta"]
    assert sup["lower_bound_grid_max"] <= sup["upper_bound_rigorous"] < 25.0


def test_phase7_fails_closed_when_stationary_oracle_breaks(tmp_path, monkeypatch) -> None:
    """Vorher gab phase7 IMMER 0 zurueck; ein gerissenes Orakel muss den Lauf rot machen."""
    monkeypatch.setattr(cli, "_STATIONARY_TOL_DIST", 0.0)  # gemessene Fehler sind ~5e-17 > 0
    path = tmp_path / "phase7.json"
    assert cli.main(["phase7", "--json", str(path)]) == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stationary_oracle"]["pass"] is False


def _phase7_with_regressed_sweep(tmp_path, monkeypatch, sweep) -> tuple[int, dict]:
    """phase7 mit einem regredierten a_kernel._sweep (der echte Sampler-Pfad aendert sich)."""
    from adaptiverg_qec import a_kernel

    monkeypatch.setattr(a_kernel, "_sweep", sweep)
    path = tmp_path / "phase7.json"
    rc = cli.main(["phase7", "--json", str(path)])
    return rc, json.loads(path.read_text(encoding="utf-8"))


def test_phase7_fails_closed_when_sweep_makes_l_plus_1_steps(tmp_path, monkeypatch) -> None:
    """Codex P2 (PR #49, cli.py:2217): eine L+1-Regression laesst das 12-Sweep-Band gruen;
    vorher schrieb phase7 trotzdem Evidenz und gab 0 zurueck. Der Paritaetstest gehoert
    in den Lauf und macht ihn rot."""
    import numpy as np

    from adaptiverg_qec import a_kernel

    orig = a_kernel._sweep

    def sweep_l_plus_1(x, H, beta, rng):
        H, acc = orig(x, H, beta, rng)
        i = int(rng.integers(0, x.size))
        dH = a_kernel._delta_H_flip(x, i)
        if dH <= 0 or rng.random() < np.exp(-beta * dH):
            x[i] = 1 - x[i]
            H += dH
            acc += 1
        return H, acc

    rc, payload = _phase7_with_regressed_sweep(tmp_path, monkeypatch, sweep_l_plus_1)
    assert rc == 1
    par = payload["empirical_sampler"]["parity_step_check"]
    assert par["pass"] is False
    assert par["ratio_L"] > 1.0
    # Die uebrigen Urteile bleiben gruen: rot macht ihn GENAU die Paritaet.
    assert payload["stationary_oracle"]["pass"] is True
    assert payload["containment"]["sup_t_rel_on_theta"]["upper_bound_rigorous"] < 25.0


def test_phase7_fails_closed_when_sweep_runs_a_different_kernel(tmp_path, monkeypatch) -> None:
    """Codex P2 (PR #49, cli.py:2217): ratio_true_kernel > 1 (Sampler auf 1.5*beta) muss
    den Lauf rot machen, nicht nur im Artefakt stehen."""
    from adaptiverg_qec import a_kernel

    orig = a_kernel._sweep

    def sweep_wrong_beta(x, H, beta, rng):
        # Nur der Band-Lauf (beta=0.8) regrediert; der Paritaetslauf (beta=0.3) bleibt
        # korrekt, damit rot GENAU vom Band-Urteil kommt (sonst waere der Test blind).
        return orig(x, H, 1.5 * beta if beta > 0.5 else beta, rng)

    rc, payload = _phase7_with_regressed_sweep(tmp_path, monkeypatch, sweep_wrong_beta)
    assert rc == 1
    emp = payload["empirical_sampler"]
    assert emp["marginal_band_ratio_true_kernel"] > 1.0
    assert emp["pass"] is False
    assert emp["parity_step_check"]["pass"] is True


def test_phase7_artifact_carries_sampler_verdicts(tmp_path) -> None:
    path = tmp_path / "phase7.json"
    assert cli.main(["phase7", "--json", str(path)]) == 0
    emp = json.loads(path.read_text(encoding="utf-8"))["empirical_sampler"]
    assert emp["pass"] is True
    par = emp["parity_step_check"]
    assert par["pass"] is True
    assert par["ratio_L"] <= 1.0 < 3.0 < min(par["ratio_L_minus_1"], par["ratio_L_plus_1"])


@pytest.mark.parametrize(
    ("r_true", "r_wrong", "ok"),
    [
        (0.5, 3.0, True),
        (1.0, 2.0 + 1e-12, True),
        (1.0 + 1e-12, 3.0, False),
        (0.5, 2.0, False),
        (float("nan"), 3.0, False),
        (0.5, float("nan"), False),
    ],
)
def test_empirical_band_acceptance_edges(r_true: float, r_wrong: float, ok: bool) -> None:
    assert cli._empirical_band_accepts(r_true, r_wrong) is ok


def test_phase7_fails_closed_when_the_freeze_contract_breaks(tmp_path, monkeypatch) -> None:
    """Gleiche Klasse wie Codex P2 (cli.py:2217): ein Freeze, der nicht greift (freeze_at
    ignoriert -> Defektklasse vor #51), darf keinen gruenen phase7-Lauf ergeben."""
    from adaptiverg_qec import mixing

    orig = mixing.adaptive_exact_tv

    def no_freeze(cfg, **kw):
        kw["freeze_at"] = kw["a_t"].size
        return orig(cfg, **kw)

    monkeypatch.setattr(mixing, "adaptive_exact_tv", no_freeze)
    path = tmp_path / "phase7.json"
    assert cli.main(["phase7", "--json", str(path)]) == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["empirical_sampler"]["pass"] is True
    assert any(r["frozen"] and r["frozen_floor_tv"] > 0.05 for r in payload["adaptive_chain"])


def test_g48_no_freeze_is_measured_against_the_infinite_schedule_limit() -> None:
    """Codex P2 (PR #49, cli.py:1615): beta_limit muss der Grenzwert der unendlichen
    Schedule sein (0.549268), nicht beta_400 = 0.548955."""
    ok, detail = cli._g48_adaptive_chain_exact_tv()
    assert ok, detail
    assert "beta_inf=0.549268 (closed form; beta_400=0.548955)" in detail


def test_phase7_records_the_infinite_schedule_limit(tmp_path) -> None:
    path = tmp_path / "phase7.json"
    assert cli.main(["phase7", "--json", str(path)]) == 0
    rows = json.loads(path.read_text(encoding="utf-8"))["adaptive_chain"]
    for row in rows:
        assert row["beta_limit_used"] == row["beta_limit_closed_form_1e6"]
    t0_1 = next(r for r in rows if not r["frozen"] and r["adapt_c"] == 0.5 and r["adapt_T0"] == 1.0)
    assert t0_1["beta_limit_used"] - t0_1["beta_limit_schedule_end"] > 1e-4
    assert t0_1["tv_to_limit_end"] > 1e-5


def _sup(**changes):
    import dataclasses

    from adaptiverg_qec import mixing

    base = mixing.RelaxationSupBound(
        grid_max=19.0,
        grid_argmax=2.0,
        upper_bound=22.0,
        step=0.01,
        n_points=191,
        lower_bound=18.8,
        eigen_radius_max=1e-12,
        exp_rel_err_max=1e-16,
    )
    return dataclasses.replace(base, **changes)


@pytest.mark.parametrize(
    ("changes", "betas", "t_rel", "ok"),
    [
        ({}, (4.0, 6.0), (994.0, 5e4), True),
        ({"lower_bound": 23.0}, (4.0, 6.0), (994.0, 5e4), False),  # invertiertes Intervall
        ({"eigen_radius_max": 1e-9}, (4.0, 6.0), (994.0, 5e4), False),  # Grenze selbst ist zu
        ({"upper_bound": 25.0}, (4.0, 6.0), (994.0, 5e4), False),
        ({"upper_bound": float("nan")}, (4.0, 6.0), (994.0, 5e4), False),
        ({"eigen_radius_max": float("nan")}, (4.0, 6.0), (994.0, 5e4), False),
        ({}, (4.0, 6.0), (994.0, 1e4), False),  # Kontrolle ausserhalb Theta greift nicht mehr
        ({}, (4.0, 6.0), (994.0, float("nan")), False),
        # Codex P2 (cli.py:1651): unter der BEWIESENEN Schranke e^8/120 = 24.8 bei beta=4
        # ist der berechnete nicht der analysierte Kern -- auch wenn t_rel(6) > 1e4.
        ({}, (4.0, 6.0), (24.0, 5e4), False),
        ({}, (4.0,), (994.0,), False),  # ohne beta=6 kein Urteil
    ],
)
def test_containment_acceptance_edges(changes, betas, t_rel, ok) -> None:
    accepted = cli._containment_accepts(
        _sup(**changes), betas_far=np.array(betas), t_rel_far=np.array(t_rel)
    )
    assert accepted is ok


@pytest.mark.parametrize("regression", ["inverted", "eigen_radius", "no_divergence_control"])
def test_phase7_applies_the_full_containment_verdict(tmp_path, monkeypatch, regression) -> None:
    """Codex P2 (PR #49, cli.py:2282): phase7 akzeptierte Containment allein ueber
    upper_bound < 25. Jede Regression laesst upper_bound < 25 stehen -- rot kommt also
    GENAU vom vollstaendigen G49-Praedikat."""
    import dataclasses

    from adaptiverg_qec import mixing

    orig_sup = mixing.relaxation_time_sup_bound
    orig_profile = mixing.relaxation_time_profile

    def sup_bound(*args, **kw):
        sup = orig_sup(*args, **kw)
        if regression == "inverted":
            return dataclasses.replace(sup, lower_bound=sup.upper_bound + 1.0)
        if regression == "eigen_radius":
            return dataclasses.replace(sup, eigen_radius_max=1e-8)
        return sup

    def profile(L, betas):
        out = orig_profile(L, betas)
        if regression == "no_divergence_control":
            out = np.where(np.asarray(betas) == 6.0, 5000.0, out)
        return out

    monkeypatch.setattr(mixing, "relaxation_time_sup_bound", sup_bound)
    monkeypatch.setattr(mixing, "relaxation_time_profile", profile)
    path = tmp_path / "phase7.json"
    assert cli.main(["phase7", "--json", str(path)]) == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    cont = payload["containment"]
    assert cont["sup_t_rel_on_theta"]["upper_bound_rigorous"] < 25.0
    assert cont["pass"] is False
    assert payload["empirical_sampler"]["pass"] is True
