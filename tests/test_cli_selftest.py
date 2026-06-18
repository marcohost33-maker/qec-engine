"""CLI/Selftest-Gate-Vertrag: alle Gates PASS, exit 0; demo exit 0."""

from __future__ import annotations

import json
import os

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
