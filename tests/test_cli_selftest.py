"""CLI/Selftest-Gate-Vertrag: alle Gates PASS, exit 0; demo exit 0."""

from __future__ import annotations

import json

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
