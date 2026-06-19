"""Tests fuer das Run-Manifest (Phase-5, Reproduzierbarkeit).

Determinismus-Vertrag (SLSA-Custody-Lehre: der aufgezeichnete Wert muss WIRKEN):
(a) Round-trip run -> Manifest -> run_from_manifest == identischer result_hash;
(b) ein VERAENDERTER Seed im Manifest ERGIBT einen anderen Hash (kein toter Record).
"""

from __future__ import annotations

import json

import pytest

from adaptiverg_qec import manifest as M

# Kleines, schnelles Manifest fuer Tests (PC RAM-limitiert; EIN Waiter, kein Spawn).
_FAST = dict(n_chains=4, n_steps=1500, burn_in=300, L=16)


def test_run_is_deterministic_same_object() -> None:
    """Zwei Laeufe desselben Manifest-Objekts -> identischer Hash."""
    mf = M.RunManifest(base_seed=42, **_FAST)
    r1 = M.run(mf)
    r2 = M.run(mf)
    assert r1.result_hash == r2.result_hash
    assert r1.chain_mean_H == r2.chain_mean_H


def test_roundtrip_via_file(tmp_path) -> None:
    """NON-VAKUOES (a): run -> write -> run_from_manifest == identischer Hash."""
    mf = M.RunManifest(base_seed=7, **_FAST)
    r_direct = M.run(mf)
    p = tmp_path / "manifest.json"
    M.write_manifest(mf, p)
    r_loaded = M.run_from_manifest(p)
    assert r_loaded.result_hash == r_direct.result_hash
    assert r_loaded.rhat == pytest.approx(r_direct.rhat, abs=0.0)


def test_changed_seed_changes_hash(tmp_path) -> None:
    """NON-VAKUOES (b): geaenderter Seed im Manifest -> ANDERER Hash (Manifest treibt den Lauf)."""
    p = tmp_path / "manifest.json"
    M.write_manifest(M.RunManifest(base_seed=100, **_FAST), p)
    h1 = M.run_from_manifest(p).result_hash
    M.write_manifest(M.RunManifest(base_seed=101, **_FAST), p)
    h2 = M.run_from_manifest(p).result_hash
    assert h1 != h2


def test_changed_nsteps_changes_hash(tmp_path) -> None:
    """Auch n_steps wirkt: anderer Parameter -> anderer Hash (Record ist nicht tot)."""
    p = tmp_path / "manifest.json"
    M.write_manifest(M.RunManifest(base_seed=5, n_chains=4, n_steps=1500, burn_in=300, L=16), p)
    h1 = M.run_from_manifest(p).result_hash
    M.write_manifest(M.RunManifest(base_seed=5, n_chains=4, n_steps=1800, burn_in=300, L=16), p)
    h2 = M.run_from_manifest(p).result_hash
    assert h1 != h2


def test_manifest_records_environment(tmp_path) -> None:
    """Das geschriebene Manifest enthaelt Versionen, git-SHA, Plattform (Provenienz)."""
    p = tmp_path / "manifest.json"
    M.write_manifest(M.RunManifest(base_seed=1, **_FAST), p)
    data = json.loads(p.read_text(encoding="utf-8"))
    env = data["environment"]
    for key in ("python", "numpy", "scipy", "package_version", "git_sha", "platform"):
        assert key in env and env[key], (key, env)
    assert data["schema"] == M.MANIFEST_SCHEMA


def test_load_rejects_bad_schema(tmp_path) -> None:
    """Silent-Failure-Gate: falsches/fehlendes Schema -> sauberer Fehler."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "wrong/v0", "base_seed": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        M.load_manifest(p)


def test_load_rejects_unknown_keys(tmp_path) -> None:
    """Silent-Failure-Gate: unbekannte Manifest-Keys -> Fehler (kein stilles Ignorieren)."""
    p = tmp_path / "extra.json"
    d = M.RunManifest(base_seed=1, **_FAST).to_dict()
    d["schema"] = M.MANIFEST_SCHEMA
    d["bogus_param"] = 999
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        M.load_manifest(p)


def test_environment_does_not_affect_hash(tmp_path) -> None:
    """Die Umgebung ist Provenienz, kein Lauf-Treiber: Hash haengt nur an Parametern."""
    mf = M.RunManifest(base_seed=3, **_FAST)
    h_obj = M.run(mf).result_hash
    p = tmp_path / "m.json"
    M.write_manifest(mf, p)  # fuegt environment hinzu
    h_file = M.run_from_manifest(p).result_hash
    assert h_obj == h_file
