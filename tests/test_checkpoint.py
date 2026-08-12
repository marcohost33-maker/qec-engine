"""Tests fuer checkpoint.py: Checkpoint/Restart + Lockfile (Phase-6)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from adaptiverg_qec import checkpoint, manifest


def _small_manifest(**overrides) -> manifest.RunManifest:
    base = dict(base_seed=4711, n_chains=2, n_steps=400, burn_in=100, L=16)
    base.update(overrides)
    return manifest.RunManifest(**base)


class TestResumeByteIdentical:
    def test_uninterrupted_resumable_equals_direct(self, tmp_path):
        mf = _small_manifest()
        direct = manifest.run(mf)
        r = checkpoint.run_resumable(mf, tmp_path / "ck.json", checkpoint_every=150)
        assert r is not None
        assert r.result_hash == direct.result_hash
        # Erfolg raeumt Checkpoint + Lock auf.
        assert not (tmp_path / "ck.json").exists()
        assert not (tmp_path / "ck.json.lock").exists()

    def test_interrupt_and_resume_matches_direct_hash(self, tmp_path):
        mf = _small_manifest(n_chains=3)
        direct = manifest.run(mf)
        p = tmp_path / "ck.json"
        r1 = checkpoint.run_resumable(mf, p, checkpoint_every=150, interrupt_after=500)
        assert r1 is None
        assert p.exists()
        r2 = checkpoint.resume(p, checkpoint_every=150)
        assert r2 is not None
        assert r2.result_hash == direct.result_hash

    def test_double_interrupt_then_resume(self, tmp_path):
        """Zwei Unterbrechungen hintereinander -> immer noch byte-identisch."""
        mf = _small_manifest(n_chains=3)
        direct = manifest.run(mf)
        p = tmp_path / "ck.json"
        assert checkpoint.run_resumable(mf, p, checkpoint_every=100, interrupt_after=250) is None
        assert checkpoint.resume(p, checkpoint_every=100, interrupt_after=300) is None
        r = checkpoint.resume(p, checkpoint_every=100)
        assert r is not None
        assert r.result_hash == direct.result_hash

    def test_interrupt_mid_first_chain(self, tmp_path):
        mf = _small_manifest()
        direct = manifest.run(mf)
        p = tmp_path / "ck.json"
        assert checkpoint.run_resumable(mf, p, checkpoint_every=50, interrupt_after=70) is None
        payload = checkpoint.load_checkpoint(p)
        assert payload["chain_index"] == 0
        assert payload["chain_state"]["t"] == 70
        r = checkpoint.resume(p)
        assert r.result_hash == direct.result_hash


class TestRngStateRoundTrip:
    def test_philox_state_json_round_trip(self):
        rng = np.random.Generator(np.random.Philox(key=123))
        rng.random(37)  # Zustand mitten im Buffer
        state = checkpoint._rng_state_to_json(rng.bit_generator.state)
        blob = json.dumps(state)  # muss JSON-serialisierbar sein
        restored = np.random.Generator(np.random.Philox())
        restored.bit_generator.state = checkpoint._rng_state_from_json(json.loads(blob))
        assert np.array_equal(rng.random(100), restored.random(100))


class TestLockfile:
    def test_double_lock_raises(self, tmp_path):
        p = tmp_path / "ck.json"
        with (
            checkpoint.checkpoint_lock(p),
            pytest.raises(checkpoint.CheckpointLockedError),
            checkpoint.checkpoint_lock(p),
        ):
            pass
        # Lock nach Kontext freigegeben -> erneut erwerbbar.
        with checkpoint.checkpoint_lock(p):
            pass

    def test_run_fails_when_locked(self, tmp_path):
        mf = _small_manifest()
        p = tmp_path / "ck.json"
        with checkpoint.checkpoint_lock(p), pytest.raises(checkpoint.CheckpointLockedError):
            checkpoint.run_resumable(mf, p)


class TestFailClosed:
    def test_tampered_checkpoint_rejected(self, tmp_path):
        mf = _small_manifest()
        p = tmp_path / "ck.json"
        assert checkpoint.run_resumable(mf, p, checkpoint_every=50, interrupt_after=60) is None
        raw = p.read_text(encoding="utf-8")
        # Byte-Flip im Payload (t-Wert manipulieren).
        tampered = raw.replace('"t": 60', '"t": 50')
        assert tampered != raw
        p.write_text(tampered, encoding="utf-8")
        with pytest.raises(checkpoint.CheckpointError):
            checkpoint.resume(p)

    def test_schema_mismatch_rejected(self, tmp_path):
        p = tmp_path / "ck.json"
        p.write_text(json.dumps({"schema": "wrong/v0"}), encoding="utf-8")
        with pytest.raises(checkpoint.CheckpointError):
            checkpoint.load_checkpoint(p)

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(checkpoint.CheckpointError):
            checkpoint.load_checkpoint(tmp_path / "nope.json")

    def test_not_json_rejected(self, tmp_path):
        p = tmp_path / "ck.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(checkpoint.CheckpointError):
            checkpoint.load_checkpoint(p)

    def test_invalid_params_rejected(self, tmp_path):
        mf = _small_manifest()
        with pytest.raises(ValueError):
            checkpoint.run_resumable(mf, tmp_path / "ck.json", checkpoint_every=0)
        with pytest.raises(ValueError):
            checkpoint.run_resumable(mf, tmp_path / "ck.json", interrupt_after=0)

    def test_resume_validates_params_too(self, tmp_path):
        """Codex-Fix: resume() validiert checkpoint_every/interrupt_after wie run_resumable
        (checkpoint_every=0 wuerde sonst endlos denselben Checkpoint schreiben)."""
        mf = _small_manifest()
        p = tmp_path / "ck.json"
        assert checkpoint.run_resumable(mf, p, checkpoint_every=50, interrupt_after=60) is None
        with pytest.raises(ValueError):
            checkpoint.resume(p, checkpoint_every=0)
        with pytest.raises(ValueError):
            checkpoint.resume(p, interrupt_after=0)


class TestChainBoundaryCheckpoint:
    def test_boundary_checkpoint_when_chains_shorter_than_interval(self, tmp_path):
        """Codex-Fix: n_steps <= checkpoint_every -> Checkpoint an der Ketten-Grenze.

        Vorher gab es in diesem Regime NIE einen periodischen Write (die
        Mid-Chain-Bedingung feuert nicht); ein Crash zwischen Ketten verlor
        allen Fortschritt.
        """
        mf = _small_manifest(n_chains=3, n_steps=100, burn_in=20)
        direct = manifest.run(mf)
        p = tmp_path / "ck.json"
        # interrupt genau an der Grenze: Kette 0 (100 Sweeps) fertig, Kette 1 Budget 0.
        r = checkpoint.run_resumable(mf, p, checkpoint_every=500, interrupt_after=100)
        assert r is None and p.exists()
        payload = checkpoint.load_checkpoint(p)
        assert payload["chain_index"] == 1
        assert payload["chain_state"]["t"] == 0  # Grenz-Checkpoint: Kette 1 frisch
        assert len(payload["done_H"]) == 1  # Kette 0 vollstaendig persistiert
        res = checkpoint.resume(p, checkpoint_every=500)
        assert res is not None
        assert res.result_hash == direct.result_hash


class TestManifestValidation:
    def test_post_init_fail_closed(self):
        with pytest.raises(ValueError):
            manifest.RunManifest(n_chains=1)
        with pytest.raises(ValueError):
            manifest.RunManifest(n_steps=10, burn_in=10)
        with pytest.raises(ValueError):
            manifest.RunManifest(beta_target=99.0)
        with pytest.raises(ValueError):
            manifest.RunManifest(L=1)
        with pytest.raises(ValueError):
            manifest.RunManifest(adapt_c=0.0)
