"""Phase-6: Checkpoint/Restart mit Lockfile fuer den Phase-5-Manifest-Lauf.

Schliesst die Phase-5-Luecke "Checkpoint/Restart-Lockfile" (ROADMAP).

DETERMINISMUS-VERTRAG (der eigentliche Punkt dieses Moduls)
-----------------------------------------------------------
Ein Lauf, der mittendrin unterbrochen und aus dem Checkpoint fortgesetzt wird,
produziert BYTE-IDENTISCH denselben RunResult.result_hash wie der
ununterbrochene manifest.run()-Lauf. Moeglich machen das:
  1. Philox-Counter-RNG: der vollstaendige Bit-Generator-Zustand
     (rng.bit_generator.state) ist exakt serialisierbar/restaurierbar.
  2. a_kernel.ChainState + advance_chain(): der Sweep-Loop ist EIN gemeinsamer
     Code-Pfad fuer run_adaptive_mcmc UND den Checkpoint-Runner (kein Fork).
  3. manifest.postprocess_multichain(): dieselbe Post-Processing-Funktion
     berechnet R-hat/CLT/Hash fuer beide Pfade.

INTEGRITAET + LOCKFILE (fail-closed)
------------------------------------
- Der Checkpoint traegt einen SHA-256 ueber seinen kanonischen JSON-Payload;
  ein korrumpierter/inkonsistent editierter Checkpoint wird LAUT abgewiesen
  (kein stilles Weiterrechnen auf kaputtem Zustand). EHRLICHE GRENZE
  (Codex-Review): der Hash ist UNKEYED -- er erkennt KORRUPTION und
  versehentliche Edits, ist aber KEINE kryptographische Authentifizierung:
  ein Akteur mit Schreibzugriff kann Payload UND Hash konsistent ersetzen.
  Schutz gegen boeswillige Writer erfordert einen separat geschuetzten
  Schluessel/Signatur und ist bewusst NICHT Teil dieses Moduls.
- Ein Lockfile (<checkpoint>.lock, O_CREAT|O_EXCL) verhindert konkurrierende
  Writer/Resumer auf demselben Checkpoint. Fail-closed: existiert das Lock,
  bricht der Aufruf mit CheckpointLockedError ab (keine Auto-Uebernahme;
  ein verwaistes Lock nach echtem Crash muss bewusst manuell entfernt werden).
- Checkpoint-Writes sind atomar (Temp-Datei + os.replace) -- ein Crash waehrend
  des Schreibens hinterlaesst nie einen halb geschriebenen Checkpoint.

EHRLICHE SCOPE-GRENZE: Checkpointing ist fuer den Phase-5-Multichain-Lauf
(A-Kernel) implementiert -- nicht fuer die 2D-Wolff/MCRG-Pipelines.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .a_kernel import ChainState, advance_chain, diminishing_step_sizes, new_chain_state
from .manifest import MANIFEST_SCHEMA, RunManifest, RunResult, postprocess_multichain
from .mvp_instance import MVPConfig

__all__ = [
    "CHECKPOINT_SCHEMA",
    "CheckpointError",
    "CheckpointLockedError",
    "checkpoint_lock",
    "run_resumable",
    "resume",
    "load_checkpoint",
]

CHECKPOINT_SCHEMA = "adaptiverg_qec.phase6.checkpoint/v1"


class CheckpointError(ValueError):
    """Checkpoint ungueltig (Schema/Integritaet/Inhalt) -- fail-closed."""


class CheckpointLockedError(CheckpointError):
    """Lockfile existiert bereits: konkurrierender Writer/Resumer."""


# ---------------------------------------------------------------------------
# Lockfile (fail-closed, O_EXCL)
# ---------------------------------------------------------------------------


class checkpoint_lock:
    """Context-Manager: exklusives Lockfile neben dem Checkpoint.

    Erzeugt <path>.lock mit O_CREAT|O_EXCL (atomar auf POSIX+NTFS). Existiert
    das Lock bereits, wird CheckpointLockedError geworfen (fail-closed, keine
    stille Uebernahme). Inhalt (PID + Zeit) dient nur der Diagnose.
    """

    def __init__(self, checkpoint_path: str | Path):
        self.lock_path = Path(str(checkpoint_path) + ".lock")

    def __enter__(self) -> Path:
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise CheckpointLockedError(
                f"lockfile {self.lock_path} exists -- another writer/resumer is active "
                "(or crashed and left an orphaned lock; remove it manually after checking)"
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "time_unix": time.time()}))
        return self.lock_path

    def __exit__(self, *exc: object) -> None:
        with contextlib.suppress(FileNotFoundError):  # pragma: no cover - defensive
            self.lock_path.unlink()


# ---------------------------------------------------------------------------
# RNG-State-(De-)Serialisierung (Philox, JSON-rund-trip-exakt)
# ---------------------------------------------------------------------------


def _rng_state_to_json(state: Any) -> Any:
    """np.uint64-Arrays/Skalare -> Python-int-Listen (JSON verlustfrei, big-int)."""
    if isinstance(state, dict):
        return {k: _rng_state_to_json(v) for k, v in state.items()}
    if isinstance(state, np.ndarray):
        return [int(v) for v in state.tolist()]
    if isinstance(state, np.integer):
        return int(state)
    return state


def _rng_state_from_json(state: Any) -> Any:
    """Invers zu _rng_state_to_json: Listen -> np.uint64-Arrays."""
    if isinstance(state, dict):
        return {k: _rng_state_from_json(v) for k, v in state.items()}
    if isinstance(state, list):
        return np.array(state, dtype=np.uint64)
    return state


# ---------------------------------------------------------------------------
# Checkpoint-Payload
# ---------------------------------------------------------------------------


def _canonical_blob(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_checkpoint(
    path: Path,
    manifest: RunManifest,
    *,
    chain_index: int,
    state: ChainState,
    rng_state: Any,
    done_H: list[list[float]],
    done_beta: list[list[float]],
    partial_H: np.ndarray,
    partial_beta: np.ndarray,
) -> None:
    """Schreibe den Checkpoint atomar (Temp + os.replace), mit Integritaets-Hash."""
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "manifest": manifest.to_dict(),
        "chain_index": int(chain_index),
        "chain_state": {
            "x": [int(v) for v in state.x.tolist()],
            "H": int(state.H),
            "beta": float(state.beta),
            "t": int(state.t),
            "accepted": int(state.accepted),
            "attempted": int(state.attempted),
        },
        "rng_state": _rng_state_to_json(rng_state),
        "done_H": done_H,
        "done_beta": done_beta,
        "partial_H": [float(v) for v in partial_H[: state.t].tolist()],
        "partial_beta": [float(v) for v in partial_beta[: state.t].tolist()],
    }
    payload["integrity_sha256"] = hashlib.sha256(_canonical_blob(payload)).hexdigest()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Lade + verifiziere einen Checkpoint (Schema + Integritaets-Hash).

    Der Integritaets-Hash ist UNKEYED (erkennt Korruption/inkonsistente Edits,
    keine kryptographische Authentifizierung -- s. Modul-Docstring).

    Raises:
        CheckpointError: Schema-Mismatch, fehlende Felder oder Hash-Mismatch
            (Korruption/inkonsistenter Edit) -- fail-closed.
    """
    p = Path(path)
    if not p.exists():
        raise CheckpointError(f"checkpoint {p} does not exist")
    with open(p, encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"checkpoint {p} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(
            f"checkpoint schema mismatch: expected {CHECKPOINT_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    stored = payload.pop("integrity_sha256", None)
    if stored is None:
        raise CheckpointError(f"checkpoint {p} lacks integrity_sha256")
    actual = hashlib.sha256(_canonical_blob(payload)).hexdigest()
    if actual != stored:
        raise CheckpointError(
            f"checkpoint {p} integrity check FAILED (stored {stored[:12]}..., "
            f"actual {actual[:12]}...) -- refusing to resume from tampered/corrupt state"
        )
    payload["integrity_sha256"] = stored
    return payload


# ---------------------------------------------------------------------------
# Resumierbarer Lauf
# ---------------------------------------------------------------------------


def run_resumable(
    manifest: RunManifest,
    checkpoint_path: str | Path,
    *,
    checkpoint_every: int = 500,
    interrupt_after: int | None = None,
) -> RunResult | None:
    """Fuehre den Phase-5-Manifest-Lauf mit periodischen Checkpoints aus.

    Args:
        manifest: derselbe RunManifest wie fuer manifest.run().
        checkpoint_path: Checkpoint-Datei (JSON; .lock daneben).
        checkpoint_every: Sweeps zwischen Checkpoints (>=1).
        interrupt_after: Nur fuer Tests/Gates -- simulierter Crash: nach so
            vielen GESAMT-Sweeps (ueber alle Ketten) wird der Zustand
            checkpointet und None zurueckgegeben.

    Returns:
        RunResult (identisch zu manifest.run()) -- oder None beim simulierten
        Interrupt (Checkpoint liegt dann auf checkpoint_path).
    """
    if checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if interrupt_after is not None and interrupt_after < 1:
        raise ValueError(f"interrupt_after must be >= 1, got {interrupt_after}")
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_lock(path):
        return _run_chains(
            manifest,
            path,
            start_chain=0,
            resume_state=None,
            checkpoint_every=checkpoint_every,
            interrupt_after=interrupt_after,
            done_H=[],
            done_beta=[],
        )


def resume(
    checkpoint_path: str | Path,
    *,
    checkpoint_every: int = 500,
    interrupt_after: int | None = None,
) -> RunResult | None:
    """Setze einen unterbrochenen Lauf aus dem Checkpoint fort.

    Ergebnis-Vertrag: RunResult.result_hash ist BYTE-IDENTISCH zu dem des
    ununterbrochenen manifest.run()-Laufs (Gate G44).

    Locking (Codex-Review-Fix, TOCTOU): das Lock wird VOR dem Laden erworben
    und ueber Laden + Validierung + Lauf gehalten -- zwei ueberlappende
    resume()-Aufrufe koennen nie beide auf demselben (ggf. inzwischen
    geloeschten) Payload weiterrechnen.
    """
    # Codex-Review-Fix: dieselbe Parameter-Validierung wie run_resumable
    # (checkpoint_every=0 wuerde sonst endlos denselben Checkpoint schreiben).
    if checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
    if interrupt_after is not None and interrupt_after < 1:
        raise ValueError(f"interrupt_after must be >= 1, got {interrupt_after}")
    path = Path(checkpoint_path)
    with checkpoint_lock(path):
        payload = load_checkpoint(path)
        manifest = RunManifest(**payload["manifest"])
        if manifest.schema != MANIFEST_SCHEMA:  # pragma: no cover - defensive
            raise CheckpointError(f"embedded manifest schema mismatch: {manifest.schema!r}")

        cs = payload["chain_state"]
        state = ChainState(
            x=np.array(cs["x"], dtype=np.int8),
            H=int(cs["H"]),
            beta=float(cs["beta"]),
            t=int(cs["t"]),
            accepted=int(cs["accepted"]),
            attempted=int(cs["attempted"]),
        )
        if state.x.size != manifest.L:
            raise CheckpointError(f"chain state length {state.x.size} != manifest L {manifest.L}")
        if not np.all((state.x == 0) | (state.x == 1)):
            raise CheckpointError("chain state x must be in {0,1}")
        if not (0 <= state.t <= manifest.n_steps):
            raise CheckpointError(f"chain state t {state.t} outside [0, {manifest.n_steps}]")

        rng = np.random.Generator(np.random.Philox())
        rng.bit_generator.state = _rng_state_from_json(payload["rng_state"])

        return _run_chains(
            manifest,
            path,
            start_chain=int(payload["chain_index"]),
            resume_state=(state, rng, payload["partial_H"], payload["partial_beta"]),
            checkpoint_every=checkpoint_every,
            interrupt_after=interrupt_after,
            done_H=[list(row) for row in payload["done_H"]],
            done_beta=[list(row) for row in payload["done_beta"]],
        )


def _run_chains(
    manifest: RunManifest,
    path: Path,
    *,
    start_chain: int,
    resume_state: tuple[ChainState, np.random.Generator, list[float], list[float]] | None,
    checkpoint_every: int,
    interrupt_after: int | None,
    done_H: list[list[float]],
    done_beta: list[list[float]],
) -> RunResult | None:
    """Gemeinsamer Kern von run_resumable() und resume()."""
    cfg = MVPConfig(L=manifest.L, beta_min=manifest.beta_min, beta_max=manifest.beta_max)
    a_t = diminishing_step_sizes(manifest.n_steps, manifest.adapt_c, manifest.adapt_T0)
    sweeps_done_this_call = 0

    for c in range(start_chain, manifest.n_chains):
        H_traj = np.empty(manifest.n_steps, dtype=np.float64)
        beta_traj = np.empty(manifest.n_steps, dtype=np.float64)
        if resume_state is not None and c == start_chain:
            state, rng, partial_H, partial_beta = resume_state
            H_traj[: state.t] = partial_H
            beta_traj[: state.t] = partial_beta
        else:
            state, rng = new_chain_state(
                cfg, seed=manifest.base_seed + c, beta_start=manifest.beta_start
            )
            if c > 0:
                # Codex-Review-Fix: Checkpoint an der KETTEN-GRENZE. Ohne ihn
                # gaebe es bei n_steps <= checkpoint_every NIE einen periodischen
                # Write (die Bedingung unten feuert nur MITTEN in einer Kette),
                # und ein echter Crash zwischen Ketten verloere allen Fortschritt.
                _write_checkpoint(
                    path,
                    manifest,
                    chain_index=c,
                    state=state,
                    rng_state=rng.bit_generator.state,
                    done_H=done_H,
                    done_beta=done_beta,
                    partial_H=H_traj,
                    partial_beta=beta_traj,
                )

        while state.t < manifest.n_steps:
            t_stop = min(state.t + checkpoint_every, manifest.n_steps)
            if interrupt_after is not None:
                budget = interrupt_after - sweeps_done_this_call
                if budget <= 0:
                    _write_checkpoint(
                        path,
                        manifest,
                        chain_index=c,
                        state=state,
                        rng_state=rng.bit_generator.state,
                        done_H=done_H,
                        done_beta=done_beta,
                        partial_H=H_traj,
                        partial_beta=beta_traj,
                    )
                    return None  # simulierter Crash NACH Checkpoint
                t_stop = min(t_stop, state.t + budget)
            advanced_from = state.t
            advance_chain(
                state,
                rng,
                cfg,
                beta_target=manifest.beta_target,
                a_t=a_t,
                t_stop=t_stop,
                H_out=H_traj,
                beta_out=beta_traj,
            )
            sweeps_done_this_call += state.t - advanced_from
            if state.t < manifest.n_steps:
                _write_checkpoint(
                    path,
                    manifest,
                    chain_index=c,
                    state=state,
                    rng_state=rng.bit_generator.state,
                    done_H=done_H,
                    done_beta=done_beta,
                    partial_H=H_traj,
                    partial_beta=beta_traj,
                )

        done_H.append([float(v) for v in H_traj.tolist()])
        done_beta.append([float(v) for v in beta_traj.tolist()])

    H = np.array([row[manifest.burn_in :] for row in done_H], dtype=np.float64)
    result = postprocess_multichain(H)
    # Erfolgreich beendet -> Checkpoint aufraeumen (Lock raeumt der Kontext auf).
    if path.exists():
        path.unlink()
    return result
