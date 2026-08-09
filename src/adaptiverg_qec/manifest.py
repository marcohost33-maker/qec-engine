"""Run-Manifest fuer vollstaendige Reproduzierbarkeit (Phase-5).

KONTEXT (Phase-5). Akzeptanz: "Run reproduzierbar aus Manifest." Diese Datei
definiert (a) einen vollstaendig parametrisierten Phase-5-Lauf (Multichain-
A-Kernel-MCMC -> R-hat + CLT-Varianz), (b) ein JSON-Manifest, das den Lauf
BYTE-IDENTISCH beschreibt (Seeds, alle Parameter, Paket-Versionen, git-SHA,
Plattform), und (c) den `--from-manifest`-Pfad, der aus dem Manifest exakt
dieselben Resultat-Hashes reproduziert (Determinismus-Vertrag).

DETERMINISMUS-VERTRAG
---------------------
- RNG: pro Kette ein Philox(key = base_seed + chain_index) -- counter-basiert,
  plattformunabhaengig reproduzierbar (Spec 10.3a, wie a_kernel.py).
- Resultat-Hash: SHA-256 ueber die kanonisch serialisierten Roh-Resultate
  (H-Trajektorien-Mittel je Kette + R-hat + sigma^2_g), gerundet auf feste
  Dezimalstellen (vermeidet ULP-Rauschen ueber numpy-Builds), dann JSON mit
  sort_keys. Zwei Laeufe mit identischem Manifest -> identischer Hash.

NON-VAKUOESER TEST (SLSA-Custody-Lehre: aufgezeichneter Wert muss WIRKEN)
------------------------------------------------------------------------
(a) Round-trip: run() -> Manifest -> run_from_manifest() == identischer
    result_hash. (b) Ein VERAENDERTER base_seed im Manifest ERGIBT einen anderen
    Hash -> beweist, dass das Manifest den Lauf wirklich treibt (kein toter
    Record). Beide als Test (tests/test_manifest.py + Gates G37/G38).

Alles numpy/scipy/stdlib-only (keine neuen Runtime-Deps).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import __version__
from .a_kernel import run_adaptive_mcmc
from .clt import clt_variance
from .mvp_instance import MVPConfig
from .rhat import split_rhat

__all__ = [
    "RunManifest",
    "RunResult",
    "run",
    "postprocess_multichain",
    "run_from_manifest",
    "write_manifest",
    "load_manifest",
]

MANIFEST_SCHEMA = "adaptiverg_qec.phase5.run_manifest/v1"
_HASH_DECIMALS = 9  # Rundung vor dem Hash (ULP-robust ueber numpy-Builds).


def _git_sha() -> str:
    """Aktueller git-SHA (kurz) oder 'unknown' ausserhalb eines Repos."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git fehlt/Timeout -> kein harter Fehler
        pass
    return "unknown"


@dataclass(frozen=True)
class RunManifest:
    """Vollstaendige, reproduzierbare Beschreibung eines Phase-5-Laufs."""

    schema: str = MANIFEST_SCHEMA
    # --- Lauf-Parameter (treiben das Resultat) ---
    base_seed: int = 20260619
    n_chains: int = 4
    L: int = 16
    beta_min: float = 0.1
    beta_max: float = 2.0
    beta_target: float = 1.0
    beta_start: float = 0.2
    n_steps: int = 3000
    burn_in: int = 500
    adapt_c: float = 0.5
    adapt_T0: float = 100.0
    # --- Umgebung (Provenienz; treibt das Resultat NICHT, dokumentiert es) ---
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail-closed-Validierung beim Konstruieren UND beim Laden (load_manifest).

        Vorher wurde ein defektes Manifest erst tief im Lauf abgelehnt (z.B.
        n_chains=1 -> split_rhat-Fehler); jetzt an der Vertrauensgrenze.
        """
        if not isinstance(self.base_seed, int):
            raise ValueError(f"base_seed must be an int, got {type(self.base_seed).__name__}")
        if not isinstance(self.n_chains, int) or self.n_chains < 2:
            raise ValueError(f"n_chains must be an int >= 2 (split-R-hat), got {self.n_chains}")
        if not isinstance(self.L, int) or self.L < 2:
            raise ValueError(f"L must be an int >= 2, got {self.L}")
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError(f"n_steps must be an int >= 1, got {self.n_steps}")
        if not isinstance(self.burn_in, int) or not (0 <= self.burn_in < self.n_steps):
            raise ValueError(f"need 0 <= burn_in < n_steps, got {self.burn_in}/{self.n_steps}")
        if not (0.0 < self.beta_min < self.beta_max):
            raise ValueError(f"need 0 < beta_min < beta_max, got {self.beta_min}/{self.beta_max}")
        for name in ("beta_target", "beta_start"):
            v = getattr(self, name)
            if not (self.beta_min <= v <= self.beta_max):
                raise ValueError(
                    f"{name} {v} outside compact Theta [{self.beta_min}, {self.beta_max}]"
                )
        if not (self.adapt_c > 0 and self.adapt_T0 > 0):
            raise ValueError(
                f"need adapt_c > 0 and adapt_T0 > 0, got {self.adapt_c}/{self.adapt_T0}"
            )

    def with_environment(self) -> RunManifest:
        """Kopie mit aktuell aufgezeichneter Umgebung (Versionen, git-SHA, Plattform)."""
        env = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "package_version": __version__,
            "git_sha": _git_sha(),
            "platform": platform.platform(),
        }
        d = asdict(self)
        d["environment"] = env
        return RunManifest(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    """Resultat eines Phase-5-Laufs + Determinismus-Hash."""

    chain_mean_H: list[float]
    """Ergodisches <H> je Kette (nach Burn-in)."""
    rhat: float
    """Rank-normalized split-R-hat ueber die M Ketten (Observable H)."""
    bulk_rhat: float
    folded_rhat: float
    ess_bulk: float
    ess_tail: float
    rhat_converged: bool
    sigma2_g_gamma: float
    """CLT-Varianz von H (Gamma-Methode), gepoolt ueber Ketten (Mittel)."""
    sigma2_g_obm: float
    """CLT-Varianz von H (OBM), gepoolt ueber Ketten (Mittel)."""
    result_hash: str
    """SHA-256 ueber die gerundeten Roh-Resultate (Determinismus-Vertrag)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _multichain_H(manifest: RunManifest) -> np.ndarray:
    """Fuehre M unabhaengige A-Kernel-Ketten -> (M, n_post) H-Trajektorien (post-burn-in)."""
    cfg = MVPConfig(L=manifest.L, beta_min=manifest.beta_min, beta_max=manifest.beta_max)
    rows = []
    for c in range(manifest.n_chains):
        res = run_adaptive_mcmc(
            cfg,
            beta_target=manifest.beta_target,
            n_steps=manifest.n_steps,
            burn_in=manifest.burn_in,
            seed=manifest.base_seed + c,  # ein eigener Philox-Stream je Kette
            beta_start=manifest.beta_start,
            adapt_c=manifest.adapt_c,
            adapt_T0=manifest.adapt_T0,
        )
        rows.append(res.H_traj[manifest.burn_in :])
    return np.vstack(rows)


def _hash_results(
    chain_means: list[float], rhat: float, sigma2_gamma: float, sigma2_obm: float
) -> str:
    """SHA-256 ueber die gerundeten Roh-Resultate (ULP-robust, kanonisch)."""
    payload = {
        "chain_mean_H": [round(float(m), _HASH_DECIMALS) for m in chain_means],
        "rhat": round(float(rhat), _HASH_DECIMALS),
        "sigma2_g_gamma": round(float(sigma2_gamma), _HASH_DECIMALS),
        "sigma2_g_obm": round(float(sigma2_obm), _HASH_DECIMALS),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run(manifest: RunManifest) -> RunResult:
    """Fuehre den Phase-5-Lauf gemaess Manifest aus (deterministisch)."""
    return postprocess_multichain(_multichain_H(manifest))


def postprocess_multichain(H: np.ndarray) -> RunResult:
    """Post-Processing (R-hat + CLT + Hash) fuer (M, n_post) H-Trajektorien.

    Von run() UND vom Phase-6-Checkpoint/Restart (checkpoint.py) genutzt --
    derselbe Code-Pfad garantiert, dass ein resumierter Lauf denselben
    result_hash produziert wie der ununterbrochene (kein Divergenz-Risiko).
    """
    chain_means = [float(row.mean()) for row in H]

    r = split_rhat(H)

    # CLT-Varianz je Kette (EIN Aufruf je Kette; vorher lief die ganze
    # FFT/tau_int-Pipeline doppelt), dann Mittel (gepoolter Skalar).
    per_chain = [clt_variance(row) for row in H]
    sigma2_gamma = float(np.mean([r.sigma2_g_gamma for r in per_chain]))
    sigma2_obm = float(np.mean([r.sigma2_g_obm for r in per_chain]))

    result_hash = _hash_results(chain_means, r.rhat, sigma2_gamma, sigma2_obm)
    return RunResult(
        chain_mean_H=chain_means,
        rhat=r.rhat,
        bulk_rhat=r.bulk_rhat,
        folded_rhat=r.folded_rhat,
        ess_bulk=r.ess_bulk,
        ess_tail=r.ess_tail,
        rhat_converged=r.converged,
        sigma2_g_gamma=sigma2_gamma,
        sigma2_g_obm=sigma2_obm,
        result_hash=result_hash,
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Schreibe das Manifest (mit Umgebung) als JSON. Gibt den Pfad zurueck."""
    m = manifest.with_environment()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(m.to_dict(), fh, indent=2, sort_keys=True)
    return p


def load_manifest(path: str | Path) -> RunManifest:
    """Lade ein Manifest-JSON. Unbekannte Top-Level-Keys werden gemeldet."""
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"manifest {p} is not a JSON object")
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"manifest schema mismatch: expected {MANIFEST_SCHEMA!r}, got {data.get('schema')!r}"
        )
    known = {f for f in RunManifest.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"manifest has unknown keys: {sorted(unknown)}")
    return RunManifest(**{k: data[k] for k in data if k in known})


def run_from_manifest(path: str | Path) -> RunResult:
    """Lade ein Manifest und reproduziere den Lauf (Determinismus-Vertrag)."""
    return run(load_manifest(path))
