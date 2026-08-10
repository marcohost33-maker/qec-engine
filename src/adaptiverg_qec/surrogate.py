"""Phase-6: Surrogate-beschleunigter MCMC via Delayed Acceptance + Drift-Guard.

Schliesst die Phase-4-Luecke "Surrogate-Beschleunigung; Surrogate-Drift-Kontrolle"
(ROADMAP) in einer BOUNDED, mathematisch sauberen Form.

------------------------------------------------------------------------------
DELAYED-ACCEPTANCE-METROPOLIS (Christen & Fox 2005, J. Comput. Graph. Stat. 14)
------------------------------------------------------------------------------
Ziel pi ~ exp(-beta H) auf dem 1D-Repetition-Code-Ring (wie a_kernel.py),
symmetrischer Single-Flip-Proposal. Statt der teuren exakten Energie wird
zuerst ein billiges SURROGAT pi~ ~ exp(-beta~ H) befragt:

  Stufe 1 (nur Surrogat):  a1 = min(1, pi~(y)/pi~(x)) = min(1, e^{-beta~ dH}).
  Stufe 2 (nur bei Stufe-1-Akzept; exakte Energie):
           a2 = min(1, [pi(y) pi~(x)] / [pi(x) pi~(y)]) = min(1, e^{-(beta-beta~) dH}).

Der Produkt-Kernel erfuellt detailed balance gegen das EXAKTE pi fuer JEDES
Surrogat (Christen-Fox-Theorem): das Surrogat beeinflusst nur die Effizienz,
NIE die Stationaritaet. Genau das wird hier gegen das Transfer-Matrix-Orakel
verifiziert -- inkl. absichtlich MISKALIBRIERTEM Surrogat.

Surrogat-Modell (bounded MVP): beta~ = beta * (1 + gamma) -- ein systematisch
miskalibriertes Energiemodell mit Fehlerparameter gamma (gamma=0: perfekt).

EXAKTHEITS-ANKER (bit-genau, non-vakuoes): fuer gamma=0 kollabiert Stufe 2 zu
a2=1 OHNE RNG-Verbrauch, und der DA-Kernel ist BIT-IDENTISCH zum Metropolis-
Kernel des A-Kernels (gleicher Philox-Stream, gleiche Flip-Reihenfolge).

SURROGATE-DRIFT-GUARD (Spec Phase-4: "Surrogate-Drift unter Schwelle"):
bei jeder Stufe-2-Auswertung wird die Log-Diskrepanz
    delta = |log pi~-ratio - log pi-ratio| = |beta~ - beta| * |dH|
akkumuliert; der Guard FEUERT, wenn der laufende Mittelwert die Schwelle
ueberschreitet (Surrogat zu schlecht -> Effizienz-Verlust; Korrektheit bleibt).

EHRLICHE SCOPE-GRENZE: das Surrogat ist hier NICHT billiger als das Original
(1D-Toy) -- demonstriert wird die KORREKTHEITS-Maschinerie (DA-Kernel + Guard),
kein Speedup-Claim. Insbesondere (Codex-Review): dH wird in DIESER Implementierung
fuer jeden Proposal ohnehin exakt berechnet (Surrogat und Ziel teilen dieselbe
lokale Groesse dH); die Ersparnis-Zaehler n_exact_evals/exact_eval_savings sind
daher eine ACCOUNTING-GROESSE -- sie zaehlen, wie oft ein REALES teures Ziel in
Stufe 2 ausgewertet werden MUESSTE (== Stufe-1-Akzepte), nicht real gesparte
Rechenzeit dieses Toys. Kombination mit Diminishing Adaptation bleibt offen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .a_kernel import _delta_H_flip, _initial_state, hamiltonian
from .mvp_instance import MVPConfig

__all__ = [
    "DriftGuardReport",
    "DAResult",
    "run_da_mcmc",
]

DEFAULT_DRIFT_THRESHOLD = 0.05


@dataclass(frozen=True)
class DriftGuardReport:
    """Ergebnis der Surrogate-Drift-Ueberwachung (Stufe-2-Auswertungen)."""

    mean_discrepancy: float
    """Mittlere Log-Diskrepanz |log pi~-ratio - log pi-ratio| je Stufe-2-Eval."""
    max_discrepancy: float
    threshold: float
    n_evals: int
    """Anzahl Stufe-2-Auswertungen (== Anzahl exakter Energie-Abfragen)."""
    fired: bool
    """True gdw mean_discrepancy > threshold (Surrogat-Drift ueber Schwelle)."""


@dataclass(frozen=True)
class DAResult:
    """Ausgabe eines Delayed-Acceptance-Laufs."""

    H_traj: np.ndarray
    mean_H: float
    """Ergodisches Mittel von H nach Burn-in."""
    acceptance: float
    """Gesamt-Akzeptanzrate (Stufe 1 UND Stufe 2 akzeptiert)."""
    n_attempts: int
    n_exact_evals: int
    """Anzahl Stufe-2-Auswertungen (== Stufe-1-Akzeptanzen) -- die Zahl der
    Ziel-Auswertungen, die ein REALES teures Target braeuchte. ACCOUNTING-
    Groesse: in diesem 1D-Toy wird dH ohnehin fuer jeden Proposal exakt
    berechnet (s. Modul-Docstring); kein gemessener Speedup."""
    exact_eval_savings: float
    """1 - n_exact_evals/n_attempts: HYPOTHETISCHER Anteil eingesparter
    Ziel-Auswertungen bei teurem Target (Accounting, kein gemessener Speedup)."""
    stage1_accept_rate: float
    stage2_reject_rate: float
    """Anteil der Stufe-1-Akzeptanzen, die Stufe 2 wieder verwirft."""
    drift_guard: DriftGuardReport
    final_state: np.ndarray


def run_da_mcmc(
    cfg: MVPConfig,
    *,
    beta: float,
    n_steps: int,
    burn_in: int,
    seed: int,
    gamma: float = 0.0,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> DAResult:
    """Delayed-Acceptance-Metropolis mit Surrogat beta~ = beta(1+gamma).

    Args:
        cfg: MVP-Konfiguration (L, kompaktes Theta).
        beta: exaktes Ziel-beta (muss in [beta_min, beta_max] liegen).
        n_steps: Sweeps (1 Sweep = L Flip-Versuche).
        burn_in: verworfene Anfangs-Sweeps.
        seed: Philox-Seed (bit-reproduzierbar; fuer gamma=0 identischer Stream
            wie a_kernel.run_adaptive_mcmc mit beta_start=beta_target=beta).
        gamma: Surrogat-Miskalibrierung; beta~ = beta(1+gamma). gamma <= -1
            wuerde beta~ <= 0 ergeben (kein gueltiges Surrogat) -> ValueError.
        drift_threshold: Schwelle des Drift-Guards (mittlere Log-Diskrepanz).

    Returns:
        DAResult (Trajektorie, Effizienz-Zaehler, Drift-Guard-Report).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not (0 <= burn_in < n_steps):
        raise ValueError(f"need 0 <= burn_in < n_steps, got {burn_in}/{n_steps}")
    if not (cfg.beta_min <= beta <= cfg.beta_max):
        raise ValueError(f"beta {beta} outside compact Theta [{cfg.beta_min}, {cfg.beta_max}]")
    if not math.isfinite(gamma):
        raise ValueError(f"gamma must be finite, got {gamma}")
    if gamma <= -1.0:
        raise ValueError(f"gamma must be > -1 (surrogate beta~ > 0), got {gamma}")
    if not (drift_threshold > 0):
        raise ValueError(f"drift_threshold must be > 0, got {drift_threshold}")

    beta_sur = beta * (1.0 + gamma)
    rng = np.random.Generator(np.random.Philox(key=seed))
    L = cfg.L
    x = _initial_state(rng, L)
    H = hamiltonian(x)

    H_traj = np.empty(n_steps, dtype=np.float64)
    attempts = 0
    accepted = 0
    stage1_accepts = 0
    stage2_rejects = 0
    disc_sum = 0.0
    disc_max = 0.0

    for t in range(n_steps):
        for _ in range(L):
            i = int(rng.integers(0, L))
            dH = _delta_H_flip(x, i)
            attempts += 1
            # Stufe 1: Surrogat. dH<=0 -> a1=1 ohne RNG-Verbrauch (wie a_kernel).
            if dH <= 0 or rng.random() < np.exp(-beta_sur * dH):
                stage1_accepts += 1
                # Stufe 2: exakte Energie (einzige exakte Auswertung).
                # a2 = min(1, e^{-(beta-beta_sur) dH}); ==1 exakt fuer gamma=0
                # ODER dH==0 -> kein RNG-Verbrauch (haelt gamma=0 bit-identisch
                # zum reinen Metropolis-Kernel).
                log_a2 = -(beta - beta_sur) * dH
                disc = abs(beta_sur - beta) * abs(dH)
                disc_sum += disc
                disc_max = max(disc_max, disc)
                if log_a2 >= 0.0 or rng.random() < np.exp(log_a2):
                    x[i] = 1 - x[i]
                    H += dH
                    accepted += 1
                else:
                    stage2_rejects += 1
        H_traj[t] = H

    post = H_traj[burn_in:]
    n_exact = stage1_accepts
    guard = DriftGuardReport(
        mean_discrepancy=disc_sum / n_exact if n_exact > 0 else 0.0,
        max_discrepancy=disc_max,
        threshold=float(drift_threshold),
        n_evals=int(n_exact),
        fired=bool(n_exact > 0 and disc_sum / n_exact > drift_threshold),
    )
    return DAResult(
        H_traj=H_traj,
        mean_H=float(np.mean(post)),
        acceptance=accepted / attempts,
        n_attempts=int(attempts),
        n_exact_evals=int(n_exact),
        exact_eval_savings=1.0 - n_exact / attempts,
        stage1_accept_rate=stage1_accepts / attempts,
        stage2_reject_rate=stage2_rejects / stage1_accepts if stage1_accepts else 0.0,
        drift_guard=guard,
        final_state=x.copy(),
    )
