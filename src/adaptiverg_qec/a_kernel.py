"""A-Kernel: adaptiver MCMC-Sampler auf z=(x, theta)  [Spec 3-5].

MVP-Instanz (mvp_instance.py): x in {0,1}^L (Repetition-Code-Ring),
theta = beta in kompaktem [beta_min, beta_max]. Ziel: pi_beta(x) ~ exp(-beta H(x)),
H = #Domaenenwaende. Single-Spin-Flip-Metropolis (detailed balance gegen pi_beta).

Adaptive Komponente (Spec 4.1, Diminishing Adaptation):
    Containment via KOMPAKTEM Theta (Spec 4.2). theta wird mit einem
    summierbaren Schedule a_t = c / (1 + t/T0)^2 (=> sum a_t < inf) Richtung
    theta_target bewegt. sum a_t < inf ist die staerkste Form (P1.2 Schritt 3,
    DOCX: starkes Gesetz d. grossen Zahlen unter sum a_t < inf).

Diese Datei implementiert REAL (kein Stub):
- exakte Metropolis-Akzeptanz mit detailliertem Gleichgewicht,
- Philox-counter-basiertes, reproduzierbares RNG (Spec 10.3a),
- Drift-Trajektorie H(z_t) fuer den Guard (drift.py),
- Diminishing-Adaptation-Schedule mit Summierbarkeits-Pruefung + Containment-Clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mvp_instance import MVPConfig


def _initial_state(rng: np.random.Generator, L: int) -> np.ndarray:
    """Zufaellige Startkonfiguration x in {0,1}^L."""
    return rng.integers(0, 2, size=L, dtype=np.int8)


def hamiltonian(x: np.ndarray) -> int:
    """H(x) = #Domaenenwaende (periodisch). Vektorisiert, exakt."""
    x = np.asarray(x)
    return int(np.sum(x != np.roll(x, -1)))


def _delta_H_flip(x: np.ndarray, i: int) -> int:
    """Aenderung von H beim Flip von Bit i (lokal, O(1)).

    Nur die beiden Kanten (i-1,i) und (i,i+1) aendern sich.
    """
    L = x.size
    left = x[(i - 1) % L]
    right = x[(i + 1) % L]
    xi = x[i]
    xi_new = 1 - xi
    before = int(xi != left) + int(xi != right)
    after = int(xi_new != left) + int(xi_new != right)
    return after - before


def diminishing_step_sizes(n: int, c: float, T0: float) -> np.ndarray:
    """Summierbarer Adaptions-Schedule a_t = c / (1 + t/T0)^2, t=0..n-1.

    sum_t a_t konvergiert (p=2 > 1). Erfuellt Spec 4.1 (Diminishing Adaptation)
    in der staerksten summierbaren Form.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}")
    if T0 <= 0:
        raise ValueError(f"T0 must be > 0, got {T0}")
    t = np.arange(n, dtype=np.float64)
    return c / (1.0 + t / T0) ** 2


@dataclass
class ChainState:
    """Resumierbarer Zustand einer A-Kernel-Kette (Phase-6 Checkpoint/Restart).

    Zusammen mit dem RNG-Bit-Generator-State (rng.bit_generator.state) beschreibt
    dieser Zustand den Sampler VOLLSTAENDIG: advance_chain() ab (state, rng)
    liefert bit-identisch dieselbe Fortsetzung wie ein ununterbrochener Lauf.
    """

    x: np.ndarray
    """Aktuelle Konfiguration x in {0,1}^L."""
    H: int
    """Aktuelles H(x) (inkrementell mitgefuehrt)."""
    beta: float
    """Aktuelles theta_t = beta_t (Diminishing-Adaptation-Zustand)."""
    t: int
    """Anzahl abgeschlossener Sweeps."""
    accepted: int
    attempted: int


def new_chain_state(
    cfg: MVPConfig, *, seed: int, beta_start: float
) -> tuple[ChainState, np.random.Generator]:
    """Initialisiere (ChainState, Philox-RNG) exakt wie run_adaptive_mcmc."""
    if not (cfg.beta_min <= beta_start <= cfg.beta_max):
        raise ValueError(f"beta_start {beta_start} outside compact Theta")
    rng = np.random.Generator(np.random.Philox(key=seed))
    x = _initial_state(rng, cfg.L)
    return (
        ChainState(x=x, H=hamiltonian(x), beta=float(beta_start), t=0, accepted=0, attempted=0),
        rng,
    )


def _sweep(x: np.ndarray, H: int, beta: float, rng: np.random.Generator) -> tuple[int, int]:
    """Ein Sweep = L Einzel-Flip-Metropolis-Schritte (detailed balance).

    Mutiert x in-place; gibt (H_neu, akzeptierte Flips) zurueck. RNG-Verbrauch:
    je Versuch 1x integers, plus 1x random NUR falls dH > 0 (bit-Stream-Vertrag,
    auf dem G6/G37 und der Checkpoint-Determinismus beruhen).
    """
    L = x.size
    accepted = 0
    for _ in range(L):
        i = int(rng.integers(0, L))
        dH = _delta_H_flip(x, i)
        # Akzeptanz min(1, exp(-beta dH)). dH<=0 immer akzeptiert.
        if dH <= 0 or rng.random() < np.exp(-beta * dH):
            x[i] = 1 - x[i]
            H += dH
            accepted += 1
    return H, accepted


def advance_chain(
    state: ChainState,
    rng: np.random.Generator,
    cfg: MVPConfig,
    *,
    beta_target: float,
    a_t: np.ndarray,
    t_stop: int,
    H_out: np.ndarray,
    beta_out: np.ndarray,
    configs_out: np.ndarray | None = None,
) -> None:
    """Fuehre die Kette von state.t bis t_stop-1 fort (mutiert state + Arrays).

    a_t ist der VOLLE Schedule des Gesamtlaufs (Index = absoluter Sweep t), damit
    ein Resume denselben Schedule-Wert sieht wie der ununterbrochene Lauf.
    """
    if t_stop > a_t.size or t_stop > H_out.size:
        raise ValueError(f"t_stop {t_stop} exceeds schedule/output length")
    x = state.x
    for t in range(state.t, t_stop):
        # Diminishing Adaptation Richtung beta_target, in Theta geclippt (Containment).
        state.beta += a_t[t] * (beta_target - state.beta)
        state.beta = min(max(state.beta, cfg.beta_min), cfg.beta_max)
        state.H, acc = _sweep(x, state.H, state.beta, rng)
        state.accepted += acc
        state.attempted += x.size
        H_out[t] = state.H
        beta_out[t] = state.beta
        if configs_out is not None:
            configs_out[t] = x
    state.t = t_stop


@dataclass
class SampleResult:
    """Ausgabe eines A-Kernel-Laufs."""

    H_traj: np.ndarray
    """H(z_t) je gespeichertem Schritt (fuer Drift-Guard + Observablen)."""
    beta_traj: np.ndarray
    """theta_t = beta_t je Schritt (Containment-Beleg)."""
    mean_H: float
    """Ergodisches Mittel von H nach Burn-in (Schaetzer)."""
    acceptance: float
    """Empirische Metropolis-Akzeptanzrate."""
    adaptation_sum: float
    """sum_t a_t (muss endlich/summierbar sein -> Diminishing Adaptation)."""
    final_state: np.ndarray = field(repr=False)
    """Endkonfiguration x (fuer Checkpoint/Restart)."""
    configs: np.ndarray | None = field(default=None, repr=False)
    """Optionale Konfigurations-Snapshots (n_steps, L) in {0,1}, je Sweep nach
    dem Sweep aufgezeichnet. Nur gefuellt, wenn run_adaptive_mcmc(record_configs=True);
    Quelle der korrelierten S/S'-Zeitreihen fuer den autokorr-bewussten Swendsen-
    Schaetzer (Phase-3a). None spart Speicher, wenn nur H_traj gebraucht wird."""


def run_adaptive_mcmc(
    cfg: MVPConfig,
    *,
    beta_target: float,
    n_steps: int,
    burn_in: int,
    seed: int,
    beta_start: float | None = None,
    adapt_c: float = 0.5,
    adapt_T0: float = 100.0,
    record_configs: bool = False,
) -> SampleResult:
    """Fuehre den adaptiven Single-Spin-Flip-Metropolis-Sampler aus.

    Args:
        cfg: MVP-Konfiguration (L, Theta-Grenzen).
        beta_target: Ziel-beta (muss in [beta_min, beta_max] liegen).
        n_steps: Sweeps (1 Sweep = L Einzel-Flip-Versuche).
        burn_in: zu verwerfende Anfangs-Sweeps.
        seed: reproduzierbarer Seed (Philox).
        beta_start: Start-beta (Default: beta_min) -> Adaptation laeuft.
        adapt_c, adapt_T0: Schedule-Parameter a_t = c/(1+t/T0)^2.
        record_configs: wenn True, speichere je Sweep die Konfiguration x in
            SampleResult.configs (n_steps, L) -- Quelle der korrelierten
            S/S'-Zeitreihen fuer den Phase-3a-Swendsen-Schaetzer. Default False
            (spart Speicher: O(n_steps*L) statt O(n_steps)).

    Returns:
        SampleResult mit Trajektorien + Schaetzern.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not (0 <= burn_in < n_steps):
        raise ValueError(f"need 0 <= burn_in < n_steps, got {burn_in}/{n_steps}")
    if not (cfg.beta_min <= beta_target <= cfg.beta_max):
        raise ValueError(
            f"beta_target {beta_target} outside compact Theta "
            f"[{cfg.beta_min}, {cfg.beta_max}] (Containment violation)"
        )

    beta_start = cfg.beta_min if beta_start is None else beta_start
    if not (cfg.beta_min <= beta_start <= cfg.beta_max):
        raise ValueError(f"beta_start {beta_start} outside compact Theta")

    state, rng = new_chain_state(cfg, seed=seed, beta_start=beta_start)
    L = cfg.L

    a_t = diminishing_step_sizes(n_steps, adapt_c, adapt_T0)
    adaptation_sum = float(np.sum(a_t))

    H_traj = np.empty(n_steps, dtype=np.float64)
    beta_traj = np.empty(n_steps, dtype=np.float64)
    configs = np.empty((n_steps, L), dtype=np.int8) if record_configs else None

    advance_chain(
        state,
        rng,
        cfg,
        beta_target=beta_target,
        a_t=a_t,
        t_stop=n_steps,
        H_out=H_traj,
        beta_out=beta_traj,
        configs_out=configs,
    )

    post = H_traj[burn_in:]
    return SampleResult(
        H_traj=H_traj,
        beta_traj=beta_traj,
        mean_H=float(np.mean(post)),
        acceptance=state.accepted / state.attempted,
        adaptation_sum=adaptation_sum,
        final_state=state.x.copy(),
        configs=configs,
    )
