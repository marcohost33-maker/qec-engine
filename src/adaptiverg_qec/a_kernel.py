"""A-Kernel: adaptiver MCMC-Sampler auf z=(x, theta)  [Spec 3-5].

MVP-Instanz (mvp_instance.py): x in {0,1}^L (Repetition-Code-Ring),
theta = beta in kompaktem [beta_min, beta_max]. Ziel: pi_beta(x) ~ exp(-beta H(x)),
H = #Domaenenwaende. Single-Spin-Flip-Metropolis (detailed balance gegen pi_beta).

Adaptive Komponente -- Freeze-Vertrag (Issue #51):
    Die Adaption bewegt das ZIEL pi_beta selbst. Mit dem summierbaren Schedule
    a_t = c / (1 + t/T0)^2 gilt e_{t+1} = (1 - a_t) e_t fuer e_t = beta_t - beta_star,
    also e_inf = e_0 prod_t (1 - a_t); fuer 0 < a_t < 1 geht das Produkt genau dann
    gegen 0, wenn sum a_t = inf. Ein summierbarer Schedule friert die Adaption
    deshalb bei einem FALSCHEN Ziel ein (c=0.5, T0=1, beta_0=0.1, beta_star=0.8:
    beta_inf = 0.8 - 0.7 sin(pi/sqrt2)/(pi/sqrt2) ~ 0.549). Auch beim Default
    (T0=100) erreicht die Gleitkomma-Iteration beta_star nie bit-exakt: sie bleibt
    einige ulp darunter stehen, weil a_t * (1 ulp) kleiner als ein halbes ulp ist
    und weggerundet wird.

    Die frueher hier stehende Begruendung ("sum a_t < inf ist die staerkste Form")
    war falsch: Andrieu & Thoms (2008) verlangen fuer die Schrittweiten einer
    stochastischen Approximation sum gamma_t = inf UND sum gamma_t^{1+lambda} < inf;
    Roberts & Rosenthal (2007) setzen ein FESTES Ziel pi voraus.

    Deshalb gilt seit Issue #51 die Pipeline
        WARM-UP (t < warmup_steps; adaptiv, nur Kalibration)
        -> FREEZE (vor Sweep t = warmup_steps: beta := beta_star, exakte Zuweisung)
        -> FIXED-TARGET BURN-IN (burn_in Sweeps bei beta_star, verworfen)
        -> PRODUCTION (alle uebrigen Sweeps bei beta_star).
    Produktionsschaetzer lesen ausschliesslich einen ProductionRecord; dessen
    Pruefung (require_frozen) verlangt fuer JEDES Sample beta == beta_star
    bit-exakt und lehnt sonst ab (fail-closed). Containment via kompaktem Theta
    (Spec 4.2) gilt weiter fuer die Warm-up-Phase.

Diese Datei implementiert REAL (kein Stub):
- exakte Metropolis-Akzeptanz mit detailliertem Gleichgewicht,
- Philox-counter-basiertes, reproduzierbares RNG (Spec 10.3a),
- Drift-Trajektorie H(z_t) fuer den Guard (drift.py),
- Warm-up-Schedule mit Containment-Clip und exaktem Freeze (Issue #51),
- getrennte Kalibrations-/Produktions-Records mit fail-closed-Pruefung.
"""

from __future__ import annotations

import math
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


class FreezeContractError(ValueError):
    """Ein Produktionsschaetzer sah ein Sample, das nicht beim eingefrorenen Ziel liegt."""


def diminishing_step_sizes(n: int, c: float, T0: float) -> np.ndarray:
    """Summierbarer Adaptions-Schedule a_t = c / (1 + t/T0)^2, t=0..n-1.

    sum_t a_t konvergiert (p=2 > 1). ACHTUNG (Issue #51): genau deshalb konvergiert
    ein damit bewegtes Ziel NICHT gegen beta_star (s. Moduldoku). Der Schedule ist
    nur in der Warm-up-Phase zulaessig; danach friert advance_chain das Ziel ein.
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
    freeze_at: int = 0,
) -> None:
    """Fuehre die Kette von state.t bis t_stop-1 fort (mutiert state + Arrays).

    a_t ist der VOLLE Schedule des Gesamtlaufs (Index = absoluter Sweep t), damit
    ein Resume denselben Schedule-Wert sieht wie der ununterbrochene Lauf.

    freeze_at (Issue #51): Sweeps t < freeze_at sind Warm-up und bewegen beta
    adaptiv; vor dem Sweep t = freeze_at wird beta EXAKT auf beta_target gesetzt
    (Zuweisung, keine Iteration), danach bleibt es dort. Der Default 0 heisst:
    keine zielwandernde Adaption. Ein Zustand mit t > freeze_at, dessen beta nicht
    bit-gleich beta_target ist (z.B. aus einem manipulierten Checkpoint), wird
    abgelehnt statt still geheilt.
    """
    if isinstance(freeze_at, bool) or not isinstance(freeze_at, int | np.integer):
        raise ValueError(f"freeze_at must be an int, got {type(freeze_at).__name__}")
    if freeze_at < 0:
        raise ValueError(f"freeze_at must be >= 0, got {freeze_at}")
    if t_stop > a_t.size or t_stop > H_out.size:
        raise ValueError(f"t_stop {t_stop} exceeds schedule/output length")
    # PR #53 (Codex P2): der Freeze weist beta_target ohne Containment-Clip zu. Ein Ziel
    # ausserhalb des kompakten Theta wird deshalb VOR jedem Sweep abgelehnt (NaN-sicher),
    # statt still zu laufen (vorher clippte das Update jeden Sweep).
    if not (cfg.beta_min <= beta_target <= cfg.beta_max):
        raise ValueError(
            f"beta_target {beta_target} outside compact Theta [{cfg.beta_min}, {cfg.beta_max}]"
        )
    beta_star = float(beta_target)
    if state.t > freeze_at and not _bit_equal(state.beta, beta_star):
        raise FreezeContractError(
            f"chain at t={state.t} is past freeze_at={freeze_at} but beta={state.beta!r} "
            f"!= beta_star={beta_star!r}"
        )
    x = state.x
    for t in range(state.t, t_stop):
        if t < freeze_at:
            # WARM-UP: Adaption Richtung beta_target, in Theta geclippt (Containment).
            state.beta += a_t[t] * (beta_target - state.beta)
            state.beta = min(max(state.beta, cfg.beta_min), cfg.beta_max)
        elif t == freeze_at:
            # FREEZE: exakte Zuweisung -- die Iteration erreicht beta_star nie bit-genau.
            state.beta = beta_star
        state.H, acc = _sweep(x, state.H, state.beta, rng)
        state.accepted += acc
        state.attempted += x.size
        H_out[t] = state.H
        beta_out[t] = state.beta
        if configs_out is not None:
            configs_out[t] = x
    state.t = t_stop


def _bit_equal(a: float, b: float) -> bool:
    """Bit-Gleichheit zweier float64 (unterscheidet -0.0/+0.0).

    Ohne NaN-Sonderarm: advance_chain prueft beta_target vorher auf [beta_min, beta_max],
    beta_star ist also nie NaN, und ein NaN-beta im Zustand hat andere Bits (Zensus PR #53).
    """
    return bool(np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64))


@dataclass(frozen=True)
class CalibrationRecord:
    """Warm-up-Samples (t < warmup_steps). NIE Eingabe eines Produktionsschaetzers (G7a)."""

    H: np.ndarray
    beta: np.ndarray
    beta_end: float
    """Letztes Warm-up-beta: dort stuende die Adaption ohne Freeze (NaN ohne Warm-up)."""


@dataclass(frozen=True)
class ProductionRecord:
    """Produktions-Samples bei eingefrorenem Ziel (G7a/G7d).

    Eigene Arrays (Kopien, kein View auf die Gesamt-Trajektorie). Jeder
    Produktionsschaetzer prueft den Record vor Gebrauch mit require_frozen().
    """

    beta_star: float
    t_start: int
    """Absoluter Sweep-Index des ersten Produktions-Samples (= warmup_steps + burn_in)."""
    H: np.ndarray
    beta: np.ndarray
    configs: np.ndarray | None = field(default=None, repr=False)


def require_frozen(record: ProductionRecord) -> ProductionRecord:
    """Fail-closed-Pruefung vor jedem Produktionsschaetzer (G7d).

    Verlangt einen ProductionRecord; H und beta 1D, gleich lang, nicht leer; JEDES
    beta BIT-gleich beta_star (kein Pre-Freeze-Sample, auch keines ein ulp daneben);
    H endlich; configs (falls vorhanden) gleich lang. Prueft bei jedem Aufruf neu,
    weil die Arrays veraenderlich sind. Gibt den Record zurueck.
    """
    if not isinstance(record, ProductionRecord):
        raise FreezeContractError(
            f"production estimators accept only a ProductionRecord, got {type(record).__name__}"
        )
    H = np.asarray(record.H, dtype=np.float64)
    beta = np.asarray(record.beta, dtype=np.float64)
    if H.ndim != 1 or beta.shape != H.shape:
        raise FreezeContractError(f"H/beta shape mismatch: {H.shape} vs {beta.shape}")
    if H.size == 0:
        raise FreezeContractError("empty production record")
    if not math.isfinite(record.beta_star):
        raise FreezeContractError(f"beta_star must be finite, got {record.beta_star!r}")
    star_bits = np.float64(record.beta_star).view(np.uint64)
    off = np.flatnonzero(beta.view(np.uint64) != star_bits)
    if off.size:
        i = int(off[0])
        raise FreezeContractError(
            f"{off.size} production sample(s) not at the frozen target; first at "
            f"t={record.t_start + i}: beta={float(beta[i])!r} != beta_star={record.beta_star!r}"
        )
    if not np.all(np.isfinite(H)):
        raise FreezeContractError("non-finite H in production record")
    if record.configs is not None and len(record.configs) != H.size:
        raise FreezeContractError(
            f"configs length {len(record.configs)} != production length {H.size}"
        )
    return record


def production_mean(record: ProductionRecord) -> float:
    """Ergodisches Mittel von H ueber die Produktionsphase (prueft require_frozen)."""
    return float(np.mean(require_frozen(record).H))


def split_phases(
    H_traj: np.ndarray,
    beta_traj: np.ndarray,
    *,
    warmup_steps: int,
    burn_in: int,
    beta_star: float,
    configs: np.ndarray | None = None,
) -> tuple[CalibrationRecord, ProductionRecord]:
    """Teile eine Gesamt-Trajektorie in Kalibrations- und Produktions-Record.

    Kopiert (G7a: kein gemeinsamer Speicher, auch nicht mit der Trajektorie) und
    prueft den Produktions-Record sofort mit require_frozen -- eine Trajektorie,
    deren Produktionsteil nicht eingefroren ist, ergibt keinen Record.
    """
    H_traj = np.asarray(H_traj, dtype=np.float64)
    beta_traj = np.asarray(beta_traj, dtype=np.float64)
    if H_traj.ndim != 1 or beta_traj.shape != H_traj.shape:
        raise ValueError(f"H/beta trajectory shape mismatch: {H_traj.shape} vs {beta_traj.shape}")
    t0 = warmup_steps + burn_in
    if not (warmup_steps >= 0 and burn_in >= 0 and t0 < H_traj.size):
        raise ValueError(
            "need 0 <= warmup_steps, 0 <= burn_in, warmup_steps + burn_in < n_steps; "
            f"got {warmup_steps}/{burn_in}/{H_traj.size}"
        )
    cal_beta = beta_traj[:warmup_steps].copy()
    calibration = CalibrationRecord(
        H=H_traj[:warmup_steps].copy(),
        beta=cal_beta,
        beta_end=float(cal_beta[-1]) if cal_beta.size else float("nan"),
    )
    production = ProductionRecord(
        beta_star=float(beta_star),
        t_start=t0,
        H=H_traj[t0:].copy(),
        beta=beta_traj[t0:].copy(),
        configs=None if configs is None else np.array(configs[t0:], copy=True),
    )
    return calibration, require_frozen(production)


@dataclass
class SampleResult:
    """Ausgabe eines A-Kernel-Laufs."""

    H_traj: np.ndarray
    """H(z_t) je Sweep, ALLE Phasen (fuer Drift-Guard; KEIN Produktionsschaetzer)."""
    beta_traj: np.ndarray
    """theta_t = beta_t je Sweep, alle Phasen (Containment- und Freeze-Beleg)."""
    mean_H: float
    """Ergodisches Mittel von H ueber die PRODUKTIONSPHASE (production_mean)."""
    acceptance: float
    """Empirische Metropolis-Akzeptanzrate (alle Sweeps)."""
    adaptation_sum: float
    """sum_t a_t ueber die Warm-up-Sweeps (0.0 ohne Warm-up). Reine Kalibrations-Info."""
    final_state: np.ndarray = field(repr=False)
    """Endkonfiguration x (fuer Checkpoint/Restart)."""
    configs: np.ndarray | None = field(default=None, repr=False)
    """Optionale Konfigurations-Snapshots (n_steps, L) in {0,1}, je Sweep nach
    dem Sweep aufgezeichnet. Nur gefuellt, wenn run_adaptive_mcmc(record_configs=True).
    Fuer Schaetzer NICHT direkt slicen, sondern production.configs nehmen."""
    calibration: CalibrationRecord | None = field(default=None, repr=False)
    """Warm-up-Record (G7a: getrennt vom Produktions-Record)."""
    production: ProductionRecord | None = field(default=None, repr=False)
    """Produktions-Record bei beta_star -- einzige Quelle von mean_H."""
    warmup_steps: int = 0
    burn_in: int = 0


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
    warmup_steps: int = 0,
) -> SampleResult:
    """Fuehre den A-Kernel nach dem Freeze-Vertrag aus (Issue #51).

    Phasen (Sweep-Indizes): [0, warmup_steps) Warm-up (adaptiv, Kalibration) ->
    Freeze (beta := beta_target exakt) -> [warmup_steps, warmup_steps + burn_in)
    Fixed-Target-Burn-in (verworfen) -> [warmup_steps + burn_in, n_steps) Produktion.

    Args:
        cfg: MVP-Konfiguration (L, Theta-Grenzen).
        beta_target: beta_star, das eingefrorene Ziel (in [beta_min, beta_max]).
        n_steps: Sweeps GESAMT (1 Sweep = L Einzel-Flip-Versuche).
        burn_in: Fixed-Target-Burn-in NACH dem Freeze (verworfen).
        seed: reproduzierbarer Seed (Philox).
        beta_start: Start-beta des Warm-ups. Ohne Warm-up (warmup_steps=0) muss es
            fehlen oder gleich beta_target sein -- ein Start-beta, das nie wirkt,
            waere ein still ignorierter Parameter. Mit Warm-up ist es Pflicht.
        adapt_c, adapt_T0: Warm-up-Schedule a_t = c/(1+t/T0)^2.
        record_configs: wenn True, speichere je Sweep die Konfiguration x in
            SampleResult.configs (n_steps, L) und die Produktionsphase davon in
            production.configs. Default False (spart Speicher).
        warmup_steps: Laenge des adaptiven Warm-ups (Default 0: kein Warm-up, beta
            liegt ab dem ersten Sweep exakt bei beta_target).

    Returns:
        SampleResult mit Trajektorien, getrennten Records und Produktionsschaetzer.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not (0 <= burn_in < n_steps):
        raise ValueError(f"need 0 <= burn_in < n_steps, got {burn_in}/{n_steps}")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int | np.integer):
        raise ValueError(f"warmup_steps must be an int, got {type(warmup_steps).__name__}")
    if not (warmup_steps >= 0 and warmup_steps + burn_in < n_steps):
        raise ValueError(
            "need 0 <= warmup_steps and warmup_steps + burn_in < n_steps, "
            f"got {warmup_steps}/{burn_in}/{n_steps}"
        )
    if not (cfg.beta_min <= beta_target <= cfg.beta_max):
        raise ValueError(
            f"beta_target {beta_target} outside compact Theta "
            f"[{cfg.beta_min}, {cfg.beta_max}] (Containment violation)"
        )

    if warmup_steps == 0:
        if beta_start is not None and beta_start != beta_target:
            raise ValueError(
                f"beta_start {beta_start} != beta_target {beta_target} without a warm-up: "
                "the start value would be ignored (set warmup_steps > 0 or drop beta_start)"
            )
        beta_start = beta_target
    elif beta_start is None:
        raise ValueError("warmup_steps > 0 needs an explicit beta_start")
    if not (cfg.beta_min <= beta_start <= cfg.beta_max):
        raise ValueError(f"beta_start {beta_start} outside compact Theta")

    state, rng = new_chain_state(cfg, seed=seed, beta_start=beta_start)
    L = cfg.L

    a_t = diminishing_step_sizes(n_steps, adapt_c, adapt_T0)
    adaptation_sum = float(np.sum(a_t[:warmup_steps]))

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
        freeze_at=int(warmup_steps),
    )

    calibration, production = split_phases(
        H_traj,
        beta_traj,
        warmup_steps=int(warmup_steps),
        burn_in=burn_in,
        beta_star=beta_target,
        configs=configs,
    )
    return SampleResult(
        H_traj=H_traj,
        beta_traj=beta_traj,
        mean_H=production_mean(production),
        acceptance=state.accepted / state.attempted,
        adaptation_sum=adaptation_sum,
        final_state=state.x.copy(),
        configs=configs,
        calibration=calibration,
        production=production,
        warmup_steps=int(warmup_steps),
        burn_in=burn_in,
    )
