"""Phase-7: exakte TV-Mischung des A-Kernels + adaptive Kette (Phase-1/2-Akzeptanz).

Schliesst zwei seit Phase 1/2 offene Akzeptanzkriterien der ROADMAP:

* Phase 1: "geometrische Ergodizitaet empirisch (TV-Distanz faellt)".
* Phase 2: "kein AdapFail; Mischzeiten stochastisch beschraenkt" (Containment).

------------------------------------------------------------------------------
WARUM EXAKT MOEGLICH
------------------------------------------------------------------------------
Die MVP-Instanz (mvp_instance.py) hat den ENDLICHEN Zustandsraum X = {0,1}^L.
Fuer kleines L (2^L <= 4096) laesst sich die Uebergangsmatrix des Samplers
vollstaendig aufschreiben. Dann sind alle Groessen dieses Moduls Orakel, nicht
Schaetzungen:

  Einzelschritt (identisch zu a_kernel._sweep): waehle i ~ Unif{0..L-1},
  flippe mit Wkt min(1, exp(-beta dH)).
      P_1(x, x^(i)) = (1/L) min(1, exp(-beta dH_i(x))),
      P_1(x, x)     = 1 - sum_i P_1(x, x^(i)).
  Ein Sweep ist P = P_1^L (a_kernel fuehrt L Einzelschritte je Sweep aus).

  P_1 ist reversibel bzgl. pi_beta(x) ~ exp(-beta H(x)) (Metropolis), damit
  auch jede Potenz. Die Aehnlichkeitstransformation A = D^{1/2} P D^{-1/2},
  D = diag(pi), ist symmetrisch; ihr Spektrum ist reell und
      lambda_* = max{|lambda| : lambda Eigenwert von P, lambda != 1}
  (Levin, Peres, Wilmer, "Markov Chains and Mixing Times", 2. Aufl. 2017,
  Kap. 12: absolute spectral gap gamma_* = 1 - lambda_*, t_rel = 1/gamma_*).

SANDWICH-ORAKEL (beide Seiten aus der Spektralzerlegung, LPW Kap. 12):
  Obere Schranke je Startzustand x (l2-Schranke, Cauchy-Schwarz):
      ||P^t(x,.) - pi||_TV <= (1/2) sqrt((1 - pi(x)) / pi(x)) * lambda_*^t.
  Untere Schranke fuer d(t) = max_x ||P^t(x,.) - pi||_TV (Eigenfunktions-
  Argument im Beweis von LPW Thm 12.5):
      d(t) >= (1/2) lambda_*^t.
  Der exakte Verlauf muss zwischen beiden liegen und mit Rate lambda_* fallen
  -- das IST "geometrische Ergodizitaet", nicht nur ein fallender Plot.

EMPIRISCHE SEITE (der ECHTE Sampler, kein Nachbau):
  n unabhaengige Ketten laufen durch a_kernel.advance_chain (derselbe Code-Pfad
  wie alle anderen Phasen) ab demselben Startzustand. Das Histogramm ueber X
  nach t Sweeps ist ein Multinomial-Sample von P^t(x0,.). Rigoroses Band
  (Dreiecksungleichung + Jensen + McDiarmid, TV ist 1/n-Lipschitz je Kette):
      |TV_emp(t) - TV_exakt(t)| <= (1/2) sum_y sqrt(p_t(y)(1-p_t(y))/n)
                                    + sqrt(ln(2/delta) / (2n))
  mit Wkt >= 1 - delta je t. Dieselbe Schranke gilt fuer TV(p_hat_t, p_t)
  selbst (marginal_band_ratio, schaerfer). Faellt der Sampler aus dem Band, ist
  er NICHT der Kern, den die Spektralanalyse beschreibt.

ADAPTIVE KETTE (Phase 2, Freeze-Vertrag aus Issue #51):
  a_kernel bewegt beta DETERMINISTISCH: im Warm-up (t < freeze_at) gilt
  beta_{t+1} = clip(beta_t + a_t (beta_target - beta_t)) vor jedem Sweep; vor dem
  Sweep t = freeze_at wird beta := beta_target per ZUWEISUNG gesetzt und bleibt dort
  (a_kernel.advance_chain). Die Randverteilung der adaptiven
  Kette ist daher exakt mu_{t+1} = mu_t P_{beta_{t+1}} -- ein zeitinhomogenes
  Matrixprodukt, keine Simulation. Zwei Aussagen:
  1. Containment: sup_{beta in Theta} t_rel(beta) < inf (Theta kompakt, Kerne
     stetig in beta). Kontrolle in Gegenrichtung: t_rel(beta) waechst mit beta
     ohne Schranke (1D-Ising: kritischer Punkt beta_c = inf) -- genau deshalb
     muss Theta den kritischen Punkt ausschliessen (Spec 4.2, mvp_instance.py).
  2. [LUECKE, hier quantifiziert; seit #51/#53 im Sampler GESCHLOSSEN] Ohne Freeze
     (freeze_at = n, die Defektklasse des alten Samplers) friert ein SUMMIERBARER
     Schedule (sum a_t < inf, P1.2) die Adaption ein: beta_t -> beta_inf mit
         beta_inf - beta_target = (beta_0 - beta_target) prod_t (1 - a_t) != 0
     (solange kein Clip greift). Die adaptive Kette konvergiert dann nach
     pi_{beta_inf}, NICHT nach pi_{beta_target}; der TV-Boden ist
     ||pi_{beta_inf} - pi_{beta_target}||_TV. Beim Default (c=0.5, T0=100) ist
     prod(1-a_t) ~ e^{-50} und der Boden numerisch 0; bei kleinem T0 nicht.
     Einordnung: Roberts & Rosenthal (2007, J. Appl. Probab. 44(2), 458-475)
     zeigen Ergodizitaet aus Diminishing Adaptation + Containment fuer Kerne
     mit GEMEINSAMEM Ziel pi. Im MVP aendert theta = beta das Ziel selbst
     (pi_beta), der Satz greift also nicht unmittelbar; was hier exakt gezeigt
     wird, ist die Konvergenz gegen pi_{beta_inf} -- und dass beta_inf vom
     Schedule abhaengt, statt stillschweigend = beta_target angenommen zu werden.
  3. [OK, Freeze-Vertrag] Mit freeze_at < n ist der Kern ab dem Freeze zeit-
     homogen P_{beta_target}; mu_t konvergiert geometrisch exakt nach pi_{beta_target},
     gleichgueltig wie schlecht der Warm-up-Schedule ist (G48 zeigt beides).

EHRLICHE SCOPE-GRENZE: exakt nur fuer den 1D-Ring mit 2^L <= 4096 Zustaenden.
Fuer grosse L (und 2D) bleiben R-hat/ESS (rhat.py) und tau_int (autocorr.py) die
Diagnostik; dieses Modul kalibriert, dass jene Diagnostik auf einem Kern arbeitet,
dessen Mischverhalten hier exakt bekannt ist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import a_kernel
from .mvp_instance import MVPConfig

__all__ = [
    "MAX_STATES",
    "all_states",
    "hamiltonians",
    "stationary_distribution",
    "single_flip_kernel",
    "sweep_kernel",
    "SpectralSummary",
    "spectral_summary",
    "tv_distance",
    "exact_tv_curve",
    "tv_sandwich",
    "EmpiricalTV",
    "empirical_tv_curve",
    "empirical_tv_band",
    "marginal_band_ratio",
    "fit_geometric_rate",
    "adaptive_beta_schedule",
    "AdaptiveTV",
    "adaptive_exact_tv",
    "relaxation_time_profile",
]

MAX_STATES = 4096
"""Obergrenze fuer 2^L (dichte 4096x4096-Matrix, ~128 MiB float64)."""

_REVERSIBILITY_RTOL = 1e-12


def _check_L(L: int) -> int:
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise ValueError(f"L must be an int, got {type(L).__name__}")
    L = int(L)
    if L < 3:
        raise ValueError(f"L must be >= 3 (ring), got {L}")
    if 2**L > MAX_STATES:
        raise ValueError(f"2^L = {2**L} exceeds MAX_STATES={MAX_STATES}; exact analysis infeasible")
    return L


def _check_beta(beta: float) -> float:
    beta = float(beta)
    if not (math.isfinite(beta) and beta >= 0.0):  # positiv formuliert: NaN faellt hier
        raise ValueError(f"beta must be finite and >= 0, got {beta}")
    return beta


def all_states(L: int) -> np.ndarray:
    """Alle 2^L Konfigurationen als (2^L, L) int8; Zeile k = Binaerdarstellung von k.

    Bit i von k ist x_i (i=0 niedrigstwertig) -- dieselbe Kodierung wie
    ``_encode`` fuer die empirischen Histogramme.
    """
    L = _check_L(L)
    k = np.arange(2**L, dtype=np.int64)
    return ((k[:, None] >> np.arange(L)) & 1).astype(np.int8)


def _encode(x: np.ndarray) -> np.ndarray:
    """(n, L) {0,1} -> Zustandsindex sum_i x_i 2^i (Umkehrung von all_states)."""
    x = np.asarray(x, dtype=np.int64)
    return x @ (1 << np.arange(x.shape[-1], dtype=np.int64))


def hamiltonians(L: int) -> np.ndarray:
    """H(x) = #Domaenenwaende (periodisch) fuer alle Zustaende, = a_kernel.hamiltonian."""
    states = all_states(L)
    walls = states != np.roll(states, -1, axis=1)
    return np.sum(walls, axis=1).astype(np.int64)


def stationary_distribution(L: int, beta: float) -> np.ndarray:
    """pi_beta(x) = exp(-beta H(x)) / Z, exakt (log-sum-exp-stabil)."""
    beta = _check_beta(beta)
    H = hamiltonians(L).astype(np.float64)
    logw = -beta * H
    w = np.exp(logw - logw.max())
    return w / w.sum()


def single_flip_kernel(L: int, beta: float) -> np.ndarray:
    """Exakte Matrix EINES Metropolis-Einzelschritts von a_kernel._sweep.

    P_1(x, x^(i)) = (1/L) min(1, exp(-beta dH_i(x))); Rest auf der Diagonale.
    """
    L = _check_L(L)
    beta = _check_beta(beta)
    n = 2**L
    H = hamiltonians(L)
    idx = np.arange(n, dtype=np.int64)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(L):
        j = idx ^ (1 << i)
        dH = (H[j] - H[idx]).astype(np.float64)
        acc = np.where(dH <= 0, 1.0, np.exp(-beta * np.maximum(dH, 0.0)))
        P[idx, j] += acc / L
    P[idx, idx] = 1.0 - P.sum(axis=1)
    return P


def sweep_kernel(L: int, beta: float) -> np.ndarray:
    """Ein Sweep = L Einzelschritte: P = P_1^L (a_kernel-Einheit "1 Sweep")."""
    return np.linalg.matrix_power(single_flip_kernel(L, beta), _check_L(L))


@dataclass(frozen=True)
class SpectralSummary:
    """Spektrale Kennzahlen eines reversiblen Kerns (LPW 2017, Kap. 12)."""

    eigenvalues: np.ndarray
    """Alle Eigenwerte, absteigend sortiert (reell, da reversibel)."""
    lambda_star: float
    """max |lambda| ueber lambda != 1 (Vielfachheit des Eigenwerts 1 beachtet)."""
    abs_spectral_gap: float
    """gamma_* = 1 - lambda_*."""
    relaxation_time: float
    """t_rel = 1 / gamma_* (inf, falls gamma_* <= gap_tol)."""
    reversibility_residual: float
    """max |pi_x P_xy - pi_y P_yx| (Detailed-Balance-Check, muss ~eps sein)."""
    geometrically_ergodic: bool
    """lambda_* < 1 - tol: der Kern mischt geometrisch (Guard)."""


def spectral_summary(P: np.ndarray, pi: np.ndarray, *, gap_tol: float = 1e-10) -> SpectralSummary:
    """Spektrum eines bzgl. pi reversiblen Kerns via symmetrischer Aehnlichkeit.

    Fail-closed: ist P nicht stochastisch oder nicht reversibel bzgl. pi, wird
    geworfen -- die Schranken dieses Moduls gelten nur fuer reversible Kerne.
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or pi.shape != (P.shape[0],):
        raise ValueError("P must be square and pi must match its dimension")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(pi))):
        raise ValueError("P and pi must be finite")
    if not (np.all(pi > 0) and abs(pi.sum() - 1.0) < 1e-12):
        raise ValueError("pi must be a strictly positive probability vector")
    if not (np.all(P >= -1e-15) and np.allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("P must be row-stochastic")
    flow = pi[:, None] * P
    residual = float(np.max(np.abs(flow - flow.T)))
    if not (residual <= _REVERSIBILITY_RTOL):
        raise ValueError(f"P is not reversible w.r.t. pi (residual {residual:.3e})")
    s = np.sqrt(pi)
    A = s[:, None] * P / s[None, :]
    A = 0.5 * (A + A.T)  # entfernt nur Rundungsasymmetrie (residual ~ eps)
    ev = np.linalg.eigvalsh(A)[::-1]
    # Eigenwert 1 genau einmal entfernen (der zu pi gehoerende). Ein ZWEITER
    # Eigenwert ~1 (reduzibel) bleibt drin und setzt lambda_* = 1.
    rest = ev[1:]
    lambda_star = float(np.max(np.abs(rest))) if rest.size else 0.0
    lambda_star = min(lambda_star, 1.0)
    gap = 1.0 - lambda_star
    ergodic = bool(gap > gap_tol)
    return SpectralSummary(
        eigenvalues=ev,
        lambda_star=lambda_star,
        abs_spectral_gap=gap,
        # Unterhalb gap_tol ist 1/gap nur Rundungsrauschen (beta=0: ~6e14) -> inf.
        relaxation_time=(1.0 / gap) if ergodic else math.inf,
        reversibility_residual=residual,
        geometrically_ergodic=ergodic,
    )


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """||p - q||_TV = (1/2) sum |p - q|."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {q.shape}")
    return 0.5 * float(np.sum(np.abs(p - q)))


def exact_tv_curve(P: np.ndarray, pi: np.ndarray, x0: int, n_steps: int) -> np.ndarray:
    """TV(delta_{x0} P^t, pi) fuer t = 0..n_steps (exakt, Vektor-Matrix-Iteration)."""
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]
    if not (0 <= int(x0) < n):
        raise ValueError(f"x0 must be a state index in [0, {n}), got {x0}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    mu = np.zeros(n)
    mu[int(x0)] = 1.0
    out = np.empty(n_steps + 1)
    for t in range(n_steps + 1):
        out[t] = tv_distance(mu, pi)
        mu = mu @ P
    return out


def tv_sandwich(
    P: np.ndarray, pi: np.ndarray, n_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SpectralSummary]:
    """(lower, d(t), upper, spektrum) fuer t = 0..n_steps.

    lower(t) = lambda_*^t / 2,
    d(t)     = max_x ||P^t(x,.) - pi||_TV   (exakt, volle Matrixpotenz),
    upper(t) = (1/2) sqrt((1 - pi_min)/pi_min) lambda_*^t  (worst start).
    """
    spec = spectral_summary(P, pi)
    P = np.asarray(P, dtype=np.float64)
    t = np.arange(n_steps + 1)
    lam_t = spec.lambda_star**t
    pmin = float(np.min(pi))
    lower = 0.5 * lam_t
    upper = 0.5 * math.sqrt((1.0 - pmin) / pmin) * lam_t
    d = np.empty(n_steps + 1)
    Pt = np.eye(P.shape[0])
    for k in range(n_steps + 1):
        d[k] = 0.5 * float(np.max(np.sum(np.abs(Pt - pi[None, :]), axis=1)))
        Pt = Pt @ P
    return lower, d, upper, spec


@dataclass(frozen=True)
class EmpiricalTV:
    """TV-Verlauf des ECHTEN Samplers (n unabhaengige Ketten, gleicher Start)."""

    tv: np.ndarray
    """TV(empirisches Histogramm nach t Sweeps, pi) fuer t = 0..n_sweeps."""
    histograms: np.ndarray
    """(n_sweeps+1, 2^L) empirische Randverteilungen p_hat_t."""
    n_chains: int
    L: int
    beta: float
    x0: int


def empirical_tv_curve(
    L: int,
    beta: float,
    *,
    x0: int,
    n_chains: int,
    n_sweeps: int,
    seed: int,
) -> EmpiricalTV:
    """Fahre n_chains Ketten durch a_kernel.advance_chain (fixes beta) und histogrammiere.

    Jede Kette bekommt einen eigenen Philox-Stream aus SeedSequence(seed).spawn
    (unabhaengig, reproduzierbar). a_t = 0: beta bleibt exakt fix, der Code-Pfad
    ist trotzdem derselbe wie im adaptiven Lauf.
    """
    L = _check_L(L)
    beta = _check_beta(beta)
    if n_chains < 2 or n_sweeps < 1:
        raise ValueError(f"need n_chains >= 2 and n_sweeps >= 1, got {n_chains}/{n_sweeps}")
    if not (0 <= int(x0) < 2**L):
        raise ValueError(f"x0 must be in [0, 2^L), got {x0}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    if not (beta > 0.0):
        raise ValueError(
            "empirical_tv_curve needs beta > 0 (MVPConfig: compact Theta, beta_min > 0)"
        )
    # Theta muss beta enthalten; mit a_t = 0 wird beta nie bewegt, die Obergrenze
    # ist nur formal (MVPConfig verlangt beta_max > beta_min).
    cfg = MVPConfig(L=L, beta_min=beta, beta_max=beta + 1.0)
    pi = stationary_distribution(L, beta)
    start = all_states(L)[int(x0)]
    zeros = np.zeros(n_sweeps)
    finals = np.empty((n_sweeps + 1, n_chains), dtype=np.int64)
    finals[0] = int(x0)
    configs = np.empty((n_sweeps, L), dtype=np.int8)
    H_out = np.empty(n_sweeps)
    beta_out = np.empty(n_sweeps)
    for c, child in enumerate(np.random.SeedSequence(int(seed)).spawn(n_chains)):
        rng = np.random.Generator(np.random.Philox(child))
        x = start.copy()
        state = a_kernel.ChainState(
            x=x, H=a_kernel.hamiltonian(x), beta=beta, t=0, accepted=0, attempted=0
        )
        a_kernel.advance_chain(
            state,
            rng,
            cfg,
            beta_target=beta,
            a_t=zeros,
            t_stop=n_sweeps,
            H_out=H_out,
            beta_out=beta_out,
            configs_out=configs,
        )
        finals[1:, c] = _encode(configs)
    n_states = 2**L
    hist = np.array([np.bincount(row, minlength=n_states) / n_chains for row in finals])
    tv = np.array([tv_distance(h, pi) for h in hist])
    return EmpiricalTV(tv=tv, histograms=hist, n_chains=int(n_chains), L=L, beta=beta, x0=int(x0))


def empirical_tv_band(
    P: np.ndarray, x0: int, n_steps: int, n_chains: int, delta: float
) -> np.ndarray:
    """Rigorose Halbbreite |TV_emp - TV_exakt| je t (Wkt >= 1-delta je t, s. Modul-Doc)."""
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}")
    if n_chains < 1:
        raise ValueError(f"n_chains must be >= 1, got {n_chains}")
    P = np.asarray(P, dtype=np.float64)
    mu = np.zeros(P.shape[0])
    mu[int(x0)] = 1.0
    mcd = math.sqrt(math.log(2.0 / delta) / (2.0 * n_chains))
    out = np.empty(n_steps + 1)
    for t in range(n_steps + 1):
        out[t] = 0.5 * float(np.sum(np.sqrt(mu * (1.0 - mu) / n_chains))) + mcd
        mu = mu @ P
    return out


def marginal_band_ratio(emp: EmpiricalTV, P: np.ndarray, *, delta: float) -> float:
    """max_t TV(p_hat_t, delta_{x0} P^t) / Band_t.

    <= 1 heisst: der Sampler ist mit dem Kern P vertraeglich (Wkt >= 1-delta je t).
    Schaerfer als der Vergleich der TV-zu-pi-Kurven, weil die GANZE Randverteilung
    verglichen wird, nicht nur ihr Abstand zu pi. Trennschaerfe ehrlich begrenzt:
    bei n=4000 Ketten (L=6) faellt ein um 50 % falsches beta (1.2 statt 0.8) mit
    Faktor ~3.6 heraus, ein um 12 % falsches (0.9) oder L+-1 statt L Einzelschritte
    je Sweep NICHT (gemessen 0.98 / 0.76 / 0.81).
    """
    P = np.asarray(P, dtype=np.float64)
    T = emp.histograms.shape[0] - 1
    band = empirical_tv_band(P, emp.x0, T, emp.n_chains, delta)
    mu = np.zeros(P.shape[0])
    mu[emp.x0] = 1.0
    worst = 0.0
    for t in range(T + 1):
        worst = max(worst, tv_distance(emp.histograms[t], mu) / band[t])
        mu = mu @ P
    return worst


def fit_geometric_rate(tv: np.ndarray, *, floor: float) -> tuple[float, int]:
    """Kleinste-Quadrate-Steigung von ln TV(t) ueber die Punkte mit TV > floor.

    Returns:
        (rate = exp(Steigung), Anzahl genutzter Punkte). Fail-closed bei < 3 Punkten.
    """
    tv = np.asarray(tv, dtype=np.float64)
    if not np.all(np.isfinite(tv)):
        raise ValueError("tv must be finite")
    if not (floor > 0):
        raise ValueError(f"floor must be > 0, got {floor}")
    t = np.arange(tv.size)
    mask = (tv > floor) & (t >= 1)
    if int(mask.sum()) < 3:
        raise ValueError("fewer than 3 points above the noise floor; cannot fit a rate")
    slope = np.polyfit(t[mask], np.log(tv[mask]), 1)[0]
    return float(math.exp(slope)), int(mask.sum())


def adaptive_beta_schedule(
    cfg: MVPConfig,
    *,
    beta_start: float,
    beta_target: float,
    a_t: np.ndarray,
    freeze_at: int,
) -> np.ndarray:
    """beta_t exakt wie a_kernel.advance_chain(freeze_at=...).

    t < freeze_at: Warm-up-Update + Containment-Clip VOR dem Sweep; ab t = freeze_at:
    beta = beta_target per Zuweisung. ``freeze_at = a_t.size`` bildet die alte,
    zielwandernde Defektklasse nach (kein Freeze); es gibt bewusst keinen Default.
    """
    a_t = np.asarray(a_t, dtype=np.float64)
    if not np.all(np.isfinite(a_t)):
        raise ValueError("a_t must be finite")
    if isinstance(freeze_at, bool) or not isinstance(freeze_at, int | np.integer):
        raise ValueError(f"freeze_at must be an int, got {type(freeze_at).__name__}")
    if not 0 <= freeze_at <= a_t.size:
        raise ValueError(f"need 0 <= freeze_at <= {a_t.size}, got {freeze_at}")
    beta = float(beta_start)
    out = np.empty(a_t.size)
    for t, a in enumerate(a_t):
        if t < freeze_at:
            beta += a * (beta_target - beta)
            beta = min(max(beta, cfg.beta_min), cfg.beta_max)
        else:
            beta = float(beta_target)
        out[t] = beta
    return out


@dataclass(frozen=True)
class AdaptiveTV:
    """Exakter TV-Verlauf der adaptiven (zeitinhomogenen) Kette."""

    tv_to_target: np.ndarray
    """TV(mu_t, pi_{beta_target}), t = 0..n_steps."""
    tv_to_limit: np.ndarray
    """TV(mu_t, pi_{beta_inf}) -- Konvergenz bzgl. des tatsaechlichen Grenzkerns."""
    beta_traj: np.ndarray
    """beta_t je Sweep (identisch zu a_kernel.run_adaptive_mcmc(...).beta_traj)."""
    beta_limit: float
    """beta_inf = letzter Schedule-Wert (bzw. geschlossene Produktformel)."""
    frozen_floor: float
    """||pi_{beta_inf} - pi_{beta_target}||_TV: nicht unterschreitbarer Boden."""


def adaptive_exact_tv(
    cfg: MVPConfig,
    *,
    beta_start: float,
    beta_target: float,
    a_t: np.ndarray,
    x0: int,
    freeze_at: int,
    beta_limit: float | None = None,
) -> AdaptiveTV:
    """mu_{t+1} = mu_t P_{beta_{t+1}} exakt; Kerne je beta gecacht.

    Args:
        freeze_at: Freeze-Sweep wie in a_kernel.advance_chain; a_t.size = kein Freeze.
        beta_limit: Grenzwert beta_inf; Default der letzte Schedule-Wert. Fuer
            einen summierbaren Schedule ist die geschlossene Form
            beta_target + (beta_start - beta_target) * prod(1 - a_t) (ohne Clip).
    """
    L = _check_L(cfg.L)
    if not (cfg.beta_min <= beta_target <= cfg.beta_max):
        raise ValueError("beta_target outside compact Theta (Containment violation)")
    if not (cfg.beta_min <= beta_start <= cfg.beta_max):
        raise ValueError("beta_start outside compact Theta")
    betas = adaptive_beta_schedule(
        cfg, beta_start=beta_start, beta_target=beta_target, a_t=a_t, freeze_at=freeze_at
    )
    b_inf = float(betas[-1]) if beta_limit is None else _check_beta(beta_limit)
    pi_target = stationary_distribution(L, beta_target)
    pi_limit = stationary_distribution(L, b_inf)
    mu = np.zeros(2**L)
    mu[int(x0)] = 1.0
    cache: dict[float, np.ndarray] = {}
    to_target = np.empty(betas.size + 1)
    to_limit = np.empty(betas.size + 1)
    to_target[0] = tv_distance(mu, pi_target)
    to_limit[0] = tv_distance(mu, pi_limit)
    for t, b in enumerate(betas):
        P = cache.get(b)
        if P is None:
            P = sweep_kernel(L, b)
            if len(cache) < 512:
                cache[b] = P
        mu = mu @ P
        to_target[t + 1] = tv_distance(mu, pi_target)
        to_limit[t + 1] = tv_distance(mu, pi_limit)
    return AdaptiveTV(
        tv_to_target=to_target,
        tv_to_limit=to_limit,
        beta_traj=betas,
        beta_limit=b_inf,
        frozen_floor=tv_distance(pi_limit, pi_target),
    )


def relaxation_time_profile(L: int, betas: np.ndarray) -> np.ndarray:
    """t_rel(beta) des Sweep-Kerns fuer ein beta-Gitter (Containment-Profil)."""
    return np.array(
        [
            spectral_summary(sweep_kernel(L, b), stationary_distribution(L, b)).relaxation_time
            for b in np.asarray(betas, dtype=np.float64)
        ]
    )
