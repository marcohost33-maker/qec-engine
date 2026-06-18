"""Phase-3b: vektorisierter 2D-Ising-Sampler + Majority-Rule-Blocking.

KONTEXT (Phase-3b). Phase-2/3a war SKALAR (1D-Ising, T_c=0 -> kein nicht-trivialer
Fixpunkt -> nur die Schaetz-Maschinerie validierbar, KEIN echter Exponent). Das
2D-Ising-Modell auf dem Quadratgitter hat dagegen einen NICHT-TRIVIALEN Fixpunkt
bei  T_c = 2 / ln(1+sqrt(2)) ~ 2.2692 J  (Onsager 1944), also echte messbare
RG-Exponenten -- die Voraussetzung fuer die Multi-Operator-Swendsen-MATRIX
(mcrg_matrix.py).

------------------------------------------------------------------------------
KONVENTION
------------------------------------------------------------------------------
Spin s_ij in {+1,-1} auf dem L x L Quadratgitter mit PERIODISCHEN Randbedingungen.
Hamiltonian (ferromagnetisch, kein Feld):
    H(s) = - J * sum_<ij> s_i s_j      (Summe ueber NN-Bonds).
Boltzmann-Ziel pi ~ exp(-H/T) = exp(K * sum_<ij> s_i s_j) mit der dimensionslosen
Kopplung  K = J/T. Wir setzen J=1, parametrisieren also direkt ueber K (bzw. T=1/K).
T_c entspricht  K_c = ln(1+sqrt(2))/2 ~ 0.4407.

------------------------------------------------------------------------------
SAMPLER: Checkerboard-(Schachbrett-)Metropolis -- VEKTORISIERT
------------------------------------------------------------------------------
Single-Spin-Metropolis, aber OHNE Python-Doppelschleife ueber Spins: das Gitter
wird in zwei interpenetrierende Untergitter (schwarz/weiss wie ein Schachbrett)
zerlegt. Innerhalb eines Untergitters sind alle Spins UNABHAENGIG (ihre Nachbarn
liegen komplett im anderen Untergitter), also kann ein ganzes Untergitter in EINEM
vektorisierten numpy-Schritt aktualisiert werden. Ein "Sweep" = beide Untergitter
einmal. Das ist exakte Single-Spin-Metropolis-Dynamik (detailed balance pro Spin),
nur datenparallel ausgewertet.

EHRLICHE SCOPE-GRENZE (AGENTS.md Sec.1): Single-Spin-Metropolis nahe T_c leidet
unter KRITISCHEM SLOWING-DOWN (tau_int ~ L^z, z~2.17). Bei kleinem L + 1 RG-Stufe
ist die Exponenten-Praezision daher nur GROB; ein Cluster-Algorithmus (Wolff/
Swendsen-Wang) + mehrere RG-Iterationen waeren fuer Hochpraezision noetig (= Phase-4).
Hier wird die MATRIX-MASCHINERIE auf Plausibilitaets-Niveau validiert, KEIN
Frontier-Wert. Das ist bewusst und ehrlich.

------------------------------------------------------------------------------
BLOCKING: Majority-Rule (Kadanoff), b=2
------------------------------------------------------------------------------
2x2-Bloecke -> ein Block-Spin = Vorzeichen der Block-Summe (Mehrheitsregel).
Bei einem GLEICHSTAND (Summe 0, zwei +1 und zwei -1) gibt es keine Mehrheit; die
Wahl muss UNBIASED sein (sonst bricht die +/- -Symmetrie und verfaelscht den
geraden/ungeraden Sektor). Wir brechen Gleichstaende deterministisch+reproduzierbar
und im Mittel unbiased ueber einen pro-Block- und pro-Konfigurations-Hash (Splitmix-
artig aus Block-Index + globalem Seed): das Vorzeichen ist eine feste Funktion von
(config_index, block_row, block_col, seed), je zur Haelfte +1/-1. Dokumentiert &
getestet (test_ising2d: Tie-Break im Mittel symmetrisch).

Referenzen (Methode, NICHT Dependency):
- L. Onsager, Phys. Rev. 65 (1944) 117 (exakte 2D-Loesung, T_c, Exponenten).
- L. P. Kadanoff, Physics 2 (1966) 263 (Block-Spin / Majority-Rule).
- R. H. Swendsen, Phys. Rev. Lett. 42 (1979) 859 (MCRG-Matrix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KC_2D",
    "TC_2D",
    "checkerboard_metropolis",
    "Ising2DChain",
    "energy_per_spin",
    "magnetization_per_spin",
    "majority_block_b2",
    "exact_energy_per_spin_2x2",
]

# Onsager: K_c = ln(1+sqrt(2))/2 ; T_c = 1/K_c = 2/ln(1+sqrt(2)).
KC_2D: float = 0.5 * math.log(1.0 + math.sqrt(2.0))  # ~ 0.4406868
TC_2D: float = 1.0 / KC_2D  # ~ 2.2691853


@dataclass(frozen=True)
class Ising2DChain:
    """Resultat eines 2D-Ising-Metropolis-Laufs (Konfigurations-Trajektorie).

    configs: (n_records, L, L) in {+1,-1}, je ein aufgezeichneter Sweep nach Burn-in.
    """

    configs: np.ndarray
    K: float
    L: int
    acceptance: float
    seed: int

    def __post_init__(self) -> None:
        if self.configs.ndim != 3:
            raise ValueError(f"configs must be 3D (n,L,L), got ndim={self.configs.ndim}")


def _neighbor_field(s: np.ndarray) -> np.ndarray:
    """Summe der 4 NN-Spins je Gitterplatz (periodisch), vektorisiert."""
    return (
        np.roll(s, 1, axis=0)
        + np.roll(s, -1, axis=0)
        + np.roll(s, 1, axis=1)
        + np.roll(s, -1, axis=1)
    )


def _checkerboard_masks(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Schwarz/Weiss-Untergitter-Masken (Schachbrett): (i+j) gerade/ungerade."""
    ii, jj = np.indices((L, L))
    even = (ii + jj) % 2 == 0
    return even, ~even


def checkerboard_metropolis(
    K: float,
    L: int,
    *,
    n_sweeps: int,
    burn_in: int,
    seed: int,
    record_every: int = 1,
) -> Ising2DChain:
    """Vektorisierter Checkerboard-Metropolis fuer das 2D-Ising-Modell.

    Ein Sweep = beide Untergitter (schwarz, weiss) je einmal vektorisiert
    aktualisiert. Metropolis-Akzeptanz: ein Flip s->-s aendert die lokale Energie
    um dE = +2*K*s*nbsum (in pi-Konvention pi~exp(K*sum bonds), also Flip-Faktor
    exp(-dlogpi) mit dlogpi = -2*K*s*nbsum). Akzeptiere mit min(1, exp(-2*K*s*nbsum)).

    Args:
        K: Kopplung J/T (J=1). Endlich, > 0 (ferromagnetisch).
        L: Gitterkante (>=4, gerade -> b=2-Blocking ergibt L/2 sauber).
        n_sweeps: Anzahl Sweeps NACH burn_in, die fuer Records zaehlen.
        burn_in: Anzahl Sweeps Equilibrierung (verworfen).
        seed: PCG64-Seed (Reproduzierbarkeit).
        record_every: zeichne jeden record_every-ten Sweep auf (Ausduennung gegen
            Autokorrelation -- der Record-Abstand reduziert tau_int der Reihe).

    Returns:
        Ising2DChain mit configs (n_records, L, L) in {+1,-1}.
    """
    if not np.isfinite(K):
        raise ValueError(f"K must be finite, got {K}")
    if K <= 0.0:
        raise ValueError(f"K must be > 0 (ferromagnetic), got {K}")
    if L < 4:
        raise ValueError(f"L must be >= 4, got {L}")
    if L % 2 != 0:
        raise ValueError(f"L must be even for b=2 majority blocking, got {L}")
    if n_sweeps < 1:
        raise ValueError(f"n_sweeps must be >= 1, got {n_sweeps}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}")

    rng = np.random.default_rng(seed)
    s = np.where(rng.random((L, L)) < 0.5, 1.0, -1.0)
    even, odd = _checkerboard_masks(L)

    n_accept = 0
    n_attempt = 0
    records: list[np.ndarray] = []
    total = burn_in + n_sweeps

    for sweep in range(total):
        for mask in (even, odd):
            nb = _neighbor_field(s)
            # dlogpi for a flip = -2*K*s*nb; accept with prob min(1, exp(dlogpi)).
            dlogpi = -2.0 * K * s * nb
            accept_prob = np.exp(np.minimum(dlogpi, 0.0))  # min(1, exp(dlogpi))
            r = rng.random((L, L))
            flip = mask & (r < accept_prob)
            if sweep >= burn_in:
                n_accept += int(flip.sum())
                n_attempt += int(mask.sum())
            s = np.where(flip, -s, s)
        if sweep >= burn_in and ((sweep - burn_in) % record_every == 0):
            records.append(s.copy())

    configs = np.asarray(records, dtype=np.int8)
    acceptance = (n_accept / n_attempt) if n_attempt > 0 else 0.0
    return Ising2DChain(
        configs=configs, K=float(K), L=int(L), acceptance=acceptance, seed=int(seed)
    )


def energy_per_spin(s: np.ndarray) -> np.ndarray:
    """Energie pro Spin E/N = -(1/N) sum_<ij> s_i s_j (J=1), periodisch.

    Args:
        s: (...,L,L) in {+1,-1}.

    Returns:
        E/N je Konfiguration: Shape (...,) (Skalar fuer 2D-Eingabe).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.shape[-1] < 2 or s.shape[-2] < 2:
        raise ValueError(f"need L>=2 in both dims, got {s.shape[-2:]}")
    # Jeder Bond einmal: rechts + unten (periodisch).
    right = s * np.roll(s, -1, axis=-1)
    down = s * np.roll(s, -1, axis=-2)
    bonds = right.sum(axis=(-1, -2)) + down.sum(axis=(-1, -2))
    n = s.shape[-1] * s.shape[-2]
    return -bonds / n


def magnetization_per_spin(s: np.ndarray) -> np.ndarray:
    """Magnetisierung pro Spin m = (1/N) sum_ij s_ij (vorzeichenbehaftet)."""
    s = np.asarray(s, dtype=np.float64)
    n = s.shape[-1] * s.shape[-2]
    return s.sum(axis=(-1, -2)) / n


def exact_energy_per_spin_2x2(K: float) -> float:
    """Exakte E/N der 2x2-periodischen 2D-Ising-Plaquette (Lehrbuch-Orakel).

    Das 2x2-Gitter mit periodischen Randbedingungen hat DOPPELTE Bonds (jeder der
    4 Spins ist mit jedem Nachbar zweimal verbunden via wrap). Statt eine
    Konvention zu erraten, enumerieren wir die 2^4=16 Zustaende mit GENAU der
    energy_per_spin-Bond-Konvention (right+down, periodisch) -> exaktes,
    selbstkonsistentes Orakel fuer den Sampler-Test.

    Args:
        K: Kopplung.

    Returns:
        <E/N> exakt fuer L=2.
    """
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be finite and > 0, got {K}")
    states = np.arange(16, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(4)[None, :]) & 1).astype(np.int8)
    s = (1 - 2 * bits).reshape(16, 2, 2).astype(np.float64)
    e = energy_per_spin(s)  # (16,)  E/N per state
    # Boltzmann-Gewicht pi ~ exp(K * sum bonds) = exp(-K*N*e), N=4.
    w = np.exp(-K * 4.0 * e)
    w = w / w.sum()
    return float((w * e).sum())


def majority_block_b2(s: np.ndarray, *, config_index: int = 0, seed: int = 0) -> np.ndarray:
    """Majority-Rule-Blocking b=2: L x L -> L/2 x L/2 Block-Spins.

    Jeder 2x2-Block -> Vorzeichen der 4er-Summe. Gleichstand (Summe 0) wird
    UNBIASED + reproduzierbar gebrochen: ein deterministischer Splitmix-Hash aus
    (config_index, block_row, block_col, seed) liefert je zur Haelfte +1/-1
    (im Mittel symmetrisch -> +/- -Symmetrie bleibt erhalten).

    Args:
        s: (L,L) in {+1,-1} (eine Konfiguration). L gerade.
        config_index: laufender Index der Konfiguration (entkoppelt Tie-Breaks
            ueber die Trajektorie -> kein systematischer Bias).
        seed: globaler Seed fuer die Tie-Break-Hash-Funktion.

    Returns:
        Block-Spins (L/2, L/2) in {+1,-1}.
    """
    s = np.asarray(s, dtype=np.int64)
    if s.ndim != 2:
        raise ValueError(f"s must be 2D (L,L), got ndim={s.ndim}")
    L = s.shape[0]
    if s.shape[1] != L:
        raise ValueError(f"s must be square, got {s.shape}")
    if L < 2 or L % 2 != 0:
        raise ValueError(f"L must be even and >= 2, got {L}")
    lb = L // 2
    block_sum = s.reshape(lb, 2, lb, 2).sum(axis=(1, 3))  # (lb, lb), in {-4,-2,0,2,4}

    # Unbiased deterministic tie-break for block_sum == 0.
    # Splitmix64-artiger Hash; das niederwertige Bit entscheidet +/- (50/50).
    # uint64-Arithmetik ist BEWUSST modular (2^64 wraparound) -- numpy warnt sonst
    # vor "overflow"; wir unterdruecken die Warnung lokal (kein Bug, gewollte Mod-Arith).
    br, bc = np.indices((lb, lb))
    with np.errstate(over="ignore"):
        h = (
            (np.uint64(config_index) * np.uint64(0x9E3779B97F4A7C15))
            ^ (br.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9))
            ^ (bc.astype(np.uint64) * np.uint64(0x94D049BB133111EB))
            ^ np.uint64(seed)
        )
        h = h ^ (h >> np.uint64(30))
        h = h * np.uint64(0xBF58476D1CE4E5B9)
        h = h ^ (h >> np.uint64(27))
    tie = np.where((h & np.uint64(1)) == np.uint64(1), 1, -1)

    out = np.sign(block_sum).astype(np.int64)
    out = np.where(block_sum == 0, tie, out)
    return out.astype(np.int8)
