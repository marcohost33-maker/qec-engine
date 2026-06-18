"""Phase-4: Wolff-Single-Cluster-Algorithmus fuer das 2D-Ising-Modell.

KONTEXT (Phase-4). Phase-3b nutzte den vektorisierten Checkerboard-Metropolis
(ising2d.py). Single-Spin-Metropolis leidet nahe T_c unter KRITISCHEM
SLOWING-DOWN: die integrierte Autokorrelationszeit waechst wie tau_int ~ L^z mit
dynamischem Exponenten z ~ 2.1 (2D-Ising). Bei kleinem L + 1 RG-Stufe -> y_t nur
GROB (Phase-3b: y_t = 0.965).

Der Wolff-Algorithmus (Wolff, PRL 62, 361, 1989) baut nahe T_c GROSSE Cluster
korrelierter Spins und flippt sie in EINEM Schritt. Der dynamische Exponent
faellt drastisch auf z ~ 0.25 (2D) -- die Autokorrelation wird nahe T_c
DRAMATISCH kuerzer als bei Single-Spin. Das ist der Zweck (web-verifiziert in
diesem Lauf: P_add=1-exp(-2*beta*J), z~0.25; Wolff/Swendsen-Wang-Literatur).

------------------------------------------------------------------------------
ALGORITHMUS (Wolff Single-Cluster)
------------------------------------------------------------------------------
1. Waehle einen zufaelligen Seed-Spin.
2. Fuege gleich-orientierte Nachbarn mit Wahrscheinlichkeit
       P_add = 1 - exp(-2*K)   (K = J/T = beta*J, J=1)
   zum Cluster hinzu (nur Bonds zwischen GLEICH orientierten Spins koennen
   wachsen; antiparallele Bonds nie).
3. Wiederhole rekursiv/iterativ (Stack/BFS) fuer jeden neu hinzugefuegten Spin.
4. Flippe den GANZEN Cluster.

Ergodizitaet + detailliertes Gleichgewicht: das Verhaeltnis der
Cluster-Konstruktions-Wahrscheinlichkeit zur Boltzmann-Gewichtsaenderung ist
so konstruiert, dass die Akzeptanz IMMER 1 ist (rejection-free). Begruendung:
die "broken bonds" am Cluster-Rand kompensieren exakt den Energieunterschied.

------------------------------------------------------------------------------
VEKTORISIERUNG (numpy-Stack-BFS)
------------------------------------------------------------------------------
Reines Python-Spin-fuer-Spin waere langsam. Wir wachsen den Cluster in WELLEN
(BFS-Frontier): pro Welle werden alle aktuellen Frontier-Spins gleichzeitig
betrachtet, ihre 4 Nachbarn vektorisiert geprueft (gleiche Orientierung + nicht
im Cluster + Bernoulli(P_add)), neue Mitglieder bilden die naechste Frontier.
Das ist EXAKT der Wolff-Cluster (jeder qualifizierte Bond wird einmal mit P_add
getestet), nur datenparallel pro Welle ausgewertet.

EHRLICHE SCOPE-GRENZE (AGENTS.md Sec.1): Wolff dekorreliert nahe T_c, aber
finite-Groessen-Systematik (kleines L + endliche RG-Stufen) bleibt. Phase-4
verbessert die STATISTIK (kleinere tau_int -> mehr effektiv-unabhaengige
Samples), nicht die thermodynamische Limes-Extrapolation L->inf.

Referenzen (Methode, NICHT Dependency):
- U. Wolff, Phys. Rev. Lett. 62 (1989) 361 (Single-Cluster-Algorithmus).
- R. H. Swendsen, J.-S. Wang, Phys. Rev. Lett. 58 (1987) 86 (Multi-Cluster).
- L. Onsager, Phys. Rev. 65 (1944) 117 (exakte 2D-Loesung, T_c).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ising2d import KC_2D, TC_2D

__all__ = [
    "KC_2D",
    "TC_2D",
    "p_add",
    "wolff_cluster_update",
    "wolff_sample",
    "WolffChain",
    "mean_cluster_fraction",
]


def p_add(K: float) -> float:
    """Wolff-Bond-Wahrscheinlichkeit P_add = 1 - exp(-2*K) (K = beta*J, J=1).

    Args:
        K: Kopplung J/T > 0.

    Returns:
        P_add in (0, 1]. Bei grossem K (Tieftemperatur, T -> 0) saettigt
        exp(-2K) -> 0 und P_add -> 1.0 (Floating-Point exakt 1.0 sobald
        2K > ~38, da exp(-38) < float-eps). P_add == 1.0 ist PHYSIKALISCH
        gueltig: jeder gleich-orientierte Nachbar wird mit Wahrscheinlichkeit 1
        zum Cluster hinzugefuegt (der Cluster fuellt die ganze aligned Domaene).
    """
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be finite and > 0 (ferromagnetic), got {K}")
    return 1.0 - math.exp(-2.0 * K)


def wolff_cluster_update(
    s: np.ndarray, padd: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """EIN Wolff-Single-Cluster-Update auf dem L x L Gitter (periodisch).

    BFS-in-Wellen, vektorisiert: waehle Seed, wachse den Cluster wellenweise
    (alle Frontier-Nachbarn gleichzeitig per numpy), flippe den ganzen Cluster.

    Args:
        s: (L,L) in {+1,-1}. WIRD NICHT mutiert (Kopie zurueck).
        padd: Bond-Wahrscheinlichkeit (= p_add(K)).
        rng: numpy-Generator (Reproduzierbarkeit).

    Returns:
        (s_new, cluster_size) -- geflipptes Gitter + Anzahl Spins im Cluster.
    """
    s = np.asarray(s, dtype=np.int8)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"s must be square (L,L), got shape {s.shape}")
    # padd == 1.0 ist physikalisch gueltig (T -> 0: alle aligned Nachbarn sicher
    # hinzugefuegt). Bei grossem K saettigt p_add() durch Floating-Point auf exakt
    # 1.0 -- ein Tieftemperatur-Scan darf daran NICHT brechen. Der prob-1-Pfad ist
    # korrekt: rng.random() liefert [0,1), also ist "< 1.0" stets True -> jeder
    # aligned Bond wird akzeptiert. Nur padd > 1.0 / <= 0.0 ist invalide.
    if not (0.0 < padd <= 1.0):
        raise ValueError(f"padd must be in (0,1], got {padd}")
    L = s.shape[0]
    s = s.copy()

    # Seed-Spin.
    si = int(rng.integers(L))
    sj = int(rng.integers(L))
    seed_spin = s[si, sj]

    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[si, sj] = True
    # Frontier: Spins, deren Nachbarn noch zu pruefen sind.
    frontier = np.zeros((L, L), dtype=bool)
    frontier[si, sj] = True

    while frontier.any():
        # Aligned-Maske: Spins gleicher Orientierung wie der Seed, noch nicht im Cluster.
        aligned_free = (s == seed_spin) & ~in_cluster
        # Kandidaten = Nachbarn der Frontier, die aligned + frei sind.
        # Pro Richtung: ein Frontier-Spin "drueckt" auf seinen Nachbarn.
        new_members = np.zeros((L, L), dtype=bool)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            # Nachbar in Richtung (shift,axis) eines Frontier-Spins = roll der Frontier.
            neigh_is_frontier = np.roll(frontier, shift, axis=axis)
            cand = neigh_is_frontier & aligned_free
            if not cand.any():
                continue
            # Bernoulli(P_add) je Kandidat-Bond.
            draw = rng.random(cand.shape) < padd
            accepted = cand & draw
            new_members |= accepted
        in_cluster |= new_members
        frontier = new_members

    cluster_size = int(in_cluster.sum())
    s[in_cluster] = -s[in_cluster]
    return s, cluster_size


@dataclass(frozen=True)
class WolffChain:
    """Resultat eines Wolff-Sampling-Laufs (Konfigurations-Trajektorie).

    configs: (n_records, L, L) in {+1,-1}, je aufgezeichnet nach n_skip Updates.
    """

    configs: np.ndarray
    K: float
    L: int
    seed: int
    mean_cluster_size: float
    """Mittlere Cluster-Groesse ueber alle Updates (Diagnostik / z-Indikator)."""
    n_updates: int
    """Gesamtzahl Cluster-Updates (inkl. burn-in)."""

    def __post_init__(self) -> None:
        if self.configs.ndim != 3:
            raise ValueError(f"configs must be 3D (n,L,L), got ndim={self.configs.ndim}")


def wolff_sample(
    K: float,
    L: int,
    *,
    n_records: int,
    burn_in: int,
    seed: int,
    n_skip: int = 1,
) -> WolffChain:
    """Erzeuge eine 2D-Ising-Trajektorie via Wolff-Single-Cluster-Updates.

    Args:
        K: Kopplung J/T (J=1), > 0.
        L: Gitterkante (>=4, gerade -> b=2-Blocking sauber).
        n_records: Anzahl aufgezeichneter Konfigurationen NACH burn-in.
        burn_in: Anzahl Cluster-Updates Equilibrierung (verworfen).
        seed: PCG64-Seed (Reproduzierbarkeit).
        n_skip: zeichne nach je n_skip Cluster-Updates auf (Ausduennung; ein
            Cluster-Update dekorreliert nahe T_c bereits stark, daher n_skip
            klein ausreichend).

    Returns:
        WolffChain mit configs (n_records, L, L) in {+1,-1}.
    """
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be finite and > 0, got {K}")
    if L < 4:
        raise ValueError(f"L must be >= 4, got {L}")
    if L % 2 != 0:
        raise ValueError(f"L must be even for b=2 majority blocking, got {L}")
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if n_skip < 1:
        raise ValueError(f"n_skip must be >= 1, got {n_skip}")

    padd = p_add(K)
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((L, L)) < 0.5, np.int8(1), np.int8(-1))

    cluster_sizes: list[int] = []
    for _ in range(burn_in):
        s, cs = wolff_cluster_update(s, padd, rng)
        cluster_sizes.append(cs)

    records: list[np.ndarray] = []
    for _ in range(n_records):
        for _ in range(n_skip):
            s, cs = wolff_cluster_update(s, padd, rng)
            cluster_sizes.append(cs)
        records.append(s.copy())

    configs = np.asarray(records, dtype=np.int8)
    mean_cs = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
    return WolffChain(
        configs=configs,
        K=float(K),
        L=int(L),
        seed=int(seed),
        mean_cluster_size=mean_cs,
        n_updates=burn_in + n_records * n_skip,
    )


def mean_cluster_fraction(chain: WolffChain) -> float:
    """Mittlere Cluster-Groesse als Bruchteil des Gitters (Plausibilitaets-Diagnostik).

    Nahe T_c sollte der Bruchteil eine substanzielle O(0.1..0.7)-Groesse sein
    (kritische Cluster); weit darunter (K klein) winzig, weit darueber (K gross)
    nahe 1. Args: chain. Returns: mean_cluster_size / (L*L).
    """
    return chain.mean_cluster_size / (chain.L * chain.L)
