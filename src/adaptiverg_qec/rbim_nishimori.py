"""ROADMAP-Inkr.4: das +/-J random-bond Ising-Modell (RBIM) auf der
Nishimori-Linie -- die RBIM-Nishimori <-> MCRG/QEC-Bruecke.

------------------------------------------------------------------------------
WARUM (Architektur-Schlussstein)
------------------------------------------------------------------------------
Das Repo behauptet, MCRG-Maschinerie (kritische Exponenten, RG-Fixpunkt) und
QEC-Threshold seien DASSELBE Objekt -- und gehoeren deshalb in EIN Repo. Diese
Bruecke ist nicht Folklore, sondern ein exaktes Mapping (Dennis, Kitaev, Landahl,
Preskill, J. Math. Phys. 43, 4452 (2002), arXiv:quant-ph/0110143):

    Code-Capacity-Threshold des Toric-/Surface-Codes unter unabhaengigem
    Bit-/Phase-Flip-Rauschen p  ==  Ordnungs-/Unordnungs-Uebergang (Nishimori-
    Multikritischer-Punkt) des 2D +/-J RBIM auf der Nishimori-Linie.

Der publizierte Wert:  p_c ~ 0.1094  (10.94 +/- 0.02 %)
  - Dennis/Kitaev/Landahl/Preskill 2002: p_c ~ 0.1094.
  - Honecker/Picco/Pujol, PRL 87, 047201 (2001): p_c = 0.1094(2).
  - Merz/Chalker, PRB 65, 054425 (2002): p_c = 0.1093(2).
  - Tensor-Network-MC (arXiv:2409.06538, 2024): bestaetigt p_c ~ 0.10919.

Damit ist die Bruecke: misst das RBIM-Tooling p_c ~ 0.109, dann misst die
MCRG-/Statistik-Maschinerie GENAU das Objekt, das der QEC-Threshold ist ->
der "ein-Repo"-Architektur-Entscheid ist empirisch gestuetzt.

------------------------------------------------------------------------------
MODELL + NISHIMORI-LINIE
------------------------------------------------------------------------------
+/-J RBIM auf dem L x L Quadratgitter, periodisch:
    H(s; J) = - sum_<ij> J_ij s_i s_j ,   s_ij in {+1,-1}, J_ij in {+1,-1}.
Quenched Disorder: J_ij = -1 mit Wahrscheinlichkeit p ("antiferromagnetischer"/
frustrierter Bond), J_ij = +1 mit Wahrscheinlichkeit (1-p). Boltzmann-Ziel
pi ~ exp(beta * sum_<ij> J_ij s_i s_j).

NISHIMORI-BEDINGUNG (Nishimori 1981) -- bindet Temperatur an Disorder:
    p = 1 / (1 + exp(2*beta))   <=>   beta = 0.5 * ln((1-p)/p).
Auf dieser Linie ist die interne Energie exakt bekannt und der frustrierte
Sektor exakt loesbar; die FM/PM-Phasengrenze schneidet sie im Nishimori-
Multikritischen-Punkt (MNP) bei p_c. Genau dieser Schnittpunkt ist der
Threshold.

------------------------------------------------------------------------------
SAMPLER: RBIM-generalisierter Wolff-Single-Cluster
------------------------------------------------------------------------------
Wir RECYCELN die Wolff-BFS-Idee (wolff2d.py), generalisieren aber auf
ortsabhaengige Bonds (web-verifiziert, Every-Algorithm / arXiv:2107.08534):

    Ein Bond (i,j) wird in den Cluster aufgenommen mit
        P_add = 1 - exp(-2*beta*J_ij*s_i*s_j)   FALLS  J_ij*s_i*s_j > 0,
        sonst nie (unbefriedigter/frustrierter Bond waechst nie).

Fuer +/-J (|J|=1) heisst das: ueber einen BEFRIEDIGTEN Bond (J_ij*s_i*s_j=+1)
mit P_add = 1 - exp(-2*beta); ueber einen frustrierten (=-1) gar nicht. Das ist
die korrekte Verallgemeinerung (befriedigte Bonds spielen die Rolle der
"aligned" Bonds im homogenen Wolff). Im 2D-RBIM nahe p_c (mehrheitlich FM-Bonds,
FM-nahe Phase) ist Wolff effizient -- der bekannte Spin-Glas-Ineffizienz-Caveat
(jeder Cluster fast das ganze Gitter) trifft hier NICHT, da wir auf/nahe der
FM-Seite der Nishimori-Linie messen.

------------------------------------------------------------------------------
OBSERVABLE + LOKALISIERUNG
------------------------------------------------------------------------------
Disorder-gemittelte Magnetisierung [<|m|>] entlang der Nishimori-Linie als
Funktion von p. FM (p < p_c): [<|m|>] gross; PM (p > p_c): [<|m|>] -> 0 (bis auf
1/sqrt(N)-Finite-Size-Floor). Der Abfall lokalisiert den Uebergang. Zusaetzlich
die Binder-Kumulante U = 1 - [<m^4>]/(3 [<m^2>]^2) als (groessen-robusteres)
Kreuzungs-Diagnostikum.

EHRLICHE SCOPE-GRENZE (AGENTS.md Sec.1): kleines L + endliches Disorder-Sampling
-> GROBE Lokalisierung auf PLAUSIBILITAETS-NIVEAU. KEINE L->inf-FSS-Extrapolation,
KEIN Frontier-Wert. Ziel: zeigen, dass der gemessene Uebergang mit p_c ~ 0.109
KONSISTENT ist (innerhalb einer systematik-begruendeten Toleranz), nicht ihn
hochpraezise zu reproduzieren.

Referenzen (Methode/Orakel, NICHT Dependency):
- H. Nishimori, Prog. Theor. Phys. 66 (1981) 1169 (Nishimori-Linie).
- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, J. Math. Phys. 43 (2002) 4452
  (Topological quantum memory; RBIM<->Threshold-Mapping, p_c~0.1094).
- A. Honecker, M. Picco, P. Pujol, PRL 87 (2001) 047201 (p_c=0.1094(2)).
- F. Merz, J. T. Chalker, PRB 65 (2002) 054425 (p_c=0.1093(2)).
- U. Wolff, PRL 62 (1989) 361 (Single-Cluster); arXiv:2107.08534 (RBIM-Cluster).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "nishimori_beta",
    "nishimori_p",
    "RBIMBonds",
    "sample_bonds",
    "rbim_energy_per_spin",
    "rbim_wolff_cluster_update",
    "rbim_metropolis_sweep",
    "rbim_wolff_sample",
    "DisorderResult",
    "nishimori_scan",
    "locate_transition",
]


# ----------------------------------------------------------------------------
# Nishimori-Linie: p <-> beta
# ----------------------------------------------------------------------------
def nishimori_beta(p: float) -> float:
    """Inverse Temperatur beta auf der Nishimori-Linie fuer Disorder p.

    Nishimori-Bedingung: p = 1/(1+exp(2*beta))  =>  beta = 0.5*ln((1-p)/p).

    Args:
        p: Anteil frustrierter (-J) Bonds, in (0, 0.5). p->0 => beta->inf
            (T->0, rein FM); p=0.5 => beta=0 (T->inf, voll ungeordnet).

    Returns:
        beta = 0.5*ln((1-p)/p) > 0 fuer p < 0.5.
    """
    if not np.isfinite(p) or not (0.0 < p < 0.5):
        raise ValueError(f"p must be in (0, 0.5) on the Nishimori line, got {p}")
    return 0.5 * math.log((1.0 - p) / p)


def nishimori_p(beta: float) -> float:
    """Disorder p auf der Nishimori-Linie fuer inverse Temperatur beta.

    Umkehrung von nishimori_beta: p = 1/(1+exp(2*beta)).

    Args:
        beta: inverse Temperatur > 0.

    Returns:
        p in (0, 0.5).
    """
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and > 0, got {beta}")
    return 1.0 / (1.0 + math.exp(2.0 * beta))


# ----------------------------------------------------------------------------
# Quenched Disorder: +/-J Bonds
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RBIMBonds:
    """Eine quenched Disorder-Realisierung der +/-J Bonds (periodisch).

    jx: (L,L) Bonds in x-Richtung  J zwischen (i,j) und (i,j+1).
    jy: (L,L) Bonds in y-Richtung  J zwischen (i,j) und (i+1,j).
    Beide in {+1,-1}; -1 mit Wahrscheinlichkeit p.
    """

    jx: np.ndarray
    jy: np.ndarray
    p: float
    L: int
    seed: int

    def __post_init__(self) -> None:
        if self.jx.shape != self.jy.shape:
            raise ValueError("jx and jy must have identical shape")
        if self.jx.shape != (self.L, self.L):
            raise ValueError(f"bonds must be (L,L)=({self.L},{self.L}), got {self.jx.shape}")

    @property
    def frustrated_fraction(self) -> float:
        """Tatsaechlicher Anteil -1-Bonds in dieser Realisierung (~ p)."""
        n_neg = int((self.jx < 0).sum() + (self.jy < 0).sum())
        return n_neg / (2 * self.L * self.L)


def sample_bonds(p: float, L: int, *, seed: int) -> RBIMBonds:
    """Ziehe eine quenched +/-J Disorder-Realisierung (J=-1 mit Wkt p).

    Args:
        p: Anteil frustrierter Bonds in [0, 0.5).
        L: Gitterkante >= 4, gerade.
        seed: PCG64-Seed (Reproduzierbarkeit der Realisierung).

    Returns:
        RBIMBonds mit jx, jy in {+1,-1}.
    """
    if not np.isfinite(p) or not (0.0 <= p < 0.5):
        raise ValueError(f"p must be in [0, 0.5), got {p}")
    if L < 4 or L % 2 != 0:
        raise ValueError(f"L must be even and >= 4, got {L}")
    rng = np.random.default_rng(seed)
    jx = np.where(rng.random((L, L)) < p, np.int8(-1), np.int8(1))
    jy = np.where(rng.random((L, L)) < p, np.int8(-1), np.int8(1))
    return RBIMBonds(jx=jx, jy=jy, p=float(p), L=int(L), seed=int(seed))


def rbim_energy_per_spin(s: np.ndarray, bonds: RBIMBonds) -> float:
    """Energie pro Spin E/N = -(1/N) sum_<ij> J_ij s_i s_j (periodisch).

    Bond-Konvention identisch zu energy_per_spin (right + down), aber je Bond
    mit dem zugehoerigen J_ij gewichtet:
        right-Bond (i,j)-(i,j+1) traegt J = jx[i,j]
        down-Bond  (i,j)-(i+1,j) traegt J = jy[i,j]

    Args:
        s: (L,L) in {+1,-1}.
        bonds: passende RBIMBonds (gleiches L).

    Returns:
        skalare E/N.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (bonds.L, bonds.L):
        raise ValueError(f"s shape {s.shape} != bonds L={bonds.L}")
    right = bonds.jx * s * np.roll(s, -1, axis=1)
    down = bonds.jy * s * np.roll(s, -1, axis=0)
    bond_sum = float(right.sum() + down.sum())
    n = bonds.L * bonds.L
    return -bond_sum / n


# ----------------------------------------------------------------------------
# RBIM-generalisierter Wolff-Single-Cluster
# ----------------------------------------------------------------------------
def rbim_wolff_cluster_update(
    s: np.ndarray, bonds: RBIMBonds, beta: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """EIN RBIM-Wolff-Single-Cluster-Update (periodisch, vektorisierte BFS).

    Generalisierung von wolff2d.wolff_cluster_update auf orts-abhaengige Bonds:
    ueber einen BEFRIEDIGTEN Bond (J_ij*s_i*s_j > 0) waechst der Cluster mit
        P_add = 1 - exp(-2*beta) (|J|=1),
    ueber frustrierte Bonds nie. Wie im homogenen Fall rejection-free: die
    befriedigten Rand-Bonds kompensieren ΔE exakt.

    Args:
        s: (L,L) in {+1,-1}. Wird nicht mutiert (Kopie zurueck).
        bonds: quenched Disorder.
        beta: inverse Temperatur > 0.
        rng: numpy-Generator.

    Returns:
        (s_new, cluster_size).
    """
    s = np.asarray(s, dtype=np.int8)
    L = bonds.L
    if s.shape != (L, L):
        raise ValueError(f"s shape {s.shape} != bonds L={L}")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and > 0, got {beta}")
    padd = 1.0 - math.exp(-2.0 * beta)  # in (0,1]; saturiert -> 1 bei grossem beta
    s = s.copy()

    si = int(rng.integers(L))
    sj = int(rng.integers(L))
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[si, sj] = True
    frontier = np.zeros((L, L), dtype=bool)
    frontier[si, sj] = True

    # Vorzeichen jedes Bonds als Funktion BEIDER Endpunkte (J_ij*s_i*s_j) wird
    # pro Welle frisch berechnet (s ist hier konstant waehrend des Wachsens, aber
    # in_cluster aendert sich; deshalb pro Welle neu maskiert).
    while frontier.any():
        free = ~in_cluster
        new_members = np.zeros((L, L), dtype=bool)
        # 4 Richtungen. Fuer jede betrachten wir das ZIEL-Gitter (Kandidat-Nachbar)
        # und pruefen: sitzt in einer bestimmten Richtung ein Frontier-Spin, und ist
        # der verbindende Bond BEFRIEDIGT (J*s_target*s_frontier > 0)?
        #
        # KORREKTHEIT (verifiziert gegen exakte L=4-Enumeration, frustriert):
        # `np.roll(frontier, shift, axis)` ist genau dort True, wo der Frontier-Spin
        # um (shift) verschoben liegt, d.h. der Frontier-Spin ist der Nachbar des
        # Ziels in der durch (shift,axis) gegebenen Richtung. Das ZUGEHOERIGE
        # Bond-Array muss am Ziel-Platz das J DIESES Bonds tragen:
        #   right-target  (Frontier links):  Bond = jx[target_row, target_col-1] = roll(jx,1,ax1)
        #   left-target   (Frontier rechts): Bond = jx[target]                    = jx
        #   down-target   (Frontier oben):   Bond = jy[target_row-1, target_col]  = roll(jy,1,ax0)
        #   up-target     (Frontier unten):  Bond = jy[target]                    = jy
        # s_neighbor (Frontier-Spin am Ziel-Platz) = roll(s, shift, axis).
        directions = (
            (1, 1, np.roll(bonds.jx, 1, axis=1)),  # right-target, frontier is left
            (-1, 1, bonds.jx),  # left-target, frontier is right
            (1, 0, np.roll(bonds.jy, 1, axis=0)),  # down-target, frontier is up
            (-1, 0, bonds.jy),  # up-target, frontier is down
        )
        for shift, axis, jbond in directions:
            neigh_is_frontier = np.roll(frontier, shift, axis=axis)
            s_neighbor = np.roll(s, shift, axis=axis)  # Frontier-Spin am Ziel-Platz
            satisfied = (jbond * s * s_neighbor) > 0
            cand = neigh_is_frontier & free & satisfied
            if not cand.any():
                continue
            draw = rng.random(cand.shape) < padd
            accepted = cand & draw
            new_members |= accepted
        in_cluster |= new_members
        frontier = new_members

    cluster_size = int(in_cluster.sum())
    s[in_cluster] = -s[in_cluster]
    return s, cluster_size


def rbim_metropolis_sweep(
    s: np.ndarray, bonds: RBIMBonds, beta: float, rng: np.random.Generator
) -> np.ndarray:
    """EIN vektorisierter Checkerboard-Metropolis-Sweep fuer das RBIM (periodisch).

    Generalisierung des homogenen Checkerboard-Metropolis (ising2d) auf
    orts-abhaengige Bonds: das lokale Feld an (i,j) ist die mit J GEWICHTETE
    Nachbarsumme  h_ij = sum_<nb> J_(ij,nb) * s_nb. Flip s->-s: dlogpi = -2*beta*s*h.
    Akzeptiere mit min(1, exp(dlogpi)). Schwarz/Weiss-Untergitter -> 2 vektorisierte
    Teilschritte.

    ZWECK (ergaenzt Wolff): tief in der FM-Phase saturiert P_add(Wolff) -> 1, der
    Single-Cluster flippt fast das ganze Gitter und kann die EINGEFRORENE
    Domaenenstruktur NICHT mehr umordnen (er flippt nur das globale Vorzeichen).
    Lokale Metropolis-Flips ordnen die Domaenen um -> Ergodizitaet im geordneten
    Sektor. Wir MISCHEN beide (rbim_wolff_sample, sweep_per_record>0).

    Args:
        s: (L,L) in {+1,-1}. Wird NICHT in-place mutiert (intern kopiert);
            das aktualisierte Gitter ist der Rueckgabewert.
        bonds: quenched Disorder.
        beta: inverse Temperatur > 0.
        rng: numpy-Generator.

    Returns:
        aktualisiertes s.
    """
    L = bonds.L
    s = s.copy()
    ii, jj = np.indices((L, L))
    even = (ii + jj) % 2 == 0
    for mask in (even, ~even):
        # gewichtetes lokales Feld: jeder der 4 Bonds mit seinem J.
        # right-bond (i,j)-(i,j+1): J=jx[i,j], Nachbar s[i,j+1]=roll(s,-1,ax1)
        # left-bond  (i,j)-(i,j-1): J=jx[i,j-1]=roll(jx,1,ax1), Nachbar roll(s,1,ax1)
        # down-bond  (i,j)-(i+1,j): J=jy[i,j], Nachbar roll(s,-1,ax0)
        # up-bond    (i,j)-(i-1,j): J=jy[i-1,j]=roll(jy,1,ax0), Nachbar roll(s,1,ax0)
        h = (
            bonds.jx * np.roll(s, -1, axis=1)
            + np.roll(bonds.jx, 1, axis=1) * np.roll(s, 1, axis=1)
            + bonds.jy * np.roll(s, -1, axis=0)
            + np.roll(bonds.jy, 1, axis=0) * np.roll(s, 1, axis=0)
        ).astype(np.float64)
        dlogpi = -2.0 * beta * s * h
        accept = np.exp(np.minimum(dlogpi, 0.0))
        flip = mask & (rng.random((L, L)) < accept)
        s = np.where(flip, -s, s).astype(np.int8)
    return s


@dataclass(frozen=True)
class DisorderResult:
    """Disorder-gemittelte Observablen an einem Nishimori-Punkt (festes p).

    Alle Werte sind ueber n_disorder Realisierungen gemittelt; err = SEM ueber
    die Realisierungen (Disorder-Fluktuation dominiert bei kleinem L).
    """

    p: float
    beta: float
    L: int
    abs_m: float  # [<|m|>]
    abs_m_err: float
    m2: float  # [<m^2>]
    m4: float  # [<m^4>]
    binder: float  # U = 1 - [<m^4>]/(3[<m^2>]^2)
    mean_cluster_frac: float
    n_disorder: int
    n_records: int


def rbim_wolff_sample(
    bonds: RBIMBonds,
    beta: float,
    *,
    n_records: int,
    burn_in: int,
    seed: int,
    n_skip: int = 1,
    sweeps_per_step: int = 1,
    aligned_start: bool = False,
) -> tuple[np.ndarray, float]:
    """Erzeuge eine RBIM-Trajektorie (HYBRID Wolff + Metropolis) fuer EINE
    Disorder-Realisierung.

    `aligned_start=True` startet vom all-+1-Zustand statt zufaellig. Standard-
    Protokoll zur Messung des FM-Ordnungsparameters: vermeidet das Einfrieren in
    metastabilen, system-spannenden Domaenenwaenden tief im geordneten Sektor
    (cold beta), wo Single-Cluster-Wolff (cf->1) + lokale Sweeps eine solche Wand
    nicht mehr ausheilen. Im PM-Sektor (p>p_c) relaxiert der aligned-Start rasch
    zu m~0 -> kein Bias auf die Uebergangs-Lokalisierung.

    Pro "Update-Schritt": 1 Wolff-Cluster-Update (dekorreliert nahe Uebergang)
    + `sweeps_per_step` Metropolis-Sweeps (ordnen die Domaenenstruktur um, noetig
    tief in der FM-Phase, wo Wolff-cf -> 1 saturiert). Burn-in und n_skip zaehlen
    Update-Schritte.

    Args:
        sweeps_per_step: Metropolis-Sweeps je Wolff-Update (>=0). 0 = reines Wolff.

    Returns:
        (configs (n_records,L,L), mean_cluster_fraction).
    """
    if n_records < 1 or burn_in < 0 or n_skip < 1 or sweeps_per_step < 0:
        raise ValueError("n_records>=1, burn_in>=0, n_skip>=1, sweeps_per_step>=0 required")
    L = bonds.L
    rng = np.random.default_rng(seed)
    if aligned_start:
        s = np.ones((L, L), dtype=np.int8)
    else:
        s = np.where(rng.random((L, L)) < 0.5, np.int8(1), np.int8(-1))
    cluster_sizes: list[int] = []

    def _step() -> None:
        nonlocal s
        s2, cs = rbim_wolff_cluster_update(s, bonds, beta, rng)
        s = s2
        cluster_sizes.append(cs)
        for _ in range(sweeps_per_step):
            s = rbim_metropolis_sweep(s, bonds, beta, rng)

    for _ in range(burn_in):
        _step()
    records: list[np.ndarray] = []
    for _ in range(n_records):
        for _ in range(n_skip):
            _step()
        records.append(s.copy())
    configs = np.asarray(records, dtype=np.int8)
    mean_cf = (float(np.mean(cluster_sizes)) / (L * L)) if cluster_sizes else 0.0
    return configs, mean_cf


def nishimori_scan(
    p: float,
    L: int,
    *,
    n_disorder: int,
    n_records: int,
    burn_in: int,
    base_seed: int,
    n_skip: int = 1,
    sweeps_per_step: int = 2,
    aligned_start: bool = True,
) -> DisorderResult:
    """Disorder-Average an einem Nishimori-Punkt p (beta via Nishimori-Linie).

    Pro Disorder-Realisierung: eigene Bonds + eigener Wolff-Lauf; |m|, m^2, m^4
    je Realisierung gemittelt, dann ueber Realisierungen gemittelt.

    Args:
        p: Disorder in (0, 0.5).
        L: Gitterkante.
        n_disorder: Anzahl quenched Realisierungen (Disorder-Average).
        n_records: aufgezeichnete Konfigurationen je Realisierung.
        burn_in: Wolff-Updates Equilibrierung je Realisierung.
        base_seed: Seed-Basis; Realisierung d nutzt base_seed + d (Bonds) und
            base_seed + 10000 + d (MCMC) -> reproduzierbar, entkoppelt.
        n_skip: Wolff-Updates zwischen Records.

    Returns:
        DisorderResult.
    """
    if n_disorder < 1:
        raise ValueError(f"n_disorder must be >= 1, got {n_disorder}")
    beta = nishimori_beta(p)
    per_real_absm: list[float] = []
    per_real_m2: list[float] = []
    per_real_m4: list[float] = []
    cfracs: list[float] = []
    n = L * L
    for d in range(n_disorder):
        bonds = sample_bonds(p, L, seed=base_seed + d)
        configs, cf = rbim_wolff_sample(
            bonds,
            beta,
            n_records=n_records,
            burn_in=burn_in,
            seed=base_seed + 10_000 + d,
            n_skip=n_skip,
            sweeps_per_step=sweeps_per_step,
            aligned_start=aligned_start,
        )
        # m je Konfiguration (vorzeichenbehaftet), dann |m|, m^2, m^4 mitteln.
        m = configs.reshape(configs.shape[0], -1).sum(axis=1) / n  # (n_records,)
        per_real_absm.append(float(np.mean(np.abs(m))))
        per_real_m2.append(float(np.mean(m * m)))
        per_real_m4.append(float(np.mean(m**4)))
        cfracs.append(cf)
    absm_arr = np.asarray(per_real_absm)
    m2 = float(np.mean(per_real_m2))
    m4 = float(np.mean(per_real_m4))
    binder = 1.0 - m4 / (3.0 * m2 * m2) if m2 > 0 else float("nan")
    sem = float(np.std(absm_arr, ddof=1) / math.sqrt(len(absm_arr))) if len(absm_arr) > 1 else 0.0
    return DisorderResult(
        p=float(p),
        beta=float(beta),
        L=int(L),
        abs_m=float(np.mean(absm_arr)),
        abs_m_err=sem,
        m2=m2,
        m4=m4,
        binder=binder,
        mean_cluster_frac=float(np.mean(cfracs)),
        n_disorder=int(n_disorder),
        n_records=int(n_records),
    )


def locate_transition(results: list[DisorderResult], threshold: float = 0.5) -> float:
    """Lokalisiere den Uebergang p* als steilsten Abfall von [<|m|>] ueber p.

    Heuristik (ehrlich GROB): finde das Intervall mit dem groessten negativen
    Gradienten von [<|m|>] gegen p und gib den Mittelpunkt zurueck. Dies ist
    KEINE FSS-Lokalisierung -- nur eine robuste Schaetzung der Abfallregion bei
    fester (kleiner) Groesse.

    Args:
        results: nach p aufsteigend sortierte DisorderResult-Liste.
        threshold: ungenutzt fuer die Gradienten-Variante (Signatur-stabil).

    Returns:
        p* = Mittelpunkt des Intervalls maximalen |m|-Abfalls.
    """
    if len(results) < 2:
        raise ValueError("need >= 2 scan points to locate a transition")
    ps = np.asarray([r.p for r in results])
    ms = np.asarray([r.abs_m for r in results])
    order = np.argsort(ps)
    ps, ms = ps[order], ms[order]
    # negativer Gradient (Abfall) pro Intervall.
    dm = np.diff(ms)
    dp = np.diff(ps)
    if np.any(dp <= 0):
        # Doppelte p-Punkte -> 0/0-Gradient, argmin auf NaN waere willkuerlich.
        raise ValueError("scan points must have pairwise distinct p values")
    slope = dm / dp
    k = int(np.argmin(slope))  # steilster Abfall
    return float(0.5 * (ps[k] + ps[k + 1]))


# ----------------------------------------------------------------------------
# Validierungs-Report (regenerierbares Artefakt) -- ROADMAP-Inkr.4
# ----------------------------------------------------------------------------
# Publiziertes Orakel: 2D +/-J RBIM Nishimori-Multikritischer-Punkt == Code-
# Capacity-Toric/Surface-Threshold p_c ~ 0.1094 (10.94 +/- 0.02 %).
P_C_NISHIMORI: float = 0.1094
"""Publizierter Nishimori-Punkt / Toric-Code-Threshold (Dennis/Kitaev/Landahl/
Preskill J.Math.Phys.43 (2002); Honecker/Picco/Pujol PRL87 (2001) 0.1094(2);
Merz/Chalker PRB65 (2002) 0.1093(2))."""


def _write_validation_report(
    path: str = "results/inkr4-rbim-nishimori.json",
    *,
    L: int = 8,
    ps: tuple[float, ...] = (0.04, 0.07, 0.09, 0.11, 0.13, 0.16, 0.20),
    n_disorder: int = 24,
    n_records: int = 120,
    burn_in: int = 150,
    base_seed: int = 2026,
    sweeps_per_step: int = 3,
) -> int:
    """Erzeuge den Inkr.4-Validierungs-Report (Nishimori-Scan + p*-Lokalisierung).

    Schreibt results/inkr4-rbim-nishimori.json mit allen [<|m|>](p), Binder U(p),
    dem lokalisierten p* und dem Abstand zum publizierten Orakel p_c~0.1094.
    Default L=8 (schnelle, regenerierbare Aufloesung -- bewusst GROB).
    """
    import json
    from pathlib import Path

    res = [
        nishimori_scan(
            p,
            L=L,
            n_disorder=n_disorder,
            n_records=n_records,
            burn_in=burn_in,
            base_seed=base_seed,
            sweeps_per_step=sweeps_per_step,
            aligned_start=True,
        )
        for p in ps
    ]
    pstar = locate_transition(res)
    payload = {
        "tool": "adaptiverg_qec.rbim_nishimori",
        "method": (
            "ROADMAP-Inkr.4: +/-J RBIM on the Nishimori line p=1/(1+exp(2*beta)); "
            "RBIM-generalized Wolff single-cluster (P_add=1-exp(-2*beta*J_ij*s_i*s_j) "
            "on satisfied bonds) hybridized with weighted checkerboard Metropolis "
            "sweeps; aligned-start FM order-parameter protocol; disorder-averaged "
            "[<|m|>] and Binder cumulant; transition located at steepest |m| drop."
        ),
        "oracle_p_c": P_C_NISHIMORI,
        "oracle_source": (
            "2D +/-J RBIM Nishimori multicritical point == code-capacity toric/surface "
            "threshold p_c~0.1094: Dennis,Kitaev,Landahl,Preskill J.Math.Phys.43 (2002) "
            "arXiv:quant-ph/0110143; Honecker,Picco,Pujol PRL87 (2001) 047201 0.1094(2); "
            "Merz,Chalker PRB65 (2002) 054425 0.1093(2); TN-MC arXiv:2409.06538 (2024) "
            "0.10919 (web-verified this run)."
        ),
        "bridge_claim": (
            "If RBIM tooling (built from the SAME ising2d/wolff2d/MCRG machinery) "
            "locates p* ~ p_c, then the MCRG critical-exponent machinery measures the "
            "SAME object as the QEC code-capacity threshold -> empirical support for "
            "the single-repo (no-split) architecture decision."
        ),
        "L": L,
        "n_disorder": n_disorder,
        "n_records": n_records,
        "burn_in": burn_in,
        "sweeps_per_step": sweeps_per_step,
        "base_seed": base_seed,
        "rows": [
            {
                "p": r.p,
                "beta": r.beta,
                "abs_m": r.abs_m,
                "abs_m_err": r.abs_m_err,
                "m2": r.m2,
                "m4": r.m4,
                "binder": r.binder,
                "mean_cluster_frac": r.mean_cluster_frac,
            }
            for r in res
        ],
        "p_star_located": pstar,
        "p_star_abs_err_vs_oracle": abs(pstar - P_C_NISHIMORI),
        "scope_note": (
            "HONEST PLAUSIBILITY LEVEL. Small L + finite disorder sampling -> COARSE "
            "localization. NO L->inf finite-size-scaling, NO frontier value. The claim "
            "is CONSISTENCY of p* with p_c~0.109 within a systematics-justified band, "
            "not high-precision reproduction. Consistent with the repo's honest "
            "diagnostic-harness positioning (AGENTS.md Sec.1)."
        ),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"inkr4-rbim-nishimori -> {out}")
    for r in res:
        print(
            f"  p={r.p:.2f} beta={r.beta:.3f} |m|={r.abs_m:.3f}+/-{r.abs_m_err:.3f} "
            f"U={r.binder:.3f} cf={r.mean_cluster_frac:.2f}"
        )
    print(
        f"  located p*={pstar:.3f} vs oracle p_c={P_C_NISHIMORI} "
        f"(|err|={abs(pstar - P_C_NISHIMORI):.3f}); HONEST coarse/plausibility."
    )
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Inkr.4 RBIM-Nishimori-Validierungs-Report (regenerierbar)."
    )
    ap.add_argument("--L", type=int, default=8, help="Gittergroesse (default 8)")
    ap.add_argument(
        "--n-disorder", type=int, default=24, help="Disorder-Realisierungen (default 24)"
    )
    ap.add_argument(
        "--out",
        default="results/inkr4-rbim-nishimori.json",
        help="Output-JSON-Pfad",
    )
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()
    sys.exit(
        _write_validation_report(
            args.out,
            L=args.L,
            n_disorder=args.n_disorder,
            base_seed=args.base_seed,
        )
    )
