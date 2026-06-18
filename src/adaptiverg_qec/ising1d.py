"""Unabhaengiges analytisches Orakel: 1D-Ising / Repetition-Code (Ring).

KEINE MCMC hier -- nur geschlossene Transfer-Matrix-Ausdruecke. Diese Datei
dient als unabhaengiges Orakel (Spec/Codie-Disziplin: Korrektheit gegen ein
unabhaengiges Orakel, nicht gegen sich selbst).

Energie (Domaenenwand-Zaehlung, periodisch):
    H(x) = sum_i [x_i != x_{i+1 mod L}],   x in {0,1}^L.

Mit s_i = 1 - 2*x_i in {+1,-1} ist  [x_i != x_{i+1}] = (1 - s_i s_{i+1}) / 2,
also  H = (L - sum_i s_i s_{i+1}) / 2. Die Boltzmann-Verteilung
pi_beta(x) ~ exp(-beta H(x)) ist die 1D-Ising-Kette mit Kopplung K = beta/2.

Transfer-Matrix (Ising, Kopplung K, kein Feld):
    Z_ising = (2 cosh K)^L + (2 sinh K)^L.
"""

from __future__ import annotations

import math


def coupling_from_beta(beta: float) -> float:
    """K = beta/2 (Domaenenwand-Konvention -> Ising-Kopplung)."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return 0.5 * beta


def log_partition(beta: float, L: int) -> float:
    """ln Z der periodischen Kette in der Domaenenwand-Konvention.

    Z_dw = sum_x exp(-beta H(x)). Mit der Ising-Identitaet und K=beta/2:
        Z_dw = exp(-beta L / 2) * [ (2 cosh K)^L + (2 sinh K)^L ].
    """
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    k = coupling_from_beta(beta)
    a = L * math.log(2.0 * math.cosh(k))
    b = L * math.log(2.0 * math.sinh(k)) if k > 0 else -math.inf
    hi = max(a, b)
    log_ising = hi + math.log1p(math.exp(min(a, b) - hi)) if math.isfinite(b) else a
    return -0.5 * beta * L + log_ising


def mean_energy(beta: float, L: int) -> float:
    """<H> = Erwartete Anzahl Domaenenwaende, analytisch.

    Mit t = tanh(K), K = beta/2, ist der NN-Korrelator der periodischen Kette
        <s_i s_{i+1}> = (t^{L-1} + t) / (1 + t^L),
    und <H> = (L/2) * (1 - <s_i s_{i+1}>).
    """
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    k = coupling_from_beta(beta)
    t = math.tanh(k)
    corr = (t ** (L - 1) + t) / (1.0 + t**L)
    return 0.5 * L * (1.0 - corr)


def exact_distribution(beta: float, L: int) -> list[float]:
    """Exakte pi_beta(x) ueber ALLE 2^L Zustaende (nur klein-L, Test-Orakel).

    Rueckgabe: Liste der Wahrscheinlichkeiten, indiziert durch int(x).
    Nur fuer kleine L (<= 24) gedacht (exponentiell).
    """
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if L > 24:
        raise ValueError(f"exact_distribution is exponential; L<=24 only, got {L}")
    weights = [0.0] * (1 << L)
    for state in range(1 << L):
        h = domain_walls(state, L)
        weights[state] = math.exp(-beta * h)
    z = math.fsum(weights)
    return [w / z for w in weights]


def domain_walls(state: int, L: int) -> int:
    """H(x) fuer x als Integer-Bitmaske (periodisch)."""
    walls = 0
    for i in range(L):
        bit_i = (state >> i) & 1
        bit_next = (state >> ((i + 1) % L)) & 1
        walls += bit_i ^ bit_next
    return walls
