"""MVP-Instanz: konkrete, dokumentierte Wahl von (Code, V, g).

=============================================================================
WICHTIG: Dies ist eine bewusst MINIMALE MVP-WAHL, NICHT die volle Spec.
Die Spec v1.0 laesst V (Lyapunov-Funktion), g (Kopplungsvektor) und die
Kalibrierung explizit fuer "v1.3" offen. Diese Datei fixiert den einfachsten
sinnvollen Fall, damit Phase 1 lauffaehig + falsifizierbar wird.
=============================================================================

Gewaehlter Code (X)
-------------------
**1D-Repetition-Code auf einem Ring** mit L Bits.
- Zustandsraum X = {0,1}^L  (Bit-Flip-Fehler auf den L Code-Bits / Kanten
  des 1D-Gitters). Endlich: |X| = 2^L. Damit ist (Spec 3.2) die Minorization
  auf endlichem Zustandsraum mit positiver Uebergangswahrscheinlichkeit
  automatisch erfuellt; der kritische Punkt ist die Drift-Bedingung.
- Der Repetition-Code ist das QEC-Lehrbuch-Minimalbeispiel und in seiner
  Statistik EXAKT die 1D-Ising-Kette (Defekt = Domaenenwand). Das gibt uns
  ein unabhaengiges analytisches Orakel (Transfer-Matrix), s. ising1d.py.

Augmentierter Zustand z = (x, theta)  [Spec 2.1]
------------------------------------------------
- x in {0,1}^L : Fehler-Konfiguration.
- theta in Theta : adaptiver Parameter. MVP: theta = beta (inverse Temperatur
  des Boltzmann-Ziels exp(-beta * H(x))). Theta = [beta_min, beta_max] KOMPAKT
  (Spec 4.2: Kompaktheit => Containment auf endlichem X). Der kritische Punkt
  beta_c = inf (1D-Ising hat T_c = 0) liegt NICHT in Theta -> kein AdapFail.

Hamiltonian / Energie  H(x)
---------------------------
H(x) = Anzahl der Domaenenwaende (benachbarte ungleiche Bits, periodisch)
     = sum_i [x_i != x_{i+1 mod L}].
Das ist die Syndrom-Gewichtung des Repetition-Codes: jeder Term ist ein
verletzter Stabilizer-Check. Ziel-Verteilung: pi_beta(x) = exp(-beta H(x)) / Z.

Lyapunov-Funktion V  [Spec 3.1 — EXPLIZIT hingeschrieben, MVP-Wahl]
-------------------------------------------------------------------
V(x) = 1 + H(x).
- V >= 1 fuer alle x (Spec verlangt V >= 1 fuer geom. Ergodizitaet). OK.
- V ist beschraenkt: 1 <= V <= 1 + L (endlicher Zustandsraum). Damit ist die
  Drift TRIVIAL erfuellbar, ABER wir erzwingen + pruefen sie zur Laufzeit als
  echten Guard (drift.py), denn die Spec verlangt einen LAUFZEIT-Guard, der
  bei Verletzung greift -- nicht nur eine Existenzaussage.
- Petite Set: C = {x : V(x) <= d}. Mit d > b/(1-lambda) (Spec 3.1).

g (RG-Kopplungsvektor)  [Spec 6.1 — MVP-Wahl]
---------------------------------------------
MVP: skalare Kopplung g = K (Ising-Kopplung, K = beta * J, J = 1).
Die RG-Map ist die EXAKTE 1D-Ising-Decimation-Rekursion (Skalenfaktor b=2):
    K' = R(K) = (1/2) * ln( cosh(2K) ).
Bekannte Fixpunkte (analytisches Orakel, s. rg_map.py):
    K* = 0       (Hochtemperatur, trivial-stabil),
    K* = +inf    (T=0-Fixpunkt, kritisch fuer 1D).
Diese Map ist KEIN Spielzeug: sie ist die lehrbuchexakte Real-Space-RG des
1D-Ising/Repetition-Codes (Decimation jedes zweiten Spins).

Validitaetsgrenzen dieser MVP-Instanz
-------------------------------------
- 1D-Ising hat T_c = 0 -> KEIN endlicher Threshold und KEIN nicht-trivialer
  kritischer Exponent im 2D-Sinne. Die MVP demonstriert die MASCHINERIE
  (Drift-Guard, Jacobian-Konsistenz, Reproduzierbarkeit, RG-Fixpunkt-Iteration)
  gegen analytische Orakel -- sie liefert KEINEN physikalischen Frontier-Wert.
- Phase-2-Akzeptanz der Spec (2D-Ising y_t=1, y_h=15/8) ist NICHT Ziel dieses
  MVP und wird NICHT behauptet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MVPConfig:
    """Eingefrorene MVP-Parameter (reproduzierbar, dokumentiert)."""

    L: int = 16
    """Ring-Laenge (Anzahl Code-Bits). Endlicher Zustandsraum 2^L."""

    beta_min: float = 0.1
    """Untere Grenze des kompakten Theta = [beta_min, beta_max]."""

    beta_max: float = 2.0
    """Obere Grenze; haelt den kritischen Punkt beta_c=inf ausserhalb Theta."""

    def __post_init__(self) -> None:
        if self.L < 2:
            raise ValueError(f"L must be >= 2 (ring needs >=2 sites), got {self.L}")
        if not (self.beta_min > 0.0):
            raise ValueError(f"beta_min must be > 0, got {self.beta_min}")
        if not (self.beta_max > self.beta_min):
            raise ValueError(f"beta_max ({self.beta_max}) must exceed beta_min ({self.beta_min})")


# Modul-Konstanten der MVP-Instanz (Single Source of Truth fuer Doku/Tests).
RG_SCALE_FACTOR_B: int = 2
"""Skalenfaktor b der Decimation-RG (jeder 2. Spin), Spec 6.1."""
