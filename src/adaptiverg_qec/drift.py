"""Foster-Lyapunov-Drift-Guard (Spec 3.1) — Laufzeit-Pruefung.

Spec 3.1 (Meyn & Tweedie, Thm 15.0.1):
    E[V(z_{t+1}) | z_t = z] <= lambda * V(z) + b * 1_C(z),   0 < lambda < 1.
mit V >= 1 und petite set C = {z : V(z) <= d}, d > b/(1-lambda).

MVP-V (mvp_instance.py):  V(x) = 1 + H(x),  H = #Domaenenwaende.

Der Guard ist KEINE Existenzaussage, sondern eine LAUFZEIT-Pruefung: er
schaetzt den empirischen Ein-Schritt-Drift-Quotienten ausserhalb von C aus
einer Trajektorie und meldet eine Verletzung (lambda_hat >= 1), wie die Spec
("Guard greift bei Verletzung") verlangt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lyapunov_V(H: np.ndarray) -> np.ndarray:
    """V(x) = 1 + H(x). Vektorisiert ueber eine H-Trajektorie."""
    H = np.asarray(H, dtype=np.float64)
    if np.any(H < 0):
        raise ValueError("H (domain-wall count) must be >= 0")
    return 1.0 + H


@dataclass(frozen=True)
class DriftReport:
    """Ergebnis der Drift-Guard-Pruefung."""

    lambda_hat: float
    """Empirischer Drift-Koeffizient ausserhalb des petite set C."""
    b_hat: float
    """Empirische Obergrenze des Drift-Ueberschusses innerhalb C."""
    d: float
    """Schwelle des petite set C = {V <= d}."""
    n_outside: int
    """Anzahl ausgewerteter Uebergaenge ausserhalb C."""
    holds: bool
    """True gdw lambda_hat < 1 (Drift kontrahiert ausserhalb C)."""

    def __str__(self) -> str:  # pragma: no cover - kosmetisch
        return (
            f"DriftReport(lambda_hat={self.lambda_hat:.4f}, b_hat={self.b_hat:.4f}, "
            f"d={self.d:.2f}, n_outside={self.n_outside}, holds={self.holds})"
        )


def estimate_drift(
    H_traj: np.ndarray,
    d: float,
    *,
    require_outside: int = 10,
    min_per_level: int = 5,
) -> DriftReport:
    """Schaetze (lambda, b) der Foster-Lyapunov-Drift aus einer Trajektorie.

    WICHTIG (Korrektheit gegen das richtige Orakel): Die Drift-Bedingung ist
    eine Aussage ueber die BEDINGTE ERWARTUNG E[V(z_{t+1}) | z_t = z], NICHT
    ueber einen einzelnen realisierten Schritt (ein einzelner Flip kann V
    pathwise erhoehen). Daher wird der Conditional-Mean-Drift geschaetzt:

      Drift(z) := E[V(z_{t+1}) | z_t = z]  ungefaehr  Mittel der V_next ueber
      alle Uebergaenge mit demselben Startwert V(z_t)=v.

    Dann ist die Drift-Ungleichung ausserhalb C (v > d):
        Drift_hat(v) <= lambda * v    =>    lambda >= Drift_hat(v) / v.
    lambda_hat = max_{v>d} Drift_hat(v) / v  (Supremum ueber die Level, denn
    die Ungleichung muss fuer jeden Startwert gelten -- aber je Level GEMITTELT,
    nicht pathwise pro Einzelschritt).

    Args:
        H_traj: Folge der Domaenenwand-Zahlen H(z_t), Laenge >= 2.
        d: petite-set-Schwelle (C = {V <= d}).
        require_outside: minimale Anzahl Ausserhalb-Uebergaenge fuer ein Urteil.
        min_per_level: minimale Stichprobe je V-Level, damit das Level zaehlt
            (sonst ist der Conditional-Mean zu verrauscht -> Level verworfen).

    Returns:
        DriftReport. Fail-closed (holds=False, lambda_hat=nan), wenn zu wenig
        Ausserhalb-Evidenz vorliegt, um die bedingte Drift zu schaetzen.
    """
    H_traj = np.asarray(H_traj, dtype=np.float64)
    if H_traj.ndim != 1:
        raise ValueError(f"H_traj must be 1D, got shape {H_traj.shape}")
    if H_traj.size < 2:
        raise ValueError(f"H_traj needs >= 2 points, got {H_traj.size}")
    if not (d >= 1.0):
        raise ValueError(f"d must be >= 1 (since V >= 1), got {d}")

    V = lyapunov_V(H_traj)
    V_t, V_next = V[:-1], V[1:]
    outside = V_t > d
    n_outside = int(np.count_nonzero(outside))

    if n_outside < require_outside:
        # Fail-closed: kein belastbares Urteil ohne genug Ausserhalb-Evidenz.
        return DriftReport(
            lambda_hat=float("nan"),
            b_hat=float("nan"),
            d=float(d),
            n_outside=n_outside,
            holds=False,
        )

    # Conditional-Mean-Drift je diskretem V-Level ausserhalb C.
    lambda_hat = -np.inf
    used_levels = 0
    for v in np.unique(V_t[outside]):
        mask = V_t == v
        if int(np.count_nonzero(mask)) < min_per_level:
            continue
        drift_mean = float(np.mean(V_next[mask]))
        lambda_hat = max(lambda_hat, drift_mean / v)
        used_levels += 1

    if used_levels == 0 or not np.isfinite(lambda_hat):
        return DriftReport(
            lambda_hat=float("nan"),
            b_hat=float("nan"),
            d=float(d),
            n_outside=n_outside,
            holds=False,
        )

    # b ueber den conditional mean INNERHALB C (Term der Drift-Ungleichung).
    inside = ~outside
    if np.any(inside):
        b_levels = []
        for v in np.unique(V_t[inside]):
            mask = V_t == v
            if int(np.count_nonzero(mask)) < min_per_level:
                continue
            drift_mean = float(np.mean(V_next[mask]))
            b_levels.append(drift_mean - lambda_hat * v)
        b_hat = float(max(b_levels)) if b_levels else 0.0
    else:
        b_hat = 0.0

    return DriftReport(
        lambda_hat=float(lambda_hat),
        b_hat=b_hat,
        d=float(d),
        n_outside=n_outside,
        holds=lambda_hat < 1.0,
    )
