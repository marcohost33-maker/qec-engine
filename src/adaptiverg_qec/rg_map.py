"""C-Kernel (minimal): MCRG-/RG-Map + Jacobian + kritische Exponenten [Spec 6-7].

MVP-RG-Map (mvp_instance.py): EXAKTE 1D-Ising-Decimation-Rekursion, b=2:
    R(K) = (1/2) * ln( cosh(2K) ).
Analytische Orakel (geschlossen, fuer Tests):
    Fixpunkte:  K*=0 (trivial stabil), K*=+inf (T=0).
    Ableitung:  R'(K) = tanh(2K).  Bei K*=0: R'(0)=0  (super-stabil).
    Exponent (Spec 6.1): y = log_b |lambda| mit lambda = R'(K*), b=2.

Jacobian-Methoden (Spec 7): Complex-Step (7.1, empfohlen) und zentrale
Differenzen (Vergleichs-Orakel). Complex-Step ist exakt bis Maschinengenauigkeit
fuer holomorphe R -- R(K) ist holomorph (cosh, ln auf passendem Gebiet).

Diese Datei implementiert REAL:
- R als (skalare + vektorwertige) holomorphe Funktion,
- DR via Complex-Step UND zentrale Differenzen (Konsistenz-Orakel),
- Eigenwerte/Exponenten + Hyperbolizitaets-Check (|lambda| != 1, Spec 6.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mvp_instance import RG_SCALE_FACTOR_B


def rg_map_scalar(K: complex | float) -> complex:
    """1D-Ising-Decimation-RG: R(K) = 0.5 * ln(cosh(2K)).

    Holomorph fortgesetzt (akzeptiert komplexe K fuer Complex-Step).
    """
    return 0.5 * np.log(np.cosh(2.0 * K))


def rg_map(g: np.ndarray) -> np.ndarray:
    """Vektorwertige RG-Map auf dem MVP-Kopplungsvektor g.

    MVP: g = [K] (skalar im Vektor verpackt, damit die Jacobian-Maschinerie
    generisch d-dimensional bleibt).
    """
    g = np.asarray(g)
    # Float64 erzwingen (komplex bleibt komplex fuer Complex-Step): mit
    # g.dtype wuerde Integer-Input das Ergebnis still auf int truncaten
    # (rg_map([1]) -> [0] statt [0.6836...]), Audit P1-8.
    out = np.empty_like(g, dtype=np.result_type(g.dtype, np.float64))
    out[0] = rg_map_scalar(g[0])
    return out


def rg_derivative_analytic(K: float) -> float:
    """R'(K) = tanh(2K). Analytisches Orakel fuer den skalaren Fall."""
    return float(np.tanh(2.0 * K))


def jacobian_complex_step(func, g_star: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    """DR(g*) via Complex-Step Differentiation [Spec 7.1].

    M_ij = Im( R_i(g* + i*eps*e_j) ) / eps. Keine Subtraktions-Ausloeschung;
    eps darf winzig sein (1e-20) -> exakt bis Maschinengenauigkeit fuer
    holomorphe func.

    Args:
        func: holomorph fortgesetzte Map R: C^d -> C^d.
        g_star: reeller Auswertungspunkt (Fixpunkt).
        eps: Complex-Step-Weite (sehr klein, kein Trunkierungs-/Rundungs-Tradeoff).
    """
    g_star = np.asarray(g_star, dtype=np.float64)
    d = g_star.size
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    M = np.empty((d, d), dtype=np.float64)
    for j in range(d):
        z = g_star.astype(np.complex128)
        z[j] = z[j] + 1j * eps
        fz = func(z)
        M[:, j] = np.imag(np.asarray(fz, dtype=np.complex128)) / eps
    return M


def jacobian_finite_difference(func, g_star: np.ndarray, *, eta: float = 1e-6) -> np.ndarray:
    """DR(g*) via zentrale Differenzen [Spec 7, Vergleichs-Orakel].

    Relative Epsilon-Skalierung [Spec 7.4]: eps_j = eta * (1 + |g*_j|).
    eta in [1e-7, 1e-4] (Goldilocks-Zone) erwartet.
    """
    g_star = np.asarray(g_star, dtype=np.float64)
    d = g_star.size
    if not (1e-9 <= eta <= 1e-2):
        raise ValueError(f"eta {eta} outside sane FD range [1e-9, 1e-2]")
    M = np.empty((d, d), dtype=np.float64)
    for j in range(d):
        h = eta * (1.0 + abs(g_star[j]))
        gp = g_star.copy()
        gm = g_star.copy()
        gp[j] += h
        gm[j] -= h
        fp = np.asarray(func(gp), dtype=np.float64)
        fm = np.asarray(func(gm), dtype=np.float64)
        M[:, j] = (fp - fm) / (2.0 * h)
    return M


@dataclass(frozen=True)
class ExponentReport:
    """RG-Eigenwerte + kritische Exponenten + Hyperbolizitaet."""

    eigenvalues: np.ndarray
    """Eigenwerte lambda_i von DR(g*)."""
    exponents: np.ndarray
    """y_i = log_b |lambda_i| fuer |lambda_i| > 0; nan fuer lambda_i = 0."""
    hyperbolic: bool
    """True gdw alle |lambda_i| != 1 (Spec 6.2). Marginale Moden sonst gemeldet."""
    marginal_mask: np.ndarray
    """Boolean-Maske der marginalen Richtungen (|lambda_i| ~ 1)."""


def exponents_from_jacobian(
    M: np.ndarray, *, b: int = RG_SCALE_FACTOR_B, marginal_tol: float = 1e-8
) -> ExponentReport:
    """Kritische Exponenten y_i = log_b |lambda_i| + Hyperbolizitaets-Check.

    Spec 6.1: y_i = log_b |lambda_i|.  Spec 6.2: Fixpunkt hyperbolisch gdw
    alle |lambda_i| != 1; marginale Moden (|lambda|=1) werden GEMELDET, nicht
    stillschweigend als Exponent interpretiert.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2D, got shape {M.shape}")
    if b < 2:
        raise ValueError(f"scale factor b must be >= 2, got {b}")
    eig = np.linalg.eigvals(M)
    absl = np.abs(eig)
    marginal = np.abs(absl - 1.0) < marginal_tol
    with np.errstate(divide="ignore"):
        y = np.where(absl > 0.0, np.log(absl) / np.log(b), np.nan)
    return ExponentReport(
        eigenvalues=eig,
        exponents=y,
        hyperbolic=bool(not np.any(marginal)),
        marginal_mask=marginal,
    )
