"""Phase-3b: Multi-Operator-Swendsen-MCRG-MATRIX (2D-Ising, gerader Sektor).

Verallgemeinerung des skalaren Phase-3a-Schaetzers (mcrg.py, EINE Kopplung) auf
MEHRERE Kopplungen / Operatoren. Das ist der eigentliche Swendsen-MCRG (PRL 42,
859, 1979): die linearisierte RG-Transformation ist eine MATRIX  T_ab = dK'_a/dK_b,
deren Eigenwerte die kritischen Exponenten liefern.

------------------------------------------------------------------------------
OPERATORBASIS (gerader / thermischer Sektor)
------------------------------------------------------------------------------
Auf dem L x L Spin-Gitter s_ij in {+1,-1} (periodisch) definieren wir gerade
(spin-flip-symmetrische) Operatoren:
    S_1 = sum_<ij>      s_i s_j        (NN-Bonds, rechts+unten)
    S_2 = sum_<<ij>>    s_i s_j        (NNN-Bonds, beide Diagonalen)
    S_3 = sum_plaq      s_a s_b s_c s_d (4-Spin-Plaquette pro Elementarquadrat)
Alle drei sind invariant unter s -> -s (gerader Sektor) -> sie koppeln NICHT an
die Magnetisierung (ungerader Sektor) und messen den THERMISCHEN Exponenten y_t.

------------------------------------------------------------------------------
SWENDSEN-MATRIX (linearisierte RG)
------------------------------------------------------------------------------
Block-Spins s' = Majority-Rule(s, b=2) (ising2d.majority_block_b2). Auf dem
geblockten L/2 x L/2 Gitter dieselben Operatoren S'_a.

Kettenregel der thermischen Ableitungen (Swendsen 1979):
    d<S'_a>/dK_b = sum_c (d<S'_a>/dK'_c) (dK'_c/dK_b)
mit den connected correlations
    A_ab := d<S'_a>/dK_b  = <S'_a S_b>_c   (Block-Operator x ORIGINAL-Operator)
    B_ac := d<S'_a>/dK'_c = <S'_a S'_c>_c  (Block-Operator x BLOCK-Operator).
Loesen nach der RG-Matrix:
    T = A . B^{-1}          (T_ab = dK'_a/dK_b),
numerisch als LINEARES GLEICHUNGSSYSTEM  T B = A  =>  T^T = solve(B^T, A^T)
(np.linalg.solve, KEINE explizite Inverse -- besser konditioniert).

------------------------------------------------------------------------------
EXPONENTEN
------------------------------------------------------------------------------
Eigenwerte lambda_i von T; kritische Exponenten  y_i = ln|lambda_i| / ln b, b=2.
Der GROESSTE (relevante) gerade Eigenwert -> thermischer Exponent y_t.

ORAKEL (2D-Ising EXAKT, Onsager; web-verifiziert in diesem Lauf):
    y_t = 1/nu = 1   (nu = 1).      [y_h = 2 - eta/2 = 15/8 = 1.875, ungerader
                                     Sektor -> optionaler Stretch, hier NICHT
                                     das Kern-Deliverable.]

------------------------------------------------------------------------------
EHRLICHE SCOPE-GRENZE (AGENTS.md Sec.1, KEIN Ueber-Claim)
------------------------------------------------------------------------------
- Single-Spin-Metropolis nahe T_c hat kritisches Slowing-Down; kleines L + 1
  RG-Stufe -> y_t nur GROB (typ. y_t ~ 1 +/- O(0.1..0.3)). Cluster-Algorithmus +
  mehrere RG-Iterationen (= Phase-4) waeren fuer Praezision noetig.
- Validiert wird die MATRIX-MASCHINERIE (>=2 Operatoren, A.B^{-1}, Eigenwert-
  Exponenten) gegen das exakte Orakel auf PLAUSIBILITAETS-Niveau. Kein Frontier-Wert.
- Fehlerbalken via Block-Jackknife ueber die Matrix-Pipeline (autocorr.py).

Referenzen (Methode, NICHT Dependency):
- R. H. Swendsen, Phys. Rev. Lett. 42 (1979) 859.
- L. Onsager, Phys. Rev. 65 (1944) 117.
- L. P. Kadanoff, Physics 2 (1966) 263.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import autocorr, ising2d

__all__ = [
    "even_operators",
    "block_even_operators",
    "OperatorTimeseries",
    "operator_timeseries",
    "swendsen_matrix",
    "exponents_from_T",
    "SwendsenMatrixEstimate",
    "estimate_y_t",
    "validate_y_t_2d",
]


# --------------------------------------------------------------------------
# Operatoren (gerader Sektor)
# --------------------------------------------------------------------------
def even_operators(s: np.ndarray) -> np.ndarray:
    """Gerade Operatoren (S_1 NN, S_2 NNN, S_3 Plaquette) je Konfiguration.

    Args:
        s: (n, L, L) in {+1,-1} (oder (L,L)).

    Returns:
        (n, 3) Operatorwerte [S_1, S_2, S_3] je Konfiguration (oder (3,)).
    """
    s = np.asarray(s, dtype=np.float64)
    squeeze = s.ndim == 2
    if squeeze:
        s = s[None, ...]
    if s.ndim != 3 or s.shape[-1] < 2 or s.shape[-2] < 2:
        raise ValueError(f"s must be (n,L,L) with L>=2, got shape {s.shape}")
    # S_1: NN-Bonds (rechts + unten), periodisch.
    s1 = (s * np.roll(s, -1, axis=-1)).sum(axis=(-1, -2)) + (s * np.roll(s, -1, axis=-2)).sum(
        axis=(-1, -2)
    )
    # S_2: NNN-Bonds (beide Diagonalen): (i,j)-(i+1,j+1) und (i,j)-(i+1,j-1).
    diag1 = s * np.roll(np.roll(s, -1, axis=-1), -1, axis=-2)
    diag2 = s * np.roll(np.roll(s, 1, axis=-1), -1, axis=-2)
    s2 = diag1.sum(axis=(-1, -2)) + diag2.sum(axis=(-1, -2))
    # S_3: 4-Spin-Plaquette pro Elementarquadrat (i,j),(i+1,j),(i,j+1),(i+1,j+1).
    plaq = (
        s
        * np.roll(s, -1, axis=-2)
        * np.roll(s, -1, axis=-1)
        * np.roll(np.roll(s, -1, axis=-2), -1, axis=-1)
    )
    s3 = plaq.sum(axis=(-1, -2))
    out = np.stack([s1, s2, s3], axis=-1)
    return out[0] if squeeze else out


def block_even_operators(configs: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Gerade Operatoren auf dem b=2-majority-geblockten Gitter, je Konfiguration.

    Args:
        configs: (n, L, L) in {+1,-1}. L gerade.
        seed: Tie-Break-Seed (majority_block_b2).

    Returns:
        (n, 3) Operatorwerte [S'_1, S'_2, S'_3] auf dem L/2 x L/2 Block-Gitter.
    """
    configs = np.asarray(configs)
    if configs.ndim != 3:
        raise ValueError(f"configs must be 3D (n,L,L), got ndim={configs.ndim}")
    n = configs.shape[0]
    blocks = np.empty((n, configs.shape[1] // 2, configs.shape[2] // 2), dtype=np.int8)
    for i in range(n):
        blocks[i] = ising2d.majority_block_b2(configs[i], config_index=i, seed=seed)
    return even_operators(blocks)


@dataclass(frozen=True)
class OperatorTimeseries:
    """S- und S'-Operator-Zeitreihen einer 2D-Ising-Trajektorie."""

    S: np.ndarray
    """(n, n_op) Original-Operatoren."""
    Sp: np.ndarray
    """(n, n_op) Block-Operatoren."""
    K: float
    L: int


def operator_timeseries(chain: ising2d.Ising2DChain, *, seed: int = 0) -> OperatorTimeseries:
    """Bilde aus einer 2D-Ising-Trajektorie die S/S'-Operator-Matrizen.

    Args:
        chain: Ising2DChain (configs (n,L,L)).
        seed: Tie-Break-Seed fuers Blocking.

    Returns:
        OperatorTimeseries mit S, Sp je (n, 3).
    """
    S = even_operators(chain.configs)
    Sp = block_even_operators(chain.configs, seed=seed)
    return OperatorTimeseries(S=S, Sp=Sp, K=chain.K, L=chain.L)


# --------------------------------------------------------------------------
# Swendsen-Matrix + Exponenten
# --------------------------------------------------------------------------
def _connected_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Connected-Correlation-Matrix C_ab = <X_a Y_b> - <X_a><Y_b>.

    Args:
        X: (n, p) Sample-Spalten.
        Y: (n, q) Sample-Spalten.

    Returns:
        (p, q) connected-correlation-Matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, Y must be 2D (n, k)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"sample mismatch: {X.shape[0]} vs {Y.shape[0]}")
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"need >=2 samples, got {n}")
    xc = X - X.mean(axis=0, keepdims=True)
    yc = Y - Y.mean(axis=0, keepdims=True)
    return (xc.T @ yc) / n


def swendsen_matrix(S: np.ndarray, Sp: np.ndarray) -> np.ndarray:
    """Linearisierte Swendsen-RG-Matrix T = A . B^{-1}.

    A_ab = <S'_a S_b>_c  (p x p),  B_ac = <S'_a S'_c>_c  (p x p).
    T loest  T B = A  (np.linalg.solve, keine explizite Inverse). Da B
    symmetrisch positiv (Kovarianzmatrix der Block-Operatoren) ist das
    well-posed; bei (numerischer) Singularitaet wird sauber geworfen.

    Args:
        S: (n, p) Original-Operatoren.
        Sp: (n, p) Block-Operatoren.

    Returns:
        (p, p) RG-Matrix T, T_ab = dK'_a/dK_b.
    """
    A = _connected_matrix(Sp, S)  # <S'_a S_b>_c
    B = _connected_matrix(Sp, Sp)  # <S'_a S'_c>_c (symmetrisch)
    # T B = A  =>  B^T T^T = A^T  =>  T^T = solve(B^T, A^T); B symmetrisch => B^T=B.
    cond = np.linalg.cond(B)
    if not np.isfinite(cond) or cond > 1e12:
        raise np.linalg.LinAlgError(
            f"block-correlation matrix B ill-conditioned (cond={cond:.3e}); "
            "operators near-degenerate -- drop a redundant operator or sample more"
        )
    T_T = np.linalg.solve(B.T, A.T)
    return T_T.T


def exponents_from_T(T: np.ndarray, *, b: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Eigenwerte lambda_i und Exponenten y_i = ln|lambda_i|/ln b der RG-Matrix.

    Args:
        T: (p, p) RG-Matrix.
        b: RG-Skalenfaktor (Majority b=2).

    Returns:
        (eigenvalues, y) -- eigenvalues (p,) komplex, y (p,) reell, absteigend
        nach |lambda| sortiert.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"T must be square 2D, got {T.shape}")
    if b < 2:
        raise ValueError(f"b must be >= 2, got {b}")
    eig = np.linalg.eigvals(T)
    order = np.argsort(-np.abs(eig))
    eig = eig[order]
    absl = np.abs(eig)
    with np.errstate(divide="ignore"):
        y = np.where(absl > 0.0, np.log(absl) / np.log(b), -np.inf)
    return eig, y


@dataclass(frozen=True)
class SwendsenMatrixEstimate:
    """Multi-Operator-Swendsen-MATRIX-Schaetzung + y_t mit Fehlerbalken."""

    K: float
    L: int
    n_op: int
    T: np.ndarray
    """RG-Matrix (n_op, n_op) (Voll-Sample-Punktschaetzer)."""
    eigenvalues: np.ndarray
    """Eigenwerte, absteigend nach |lambda|."""
    exponents: np.ndarray
    """y_i = ln|lambda_i|/ln 2, absteigend."""
    y_t: float
    """Groesster gerader Exponent (thermisch); = exponents[0]."""
    y_t_error: float
    """Block-Jackknife-Fehler von y_t (autokorr-bewusst)."""
    oracle_y_t: float
    """Exaktes 2D-Orakel y_t = 1 (Onsager)."""
    lambda_max: float
    """|lambda_max| (groesster Eigenwert-Betrag)."""
    cond_B: float
    """Konditionszahl von B (Block-Korrelationsmatrix) -- Degeneriertheits-Check."""
    block_size: int
    n_blocks: int
    n_samples: int
    tau_int_max: float
    """max tau_int der Operator-Reihen (Slowing-Down-Indikator)."""

    @property
    def abs_error(self) -> float:
        """|y_t - 1|."""
        return abs(self.y_t - self.oracle_y_t)

    @property
    def n_sigma(self) -> float:
        """Abweichung vom Orakel in Einheiten des Jackknife-Fehlers."""
        return self.abs_error / self.y_t_error if self.y_t_error > 0 else float("inf")


def _y_t_from_columns(S: np.ndarray, Sp: np.ndarray, *, b: int = 2) -> float:
    """y_t = ln|lambda_max(T)|/ln b aus den Operator-Spalten (eine Auswertung).

    Reine Funktion der Sample-Mittel/-Kovarianzen -> direkt im Jackknife verwendbar.
    """
    T = swendsen_matrix(S, Sp)
    _, y = exponents_from_T(T, b=b)
    return float(y[0])


def estimate_y_t(
    ts: OperatorTimeseries,
    *,
    n_op: int = 2,
    b: int = 2,
    c_window: float = 1.5,
    block_size: int | None = None,
) -> SwendsenMatrixEstimate:
    """Schaetze y_t aus der Swendsen-MATRIX + Block-Jackknife-Fehlerbalken.

    Schritte:
      1. Waehle die ersten n_op geraden Operatoren (>=2 -> echte Matrix).
      2. tau_int je Operator-Reihe (S und S') via Gamma-Methode (autocorr.py);
         Block-Groesse b_jk := ceil(2*max tau_int) (>~ Autokorr -> Bloecke
         quasi-unabhaengig), >=16 Bloecke erzwungen.
      3. Voll-Sample T = A.B^{-1}, Eigenwerte -> y_t = ln|lambda_max|/ln b.
      4. Block-Jackknife (delete-one-block) auf die GANZE Matrix-Pipeline -> y_t
         je Loeschung neu berechnet -> autokorr-bewusster Fehler von y_t.

    Args:
        ts: OperatorTimeseries (S, Sp je (n, >=n_op)).
        n_op: Anzahl Operatoren (>=2). Default 2 (NN + NNN), gut konditioniert.
        b: RG-Skalenfaktor (Majority b=2).
        c_window: Wolff-Windowing-Parameter.
        block_size: Jackknife-Block; Default ceil(2*max tau_int).

    Returns:
        SwendsenMatrixEstimate.
    """
    if n_op < 2:
        raise ValueError(f"n_op must be >= 2 for a MATRIX estimate, got {n_op}")
    S = np.asarray(ts.S, dtype=np.float64)[:, :n_op]
    Sp = np.asarray(ts.Sp, dtype=np.float64)[:, :n_op]
    n = S.shape[0]
    if S.shape[1] < n_op or Sp.shape[1] < n_op:
        raise ValueError(f"need >= {n_op} operator columns, got {S.shape[1]}")
    if n < 64:
        raise ValueError(f"need >=64 samples for autocorr-aware errors, got {n}")

    # tau_int je Operator-Reihe (S und S'); max -> Block-Groesse.
    taus = []
    for col in range(n_op):
        for arr in (S[:, col], Sp[:, col]):
            if np.var(arr) > 0:
                taus.append(autocorr.integrated_autocorr_time(arr, c_window=c_window).tau_int)
    tau_max = max(taus) if taus else 0.5

    if block_size is None:
        block_size = max(1, int(np.ceil(2.0 * tau_max)))
    if n // block_size < 16:
        block_size = max(1, n // 16)
    n_blocks = n // block_size
    if n_blocks < 2:
        raise ValueError(f"need >=2 jackknife blocks; n={n}, block={block_size}")

    # Voll-Sample-Matrix + Punktschaetzer.
    B_full = _connected_matrix(Sp, Sp)
    cond_B = float(np.linalg.cond(B_full))
    T_full = swendsen_matrix(S, Sp)
    eig, y = exponents_from_T(T_full, b=b)
    y_t_full = float(y[0])

    # Block-Jackknife auf y_t: delete-one-block, y_t neu berechnen.
    keep = n_blocks * block_size
    Sk = S[:keep]
    Spk = Sp[:keep]
    jack = np.empty(n_blocks, dtype=np.float64)
    for j in range(n_blocks):
        sl = np.ones(keep, dtype=bool)
        sl[j * block_size : (j + 1) * block_size] = False
        jack[j] = _y_t_from_columns(Sk[sl], Spk[sl], b=b)
    jack_mean = float(jack.mean())
    y_t_error = float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((jack - jack_mean) ** 2)))

    return SwendsenMatrixEstimate(
        K=float(ts.K),
        L=int(ts.L),
        n_op=int(n_op),
        T=T_full,
        eigenvalues=eig,
        exponents=y,
        y_t=y_t_full,
        y_t_error=y_t_error,
        oracle_y_t=1.0,
        lambda_max=float(np.abs(eig[0])),
        cond_B=cond_B,
        block_size=int(block_size),
        n_blocks=int(n_blocks),
        n_samples=int(n),
        tau_int_max=float(tau_max),
    )


def validate_y_t_2d(
    *,
    L: int = 16,
    n_op: int = 2,
    K: float = ising2d.KC_2D,
    n_sweeps: int = 12000,
    burn_in: int = 3000,
    record_every: int = 2,
    seed: int = 0,
) -> SwendsenMatrixEstimate:
    """End-to-end: 2D-Ising-Trajektorie bei K (Default K_c) -> y_t aus der Matrix.

    Klein gehalten (Anti-Stall): L<=32, gekappte Sweeps. Single-Spin-Metropolis
    -> y_t nur GROB (kritisches Slowing-Down), ehrlich dokumentiert.

    Args:
        L: Gitterkante (klein, gerade).
        n_op: Anzahl gerade Operatoren (>=2).
        K: Kopplung (Default K_c, dort lebt der nicht-triviale Fixpunkt).
        n_sweeps, burn_in, record_every: Ketten-Parameter (klein halten).
        seed: Reproduzierbarkeit.

    Returns:
        SwendsenMatrixEstimate.
    """
    chain = ising2d.checkerboard_metropolis(
        K, L, n_sweeps=n_sweeps, burn_in=burn_in, seed=seed, record_every=record_every
    )
    ts = operator_timeseries(chain, seed=seed)
    return estimate_y_t(ts, n_op=n_op)


def _write_validation_report(path: str = "results/phase3b-swendsen-matrix.json") -> int:
    """Schreibe die Phase-3b-Validierung als JSON-Gate-Log (Evidenz)."""
    import json
    from pathlib import Path

    # Multi-Seed-Lauf bei K_c (ehrliche Streuung).
    seeds = (0, 1, 2)
    ests = [validate_y_t_2d(L=16, n_op=2, seed=sd) for sd in seeds]
    y_ts = np.array([e.y_t for e in ests])
    y_t_mean = float(y_ts.mean())
    y_t_spread = float(y_ts.std(ddof=1))
    payload = {
        "tool": "adaptiverg_qec.mcrg_matrix",
        "method": "Multi-operator Swendsen-MCRG matrix T=A.B^-1 (2D-Ising, even sector)",
        "operators": ["S1=NN-bonds", "S2=NNN-bonds (diag)", "S3=plaquette"][:2],
        "blocking": "majority-rule b=2 (Kadanoff), unbiased deterministic tie-break",
        "oracle_y_t": 1.0,
        "oracle_source": "Onsager 1944 (y_t=1/nu=1); web-verified this run",
        "K": ising2d.KC_2D,
        "T_c": ising2d.TC_2D,
        "L": 16,
        "n_op": 2,
        "scope_note": (
            "Single-spin Metropolis near T_c -> critical slowing-down; small L + "
            "1 RG step => y_t only coarse (plausibility, NOT a frontier value). "
            "Cluster algo + multi RG steps = Phase-4."
        ),
        "rows": [
            {
                "seed": sd,
                "y_t": e.y_t,
                "y_t_error_jackknife": e.y_t_error,
                "abs_error_vs_oracle": e.abs_error,
                "n_sigma": e.n_sigma,
                "lambda_max": e.lambda_max,
                "cond_B": e.cond_B,
                "tau_int_max": e.tau_int_max,
                "block_size": e.block_size,
                "n_blocks": e.n_blocks,
                "n_samples": e.n_samples,
                "eigenvalues_abs": [float(abs(z)) for z in e.eigenvalues],
                "exponents": [float(v) for v in e.exponents],
            }
            for sd, e in zip(seeds, ests, strict=True)
        ],
        "y_t_mean": y_t_mean,
        "y_t_multiseed_spread": y_t_spread,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"phase3b-validation -> {out}")
    for sd, e in zip(seeds, ests, strict=True):
        print(
            f"  seed={sd}: y_t={e.y_t:.4f} +/- {e.y_t_error:.4f}  "
            f"(oracle 1.0, |err|={e.abs_error:.4f}, {e.n_sigma:.2f}sigma) "
            f"lambda_max={e.lambda_max:.4f} tau={e.tau_int_max:.2f}"
        )
    print(f"  multi-seed: y_t = {y_t_mean:.4f} +/- {y_t_spread:.4f} (3 seeds)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_write_validation_report())
