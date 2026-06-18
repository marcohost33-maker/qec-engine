"""Phase-2: sample-geschaetzte stochastische RG-Map R-hat via Swendsen-MCRG.

Skalarer 1D-Ising-Fall (Decimation b=2). Diese Datei implementiert REAL den
Monte-Carlo-Renormalization-Group-Schaetzer nach Swendsen (1979) und validiert
ihn gegen das ANALYTISCHE Orakel der exakten Decimation-Rekursion.

------------------------------------------------------------------------------
KONVENTION (kompatibel mit ising1d.py / rg_map.py)
------------------------------------------------------------------------------
Spin  s_i = 1 - 2*x_i in {+1,-1}  (x in {0,1} = Code-Bit/Fehler).
Energie-Operator (NN, EINE Kopplung): S = sum_i s_i s_{i+1}.
Boltzmann-Ziel  pi ~ exp(-beta H)  mit  H = (L - S)/2  =>  pi ~ exp(K * S),
also Ising-Kopplung  K = beta/2  (s. ising1d.coupling_from_beta).

Block-Transform (Decimation b=2): jeder 2. Spin bleibt,  s'_j = s_{2j}.
Geblockter Operator auf dem ausgeduennten Gitter:  S' = sum_j s'_j s'_{j+1}.

------------------------------------------------------------------------------
SWENDSEN-MCRG-SCHAETZER  (Swendsen, PRL 42, 859 (1979))
------------------------------------------------------------------------------
Linearisierte RG-Matrix  T_ab = d K'_a / d K_b  am Sample-Niveau ueber die
Ketten-Regel der thermischen Ableitungen:
    d<S'_a>/dK_b = sum_c (d<S'_a>/dK'_c) (dK'_c/dK_b)
mit  d<S'_a>/dK_b   = <S'_a S_b>_c      (connected correlation, ableitung nach
                                          der MIKROSKOPISCHEN Kopplung)
und  d<S'_a>/dK'_c  = <S'_a S'_c>_c     (ableitung nach der GEBLOCKTEN Kopplung).
Skalar (eine Kopplung) reduziert sich das auf:

    T-hat(K) = <S' S>_c / <S' S'>_c ,     <AB>_c := <AB> - <A><B>.

ORAKEL:  fuer die exakte 1D-Decimation gilt  R'(K) = tanh(2K)  (rg_map.py).
Der Swendsen-Schaetzer konvergiert mit den Samples gegen tanh(2K); der
statistische Fehler skaliert wie 1/sqrt(N). Die Identitaet ist im EXAKTEN
Ensemble exakt (per Enumeration bestaetigt, s. tests/test_mcrg.py) -- der
Schaetzer ist also unbiased, nur sampling-rausch-behaftet.

------------------------------------------------------------------------------
EHRLICHE SCOPE-GRENZE (AGENTS.md Sec.1)
------------------------------------------------------------------------------
- 1D-Ising = EINE Kopplung => T-hat ist SKALAR. Die volle Multi-Operator-
  Swendsen-MATRIX (mehrere S_a, lineares Gleichungssystem) ist PHASE-3 und wird
  hier NICHT behauptet.
- T_c = 0 (1D) => kein nicht-trivialer kritischer Exponent. Validiert wird die
  SCHAETZ-MASCHINERIE gegen das analytische Orakel, KEIN Frontier-Wert.
- Sampler hier: EXAKTER i.i.d.-Sampler der offenen 1D-Kette (vektorisiert,
  unkorreliert) -- bewusst gewaehlt, damit der 1/sqrt(N)-Test sauber ist. Der
  korrelierte adaptive A-Kernel (a_kernel.py) bleibt die MCMC-Quelle der
  uebrigen Pipeline; die Korrektheit des Schaetzers haengt nicht von der
  Sampler-Wahl ab (Identitaet gilt fuer JEDES korrekte Ensemble).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "spins_from_bits",
    "nn_operator",
    "decimate_b2",
    "connected_correlation",
    "sample_ising_open_chain",
    "swendsen_T_scalar",
    "SwendsenEstimate",
    "validate_swendsen",
]


def spins_from_bits(x: np.ndarray) -> np.ndarray:
    """s = 1 - 2*x  (x in {0,1} -> s in {+1,-1}). Akzeptiert (...,L)-Arrays."""
    x = np.asarray(x)
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must be in {0,1}")
    return 1.0 - 2.0 * x.astype(np.float64)


def nn_operator(s: np.ndarray, *, periodic: bool = False) -> np.ndarray:
    """NN-Energie-Operator S = sum_i s_i s_{i+1} je Konfiguration.

    Args:
        s: Spin-Array (N, L) (oder (L,)) in {+1,-1} (oder reellwertig).
        periodic: True -> Ring (Wrap-Bond L-1<->0); False -> offene Kette.

    Returns:
        S je Zeile: Shape (N,) (oder Skalar fuer 1D-Eingabe).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
        squeeze = True
    else:
        squeeze = False
    if s.shape[-1] < 2:
        raise ValueError(f"need L>=2 for NN operator, got L={s.shape[-1]}")
    if periodic:
        bonds = s * np.roll(s, -1, axis=-1)
        out = bonds.sum(axis=-1)
    else:
        out = (s[:, :-1] * s[:, 1:]).sum(axis=-1)
    return out[0] if squeeze else out


def decimate_b2(s: np.ndarray) -> np.ndarray:
    """Decimation-Block b=2: behalte jeden 2. Spin, s'_j = s_{2j}.

    Args:
        s: Spin-Array (N, L) (oder (L,)). L sollte gerade sein (sonst wird der
           letzte ungepaarte Spin verworfen -- das ist die Standard-Decimation).

    Returns:
        Geblockte Spins (N, L//2) (oder (L//2,)).
    """
    s = np.asarray(s)
    if s.ndim == 1:
        return s[::2]
    return s[:, ::2]


def connected_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """<a b>_c = <a b> - <a><b>, bias-bewusst geschaetzt ueber Samples.

    Verwendet das Plug-in- (Sample-Moment-)Mittel. Bias-Hinweis: das Plug-in
    <a><b> hat O(1/N)-Bias gegenueber dem unbiased Kovarianz-Schaetzer
    (Faktor N/(N-1)); fuer die T-hat-RATIO kuerzt sich dieser Faktor exakt
    heraus (Zaehler und Nenner teilen ihn), siehe swendsen_T_scalar.

    Args:
        a, b: gleichlange 1D-Sample-Arrays (N,).

    Returns:
        Geschaetzte connected correlation (float).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a, b must have equal shape, got {a.shape} vs {b.shape}")
    if a.size < 2:
        raise ValueError(f"need >=2 samples for a connected correlation, got {a.size}")
    return float(np.mean(a * b) - np.mean(a) * np.mean(b))


def sample_ising_open_chain(K: float, L: int, n_samples: int, *, seed: int) -> np.ndarray:
    """EXAKTE i.i.d.-Stichprobe der offenen 1D-Ising-Kette (vektorisiert).

    Konstruktion (geschlossen, kein MCMC-Burn-in noetig): s_0 uniform +-1; jeder
    Bond stimmt unabhaengig mit  p = e^K/(e^K+e^{-K}) = (1+tanh K)/2  ueberein.
    Damit ist  <s_i s_{i+1}> = tanh K  exakt -- ein unabhaengig pruefbares Orakel
    (Validierung in validate_swendsen / tests).

    Args:
        K: Ising-Kopplung (= beta/2). Endlich; K<=0 oder nicht-endlich verboten.
        L: Kettenlaenge (>=2).
        n_samples: Anzahl unabhaengiger Konfigurationen (>=2).
        seed: reproduzierbarer Seed (PCG64).

    Returns:
        Spin-Array (n_samples, L) in {+1,-1}.
    """
    if not np.isfinite(K):
        raise ValueError(f"K must be finite, got {K}")
    if K <= 0.0:
        raise ValueError(f"K must be > 0, got {K}")
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    rng = np.random.default_rng(seed)
    # p_agree = e^K / (e^K + e^-K) = sigmoid(2K) = (1 + tanh K)/2.
    p_agree = float(np.exp(K) / (np.exp(K) + np.exp(-K)))
    s = np.empty((n_samples, L), dtype=np.float64)
    s[:, 0] = 1.0 - 2.0 * rng.integers(0, 2, size=n_samples)
    if L > 1:
        agree = rng.random((n_samples, L - 1)) < p_agree
        bond = np.where(agree, 1.0, -1.0)
        s[:, 1:] = s[:, 0:1] * np.cumprod(bond, axis=1)
    return s


@dataclass(frozen=True)
class SwendsenEstimate:
    """Ergebnis einer Swendsen-T-hat-Schaetzung (skalar)."""

    K: float
    """Mikroskopische Ising-Kopplung, bei der gesampelt wurde."""
    T_hat: float
    """Swendsen-Schaetzer <S'S>_c / <S'S'>_c."""
    oracle: float
    """Analytisches Orakel R'(K) = tanh(2K)."""
    cov_SpS: float
    """<S' S>_c (Zaehler)."""
    cov_SpSp: float
    """<S' S'>_c (Nenner)."""
    n_samples: int
    """Verwendete Stichprobengroesse."""

    @property
    def abs_error(self) -> float:
        """|T_hat - tanh(2K)|."""
        return abs(self.T_hat - self.oracle)


def swendsen_T_scalar(s: np.ndarray, *, periodic: bool = False) -> SwendsenEstimate:
    """Skalarer Swendsen-MCRG-Schaetzer T-hat aus einer Spin-Stichprobe.

    T-hat(K) = <S' S>_c / <S' S'>_c, mit S = NN-Operator, S' = NN-Operator auf
    dem b=2-decimierten Gitter. Erfordert, dass die Samples aus pi ~ exp(K*S)
    gezogen wurden; K wird fuers Orakel aus der NN-Korrelation rueckgewonnen
    ( <s_i s_{i+1}> = tanh K ), damit das Orakel nicht extern eingespeist werden
    muss (Selbst-Konsistenz -> keine erfundene Referenz).

    Args:
        s: Spin-Stichprobe (n_samples, L) in {+1,-1}.
        periodic: Randbedingung fuer S/S' (muss zur Sampler-Konvention passen).

    Returns:
        SwendsenEstimate (T_hat + Orakel + Kovarianzen).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"s must be 2D (n_samples, L), got ndim={s.ndim}")
    n_samples, L = s.shape
    if n_samples < 2:
        raise ValueError(f"need >=2 samples, got {n_samples}")
    if L < 4:
        raise ValueError(f"need L>=4 so the decimated chain has >=2 sites, got L={L}")

    S = nn_operator(s, periodic=periodic)
    sp = decimate_b2(s)
    Sp = nn_operator(sp, periodic=periodic)

    cov_SpS = connected_correlation(Sp, S)
    cov_SpSp = connected_correlation(Sp, Sp)
    if cov_SpSp == 0.0:
        raise ValueError("degenerate sample: <S'S'>_c == 0 (no fluctuation)")
    T_hat = cov_SpS / cov_SpSp

    # K aus der NN-Korrelation rueckgewinnen (unabhaengiges Orakel im Sample):
    # offene Kette: mean ueber alle inneren Bonds; <s_i s_{i+1}> = tanh K.
    nn_corr = float(np.mean(s[:, :-1] * s[:, 1:]))
    nn_corr = min(max(nn_corr, -0.999999), 0.999999)
    K_est = float(np.arctanh(nn_corr))
    oracle = float(np.tanh(2.0 * K_est))

    return SwendsenEstimate(
        K=K_est,
        T_hat=float(T_hat),
        oracle=oracle,
        cov_SpS=float(cov_SpS),
        cov_SpSp=float(cov_SpSp),
        n_samples=int(n_samples),
    )


@dataclass(frozen=True)
class ValidationRow:
    """Eine K-Zeile der Multi-Seed-Validierung (mit Fehlerbalken)."""

    K: float
    oracle: float
    """tanh(2K)."""
    T_hat_mean: float
    """Mittel der T-hat ueber die Seeds."""
    T_hat_std: float
    """Multi-Seed-Standardabweichung (Fehlerbalken)."""
    T_hat_sem: float
    """Standardfehler des Mittels = std/sqrt(n_seeds)."""
    n_sigma: float
    """|mean - oracle| / std (0 falls std==0)."""
    seeds: tuple[int, ...]
    n_samples: int


def validate_swendsen(
    K_values: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
    *,
    L: int = 64,
    n_samples: int = 3000,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    periodic: bool = False,
) -> list[ValidationRow]:
    """Multi-Seed-Validierung T-hat(K) vs tanh(2K) mit Fehlerbalken.

    Fuer jedes K: ziehe pro Seed eine unabhaengige Stichprobe, schaetze T-hat,
    bilde Mittel + Multi-Seed-std (Fehlerbalken, schliesst Equalita-G1-Nit).

    Args:
        K_values: zu testende Kopplungen.
        L: Kettenlaenge (klein halten; b=2 braucht L>=4).
        n_samples: Stichprobengroesse je Seed (klein halten, Anti-Stall).
        seeds: Multi-Seed-Liste (>=2 fuer eine std).
        periodic: Randbedingung.

    Returns:
        Liste von ValidationRow, eine je K.
    """
    if len(seeds) < 2:
        raise ValueError(f"need >=2 seeds for an error bar, got {len(seeds)}")
    rows: list[ValidationRow] = []
    for K in K_values:
        ests = np.array(
            [
                swendsen_T_scalar(
                    sample_ising_open_chain(K, L, n_samples, seed=K_seed(K, sd)),
                    periodic=periodic,
                ).T_hat
                for sd in seeds
            ]
        )
        oracle = float(np.tanh(2.0 * K))
        mean = float(ests.mean())
        std = float(ests.std(ddof=1))
        sem = std / np.sqrt(len(seeds))
        n_sigma = abs(mean - oracle) / std if std > 0 else 0.0
        rows.append(
            ValidationRow(
                K=float(K),
                oracle=oracle,
                T_hat_mean=mean,
                T_hat_std=std,
                T_hat_sem=float(sem),
                n_sigma=float(n_sigma),
                seeds=tuple(int(s) for s in seeds),
                n_samples=int(n_samples),
            )
        )
    return rows


def K_seed(K: float, sd: int) -> int:
    """Deterministischer, K-abhaengiger Seed (reproduzierbar, dekorreliert K's)."""
    return (int(round(K * 1000)) * 100003 + sd) % (2**31 - 1)


def _write_validation_report(path: str = "results/phase2-swendsen.json") -> int:
    """Schreibe die Phase-2-Validierungstabelle als JSON-Gate-Log (Evidenz)."""
    import json
    from pathlib import Path

    rows = validate_swendsen()
    payload = {
        "tool": "adaptiverg_qec.mcrg",
        "method": "Swendsen-MCRG scalar (1D-Ising decimation b=2)",
        "estimator": "T_hat(K) = <S'S>_c / <S'S'>_c",
        "oracle": "R'(K) = tanh(2K)",
        "sampler": "exact i.i.d. open 1D-Ising chain (vectorized)",
        "L": 64,
        "rows": [
            {
                "K": r.K,
                "oracle_tanh2K": r.oracle,
                "T_hat_mean": r.T_hat_mean,
                "T_hat_std": r.T_hat_std,
                "T_hat_sem": r.T_hat_sem,
                "n_sigma": r.n_sigma,
                "n_seeds": len(r.seeds),
                "n_samples": r.n_samples,
            }
            for r in rows
        ],
        "all_within_3sigma": all(r.n_sigma <= 3.0 for r in rows),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"phase2-validation -> {out}")
    for r in rows:
        print(
            f"  K={r.K}: T_hat={r.T_hat_mean:.5f} +/- {r.T_hat_std:.5f}  "
            f"tanh(2K)={r.oracle:.5f}  n_sigma={r.n_sigma:.2f}"
        )
    return 0 if payload["all_within_3sigma"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(_write_validation_report())
