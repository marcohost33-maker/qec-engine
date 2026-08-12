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

from . import autocorr

__all__ = [
    "spins_from_bits",
    "nn_operator",
    "decimate_b2",
    "connected_correlation",
    "sample_ising_open_chain",
    "swendsen_T_scalar",
    "SwendsenEstimate",
    "validate_swendsen",
    "operator_timeseries_from_configs",
    "swendsen_T_from_chain",
    "AKernelSwendsenEstimate",
    "validate_swendsen_akernel",
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
    # <s_i s_{i+1}> = tanh K. Die Bond-Menge MUSS zur Randbedingung passen
    # (periodisch: inkl. Wrap-Bond; offen: nur innere Bonds) -- Audit P1-9.
    if periodic:
        nn_corr = float(np.mean(s * np.roll(s, -1, axis=1)))
    else:
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
    """Multi-Seed-Standardabweichung (Einzel-Seed-Streuung)."""
    T_hat_sem: float
    """Standardfehler des Mittels = std/sqrt(n_seeds)."""
    n_sigma: float
    """|mean - oracle| / sem (Signifikanz des MITTELS; inf falls sem==0 bei err>0)."""
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
        # Signifikanz des MITTELS gehoert auf den Standardfehler des Mittels
        # (sem = std/sqrt(n_seeds)), nicht auf die Einzel-Seed-Streuung std --
        # sonst ist der 3-sigma-Test um sqrt(n_seeds) zu lax (Audit P0-3).
        abs_err = abs(mean - oracle)
        n_sigma = abs_err / sem if sem > 0 else (0.0 if abs_err == 0.0 else float("inf"))
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


# ===========================================================================
# PHASE-3a: Swendsen-T_hat aus dem KORRELIERTEN A-Kernel-MCMC + autokorr-Fehler
# ===========================================================================
#
# Phase-2 nutzte den exakten i.i.d.-Sampler (sample_ising_open_chain) -- bewusst,
# damit der 1/sqrt(N)-Test sauber war. Phase-3a verdrahtet die ECHTE Sample-Quelle:
# die korrelierte Markov-Kette des adaptiven A-Kernels (a_kernel.run_adaptive_mcmc).
# Aufeinanderfolgende Samples sind nun NICHT unabhaengig; die Fehlerbalken muessen
# autokorrelations-bewusst sein (autocorr.py: Gamma-Methode + Binning-Cross-Check;
# Jackknife-ueber-Bloecke fuer das Verhaeltnis T_hat = <S'S>_c / <S'S'>_c).
#
# WISSENSCHAFTLICHER KERNPUNKT: weil korrelierte Samples weniger Information tragen
# (N_eff = N/(2 tau_int) < N), sind die korrekten Fehler GROESSER als die naiven
# i.i.d.-Fehler aus Phase-2. Das ehrlich zu zeigen ist der Zweck dieser Phase.


def operator_timeseries_from_configs(
    configs: np.ndarray, *, periodic: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Bilde aus A-Kernel-Konfigurations-Snapshots die S- und S'-Zeitreihen.

    Der A-Kernel speichert je Sweep x in {0,1}^L (record_configs=True). Hier
    werden daraus pro Sweep der NN-Operator S = sum s_i s_{i+1} und der geblockte
    Operator S' (auf dem b=2-decimierten Gitter) berechnet -- vektorisiert ueber
    die ganze Trajektorie. Die Randbedingung MUSS zur Sampler-Konvention passen:
    der A-Kernel ist PERIODISCH (H = #Domaenenwaende mit np.roll), also periodic=True.

    Args:
        configs: (n_steps, L) in {0,1} (SampleResult.configs).
        periodic: Randbedingung fuer S/S' (A-Kernel -> True).

    Returns:
        (S_traj, Sp_traj): je (n_steps,) float-Zeitreihen.
    """
    configs = np.asarray(configs)
    if configs.ndim != 2:
        raise ValueError(f"configs must be 2D (n_steps, L), got ndim={configs.ndim}")
    if configs.shape[1] < 4:
        raise ValueError(
            f"need L>=4 so the decimated chain has >=2 sites, got L={configs.shape[1]}"
        )
    s = spins_from_bits(configs)  # (n_steps, L) in {+1,-1}, validiert {0,1}
    S = nn_operator(s, periodic=periodic)
    Sp = nn_operator(decimate_b2(s), periodic=periodic)
    return S, Sp


@dataclass(frozen=True)
class AKernelSwendsenEstimate:
    """Swendsen-T_hat aus korrelierten A-Kernel-Samples, autokorr-bewusst.

    EHRLICHE NUANCE (kein Ueber-Claim): tau_int_S/Sp/product sind die Autokorr-
    Zeiten der ROHEN Operatoren. Der RATIO-Schaetzer T_hat erbt diese Autokorr-
    Zeit aber NICHT voll: Zaehler und Nenner sind aus denselben korrelierten
    Konfigurationen gebaut, ihre fuehrenden Fluktuationen kuerzen sich im
    Verhaeltnis teilweise, sodass die effektive Autokorr-Zeit von T_hat (und
    damit error_inflation) kleiner ist als sqrt(2*tau_int_S). Der Block-Jackknife
    misst die KORREKTE Inflation des Verhaeltnisses direkt (Plateau in b), nicht
    die der Rohfelder. error_inflation > 1 bleibt der belegte Kernpunkt.
    """

    K: float
    """Ising-Kopplung K = beta/2 (Sampler-Konvention)."""
    T_hat: float
    """Swendsen-Schaetzer <S'S>_c / <S'S'>_c (Voll-Sample-Punktschaetzer)."""
    oracle: float
    """Analytisches Orakel tanh(2K)."""
    error_correlated: float
    """Autokorr-bewusster Fehler von T_hat (Block-Jackknife, b >~ tau_int)."""
    error_iid_naive: float
    """Naiver i.i.d.-Jackknife-Fehler (b=1) -- IGNORIERT Autokorrelation."""
    tau_int_S: float
    """tau_int des NN-Operators S (Gamma-Methode)."""
    tau_int_Sp: float
    """tau_int des geblockten Operators S' (Gamma-Methode)."""
    tau_int_product: float
    """tau_int des Produkts S'*S (relevant fuer den Zaehler-Korrelator)."""
    block_size: int
    """Jackknife-Block-Groesse b (>~ max tau_int)."""
    n_blocks: int
    """Anzahl Jackknife-Bloecke."""
    n_samples: int
    """Kettenlaenge nach Burn-in."""
    n_eff: float
    """Effektive Stichprobe N/(2 max(tau_int)) (Faustwert)."""

    @property
    def abs_error(self) -> float:
        """|T_hat - tanh(2K)|."""
        return abs(self.T_hat - self.oracle)

    @property
    def n_sigma(self) -> float:
        """Abweichung vom Orakel in Einheiten des KORRELIERTEN Fehlers."""
        return self.abs_error / self.error_correlated if self.error_correlated > 0 else 0.0

    @property
    def error_inflation(self) -> float:
        """Wievielmal groesser der korrelierte Fehler ggue. dem naiven i.i.d. ist."""
        return (
            self.error_correlated / self.error_iid_naive
            if self.error_iid_naive > 0
            else float("nan")
        )


def _T_ratio_combine(num_means: np.ndarray, den_means: np.ndarray) -> float:
    """combine() fuer den Jackknife: T_hat = <S'S>_c / <S'S'>_c aus Spalten-Mitteln.

    Spalten-Layout:
        num_means = [<S'S>, <S'>, <S>]   -> Zaehler = <S'S> - <S'><S>
        den_means = [<S'S'>, <S'>]       -> Nenner  = <S'S'> - <S'>^2
    """
    cov_sps = num_means[0] - num_means[1] * num_means[2]
    cov_spsp = den_means[0] - den_means[1] * den_means[1]
    if cov_spsp == 0.0:
        return float("nan")
    return float(cov_sps / cov_spsp)


def swendsen_T_from_chain(
    S: np.ndarray,
    Sp: np.ndarray,
    *,
    K: float,
    block_size: int | None = None,
    c_window: float = 1.5,
) -> AKernelSwendsenEstimate:
    """Skalarer Swendsen-T_hat aus korrelierten S/S'-Zeitreihen + autokorr-Fehler.

    Schritte:
      1. tau_int von S, S' und dem Produkt S'*S via Gamma-Methode (autocorr.py).
      2. Block-Groesse b := ceil(2 * max tau_int) (>~ Autokorr-Zeit -> Bloecke
         quasi-unabhaengig), sofern nicht explizit gesetzt.
      3. Block-Jackknife auf das VERHAELTNIS T_hat = <S'S>_c/<S'S'>_c -> korrekter,
         autokorr-bewusster Fehler (propagiert korrelierte Zaehler/Nenner sauber).
      4. Zum Vergleich: derselbe Jackknife mit b=1 (naiv i.i.d., ignoriert
         Autokorrelation) -- demonstriert, dass der korrelierte Fehler GROESSER ist.

    Args:
        S: NN-Operator-Zeitreihe (n_samples,) aus dem A-Kernel (nach Burn-in).
        Sp: geblockte Operator-Zeitreihe (n_samples,).
        K: Ising-Kopplung der Kette (= beta/2), fuer das Orakel tanh(2K).
        block_size: Jackknife-Block-Groesse b; Default ceil(2*max tau_int).
        c_window: Wolff-Windowing-Parameter S.

    Returns:
        AKernelSwendsenEstimate.
    """
    S = np.asarray(S, dtype=np.float64).ravel()
    Sp = np.asarray(Sp, dtype=np.float64).ravel()
    if S.shape != Sp.shape:
        raise ValueError(f"S, Sp must have equal length, got {S.shape} vs {Sp.shape}")
    n = S.size
    if n < 64:
        raise ValueError(f"need >=64 samples for autocorr-aware errors, got {n}")
    if not np.isfinite(K):
        raise ValueError(f"K must be finite, got {K}")

    prod = Sp * S
    ac_S = autocorr.integrated_autocorr_time(S, c_window=c_window)
    ac_Sp = autocorr.integrated_autocorr_time(Sp, c_window=c_window)
    ac_prod = autocorr.integrated_autocorr_time(prod, c_window=c_window)
    tau_max = max(ac_S.tau_int, ac_Sp.tau_int, ac_prod.tau_int)

    if block_size is None:
        block_size = max(1, int(np.ceil(2.0 * tau_max)))
    # Genug Bloecke sicherstellen (>=16) -> Block ggf. verkleinern.
    if n // block_size < 16:
        block_size = max(1, n // 16)

    # Spalten fuer den Jackknife (pro-Sample-Beitraege).
    num_terms = np.column_stack([prod, Sp, S])  # [<S'S>, <S'>, <S>]
    den_terms = np.column_stack([Sp * Sp, Sp])  # [<S'S'>, <S'>]

    jk = autocorr.jackknife_ratio(
        num_terms, den_terms, block_size=block_size, combine=_T_ratio_combine
    )
    # Naiver i.i.d.-Vergleich: derselbe Jackknife mit b=1.
    jk_iid = autocorr.jackknife_ratio(num_terms, den_terms, block_size=1, combine=_T_ratio_combine)

    oracle = float(np.tanh(2.0 * K))
    n_eff = n / (2.0 * tau_max) if tau_max > 0 else float(n)
    return AKernelSwendsenEstimate(
        K=float(K),
        T_hat=float(jk.ratio),
        oracle=oracle,
        error_correlated=float(jk.error),
        error_iid_naive=float(jk_iid.error),
        tau_int_S=float(ac_S.tau_int),
        tau_int_Sp=float(ac_Sp.tau_int),
        tau_int_product=float(ac_prod.tau_int),
        block_size=int(block_size),
        n_blocks=int(jk.n_blocks),
        n_samples=int(n),
        n_eff=float(n_eff),
    )


def validate_swendsen_akernel(
    K_values: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
    *,
    L: int = 64,
    n_steps: int = 20000,
    burn_in: int = 2000,
    seed: int = 0,
    c_window: float = 1.5,
) -> list[AKernelSwendsenEstimate]:
    """T_hat(K) aus dem A-Kernel-MCMC mit autokorr-Fehlern, fuer mehrere K.

    Fuer jedes K: laufe den adaptiven A-Kernel bei beta_target = 2K (Konvention
    K = beta/2), zeichne Konfigurationen auf, bilde S/S'-Zeitreihen (periodisch),
    schaetze T_hat + autokorr-Fehler. Klein gehalten (Anti-Stall): kurze Ketten.

    Args:
        K_values: Kopplungen (beta_target = 2K muss in [beta_min, beta_max] des
            cfg liegen; hier cfg = MVPConfig(L, 0.1, 2.0) -> K in [0.05, 1.0]).
        L, n_steps, burn_in: Ketten-Parameter (klein halten).
        seed: Basis-Seed.
        c_window: Wolff-Parameter.

    Returns:
        Liste von AKernelSwendsenEstimate, eine je K.
    """
    from .a_kernel import run_adaptive_mcmc
    from .mvp_instance import MVPConfig

    cfg = MVPConfig(L=L, beta_min=0.1, beta_max=2.0)
    out: list[AKernelSwendsenEstimate] = []
    for K in K_values:
        beta = 2.0 * K  # K = beta/2
        res = run_adaptive_mcmc(
            cfg,
            beta_target=beta,
            n_steps=n_steps,
            burn_in=burn_in,
            seed=K_seed(K, seed),
            beta_start=beta,  # bei beta_target starten -> kuerzere Equilibrierung
            record_configs=True,
        )
        if res.configs is None:  # pragma: no cover - defensive
            raise RuntimeError("record_configs=True did not populate configs")
        post = res.configs[burn_in:]
        S, Sp = operator_timeseries_from_configs(post, periodic=True)
        out.append(swendsen_T_from_chain(S, Sp, K=K, c_window=c_window))
    return out


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
