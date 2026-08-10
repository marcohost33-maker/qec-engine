"""Phase-6: Self-Normalized Importance Sampling (SNIS) mit chi^2-Varianz-Bound.

Schliesst die Phase-4-Luecke "SNIS mit chi^2-Divergenz-Varianz-Bound" (ROADMAP).

------------------------------------------------------------------------------
SETTING (bewusst gewaehlt, damit ALLE Groessen geschlossen prueffbar sind)
------------------------------------------------------------------------------
Offene 1D-Ising-Kette (mcrg.sample_ising_open_chain, exakter i.i.d.-Sampler):
    pi_K(s) = exp(K * S(s)) / Z(K),   S(s) = sum_{i<L-1} s_i s_{i+1},
    Z(K) = 2 * (2 cosh K)^{L-1}      (L-1 unabhaengige Bonds),
    <S>_K = (L-1) * tanh(K).

SNIS: Samples aus der PROPOSAL pi_{K_p}, Schaetzung unter der TARGET pi_{K_t}:
    w~(s) = exp((K_t - K_p) * S(s))          (unnormalisiert),
    g_hat = sum_i w~_i g_i / sum_i w~_i      (selbst-normalisiert).

GESCHLOSSENE ORAKEL (alle exponential-family-exakt, keine Simulation noetig):
  1. chi^2-Divergenz:
       1 + chi^2(pi_t || pi_p) = E_p[(pi_t/pi_p)^2] = Z(2K_t-K_p) Z(K_p) / Z(K_t)^2
                               = [cosh(2K_t-K_p) cosh(K_p) / cosh^2(K_t)]^(L-1).
  2. ESS (Kong):  ESS/N = (sum w~)^2 / (N sum w~^2)  ->  1/(1+chi^2)  (N->inf).
  3. Fuehrender SNIS-Bias des Ratio-Schaetzers (Delta-Methode, O(1/N)):
       bias(g_hat) = (1/N) (1+chi^2) (<g>_{K_t} - <g>_{2K_t-K_p}) + O(1/N^2),
     denn E_p[w-bar^2] = 1+chi^2 und E_p[w-bar^2 g]/(E_p w-bar)^2
     = (1+chi^2) <g>_{2K_t-K_p} (Reweighting bleibt in der Familie).
     Fuer g = S/(L-1):  <g>_K = tanh K  =>  Bias-Koeffizient
       c_bias = (1+chi^2) (tanh K_t - tanh(2K_t - K_p)).
  4. MSE-Bound (Agapiou, Papaspiliopoulos, Sanz-Alonso, Stuart 2017,
     Statist. Sci. 32(3), Thm 2.1): fuer |g| <= 1 gilt
       MSE(g_hat) <= 4 * (1+chi^2) / N.

EHRLICHE SCOPE-GRENZE (AGENTS.md Sec.1): validiert wird die SNIS-MASCHINERIE
(Gewichte, ESS, Bias-Ordnung, Bound) gegen geschlossene Orakel der 1D-Familie.
Defensive Mixture und SNIS auf 2D/RBIM-Targets sind NICHT implementiert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mcrg import sample_ising_open_chain

__all__ = [
    "log_partition_open_chain",
    "mean_bond_correlation",
    "chi2_divergence_open_chain",
    "snis_bias_coefficient",
    "SNISEstimate",
    "snis_reweight",
    "snis_from_couplings",
    "BiasScalingReport",
    "measure_bias_scaling",
]

DEFAULT_MIN_ESS = 64.0


def _check_coupling(K: float, name: str) -> float:
    K = float(K)
    if not math.isfinite(K):
        raise ValueError(f"{name} must be finite, got {K}")
    return K


def log_partition_open_chain(K: float, L: int) -> float:
    """ln Z(K) = ln 2 + (L-1) ln(2 cosh K) der offenen Kette (exakt)."""
    K = _check_coupling(K, "K")
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    # ln(2 cosh K) numerisch stabil: |K| + ln(1+e^{-2|K|}).
    return math.log(2.0) + (L - 1) * (abs(K) + math.log1p(math.exp(-2.0 * abs(K))))


def mean_bond_correlation(K: float) -> float:
    """<s_i s_{i+1}>_K = tanh K (offene Kette, exakt; = <S>_K/(L-1))."""
    return math.tanh(_check_coupling(K, "K"))


def chi2_divergence_open_chain(K_target: float, K_proposal: float, L: int) -> float:
    """chi^2(pi_t || pi_p) geschlossen: [cosh(2Kt-Kp) cosh(Kp)/cosh^2(Kt)]^(L-1) - 1.

    Exakt fuer die offene 1D-Ising-Kette (L-1 unabhaengige Bonds). Immer >= 0;
    == 0 gdw K_target == K_proposal.
    """
    Kt = _check_coupling(K_target, "K_target")
    Kp = _check_coupling(K_proposal, "K_proposal")
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")

    # In log-Raum (vermeidet Overflow bei grossem L / grossem Delta-K).
    def _lc(k: float) -> float:  # ln cosh(k), stabil
        return abs(k) + math.log1p(math.exp(-2.0 * abs(k))) - math.log(2.0)

    log_ratio = (L - 1) * (_lc(2.0 * Kt - Kp) + _lc(Kp) - 2.0 * _lc(Kt))
    # Codex-Review-Fix: bei extremem Mismatch (grosses L * grosses Delta-K)
    # ueberschreitet log_ratio den float64-Exponentenbereich (~709). Genau das
    # IST der ESS-Kollaps-Fall -> chi^2 = inf zurueckgeben (ess_rel_oracle -> 0,
    # ess_adequate=False), statt mit OverflowError abzubrechen.
    if log_ratio > 700.0:
        return float("inf")
    return math.expm1(log_ratio)


def snis_bias_coefficient(K_target: float, K_proposal: float, L: int) -> float:
    """Fuehrender Bias-Koeffizient c_bias fuer g = S/(L-1) (geschlossen).

    bias(g_hat) ~ c_bias / N  mit
    c_bias = (1+chi^2) * (tanh K_t - tanh(2 K_t - K_p)).
    """
    chi2 = chi2_divergence_open_chain(K_target, K_proposal, L)
    Kt = float(K_target)
    Kp = float(K_proposal)
    return (1.0 + chi2) * (math.tanh(Kt) - math.tanh(2.0 * Kt - Kp))


@dataclass(frozen=True)
class SNISEstimate:
    """Ergebnis einer SNIS-Reweighting-Schaetzung (mit Diagnostik + Orakeln)."""

    K_proposal: float
    """Kopplung der Proposal-Verteilung (Sample-Quelle)."""
    K_target: float
    """Kopplung der Target-Verteilung (Schaetz-Ziel)."""
    n_samples: int
    g_hat: float
    """SNIS-Schaetzer der Per-Bond-Korrelation g = S/(L-1) unter pi_{K_t}."""
    oracle: float
    """Geschlossenes Orakel tanh(K_target)."""
    error: float
    """Delta-Methoden-Fehler sqrt(sum w_i^2 (g_i - g_hat)^2), w_i normiert."""
    ess: float
    """Kong-ESS (sum w~)^2 / sum w~^2."""
    ess_rel: float
    """ESS/N (empirisch)."""
    ess_rel_oracle: float
    """Geschlossenes Orakel 1/(1+chi^2)."""
    chi2_hat: float
    """Empirische chi^2-Schaetzung N * sum(w_i^2) - 1 (w_i normiert)."""
    chi2_oracle: float
    """Geschlossene chi^2-Divergenz."""
    mse_bound: float
    """Agapiou-et-al.-Bound 4 (1+chi^2)/N fuer |g|<=1 (MSE-Obergrenze)."""
    ess_adequate: bool
    """False bei ESS-Kollaps (ess < min_ess) -> Schaetzer NICHT belastbar."""
    min_ess: float

    @property
    def abs_error(self) -> float:
        return abs(self.g_hat - self.oracle)

    @property
    def n_sigma(self) -> float:
        """Abweichung vom Orakel in Fehler-Einheiten.

        error==0 bei abs_error>0 (degenerierte Stichprobe, z.B. alle Spins
        aligned) ist ein MAXIMALER Miss, kein perfekter Treffer -> inf
        (Codex-Review-Fix, spiegelt die gehaerteten QEC-Diagnostiken).
        """
        if self.error > 0:
            return self.abs_error / self.error
        return 0.0 if self.abs_error == 0.0 else float("inf")


def snis_reweight(
    s: np.ndarray,
    *,
    K_proposal: float,
    K_target: float,
    min_ess: float = DEFAULT_MIN_ESS,
) -> SNISEstimate:
    """SNIS-Schaetzung der Per-Bond-Korrelation unter pi_{K_target}.

    Args:
        s: Spin-Stichprobe (N, L) in {+1,-1} aus pi_{K_proposal} (offene Kette).
        K_proposal: Kopplung, bei der gesampelt wurde.
        K_target: Ziel-Kopplung des Reweightings.
        min_ess: ESS-Schwelle; darunter wird ess_adequate=False gemeldet
            (Degenerations-Guard -- Gewichts-Kollaps macht SNIS unbrauchbar).

    Returns:
        SNISEstimate mit Punktschaetzer, Fehler, ESS/chi^2 (+ Orakeln).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"s must be 2D (n_samples, L), got ndim={s.ndim}")
    n, L = s.shape
    if n < 2:
        raise ValueError(f"need >=2 samples, got {n}")
    if L < 2:
        raise ValueError(f"need L>=2, got {L}")
    Kp = _check_coupling(K_proposal, "K_proposal")
    Kt = _check_coupling(K_target, "K_target")
    if not (min_ess > 0):
        raise ValueError(f"min_ess must be > 0, got {min_ess}")
    # Codex-Review-Fix: Spin-Alphabet validieren. {0,1}-Konfigurationen (etwa
    # A-Kernel-Bits) wuerden sonst still als {-1,+1}-Spins interpretiert --
    # alle Gewichte/ESS blieben endlich, aber fuer das FALSCHE Modell.
    if not np.all(np.abs(s) == 1.0):
        raise ValueError("s must contain only spins in {-1, +1} (got other values)")

    S = (s[:, :-1] * s[:, 1:]).sum(axis=1)  # (N,)
    g = S / (L - 1)

    # Log-sum-exp-stabile normierte Gewichte.
    logw = (Kt - Kp) * S
    logw -= logw.max()
    w = np.exp(logw)
    w_sum = float(w.sum())
    w_norm = w / w_sum

    g_hat = float(np.sum(w_norm * g))
    # Delta-Methoden-Varianz des SNIS-Ratios (Owen, "Monte Carlo", Kap. 9):
    # Var(g_hat) ~ sum_i w_i^2 (g_i - g_hat)^2  (w_i normiert).
    var_hat = float(np.sum(w_norm**2 * (g - g_hat) ** 2))
    error = math.sqrt(max(var_hat, 0.0))

    sum_w2 = float(np.sum(w_norm**2))
    ess = 1.0 / sum_w2 if sum_w2 > 0 else float("nan")
    chi2_hat = n * sum_w2 - 1.0
    chi2_oracle = chi2_divergence_open_chain(Kt, Kp, L)

    return SNISEstimate(
        K_proposal=Kp,
        K_target=Kt,
        n_samples=int(n),
        g_hat=g_hat,
        oracle=mean_bond_correlation(Kt),
        error=error,
        ess=float(ess),
        ess_rel=float(ess / n),
        ess_rel_oracle=1.0 / (1.0 + chi2_oracle),
        chi2_hat=float(chi2_hat),
        chi2_oracle=float(chi2_oracle),
        mse_bound=4.0 * (1.0 + chi2_oracle) / n,
        ess_adequate=bool(ess >= min_ess),
        min_ess=float(min_ess),
    )


def snis_from_couplings(
    *,
    K_proposal: float,
    K_target: float,
    L: int,
    n_samples: int,
    seed: int,
    min_ess: float = DEFAULT_MIN_ESS,
) -> SNISEstimate:
    """Bequemer Endpunkt: sample aus pi_{K_p} (exakt i.i.d.) + SNIS nach K_t."""
    s = sample_ising_open_chain(K_proposal, L, n_samples, seed=seed)
    return snis_reweight(s, K_proposal=K_proposal, K_target=K_target, min_ess=min_ess)


@dataclass(frozen=True)
class BiasScalingReport:
    """Empirische Bias-Skalierung von SNIS vs geschlossenes O(1/N)-Orakel."""

    K_proposal: float
    K_target: float
    L: int
    n_values: tuple[int, ...]
    n_replicates: tuple[int, ...]
    bias_hat: tuple[float, ...]
    """Gemessener Bias mean_r(g_hat_r) - tanh(K_t), je N."""
    bias_sem: tuple[float, ...]
    """Standardfehler des gemessenen Bias (ueber Replikate), je N."""
    bias_coefficient_oracle: float
    """Geschlossenes c_bias: bias ~ c_bias/N."""
    scaled_bias: tuple[float, ...]
    """bias_hat * N / c_bias -- muss ~1 sein, wenn die O(1/N)-Ordnung stimmt."""
    mse_hat: tuple[float, ...]
    """Empirische MSE ueber Replikate, je N."""
    mse_bound: tuple[float, ...]
    """Agapiou-Bound 4(1+chi^2)/N, je N."""
    var_deltamethod_mean: tuple[float, ...]
    """Mittlere Delta-Methoden-Varianzschaetzung, je N (Kalibrier-Check)."""


def measure_bias_scaling(
    *,
    K_proposal: float = 0.3,
    K_target: float = 0.6,
    L: int = 16,
    n_values: tuple[int, ...] = (100, 400),
    n_replicates: tuple[int, ...] = (6000, 12000),
    seed: int = 0,
    chunk: int = 250,
) -> BiasScalingReport:
    """Messe den SNIS-Bias ueber Replikate und pruefe die O(1/N)-Ordnung.

    Fuer jedes N: n_replicates unabhaengige SNIS-Schaetzungen (je N Samples);
    Bias = Replikat-Mittel minus geschlossenes Orakel tanh(K_t). Der geschlossene
    Bias-Koeffizient c_bias (Delta-Methode, exponential-family-exakt) macht den
    Test scharf: bias_hat * N / c_bias ~ 1.

    RNG-Vertrag (Codex-Review-Fix): jedes Replikat hat einen EIGENEN,
    replikat-indexierten Seed -- das Ergebnis ist unabhaengig vom
    Speicher-Tuning-Parameter `chunk` (der nur die Batch-Groesse der
    vektorisierten Auswertung steuert, nie die Zufallsstroeme).
    """
    if len(n_values) != len(n_replicates):
        raise ValueError("n_values and n_replicates must have equal length")
    if len(n_values) < 1:
        raise ValueError("need at least one N value")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    Kp = _check_coupling(K_proposal, "K_proposal")
    Kt = _check_coupling(K_target, "K_target")
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")

    oracle_g = mean_bond_correlation(Kt)
    c_bias = snis_bias_coefficient(Kt, Kp, L)
    chi2 = chi2_divergence_open_chain(Kt, Kp, L)

    bias_hat: list[float] = []
    bias_sem: list[float] = []
    scaled: list[float] = []
    mse_hat: list[float] = []
    mse_bound: list[float] = []
    var_dm: list[float] = []

    for j, (N, R) in enumerate(zip(n_values, n_replicates, strict=True)):
        if N < 2 or R < 2:
            raise ValueError(f"need N>=2 and R>=2, got N={N}, R={R}")
        g_hats = np.empty(R, dtype=np.float64)
        var_est = np.empty(R, dtype=np.float64)
        done = 0
        while done < R:
            r = min(chunk, R - done)
            # Ein eigener Seed JE REPLIKAT (nicht je Chunk): Ergebnisse sind
            # damit byte-identisch fuer jede chunk-Wahl (SeedSequence haengt
            # nur vom Replikat-Index ab, nicht von der Batch-Partitionierung).
            s = np.stack(
                [
                    sample_ising_open_chain(Kp, L, N, seed=seed + 1_000_000_007 * j + (done + k))
                    for k in range(r)
                ]
            )  # (r, N, L)
            S = (s[:, :, :-1] * s[:, :, 1:]).sum(axis=2)  # (r, N)
            g = S / (L - 1)
            logw = (Kt - Kp) * S
            logw -= logw.max(axis=1, keepdims=True)
            w = np.exp(logw)
            w /= w.sum(axis=1, keepdims=True)
            gh = np.sum(w * g, axis=1)  # (r,)
            g_hats[done : done + r] = gh
            var_est[done : done + r] = np.sum(w**2 * (g - gh[:, None]) ** 2, axis=1)
            done += r
        b = float(g_hats.mean() - oracle_g)
        sem = float(g_hats.std(ddof=1) / math.sqrt(R))
        bias_hat.append(b)
        bias_sem.append(sem)
        scaled.append(b * N / c_bias if c_bias != 0.0 else float("nan"))
        mse_hat.append(float(np.mean((g_hats - oracle_g) ** 2)))
        mse_bound.append(4.0 * (1.0 + chi2) / N)
        var_dm.append(float(var_est.mean()))

    return BiasScalingReport(
        K_proposal=Kp,
        K_target=Kt,
        L=int(L),
        n_values=tuple(int(n) for n in n_values),
        n_replicates=tuple(int(r) for r in n_replicates),
        bias_hat=tuple(bias_hat),
        bias_sem=tuple(bias_sem),
        bias_coefficient_oracle=float(c_bias),
        scaled_bias=tuple(scaled),
        mse_hat=tuple(mse_hat),
        mse_bound=tuple(mse_bound),
        var_deltamethod_mean=tuple(var_dm),
    )
