"""MCMC-CLT-Varianz sigma^2_g + Konfidenzintervalle (Phase-5).

KONTEXT (Phase-5). Phase-3a (autocorr.py) liefert tau_int (Wolff-Gamma-Methode)
und daraus den autokorr-korrekten Standardfehler des Mittels. Diese Datei macht
die zugrundeliegende CLT-Aussage explizit + liefert einen UNABHAENGIGEN zweiten
Varianz-Schaetzer (Overlapping Batch Means, OBM), damit das Resultat nicht nur
gegen sich selbst geprueft wird.

DIE MCMC-CLT (Markov-Chain Central Limit Theorem)
-------------------------------------------------
Fuer eine stationaere, geometrisch ergodische Kette mit Observable g und
Mittel g_bar_N = (1/N) sum_t g(X_t) gilt

    sqrt(N) * (g_bar_N - E[g])  ->  N(0, sigma^2_g)   (in Verteilung),

mit der ASYMPTOTISCHEN ("long-run") Varianz

    sigma^2_g = Var(g) * (1 + 2 sum_{t>=1} rho_g(t)) = Var(g) * 2 * tau_int .

(Letzte Gleichung mit tau_int = 1/2 + sum_{t>=1} rho(t), der Konvention aus
autocorr.py.) Der Standardfehler des Mittels ist dann

    sem = sqrt(sigma^2_g / N) = sqrt(2 tau_int Var(g) / N) ,

genau der Wert, den autocorr.AutocorrResult.sem liefert. Diese Datei macht
sigma^2_g zur Erststelle (statt nur sem) und stellt zwei unabhaengige
Schaetzer gegenueber.

ZWEI UNABHAENGIGE SCHAETZER FUER sigma^2_g
------------------------------------------
1. Gamma-/tau_int-Methode (ueber autocorr.py): sigma^2_g = 2 tau_int Var(g).
2. Overlapping Batch Means (OBM, Flegal & Jones 2010): mit Batch-Laenge
   b = floor(N^{1/2}) bilde alle N-b+1 ueberlappenden Batch-Mittel Y_k und

       sigma^2_g(OBM) = (N b) / ((N-b)(N-b+1)) * sum_k (Y_k - g_bar_N)^2 .

   OBM ist ein konsistenter Schaetzer der long-run-Varianz und METHODISCH
   UNABHAENGIG von der spektralen/Gamma-Methode (Batch-basiert statt
   Autokorrelations-summen-basiert) -> echter Cross-Check.

UNABHAENGIGES ANALYTISCHES ORAKEL (AR(1))
-----------------------------------------
Ein AR(1)-Prozess x_t = phi x_{t-1} + eps_t (eps ~ N(0, sigma_eps^2)) hat:
  - Marginal-Varianz       Var(x) = sigma_eps^2 / (1 - phi^2),
  - Autokorrelation        rho(t) = phi^t,
  - long-run / CLT-Varianz sigma^2_g = sigma_eps^2 / (1 - phi)^2
                                     = Var(x) * (1 + phi) / (1 - phi).
Die letzte Form ist die "inefficiency factor"-Darstellung (1+phi)/(1-phi);
sie folgt aus 1 + 2 sum_{t>=1} phi^t = 1 + 2 phi/(1-phi) = (1+phi)/(1-phi).
Quelle: Standard-Resultat fuer die long-run-Varianz von AR(1); vgl. MCMC-CLT
(Geyer 1992; Flegal & Jones 2010) und die long-run-Varianz-Identitaet
sigma^2_LR = sigma^2 (1+phi)/(1-phi) (siehe SOURCES.md).

COVERAGE (non-vakuoeser, BEIDSEITIGER Test)
-------------------------------------------
Das nominale (1-alpha)-CI muss den WAHREN Mittelwert ueber viele Replikate mit
Rate ~ (1-alpha) decken (nicht 60 %, nicht 100 %). Ein ABSICHTLICH falsches
sigma^2_g (die iid-Annahme sem_iid = sqrt(Var/N) bei korreliertem Sample)
UNTERdeckt -> Coverage VERFEHLT die Nominalrate. Beide Richtungen sind als
Test kodiert (tests/test_clt.py + Gates G33/G34).

Alles numpy/scipy-only (keine neuen Runtime-Deps).

Referenzen (Methode, NICHT Dependency):
- C. J. Geyer, "Practical Markov Chain Monte Carlo", Statist. Sci. 7 (1992) 473.
- J. M. Flegal, G. L. Jones, "Batch means and spectral variance estimators in
  Markov chain Monte Carlo", Ann. Statist. 38 (2010) 1034 (OBM).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .autocorr import integrated_autocorr_time

__all__ = [
    "CLTResult",
    "clt_variance",
    "obm_variance",
    "confidence_interval",
    "ar1_clt_variance",
]


@dataclass(frozen=True)
class CLTResult:
    """MCMC-CLT-Ergebnis fuer ein Observable g."""

    mean: float
    """Stichprobenmittel g_bar_N."""
    var_marginal: float
    """Marginal-Varianz Var(g) (biased, = gamma(0))."""
    tau_int: float
    """Integrierte Autokorrelationszeit (Gamma-Methode)."""
    n: int
    """Kettenlaenge N."""
    sigma2_g_gamma: float
    """CLT-Varianz via Gamma-Methode: 2 tau_int Var(g)."""
    sigma2_g_obm: float
    """CLT-Varianz via Overlapping Batch Means (unabhaengiger Schaetzer)."""
    sem_gamma: float
    """Standardfehler sqrt(sigma2_g_gamma / N)."""
    sem_obm: float
    """Standardfehler sqrt(sigma2_g_obm / N)."""
    sem_iid: float
    """Naiver i.i.d.-Standardfehler sqrt(Var(g) / N) (zum Vergleich; ABSICHTLICH
    falsch bei korreliertem Sample -> unterdeckt im Coverage-Test)."""

    @property
    def sigma2_g(self) -> float:
        """Default-CLT-Varianz (Gamma-Methode)."""
        return self.sigma2_g_gamma


def obm_variance(x: np.ndarray, *, batch_size: int | None = None) -> float:
    """Overlapping-Batch-Means-Schaetzer der CLT-Varianz sigma^2_g.

    sigma^2_g(OBM) = (N b) / ((N-b)(N-b+1)) * sum_k (Y_k - g_bar_N)^2,
    Y_k = Mittel des k-ten ueberlappenden Batches der Laenge b, k=0..N-b.

    Args:
        x: 1D-Zeitreihe (N,), N >= 4.
        batch_size: Batch-Laenge b (Default floor(sqrt(N)), Flegal-Jones-Wahl).
            Muss 1 <= b < N erfuellen.

    Returns:
        Geschaetzte long-run / CLT-Varianz (Skala von sigma^2_g, NICHT /N).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        raise ValueError(f"need >=4 samples for OBM, got {n}")
    if batch_size is None:
        batch_size = max(1, int(np.floor(np.sqrt(n))))
    b = int(batch_size)
    if not (1 <= b < n):
        raise ValueError(f"batch_size must satisfy 1 <= b < N={n}, got {b}")
    grand_mean = float(x.mean())
    # Vektorisierte ueberlappende Batch-Mittel via Cumsum (O(N)).
    cs = np.concatenate(([0.0], np.cumsum(x)))
    # Y_k = (cs[k+b] - cs[k]) / b, k = 0 .. n-b
    batch_means = (cs[b:] - cs[: n - b + 1]) / b
    ssq = float(np.sum((batch_means - grand_mean) ** 2))
    sigma2 = (n * b) / ((n - b) * (n - b + 1)) * ssq
    return sigma2


def clt_variance(x: np.ndarray, *, c_window: float = 1.5) -> CLTResult:
    """CLT-Varianz sigma^2_g via Gamma-Methode UND OBM (unabhaengiger Cross-Check).

    Args:
        x: 1D-Zeitreihe (N,), N >= 4.
        c_window: Wolff-Parameter S fuer das tau_int-Windowing.

    Returns:
        CLTResult mit beiden Varianz-Schaetzern + SEMs.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        raise ValueError(f"need >=4 samples for a CLT estimate, got {n}")
    ac = integrated_autocorr_time(x, c_window=c_window)
    var_marg = ac.variance
    sigma2_gamma = 2.0 * ac.tau_int * var_marg
    sigma2_obm = obm_variance(x)
    return CLTResult(
        mean=ac.mean,
        var_marginal=var_marg,
        tau_int=ac.tau_int,
        n=n,
        sigma2_g_gamma=float(sigma2_gamma),
        sigma2_g_obm=float(sigma2_obm),
        sem_gamma=float(np.sqrt(sigma2_gamma / n)),
        sem_obm=float(np.sqrt(sigma2_obm / n)),
        sem_iid=float(np.sqrt(var_marg / n)),
    )


def confidence_interval(
    mean: float, sigma2_g: float, n: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Nominales (1-alpha)-CLT-Konfidenzintervall fuer E[g].

    CI = mean +/- z_{1-alpha/2} * sqrt(sigma2_g / N), mit z aus der Normal-
    Quantilfunktion (CLT-Limes). Gibt (low, high) zurueck.

    Args:
        mean: Stichprobenmittel.
        sigma2_g: CLT-Varianz (long-run, NICHT /N).
        n: Kettenlaenge N.
        alpha: Signifikanzniveau (Default 0.05 -> 95 %-CI).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if sigma2_g < 0.0:
        raise ValueError(f"sigma2_g must be >= 0, got {sigma2_g}")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    half = z * np.sqrt(sigma2_g / n)
    return (mean - half, mean + half)


def ar1_clt_variance(phi: float, sigma_eps: float = 1.0) -> dict[str, float]:
    """Analytisches AR(1)-Orakel: long-run/CLT-Varianz in geschlossener Form.

    Var(x)    = sigma_eps^2 / (1 - phi^2)            (Marginal-Varianz)
    sigma^2_g = sigma_eps^2 / (1 - phi)^2            (CLT-Varianz)
              = Var(x) * (1 + phi) / (1 - phi)
    tau_int   = 1/2 + phi/(1 - phi) = (1+phi)/(2(1-phi))

    Args:
        phi: AR(1)-Koeffizient, |phi| < 1.
        sigma_eps: Innovations-Standardabweichung.

    Returns:
        dict mit var_marginal, sigma2_g, tau_int (alle exakt).
    """
    if not (-1.0 < phi < 1.0):
        raise ValueError(f"AR(1) requires |phi| < 1, got {phi}")
    if sigma_eps <= 0.0:
        raise ValueError(f"sigma_eps must be > 0, got {sigma_eps}")
    var_marg = sigma_eps**2 / (1.0 - phi**2)
    sigma2_g = sigma_eps**2 / (1.0 - phi) ** 2
    tau_int = (1.0 + phi) / (2.0 * (1.0 - phi))
    return {"var_marginal": var_marg, "sigma2_g": sigma2_g, "tau_int": tau_int}
