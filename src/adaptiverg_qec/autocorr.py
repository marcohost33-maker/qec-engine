"""Autokorrelations-bewusste Fehler-Statistik fuer korrelierte MCMC-Ketten.

KONTEXT (Phase-3a). Der A-Kernel (a_kernel.py) erzeugt eine *korrelierte*
Markov-Kette: aufeinanderfolgende Samples sind nicht unabhaengig. Naive
i.i.d.-Fehlerbalken (sigma = std/sqrt(N)) UNTERSCHAETZEN den wahren Fehler,
weil die effektive Stichprobe N_eff = N/(2*tau_int) < N ist (korrelierte
Samples tragen weniger Information). Diese Datei implementiert die korrekte
Behandlung.

METHODE (Wolff, "Monte Carlo errors with less errors", hep-lat/0306017):
- Normierte Autokorrelationsfunktion  rho_O(t) = [<O_i O_{i+t}> - <O>^2] / Var(O).
  Berechnet via FFT (Wiener-Khinchin-Theorem) -> O(N log N) statt O(N^2).
- Integrierte Autokorrelationszeit  tau_int(O) = 1/2 + sum_{t=1}^{W} rho_O(t).
- Automatic windowing (Wolff g-Funktion): waehle das Summen-Fenster W als
  erstes W, bei dem g(W) = exp(-W/tau_W(W)) - tau_W(W)/sqrt(W*N) < 0, mit der
  exponentiellen Autokorr-Zeit tau_W(W) = S / ln((2 tau_int(W)+1)/(2 tau_int(W)-1)),
  S ~ 1.5. Das ist Wolffs rigoroses Kriterium (balanciert statistischen Fehler
  ~ tau_W/sqrt(W N) gegen systematischen Trunkierungs-Bias ~ exp(-W/tau_W));
  die naive Variante "kleinstes W mit W >= S tau(W)" unterschaetzt tau_int bei
  stark korrelierten Reihen deutlich (empirisch bestaetigt gegen AR(1)-Orakel).
- Fehler des Mittels:  sigma^2(<O>) = (2 * tau_int / N) * Var(O).
  effektive Stichprobe  N_eff = N / (2 * tau_int).

CROSS-CHECK (Pflicht, rigorose Validierung):
- Binning/Blocking: Kette in Bloecke der Groesse b; Block-Mittel werden fuer
  b >> tau_int unabhaengig; der Blocking-Fehler-Schaetzer plateaut mit b.
  Gamma-Methode (tau_int) und Binning muessen im Plateau UEBEREINSTIMMEN.

VERHAELTNIS-SCHAETZER (T_hat = <S'S>_c / <S'S'>_c):
- Jackknife ueber Bloecke auf das gesamte Verhaeltnis -> propagiert die Fehler
  der (korrelierten) Zaehler/Nenner korrekt, ohne naive Gauss-Propagation.

Alles numpy-only (keine neuen Runtime-Deps). Vektorisiert.

Referenzen (Methode, NICHT Dependency):
- U. Wolff, Comput. Phys. Commun. 156 (2004) 143, hep-lat/0306017.
- pyerrors (arXiv:2209.14371) als unabhaengige Methoden-Referenz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "autocorr_function_fft",
    "integrated_autocorr_time",
    "AutocorrResult",
    "mean_with_autocorr_error",
    "binning_error",
    "BinningResult",
    "jackknife_ratio",
    "JackknifeRatioResult",
]


def autocorr_function_fft(x: np.ndarray) -> np.ndarray:
    """Normierte Autokorrelationsfunktion rho(t), t=0..N-1, via FFT.

    rho(t) = [<x_i x_{i+t}> - <x>^2] / Var(x), mit rho(0) = 1.

    Wiener-Khinchin: die Autokovarianz ist die inverse FFT des Leistungs-
    spektrums der zentrierten, null-gepaddeten Reihe. O(N log N).

    Args:
        x: 1D-Zeitreihe eines Observablen (N,), N >= 2.

    Returns:
        rho: (N,) mit rho[0] == 1.0. Fuer eine konstante Reihe (Var=0) wird
        ein Fehler geworfen (keine definierte Autokorrelation).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        raise ValueError(f"need >=2 samples for an autocorrelation, got {n}")
    xc = x - x.mean()
    var0 = float(np.dot(xc, xc) / n)  # = Var(x) (biased, = gamma(0))
    if var0 <= 0.0:
        raise ValueError("constant series: Var(x)=0, autocorrelation undefined")
    # Null-Padding auf >= 2N-1 (verhindert zyklische Faltung); naechste 2er-Potenz.
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, n=nfft)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    # acov[t] = sum_{i} xc_i xc_{i+t}; unbiased-in-Anzahl: durch (n - t) teilen
    # waere der Standard, ABER fuer tau_int ist die 1/n-Normierung (gamma(t)=acov/n)
    # stabiler bei grossem t (Wolff verwendet die 1/n-Form implizit). rho = gamma/gamma(0).
    gamma = acov / n
    rho = gamma / gamma[0]
    return rho


@dataclass(frozen=True)
class AutocorrResult:
    """Ergebnis der Gamma-Methode fuer ein Observable."""

    mean: float
    """Stichprobenmittel <O>."""
    variance: float
    """Stichprobenvarianz Var(O) (biased, = gamma(0))."""
    tau_int: float
    """Integrierte Autokorrelationszeit tau_int (>= 0.5)."""
    window: int
    """Gewaehltes Summen-Fenster W (Wolff automatic windowing)."""
    n: int
    """Kettenlaenge N."""
    n_eff: float
    """Effektive Stichprobe N_eff = N / (2 tau_int)."""
    sem: float
    """Autokorr-korrekter Standardfehler des Mittels sqrt(2 tau_int Var / N)."""
    sem_iid: float
    """Naiver i.i.d.-Standardfehler sqrt(Var / N) (zum Vergleich)."""

    @property
    def inflation(self) -> float:
        """Faktor, um den der korrelierte Fehler den i.i.d.-Fehler uebersteigt.

        = sem / sem_iid = sqrt(2 tau_int). >1 gdw tau_int > 0.5.
        """
        return self.sem / self.sem_iid if self.sem_iid > 0 else float("nan")


def integrated_autocorr_time(
    x: np.ndarray, *, c_window: float = 1.5, rho: np.ndarray | None = None
) -> AutocorrResult:
    """tau_int via Gamma-Methode + Wolff automatic windowing (g-Funktion).

    tau_int(W) = 1/2 + sum_{t=1}^{W} rho(t). Das Summen-Fenster W wird per
    Wolffs g-Funktion bestimmt (hep-lat/0306017): mit der exponentiellen
    Autokorr-Zeit tau_W(W) = S / ln((2 tau+1)/(2 tau-1)) stoppe beim ersten
    W>=1 mit g(W) = exp(-W/tau_W) - tau_W/sqrt(W*N) < 0. Das balanciert
    statistischen Fehler gegen systematischen Trunkierungs-Bias.
    Fuer eine i.i.d.-Reihe (rho(t)~0 fuer t>=1) ergibt sich tau_int = 0.5 und
    ein kleines W, d.h. sem = sem_iid (kein Aufbloehen) -- der korrekte Grenzfall.

    Args:
        x: 1D-Zeitreihe (N,).
        c_window: Wolff-Parameter S (Standard 1.5).
        rho: optional vorberechnetes rho (spart eine FFT).

    Returns:
        AutocorrResult mit tau_int, Fenster, N_eff, korreliertem + i.i.d.-Fehler.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if c_window <= 0:
        raise ValueError(f"c_window must be > 0, got {c_window}")
    if rho is None:
        rho = autocorr_function_fft(x)
    # Kumulative tau_int(W) = 0.5 + cumsum(rho[1:]); tau_of_w[k] = tau_int(W=k+1).
    tau_of_w = 0.5 + np.cumsum(rho[1:])
    n_w = tau_of_w.size
    window = n_w  # Fallback: ganzes Fenster (konservativ), falls g nie < 0 wird.
    tau_int = float(max(tau_of_w[-1], 0.5))
    for k in range(n_w):
        w = k + 1  # W = 1, 2, ...
        tau = max(float(tau_of_w[k]), 0.5)
        if tau <= 0.5:
            # Praktisch unkorreliert -> tau_W -> 0, g sofort negativ; stoppe.
            window, tau_int = w, tau
            break
        # tau_W aus tau_int (Wolff Gl. fuer einen einzelnen Exponential-Mode).
        tau_w = c_window / np.log((2.0 * tau + 1.0) / (2.0 * tau - 1.0))
        g = np.exp(-w / tau_w) - tau_w / np.sqrt(w * n)
        if g < 0.0:
            window, tau_int = w, tau
            break
    # EHRLICHE KONVENTION (Audit P0-5): tau_int wird bei 0.5 GEKLEMMT. Fuer
    # positiv korrelierte MCMC-Reihen (der Zweck dieses Moduls) ist das die
    # korrekte i.i.d.-Untergrenze; fuer ANTI-korrelierte Reihen (tau_int < 0.5)
    # ist der zurueckgegebene Fehler dadurch bewusst KONSERVATIV (zu gross),
    # nie zu klein. Konsequenz: n_eff <= n gilt per Konstruktion.
    tau_int = max(tau_int, 0.5)

    variance = float(np.var(x))  # = gamma(0) (biased), konsistent mit rho-Normierung
    sem_iid = float(np.sqrt(variance / n)) if n > 0 else float("nan")
    sem = float(np.sqrt(2.0 * tau_int * variance / n)) if n > 0 else float("nan")
    n_eff = n / (2.0 * tau_int)
    return AutocorrResult(
        mean=float(x.mean()),
        variance=variance,
        tau_int=tau_int,
        window=window,
        n=n,
        n_eff=float(n_eff),
        sem=sem,
        sem_iid=sem_iid,
    )


def mean_with_autocorr_error(x: np.ndarray, *, c_window: float = 1.5) -> AutocorrResult:
    """Bequemer Alias: Mittel + autokorr-korrekter Fehler in einem Aufruf."""
    return integrated_autocorr_time(x, c_window=c_window)


@dataclass(frozen=True)
class BinningResult:
    """Ergebnis der Binning/Blocking-Fehlerschaetzung."""

    block_sizes: np.ndarray
    """Geprueftes b (Block-Groessen)."""
    sem_of_b: np.ndarray
    """Standardfehler des Mittels als Funktion von b (plateaut fuer b>>tau)."""
    sem_plateau: float
    """Plateau-Schaetzer (Maximum bzw. konvergiertes sem)."""
    tau_int_equiv: float
    """Aequivalentes tau_int aus dem Plateau: 0.5*(sem_plateau/sem_iid)^2."""
    sem_iid: float
    """i.i.d.-Referenz sqrt(Var/N)."""


def binning_error(x: np.ndarray, *, max_block: int | None = None) -> BinningResult:
    """Binning/Blocking-Fehler-Schaetzer (unabhaengiger Cross-Check zur Gamma-Methode).

    Teile die Kette in floor(N/b) nicht-ueberlappende Bloecke der Groesse b,
    bilde die Block-Mittel; deren Standardfehler des Mittels schaetzt den wahren
    Fehler von <O>. Fuer b >> tau_int werden die Block-Mittel unabhaengig und der
    Schaetzer plateaut. Das Plateau muss mit dem Gamma-sem uebereinstimmen.

    sem(b) = std(block_means, ddof=1) / sqrt(n_blocks).

    Args:
        x: 1D-Zeitreihe (N,).
        max_block: groesstes b (Default n // 32, damit >= 32 Bloecke bleiben --
            grosse b mit wenigen Bloecken liefern verrauschte sem-Schaetzer und
            verfaelschen das Plateau; das Plateau wird ohnehin schon bei
            b >~ wenige*tau_int erreicht).

    Returns:
        BinningResult mit sem(b)-Kurve + Plateau + aequivalentem tau_int.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 64:
        raise ValueError(f"need >=64 samples for a meaningful binning curve, got {n}")
    if max_block is None:
        max_block = max(2, n // 32)
    if max_block < 1:
        raise ValueError(f"max_block must be >= 1, got {max_block}")
    var = float(np.var(x))
    sem_iid = float(np.sqrt(var / n))
    bs: list[int] = []
    sems: list[float] = []
    for b in range(1, max_block + 1):
        n_blocks = n // b
        if n_blocks < 2:
            break
        # Vektorisiert: trimmen + reshapen + Block-Mittel.
        trimmed = x[: n_blocks * b].reshape(n_blocks, b)
        block_means = trimmed.mean(axis=1)
        sem_b = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
        bs.append(b)
        sems.append(sem_b)
    if not sems:
        raise ValueError("not enough data to form >=2 blocks at any block size")
    block_sizes = np.asarray(bs, dtype=np.int64)
    sem_of_b = np.asarray(sems, dtype=np.float64)
    # Plateau-Schaetzer: das Plateau liegt im Bereich b >~ wenige*tau_int, wo die
    # sem(b)-Kurve flach ist. Median ueber die obere Haelfte der b-Werte (robust
    # gegen das Endpunkt-Rauschen, das bei sehr grossem b / wenigen Bloecken
    # auftritt; deshalb deckelt max_block die Block-Anzahl auf >= 32).
    upper = sem_of_b[len(sem_of_b) // 2 :]
    sem_plateau = float(np.median(upper)) if upper.size else float(sem_of_b[-1])
    tau_int_equiv = 0.5 * (sem_plateau / sem_iid) ** 2 if sem_iid > 0 else float("nan")
    return BinningResult(
        block_sizes=block_sizes,
        sem_of_b=sem_of_b,
        sem_plateau=sem_plateau,
        tau_int_equiv=float(tau_int_equiv),
        sem_iid=sem_iid,
    )


@dataclass(frozen=True)
class JackknifeRatioResult:
    """Jackknife-Schaetzung fuer ein Verhaeltnis von (korrelierten) Korrelatoren."""

    ratio: float
    """Voll-Stichproben-Verhaeltnis num/den (Punktschaetzer)."""
    ratio_jackknife: float
    """Bias-korrigierter Jackknife-Mittelwert der Block-loeschungen."""
    error: float
    """Jackknife-Standardfehler des Verhaeltnisses (autokorr-bewusst via Bloecke)."""
    bias: float
    """Geschaetzter Bias (ratio_jackknife - ratio)."""
    n_blocks: int
    """Anzahl Jackknife-Bloecke."""
    block_size: int
    """Block-Groesse b (sollte >~ tau_int sein -> unabhaengige Bloecke)."""


def jackknife_ratio(
    num_terms: np.ndarray,
    den_terms: np.ndarray,
    *,
    block_size: int,
    combine,
) -> JackknifeRatioResult:
    """Block-Jackknife-Fehler eines Verhaeltnisses von Korrelator-Mittelwerten.

    Sauberer als naive Gauss-Propagation bei korrelierten Zaehler/Nenner: der
    Schaetzer ist ein nichtlinearer Funktional der Sample-Mittel, und Block-
    Jackknife respektiert (a) die Nichtlinearitaet und (b) die Autokorrelation
    (Block-Groesse b >~ tau_int -> Bloecke quasi-unabhaengig).

    Args:
        num_terms: pro-Sample-Beitraege zum ZAEHLER, Shape (N, k_num). Das
            Verhaeltnis ist combine(mean over kept samples) -- z.B. eine
            connected correlation <AB>_c = <AB> - <A><B> aus mehreren Spalten.
        den_terms: pro-Sample-Beitraege zum NENNER, Shape (N, k_den).
        block_size: b; die Kette wird in floor(N/b) Bloecke geteilt, je einer
            wird geloescht. b sollte >~ tau_int sein.
        combine: callable(num_means: (k_num,), den_means: (k_den,)) -> float,
            das aus den Spalten-Mitteln das Verhaeltnis bildet (Nichtlinearitaet
            steckt hier). Wird auf das Voll-Sample und jede Loeschung angewandt.

    Returns:
        JackknifeRatioResult.
    """
    num_terms = np.asarray(num_terms, dtype=np.float64)
    den_terms = np.asarray(den_terms, dtype=np.float64)
    if num_terms.ndim != 2 or den_terms.ndim != 2:
        raise ValueError("num_terms and den_terms must be 2D (N, k)")
    n = num_terms.shape[0]
    if den_terms.shape[0] != n:
        raise ValueError(f"num/den sample count mismatch: {n} vs {den_terms.shape[0]}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    n_blocks = n // block_size
    if n_blocks < 2:
        raise ValueError(f"need >=2 blocks; N={n}, block_size={block_size} -> {n_blocks} blocks")
    keep = n_blocks * block_size
    num = num_terms[:keep]
    den = den_terms[:keep]

    full = float(combine(num.mean(axis=0), den.mean(axis=0)))

    # Vektorisierte Block-Loeschung ueber Block-Summen (delete-one-block).
    num_b = num.reshape(n_blocks, block_size, num.shape[1]).sum(axis=1)  # (n_blocks, k_num)
    den_b = den.reshape(n_blocks, block_size, den.shape[1]).sum(axis=1)
    num_total = num_b.sum(axis=0)
    den_total = den_b.sum(axis=0)
    denom_count = keep - block_size
    jack = np.empty(n_blocks, dtype=np.float64)
    for j in range(n_blocks):
        num_mean_j = (num_total - num_b[j]) / denom_count
        den_mean_j = (den_total - den_b[j]) / denom_count
        jack[j] = combine(num_mean_j, den_mean_j)

    jack_mean = float(jack.mean())
    # Jackknife-Varianz: (n_blocks-1)/n_blocks * sum (jack_j - jack_mean)^2.
    error = float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((jack - jack_mean) ** 2)))
    # Bias-korrigierter Schaetzer: n*full - (n-1)*jack_mean (Standard-Jackknife).
    bias_corrected = n_blocks * full - (n_blocks - 1) * jack_mean
    bias = bias_corrected - full
    return JackknifeRatioResult(
        ratio=full,
        ratio_jackknife=float(bias_corrected),
        error=error,
        bias=float(bias),
        n_blocks=int(n_blocks),
        block_size=int(block_size),
    )
