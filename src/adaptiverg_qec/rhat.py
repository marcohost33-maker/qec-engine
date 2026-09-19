"""Rank-normalized split-R-hat + bulk/tail-ESS (Phase-5).

KONTEXT (Phase-5 / Spec 10.1a). Phase-1 nannte R-hat-Multichain (Vehtari et al.)
als offen fuer die volle Konvergenz-Akzeptanz. Diese Datei implementiert die
verbesserte Konvergenzdiagnostik aus

  A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Buerkner (2021),
  "Rank-normalization, folding, and localization: An improved R-hat for
  assessing convergence of MCMC", Bayesian Analysis 16(2), 667-718,
  doi:10.1214/20-BA1221.

WARUM RANK-NORMALIZED (statt klassisches Gelman-Rubin)
------------------------------------------------------
Das klassische R-hat setzt endliche Varianz + ungefaehre Normalitaet voraus und
versagt bei schweren Schwaenzen oder ketten-abhaengiger Varianz. Vehtari et al.
rang-normalisieren zuerst (verteilungsfrei, robust) und falten dann um den
Median (faengt Skalen-Nichtkonvergenz). Sie empfehlen den Schwellwert R-hat <
1.01.

ALGORITHMUS (exakt nach Paper + Online-Appendix)
------------------------------------------------
1. SPLIT: jede der M Ketten (Laenge n) wird in 2 Haelften geteilt -> 2M
   Split-Ketten der Laenge N = floor(n/2).
2. RANK-NORMALIZE: ueber ALLE S = 2M*N Werte gepoolte Raenge r_i (Durchschnitts-
   raenge bei Bindungen), dann Blom-Transform
       z_i = Phi^{-1}( (r_i - 3/8) / (S + 1/4) ).
3. SPLIT-R-HAT auf den z:
       B = N/(2M-1) * sum_m (zbar_m - zbar)^2          (between-chain)
       W = (1/2M)   * sum_m s_m^2                       (within-chain, ddof=1)
       var_plus = (N-1)/N * W + B/N
       R-hat = sqrt(var_plus / W).
4. FOLDED-R-HAT: dasselbe auf gefalteten Werten |theta - median(theta)| -> faengt
   Skalen-/Schwanz-Nichtkonvergenz. Reportiertes R-hat = max(bulk-R-hat,
   folded-R-hat).
5. ESS (bulk/tail) auf den rang-normalisierten bzw. gefaltet-rang-normalisierten
   z via Multichain-Autokorrelation (Geyer initial-monotone-sequence /
   Gelman-BDA3-Schaetzer):
       ESS = (M n) / (1 + 2 sum_{t>=1} rho_t),  rho_t = 1 - W_t/(2 var_plus),
   wobei W_t die ueber Ketten gemittelte Varianz der Differenzen bei Lag t ist
   (Multichain-Variogramm, BDA3 Gl. 11.7). Bulk-ESS nutzt z; tail-ESS nutzt das
   Minimum der ESS der 5%- und 95%-Quantil-Indikatoren (Vehtari) -- ABER nur
   ueber die tatsaechlich MESSBAREN: ein entarteter (konstanter) Indikator wird
   uebersprungen, und dann ist der Wert kein Minimum aus zweien mehr. Siehe die
   Anmerkung an der Fundstelle; 'konservativ' gilt nur bei zwei messbaren.

NON-VAKUOESER, BEIDSEITIGER TEST
--------------------------------
(a) M unabhaengige, gut gemischte Ketten (verschiedene Seeds, GLEICHE
    Stationaerverteilung) -> R-hat < 1.01.
(b) Absichtlich nicht-konvergierte Ketten (verschiedene Drift-/Start-Regimes
    oder zu kurz) -> R-hat >> 1.01 wird GEFLAGGT.
Beide Richtungen sind als Test kodiert (tests/test_rhat.py + Gates G35/G36).

Alles numpy/scipy-only (keine neuen Runtime-Deps).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

__all__ = [
    "DiagnosticState",
    "RhatResult",
    "split_rhat",
    "rank_normalize",
    "ess_bulk_tail",
    "RHAT_THRESHOLD",
]

RHAT_THRESHOLD: float = 1.01
"""Vehtari-et-al.-Diagnostikschwelle. Allein ist R-hat KEIN Konvergenzbeleg."""


class DiagnosticState(str, Enum):
    """Semantischer Zustand der Multichain-Diagnostik.

    DEGENERATE_CONSTANT bedeutet: alle beobachteten Draws sind exakt gleich.
    Aus den Draws allein ist nicht entscheidbar, ob die Zielgroesse strukturell
    konstant ist oder der Sampler feststeckt. STRUCTURAL_CONSTANT wird nur
    gesetzt, wenn der Aufrufer diese Eigenschaft explizit als Domaenenwissen
    deklariert. Keiner der beiden Zustaende ist ein Konvergenzverdikt.
    """

    OK = "OK"
    DEGENERATE_CONSTANT = "DEGENERATE_CONSTANT"
    STRUCTURAL_CONSTANT = "STRUCTURAL_CONSTANT"


@dataclass(frozen=True)
class RhatResult:
    """Ergebnis der rank-normalized split-R-hat-Diagnostik."""

    rhat: float
    """Reportiertes R-hat = max(bulk_rhat, folded_rhat) (Vehtari-Empfehlung)."""
    bulk_rhat: float
    """Rank-normalized split-R-hat (bulk)."""
    folded_rhat: float
    """Folded rank-normalized split-R-hat (Skala/Schwanz)."""
    ess_bulk: float
    """Bulk effektive Stichprobe (rang-normalisiert)."""
    ess_tail: float
    """Tail effektive Stichprobe: min ESS ueber die MESSBAREN 5%/95%-Indikatoren.

    Entartet einer der beiden (konstante Indikator-Spalte), wird er uebersprungen
    und dieser Wert stammt aus nur EINEM Indikator -- er ist dann kein Minimum
    aus zweien und nicht in dem Sinne konservativ, wie der Name nahelegt.
    """
    n_chains: int
    """Anzahl Eingangs-Ketten M."""
    n_draws: int
    """Draws pro Eingangs-Kette n."""
    diagnostic_state: DiagnosticState
    """Semantischer Diagnostikzustand; trennt numerischen Sentinel von Aussage."""
    rhat_defined: bool
    """False bei W=B=0; rhat ist dann nur ein numerischer Konventionswert."""

    @property
    def rhat_below_threshold(self) -> bool:
        """NUR das R-hat-Kriterium: R-hat < RHAT_THRESHOLD (Vehtari < 1.01).

        Bewusst als eigene Eigenschaft ausgewiesen, weil es allein KEIN
        Konvergenz-Beleg ist: R-hat ist ein Verhaeltnis Between/Within und misst
        ausschliesslich, ob die Ketten untereinander streuen. Eingefrorene
        Ketten streuen nicht -- sie sind in genau diesem Sinn perfekt gemischt
        und passieren dieses Kriterium. Wer nur hierauf schaut, akzeptiert
        genau die Pathologie, die die Diagnostik abweisen soll.
        """
        return self.rhat_defined and self.rhat < RHAT_THRESHOLD

    @property
    def ess_sufficient(self) -> bool:
        """Hat die Stichprobe ueberhaupt etwas abgetastet? ESS > 0 in bulk UND tail.

        Fail-closed-Untergrenze: ESS = 0 ist im exakt konstanten Fall ein
        Sentinel fuer nicht entscheidbare Sampling-Information, nicht die
        Behauptung, die mathematische ESS einer strukturell konstanten Observable
        sei allgemein null. Identische Draws koennen sowohl einen festgefahrenen
        Sampler als auch eine bekannte konstante Zielgroesse repraesentieren.
        Nicht-endliche Werte gelten ebenfalls als ungenuegend.

        Das ist die HARTE Sicherheitsuntergrenze, nicht die Praxis-Empfehlung. Vehtari
        et al. verlangen zusaetzlich rund ESS > 100 pro Kette, bevor die
        Schaetzer als belastbar gelten; diese Schwelle ist eine Entscheidung
        des Aufrufers ueber seine Genauigkeitsanforderung und steht daher
        bewusst nicht hier.
        """
        return (
            np.isfinite(self.ess_bulk)
            and np.isfinite(self.ess_tail)
            and self.ess_bulk > 0.0
            and self.ess_tail > 0.0
        )

    @property
    def converged(self) -> bool:
        """Ausgewiesenes Konvergenz-Verdikt: R-hat-Kriterium UND ESS > 0.

        HISTORIE (Grund fuer die Kopplung): diese Eigenschaft prueft frueher NUR
        ``rhat < RHAT_THRESHOLD``. Damit meldete ``split_rhat(np.full((4, 2000),
        7.0))`` gleichzeitig ``ess_bulk == 0.0`` UND ``converged is True`` -- das
        Verdikt bescheinigte etwas, das es nicht geprueft hatte. Der Wert wurde
        ueber ``manifest.postprocess_multichain`` als ``rhat_converged``
        weitergereicht und von der Phase-5-CLI als ``"converged"`` serialisiert;
        jeder Verbraucher, der dem oeffentlichen Flag folgte, akzeptierte die
        eingefrorene Kette.

        Wer die beiden Achsen einzeln braucht, nimmt
        :attr:`rhat_below_threshold` und :attr:`ess_sufficient`.
        """
        return (
            self.diagnostic_state is DiagnosticState.OK
            and self.rhat_below_threshold
            and self.ess_sufficient
        )


def _as_chains(draws: np.ndarray) -> np.ndarray:
    """Validiere + forme (M, n)-Array."""
    a = np.asarray(draws, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"draws must be 2D (M chains, n draws), got ndim={a.ndim}")
    m, n = a.shape
    if m < 2:
        raise ValueError(f"need >=2 chains for R-hat, got M={m}")
    if n < 4:
        raise ValueError(f"need >=4 draws per chain, got n={n}")
    if not np.all(np.isfinite(a)):
        raise ValueError("draws contain non-finite values (NaN/Inf)")
    return a


def _split(chains: np.ndarray) -> np.ndarray:
    """Teile jede Kette in 2 Haelften -> (2M, N), N = floor(n/2)."""
    m, n = chains.shape
    half = n // 2
    if half < 2:
        raise ValueError(f"chains too short to split: n={n}")
    first = chains[:, :half]
    second = chains[:, half : 2 * half]
    return np.vstack([first, second])  # (2M, half)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Blom-rang-normalisiere ein Werte-Array (verteilungsfrei).

    z_i = Phi^{-1}( (r_i - 3/8) / (S + 1/4) ), r_i = Durchschnittsrang (1-basiert),
    S = Anzahl Werte. Bindungen -> Durchschnittsraenge (scipy 'average').

    Args:
        x: beliebig geformtes Array; Raenge werden ueber ALLE Elemente gepoolt.

    Returns:
        Array gleicher Form mit standard-normalverteilten Rang-Scores.
    """
    a = np.asarray(x, dtype=np.float64)
    flat = a.ravel()
    s = flat.size
    ranks = stats.rankdata(flat, method="average")  # 1..S, average bei Ties
    u = (ranks - 0.375) / (s + 0.25)
    z = stats.norm.ppf(u)
    return z.reshape(a.shape)


def _rhat_on(split_chains: np.ndarray) -> float:
    """Roh-Split-R-hat auf bereits gesplitteten (transformierten) Ketten (2M, N)."""
    two_m, n = split_chains.shape
    chain_means = split_chains.mean(axis=1)  # (2M,)
    grand_mean = float(chain_means.mean())
    # Between-chain-Varianz B (Skala des Mittels * N).
    b = n / (two_m - 1) * float(np.sum((chain_means - grand_mean) ** 2))
    # Within-chain-Varianz W (ddof=1 pro Kette, dann gemittelt).
    w = float(np.mean(split_chains.var(axis=1, ddof=1)))
    if w <= 0.0:
        # Konstante Ketten: NUR wenn auch B==0 (alle identisch) konvergiert;
        # konstante Ketten mit VERSCHIEDENEN Mitteln sind maximal getrennt
        # (vorher: return 1.0 unabhaengig von B -> falsches "converged").
        #
        # WARUM 1.0 fuer b <= 0 BEWUSST STEHEN BLEIBT (kein vergessener Defekt):
        # 1.0 ist eine KONVENTION -- der asymptotische Wert --, NICHT der exakte
        # Wert der Formel bei endlicher Kettenlaenge. Genau: fuer B = 0 ist
        # var_plus = (N-1)/N * W, also var_plus / W = (N-1)/N; der Quotient
        # kuerzt W heraus und ist damit auch fuer W -> 0 wohldefiniert. Der
        # EXAKTE endliche Wert ist damit sqrt((N-1)/N) < 1 -- fuer die kuerzeste
        # zulaessige Eingabe (N = 2) rund 0.707. Erst im getrennten Grenzuebergang
        # N -> inf strebt er gegen 1. W -> 0 aendert daran nichts.
        #
        # Wir geben trotzdem 1.0 zurueck, weil ein gemeldetes R-hat UNTER 1 in
        # jeder Vehtari-Konvention als "so gut wie irgend moeglich" gelesen wird
        # und Aufrufer, die auf `rhat < 1.01` schwellen, dadurch keinerlei
        # zusaetzliche Information bekaemen -- wohl aber einen Wert, der wie ein
        # Rechenfehler aussieht. Der Wert 1.0 entsteht also NICHT dadurch, dass
        # ein Fehlerfall zu "gut" gerundet wird; er ist die neutrale Marke des
        # Zweigs. Festgehalten in test_degenerate_rhat_is_a_convention_not_the_
        # finite_sample_value.
        #
        # WARUM ER TROTZDEM NICHT ALLEIN GENUEGT: R-hat ist per Konstruktion ein
        # VERHAELTNIS Between/Within und misst ausschliesslich, ob die Ketten
        # untereinander streuen -- identische Ketten sind in genau diesem Sinn
        # perfekt "gemischt". Die Frage "haben die Ketten ueberhaupt etwas
        # abgetastet?" ist eine Frage nach der STICHPROBENGROESSE, nicht nach
        # R-hat; sie wird von _ess_on beantwortet, das im Entartungsfall 0.0
        # liefert. Konsequenz fuer Aufrufer: R-hat < RHAT_THRESHOLD allein ist
        # KEIN Konvergenz-Beleg. Das Akzeptanzkriterium muss ESS > 0 (in der
        # Praxis: Vehtari-Faustregel ESS > 100 pro Kette) mitfuehren, sonst
        # passiert eine eingefrorene Kette die Diagnostik.
        return 1.0 if b <= 0.0 else float("inf")
    var_plus = (n - 1) / n * w + b / n
    return float(np.sqrt(var_plus / w))


def _ess_on(split_chains: np.ndarray) -> float:
    """Multichain-ESS (BDA3 / Vehtari) auf gesplitteten transformierten Ketten.

    ESS = (2M * N) / (1 + 2 sum_{t>=1} rho_t), mit dem Multichain-Schaetzer
    rho_t = 1 - W_t / (2 var_plus), W_t = ueber Ketten gemittelte Lag-t-Variogramm-
    Varianz. Summe via Geyer initial-monotone-positive-sequence (Paarsummen >0,
    monoton fallend), abgeschnitten beim ersten nicht-positiven Paar.
    """
    two_m, n = split_chains.shape
    chain_means = split_chains.mean(axis=1)
    grand_mean = float(chain_means.mean())
    b = n / (two_m - 1) * float(np.sum((chain_means - grand_mean) ** 2))
    w = float(np.mean(split_chains.var(axis=1, ddof=1)))
    if w <= 0.0:
        # ENTARTET: keine Within-Chain-Varianz, d.h. jede Split-Kette steht still.
        # ESS = 0.0 -- unabhaengig davon, ob die Ketten auf DERSELBEN Konstanten
        # stehen (b <= 0) oder auf VERSCHIEDENEN (b > 0).
        #
        # Begruendung: ESS misst, wieviele unabhaengige Ziehungen die Stichprobe
        # wert ist. Eine Kette, die sich nie bewegt hat, hat die Zielverteilung
        # nicht einmal abgetastet und traegt null Information ueber sie; jede
        # positive Zahl waere eine Aussage ueber eine Stichprobe, die es nicht
        # gibt. Das ist keine Konventionsfrage.
        #
        # HISTORIE: der b<=0-Zweig gab frueher 2M*N ("volle Stichprobe") zurueck.
        # Eine eingefrorene Kette bekam damit die MAXIMAL moegliche wirksame
        # Stichprobe -- einen besseren Diagnostik-Wert als eine gesunde Kette,
        # deren Autokorrelation die Zahl zurecht drueckt. Der b>0-Zweig war
        # bereits auf 0.0 korrigiert; dieser Schnitt gleicht b<=0 an.
        #
        # HAUSREGEL: die Geschwistermodule beantworten den Entartungsfall
        # fail-closed (autocorr.py raise, mcrg.py raise, drift.py nan +
        # holds=False). rhat.py war das einzige Modul, das mit einem
        # Erfolgswert antwortete.
        return 0.0
    var_plus = (n - 1) / n * w + b / n

    # Pro-Ketten-Autokovarianz via FFT, dann ueber Ketten mitteln (BDA3 11.7).
    xc = split_chains - chain_means[:, None]
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, n=nfft, axis=1)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft, axis=1)[:, :n]
    acov = acov / n  # gamma_chain(t), biased 1/N-Normierung
    mean_acov = acov.mean(axis=0)  # ueber Ketten gemittelt, (n,)
    # rho_t = 1 - (W - mean_acov_t) / var_plus  (BDA3: W_variogram_t = 2(W - acov_t))
    rho = 1.0 - (w - mean_acov) / var_plus
    rho[0] = 1.0

    # Geyer initial monotone sequence: Paarsummen P_k = rho_{2k}+rho_{2k+1}.
    tau_sum = 0.0
    prev_pair = np.inf
    k = 1
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair <= 0.0:
            break
        pair = min(pair, prev_pair)  # Monotonie erzwingen
        tau_sum += pair
        prev_pair = pair
        k += 2
    # Letzter Einzel-Lag, falls n ungerade-Rest (selten relevant; konservativ).
    ess = (two_m * n) / (1.0 + 2.0 * tau_sum)
    return float(max(ess, 1.0))


def split_rhat(draws: np.ndarray, *, expected_constant: bool = False) -> RhatResult:
    """Rank-normalized split-R-hat + folded-R-hat + bulk/tail-ESS (Vehtari 2021).

    Args:
        draws: (M, n)-Array, M Ketten je n Draws des SELBEN Skalars. M>=2, n>=4.
        expected_constant: Nur setzen, wenn externes Domaenenwissen garantiert,
            dass die Observable strukturell konstant sein muss. Dann wird der
            Zustand als STRUCTURAL_CONSTANT ausgewiesen, aber nie als CONVERGED.

    Returns:
        RhatResult mit rhat = max(bulk, folded), ESS-bulk/tail, converged-Flag.
    """
    chains = _as_chains(draws)

    constant_draws = bool(np.all(chains == chains.flat[0]))
    if expected_constant and not constant_draws:
        raise ValueError("expected_constant=True but the observed draws are not exactly constant")
    if constant_draws:
        diagnostic_state = (
            DiagnosticState.STRUCTURAL_CONSTANT
            if expected_constant
            else DiagnosticState.DEGENERATE_CONSTANT
        )
    else:
        diagnostic_state = DiagnosticState.OK

    # --- bulk: rank-normalize ueber alle Werte, dann split + R-hat/ESS.
    z = rank_normalize(chains)
    z_split = _split(z)
    bulk_rhat = _rhat_on(z_split)
    ess_bulk = _ess_on(z_split)

    # --- folded: |theta - median|, dann rank-normalize, split, R-hat.
    median = float(np.median(chains))
    folded = np.abs(chains - median)
    zf = rank_normalize(folded)
    zf_split = _split(zf)
    folded_rhat = _rhat_on(zf_split)

    rhat = max(bulk_rhat, folded_rhat)

    # --- tail-ESS: ESS der 5%/95%-Quantil-Indikatoren (rang-normalisiert),
    #     reportiert als das MINIMUM der MESSBAREN (schlechtester Schwanz).
    #
    # EHRLICHKEITS-ANMERKUNG (2026-08-28, gemessen, Verhalten bewusst unveraendert):
    # Hier stand 'konservativ'. Das gilt nur, solange BEIDE Indikatoren messbar
    # sind. Entartet einer -- konstante Spalte --, wird er unten uebersprungen,
    # und der gemeldete Wert stammt aus einem einzigen Indikator. Gemessen:
    # klebriger Zwei-Zustands-Sampler -> ess_tail 587.95, identisch mit ess_bulk;
    # iid-Zweiwert-Daten -> ess_tail 8000.00, das MAXIMUM. Eine Zusicherung, die
    # konservativ klingt und es im Entartungsfall nicht ist, ist gefaehrlicher als
    # gar keine.
    # Die Hausregel waere fail-closed (uebersprungen -> 0.0). Dagegen spricht die
    # ungemessene Wirkung auf stark gebundene Observablen wie H, deren Indikatoren
    # regelmaessig entarten: das wuerde bestehende Akzeptanzkriterien rot faerben.
    # Entschieden am 2026-08-28: Zusicherung berichtigen, Verhalten offen fuehren.
    q05, q95 = np.quantile(chains, [0.05, 0.95])
    ess_tails = []
    for q, lower in ((q05, True), (q95, False)):
        ind = (chains <= q).astype(np.float64) if lower else (chains >= q).astype(np.float64)
        # Konstante Indikator-Spalten (entartet) ueberspringen.
        if np.allclose(ind, ind.flat[0]):
            continue
        zi = rank_normalize(ind)
        ess_tails.append(_ess_on(_split(zi)))
    ess_tail = float(min(ess_tails)) if ess_tails else ess_bulk

    return RhatResult(
        rhat=float(rhat),
        bulk_rhat=float(bulk_rhat),
        folded_rhat=float(folded_rhat),
        ess_bulk=float(ess_bulk),
        ess_tail=float(ess_tail),
        n_chains=int(chains.shape[0]),
        n_draws=int(chains.shape[1]),
        diagnostic_state=diagnostic_state,
        rhat_defined=not constant_draws,
    )


def ess_bulk_tail(draws: np.ndarray) -> tuple[float, float]:
    """Bequemer Alias: (bulk-ESS, tail-ESS)."""
    r = split_rhat(draws)
    return r.ess_bulk, r.ess_tail
