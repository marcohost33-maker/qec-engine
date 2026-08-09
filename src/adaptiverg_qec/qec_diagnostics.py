"""QEC-Diagnostik-Harness (Inkrement 1): Logical-Error-Rate vs. physical p.

ZWECK / POSITIONIERUNG (ehrlich, Repositioning-Entscheid 2026-06-18):
Das uebrige Repo (`a_kernel`, `mcrg*`, `ising2d`, `wolff2d`, ...) ist ein
MCMC/MCRG-Harness fuer KRITISCHE EXPONENTEN (statistische Mechanik). Es enthaelt
KEINE eigentliche Fehlerkorrektur-Diagnostik: keinen Decoder, keine logische
Fehlerrate, keinen Sweep ueber die physikalische Fehlerrate. Dieses Modul ist der
ERSTE Inkrement-Schritt eines ehrlichen QEC-Diagnostik-/Vergleichs-Geruests:

    physikalische Fehlerrate p  --->  logische Fehlerrate p_L

am minimalen, vollstaendig analytisch orakelbaren Modell:
**n-Bit-Repetition-Code unter unabhaengigem Bit-Flip-Kanal + Majority-Vote-Decoder.**

Das ist BEWUSST das einfachste QEC-Modell (Nielsen & Chuang Kap. 10; Roffe,
"Quantum error correction: an introductory guide", 2019). Es ist KEIN
Surface-Code, KEIN MWPM/Tensor-Network-Decoder und KEIN SOTA-Threshold-Tool --
aber es ist exakt orakelbar und damit der korrekte Reality-Anchor fuer einen
ehrlichen Diagnostik-Harness: jede Monte-Carlo-Zahl wird gegen ein
geschlossenes Binomial-Orakel geprueft, nicht gegen sich selbst.

Modell (klassischer Bit-Flip-Kanal, der X-Sektor eines CSS-Repetition-Codes):
- Logisches Bit in n physikalische Bits kopiert (n ungerade).
- Jedes physikalische Bit flippt unabhaengig mit Wahrscheinlichkeit p.
- Decoder = Majority Vote (= MWPM/ML auf dem Repetition-Code, hier exakt optimal).
- Logischer Fehler <=> mehr als die Haelfte der Bits sind geflippt, d.h. k >= ceil(n/2).

Exaktes Orakel (Binomialsumme):
    p_L(n, p) = sum_{k=ceil(n/2)}^{n} C(n, k) p^k (1-p)^(n-k).

ACHTUNG (Korrektheits-Falle, web-verifiziert): eine im Netz haeufig zitierte
Formel "p_L = p^3 + 2 p^2 (1-p)" fuer n=3 ist FALSCH. Korrekt ist
    p_L(3, p) = C(3,2) p^2 (1-p) + C(3,3) p^3 = 3 p^2 (1-p) + p^3 = 3 p^2 - 2 p^3.
Genau deshalb prueft dieser Harness die MC gegen das exakte Binomial-Orakel
(unabhaengig hergeleitet), nicht gegen eine Literatur-Formel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def logical_error_rate_exact(n: int, p: float) -> float:
    """Exakte logische Fehlerrate p_L(n, p) des n-Bit-Repetition-Codes.

    Geschlossene Binomialsumme (unabhaengiges Orakel):
        p_L = sum_{k=ceil(n/2)}^{n} C(n,k) p^k (1-p)^(n-k).

    Args:
        n: Code-Distanz / Anzahl physikalischer Bits. MUSS ungerade und >= 1 sein
           (gerade n haben Stimmengleichheits-Ambiguitaet -> bewusst abgelehnt).
        p: physikalische Bit-Flip-Wahrscheinlichkeit in [0, 1].

    Returns:
        p_L in [0, 1].

    Raises:
        ValueError: n nicht ungerade/>=1, p ausserhalb [0,1] oder nicht endlich.
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1 or n % 2 == 0:
        raise ValueError(f"n must be odd and >= 1 (no tie-breaking), got {n}")
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"p must be a finite probability in [0, 1], got {p}")
    k_min = n // 2 + 1  # = ceil(n/2) fuer ungerades n
    # math.comb + math.fsum: exakte Binomialkoeffizienten, fehler-kompensierte Summe.
    terms = [math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k)) for k in range(k_min, n + 1)]
    return math.fsum(terms)


@dataclass(frozen=True)
class LogicalErrorEstimate:
    """Monte-Carlo-Schaetzung der logischen Fehlerrate + Beleg-Felder.

    Attributes:
        n: Code-Distanz (ungerade).
        p: physikalische Fehlerrate.
        shots: Anzahl unabhaengiger Code-Block-Realisierungen.
        n_logical_errors: beobachtete logische Fehler.
        p_L_mc: MC-Schaetzer = n_logical_errors / shots.
        std_err: Jeffreys-regularisierter Binomial-Standardfehler
            sqrt(p~ (1-p~) / shots) mit p~ = (k+1/2)/(shots+1) -- nie 0, damit
            auch Null-Ereignis-Zellen falsifizierbar bleiben.
        p_L_exact: exaktes Binomial-Orakel logical_error_rate_exact(n, p).
        n_sigma: |p_L_mc - p_L_exact| / std_err (immer wohldefiniert).
    """

    n: int
    p: float
    shots: int
    n_logical_errors: int
    p_L_mc: float
    std_err: float
    p_L_exact: float
    n_sigma: float


def simulate_logical_error_rate(n: int, p: float, shots: int, seed: int) -> LogicalErrorEstimate:
    """Monte-Carlo-Schaetzung von p_L(n, p) + Vergleich gegen das exakte Orakel.

    Vektorisiert (numpy, kein Python-Shot-Loop): erzeugt eine (shots x n)-
    Bit-Flip-Maske ~ Bernoulli(p), zaehlt Flips pro Block, logischer Fehler gdw
    Flip-Zahl > n/2 (Majority Vote falsch). Reproduzierbar ueber Philox-Seed.

    Args:
        n: Code-Distanz (ungerade, >= 1).
        p: physikalische Fehlerrate in [0, 1].
        shots: Anzahl Realisierungen (>= 1).
        seed: Seed fuer np.random.default_rng (bit-reproduzierbar).

    Returns:
        LogicalErrorEstimate mit MC-Wert, Standardfehler, Orakel und n_sigma.

    Raises:
        ValueError/TypeError: bei invaliden Eingaben (fail-closed, Silent-Failure-Gate).
    """
    # Eingangs-Validierung zuerst (logical_error_rate_exact validiert n & p mit;
    # wir rufen es auf, BEVOR teuer gesampelt wird -> fail-closed-early).
    p_L_exact = logical_error_rate_exact(n, p)
    if not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    rng = np.random.default_rng(seed)
    # (shots, n) Bernoulli(p)-Flips; sum entlang Achse 1 = Flips pro Block.
    flips = rng.random((shots, n)) < p
    flips_per_block = flips.sum(axis=1)
    logical_errors = int(np.count_nonzero(flips_per_block > n / 2.0))

    p_L_mc = logical_errors / shots
    # Jeffreys-regularisierter Standardfehler: p~ = (k+1/2)/(shots+1) statt des
    # Plug-in p_hat. Damit ist std_err NIE 0 -- auch Null-Ereignis-Zellen bleiben
    # falsifizierbar (vorher: k=0 -> std_err=0 -> n_sigma=0 == "perfekter Treffer",
    # ein unfalsifizierbares Gate genau im uninformativsten Regime).
    p_tilde = (logical_errors + 0.5) / (shots + 1.0)
    std_err = math.sqrt(p_tilde * (1.0 - p_tilde) / shots)
    n_sigma = abs(p_L_mc - p_L_exact) / std_err
    return LogicalErrorEstimate(
        n=n,
        p=p,
        shots=shots,
        n_logical_errors=logical_errors,
        p_L_mc=p_L_mc,
        std_err=std_err,
        p_L_exact=p_L_exact,
        n_sigma=n_sigma,
    )


def pseudo_threshold(n_small: int, n_large: int) -> float:
    """Pseudo-Threshold p* des Repetition-Codes: p_L(n_small,p*) = p_L(n_large,p*).

    Fuer den unabhaengigen Bit-Flip-Kanal mit Majority-Vote schneiden sich die
    Kurven verschiedener (ungerader) Code-Distanzen bei p* = 1/2: unterhalb
    hilft mehr Distanz (p_L faellt), oberhalb schadet sie. Das ist der exakte,
    analytisch bekannte Wert (kein Frontier-Threshold eines Surface-Codes).

    Wir loesen es NICHT numerisch, sondern geben den geschlossenen Wert zurueck und
    erlauben dem Aufrufer, ihn gegen einen MC-/Bisektions-Schnittpunkt zu pruefen.

    Args:
        n_small, n_large: zwei ungerade Code-Distanzen mit n_small != n_large.

    Returns:
        p* = 0.5 (geschlossener Wert fuer diesen Kanal).

    Raises:
        ValueError: n nicht ungerade oder gleich.
    """
    for nm in (n_small, n_large):
        if not isinstance(nm, (int, np.integer)) or nm < 1 or nm % 2 == 0:
            raise ValueError(f"distances must be odd ints >= 1, got {n_small}, {n_large}")
    if n_small == n_large:
        raise ValueError("n_small and n_large must differ")
    return 0.5


def cell_seed(base_seed: int, n: int, p: float) -> int:
    """Deterministischer, zell-eigener Seed fuer eine (n, p)-Sweep-Zelle.

    Injektiv genug fuer Sweep-Grids (p auf 1e-9 quantisiert); dekorreliert die
    Zellen, bleibt aber vollstaendig durch base_seed reproduzierbar.
    """
    return (base_seed * 1_000_003 + int(n) * 7_919 + int(round(p * 1e9))) % (2**63 - 1)


def run_diagnostic_sweep(
    distances: tuple[int, ...] = (1, 3, 5, 7),
    p_values: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    shots: int = 200_000,
    seed: int = 2026,
) -> dict:
    """Fuehre den vollen p_L-vs-p-Sweep aus + sammle MC-vs-Orakel-Belege.

    Schreibt KEINE Datei; gibt ein JSON-serialisierbares dict zurueck (der
    `__main__`-Block persistiert es nach results/). Jede Zelle traegt MC-Wert,
    Standardfehler, exaktes Orakel und n_sigma -> der Reality-Anchor des Harness.
    """
    rows = []
    worst_sigma = 0.0
    for n in distances:
        for p in p_values:
            # Ein EIGENER, deterministisch abgeleiteter Seed je Sweep-Zelle.
            # Vorher: EIN Seed fuer alle Zellen -> fuer festes n wurde dieselbe
            # (shots, n)-Uniform-Matrix fuer jedes p wiederverwendet, d.h. die
            # p-Zeilen waren perfekt rangkorreliert (keine unabhaengige Evidenz,
            # worst_n_sigma kein Max ueber unabhaengige Tests).
            est = simulate_logical_error_rate(n, p, shots=shots, seed=cell_seed(seed, n, p))
            worst_sigma = max(worst_sigma, est.n_sigma)
            rows.append(
                {
                    "n": est.n,
                    "p": est.p,
                    "p_L_mc": est.p_L_mc,
                    "std_err": est.std_err,
                    "p_L_exact": est.p_L_exact,
                    "n_sigma": est.n_sigma,
                    "n_logical_errors": est.n_logical_errors,
                }
            )
    return {
        "tool": "adaptiverg_qec.qec_diagnostics",
        "model": "n-bit repetition code, independent bit-flip channel, majority-vote decoder",
        "oracle": "p_L(n,p) = sum_{k=ceil(n/2)}^{n} C(n,k) p^k (1-p)^(n-k)",
        "shots": shots,
        "seed": seed,
        "distances": list(distances),
        "p_values": list(p_values),
        "worst_n_sigma": worst_sigma,
        # len==2: distances[1] == distances[-1] wuerde pseudo_threshold zu Recht
        # ablehnen -> erstes/letztes Paar nehmen (immer verschieden fuer len>=2).
        "pseudo_threshold": pseudo_threshold(
            distances[1] if len(distances) > 2 else distances[0], distances[-1]
        )
        if len(distances) >= 2
        else None,
        "rows": rows,
    }


def _main() -> int:
    """Schreibe results/qec-diagnostics-rep-code.json + Konsolen-Tabelle (Evidenz)."""
    import json
    from pathlib import Path

    payload = run_diagnostic_sweep()
    out = Path("results") / "qec-diagnostics-rep-code.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("QEC-Diagnostik (Inkrement 1): logical-error-rate p_L vs physical p")
    print("Repetition-Code, independent bit-flip, majority-vote; MC vs Binomial-Orakel")
    print("-" * 78)
    print(f"{'n':>3} {'p':>5} {'p_L_mc':>10} {'std_err':>9} {'p_L_exact':>11} {'n_sigma':>8}")
    for r in payload["rows"]:
        print(
            f"{r['n']:>3} {r['p']:>5.2f} {r['p_L_mc']:>10.5f} {r['std_err']:>9.5f} "
            f"{r['p_L_exact']:>11.5f} {r['n_sigma']:>8.2f}"
        )
    print("-" * 78)
    print(
        f"shots={payload['shots']} seed={payload['seed']} "
        f"worst_n_sigma={payload['worst_n_sigma']:.2f} "
        f"pseudo_threshold p*={payload['pseudo_threshold']}"
    )
    print(f"evidence -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
