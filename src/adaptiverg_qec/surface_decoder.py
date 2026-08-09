"""QEC-Diagnostik (Inkrement 3): ECHTES MWPM-Decoding via Stim + PyMatching.

ZWECK / POSITIONIERUNG (ehrlich):
Inkrement 1+2 lieferten den vollstaendig analytisch orakelbaren Repetition-Code-Teil
(Binomial-Orakel + Fit-Diagnostik), aber KEINEN echten Decoder und KEINEN Surface-Code.
Inkrement 3 schliesst genau diese Luecke -- ohne einen eigenen Decoder zu erfinden:

  * Decoder = **PyMatching 2** (Higgott & Gidney, etablierter Minimum-Weight-Perfect-
    Matching-Decoder).
  * Sampling/Schaltkreis-Konstruktion = **Stim** (Gidney) fuer den Surface-Code.

Beides liegt HINTER einem optional-dependency-Gate (`pip install adaptiverg-qec[surface]`)
und ist KEINE Kern-Dependency. Ohne die Extras importiert dieses Modul nicht -- die
zugehoerigen Tests SKIPPEN (sie failen nicht hart), s. tests/test_surface_decoder.py.

WAS HIER VALIDIERT WIRD (zwei unabhaengige Orakel):

  A. **Repetition-Code, code-capacity bit-flip**, MWPM (PyMatching aus der Paritaets-
     Pruefmatrix) gegen das EXAKTE Binomial-Orakel aus Inkrement 1
     (`qec_diagnostics.logical_error_rate_exact`). Auf dem 1D-Matching-Graphen IST MWPM
     der ML-optimale Decoder, also MUSS die MWPM-Monte-Carlo das Binomial-Orakel innerhalb
     weniger Sigma treffen. Das ist der von Inkrement 2 explizit zurueckgestellte
     "Cross-Anchor-Decoder" -- jetzt mit der etablierten Bibliothek statt einem
     hand-gerollten (in Inkr.2 verworfenen, weil den Codespace verlassenden) Prototyp.

  B. **Rotated-Surface-Code, code-capacity pure-X-flip** (X_ERROR(p) nur auf Daten-Qubits,
     perfekte Messung, 1 Runde), MWPM via Stim-Detector-Error-Model + PyMatching. Der
     Schwellen-Schaetzer (Kurven-Kreuzung verschiedener Code-Distanzen) wird gegen den
     PUBLIZIERTEN MWPM-Threshold ~10.3% verglichen (Dennis, Kitaev, Landahl & Preskill,
     "Topological quantum memory", J. Math. Phys. 43, 4452 (2002) = Nulltemperatur-
     Uebergang des Random-Bond-Ising-Modells).

KORREKTHEITS-DISZIPLIN (Codie, ehrlich):
- Der MWPM-Threshold ~10.3% ist BEWUSST NICHT der optimale (ML/Tensor-Network)-Threshold
  10.94% (Nishimori-Punkt des 2D-RBIM, Honecker/Picco/Pujol 2001 bzw. Bravyi/Suchara/Vargo
  2014). MWPM ist near-optimal, aber sub-optimal; ein Schaetzer, der 10.9% "erreicht",
  waere VERDAECHTIG. Validiert wird gegen 10.3%, nicht gegen 10.9%.
- Keine Literatur-Zahl wird blind als "erreicht" behauptet: der Schaetzer gibt seinen
  eigenen Wert + |err| zur publizierten Zahl + Toleranz aus; jede Zahl ist aus dem
  `__main__`-Lauf regenerierbar (results/qec-surface-mwpm.json).
- Endliche Code-Distanzen liefern KEINE L->oo-FSS; der Kreuzungs-Schaetzer driftet mit
  wachsender Distanz von unten zum Threshold. Das ist als Scope-Grenze benannt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .qec_diagnostics import cell_seed, logical_error_rate_exact

# --- optional-dependency-Gate -------------------------------------------------
# Ohne das [surface]-Extra bleibt HAVE_SURFACE False; die Funktionen werfen einen
# klaren ImportError, und die Tests skippen (sie failen NICHT hart).
try:  # pragma: no cover - reiner Import-Pfad
    import pymatching  # type: ignore
    import stim  # type: ignore
    from scipy.sparse import csc_matrix  # type: ignore

    HAVE_SURFACE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - nur ohne Extra
    HAVE_SURFACE = False
    _IMPORT_ERROR = exc


# Publizierte Orakel-Werte (Quellen: docs/QEC_DIAGNOSTICS_ROADMAP.md, SOURCES.md).
MWPM_THRESHOLD_LITERATURE = 0.103  # Dennis et al. 2002, code-capacity bit-flip, MWPM.
ML_THRESHOLD_LITERATURE = 0.1094  # Nishimori-Punkt 2D-RBIM (optimaler/ML-Threshold; NICHT MWPM).


def _require_surface() -> None:
    """Fail-closed: ohne das [surface]-Extra ein klarer, handlungsleitender Fehler."""
    if not HAVE_SURFACE:
        raise ImportError(
            "Inkrement-3-Funktionen brauchen die optionalen Abhaengigkeiten 'stim' und "
            "'pymatching'. Installiere sie mit:  pip install 'adaptiverg-qec[surface]'  "
            f"(urspruenglicher Import-Fehler: {_IMPORT_ERROR!r})"
        )


# =============================================================================
# Orakel A: Repetition-Code MWPM vs. exaktes Binomial-Orakel (Inkrement 1)
# =============================================================================


def _repetition_check_matrix(d: int):
    """Paritaets-Pruefmatrix H des d-Bit-Repetition-Codes (d-1 Checks, d Daten-Bits).

    Check i koppelt Bit i und i+1: Syndrom = benachbarte Bit-Paritaeten -> 1D-Matching-
    Graph. Auf diesem Graphen ist MWPM der ML-optimale Decoder.
    """
    _require_surface()
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for i in range(d - 1):
        rows += [i, i]
        cols += [i, i + 1]
        data += [1, 1]
    return csc_matrix((data, (rows, cols)), shape=(d - 1, d), dtype=np.uint8)


@dataclass(frozen=True)
class RepetitionMwpmEstimate:
    """MWPM-Monte-Carlo der logischen Fehlerrate des Repetition-Codes vs. Orakel.

    Attributes:
        d: Code-Distanz (ungerade, >= 3).
        p: physikalische Bit-Flip-Wahrscheinlichkeit in (0, 0.5).
        shots: Anzahl unabhaengiger Realisierungen.
        p_L_mwpm: MWPM-Schaetzer der logischen Fehlerrate (PyMatching).
        std_err: Jeffreys-regularisierter Binomial-Standardfehler (nie 0).
        p_L_exact: exaktes Binomial-Orakel (Inkrement 1).
        n_sigma: |p_L_mwpm - p_L_exact| / std_err (immer wohldefiniert).
    """

    d: int
    p: float
    shots: int
    p_L_mwpm: float
    std_err: float
    p_L_exact: float
    n_sigma: float


def repetition_mwpm_vs_oracle(d: int, p: float, shots: int, seed: int) -> RepetitionMwpmEstimate:
    """MWPM (PyMatching) auf dem Repetition-Code gegen das exakte Binomial-Orakel.

    Auf dem 1D-Matching-Graphen IST MWPM ML-optimal, also muss die MWPM-Monte-Carlo
    das Binomial-Orakel (Inkrement 1) innerhalb weniger Sigma treffen -- der von
    Inkrement 2 zurueckgestellte unabhaengige Cross-Anchor, jetzt mit der etablierten
    Bibliothek.

    Args:
        d: Code-Distanz, ungerade und >= 3.
        p: Bit-Flip-Wahrscheinlichkeit in (0, 0.5) (sub-threshold-Regime).
        shots: Anzahl Realisierungen (>= 1).
        seed: Seed fuer np.random.default_rng (reproduzierbar).

    Returns:
        RepetitionMwpmEstimate mit MWPM-Wert, Standardfehler, Orakel, n_sigma.

    Raises:
        ImportError: ohne das [surface]-Extra.
        ValueError/TypeError: invalide Eingaben (Silent-Failure-Gate, fail-closed).
    """
    _require_surface()
    if not isinstance(d, (int, np.integer)):
        raise TypeError(f"d must be an integer, got {type(d).__name__}")
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d must be odd and >= 3, got {d}")
    if not math.isfinite(p) or p <= 0.0 or p >= 0.5:
        raise ValueError(f"p must be in the sub-threshold range (0, 0.5), got {p}")
    if not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    d = int(d)
    h = _repetition_check_matrix(d)
    # Logische Observable = Paritaet aller Daten-Bits (logisches X des Repetition-Codes).
    logical = csc_matrix(np.ones((1, d), dtype=np.uint8))
    # Uniforme Kantengewichte log((1-p)/p): MWPM minimiert die Fehlerzahl (ML auf 1D).
    weights = math.log((1.0 - p) / p) * np.ones(d)
    matching = pymatching.Matching.from_check_matrix(h, weights=weights, faults_matrix=logical)

    rng = np.random.default_rng(seed)
    errors = (rng.random((shots, d)) < p).astype(np.uint8)
    h_dense = h.toarray().astype(np.uint8)
    syndrome = (errors @ h_dense.T) % 2
    predicted = matching.decode_batch(syndrome.astype(np.uint8)).astype(np.uint8)
    actual_logical = (errors @ logical.toarray().astype(np.uint8).T) % 2
    logical_error = (predicted ^ actual_logical.astype(np.uint8)).reshape(-1)

    p_l_mwpm = float(logical_error.mean())
    p_l_exact = logical_error_rate_exact(d, p)
    # Jeffreys-regularisierter Standardfehler (nie 0): auch Null-Ereignis-Zellen
    # bleiben falsifizierbar (s. qec_diagnostics.simulate_logical_error_rate).
    k = int(logical_error.sum())
    p_tilde = (k + 0.5) / (shots + 1.0)
    std_err = math.sqrt(p_tilde * (1.0 - p_tilde) / shots)
    n_sigma = abs(p_l_mwpm - p_l_exact) / std_err
    return RepetitionMwpmEstimate(
        d=d,
        p=p,
        shots=shots,
        p_L_mwpm=p_l_mwpm,
        std_err=std_err,
        p_L_exact=p_l_exact,
        n_sigma=n_sigma,
    )


# =============================================================================
# Orakel B: Surface-Code MWPM-Threshold vs. publizierter Wert ~10.3%
# =============================================================================


def _data_qubits(circuit) -> list[int]:
    """Daten-Qubit-Indizes = Ziele der finalen M-Instruktion im Stim-Schaltkreis."""
    data_q: list[int] | None = None
    for inst in circuit:
        if inst.name == "M":
            data_q = [t.value for t in inst.targets_copy()]
    if data_q is None:
        raise RuntimeError("no final M instruction found; unexpected stim circuit shape")
    return data_q


def _surface_codecap_xflip_circuit(d: int, p: float):
    """Rotated-Surface-Code, code-capacity pure-X-flip: X_ERROR(p) NUR auf Daten-Qubits.

    Basis: stim 'surface_code:rotated_memory_z', 1 Runde, perfekte Messung. Wir injizieren
    X_ERROR(p) direkt nach dem initialen Reset auf exakt den Daten-Qubits -> reiner
    Bit-Flip-Kanal (kein Mess-Fehler, kein Depolarisieren), fuer den der publizierte
    MWPM-Threshold 10.3% gilt.
    """
    base = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=1)
    data_q = _data_qubits(base)
    out = stim.Circuit()
    inserted = False
    for inst in base:
        out.append(inst)
        if inst.name == "R" and not inserted:
            out.append("X_ERROR", data_q, p)
            inserted = True
    if not inserted:
        raise RuntimeError("no initial R instruction found; unexpected stim circuit shape")
    return out


def surface_logical_error_rate(d: int, p: float, shots: int, seed: int) -> float:
    """Logische Fehlerrate des Rotated-Surface-Codes unter MWPM (code-capacity X-flip).

    Stim sampelt Detektor-Ereignisse, PyMatching dekodiert aus dem Detector-Error-Model.

    Args:
        d: ungerade Code-Distanz >= 3.
        p: Bit-Flip-Wahrscheinlichkeit in (0, 0.5).
        shots: Anzahl Sampling-Shots (>= 1).
        seed: Stim-Sampler-Seed.

    Returns:
        logische Fehlerrate (Anteil falsch dekodierter Observablen).

    Raises:
        ImportError: ohne das [surface]-Extra.
        ValueError/TypeError: invalide Eingaben (Silent-Failure-Gate).
    """
    _require_surface()
    if not isinstance(d, (int, np.integer)):
        raise TypeError(f"d must be an integer, got {type(d).__name__}")
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d must be odd and >= 3, got {d}")
    if not math.isfinite(p) or p <= 0.0 or p >= 0.5:
        raise ValueError(f"p must be in (0, 0.5), got {p}")
    if not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    circuit = _surface_codecap_xflip_circuit(int(d), float(p))
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(int(shots), separate_observables=True)
    predicted = matching.decode_batch(detectors)
    return float(np.any(predicted != observables, axis=1).mean())


@dataclass(frozen=True)
class ThresholdEstimate:
    """MWPM-Threshold-Schaetzer (Kurven-Kreuzung) vs. publizierter Literatur-Wert.

    Attributes:
        d_small, d_large: zwei ungerade Code-Distanzen.
        p_threshold: geschaetzter Kreuzungspunkt p_L(d_small,.)==p_L(d_large,.).
        p_literature: publizierter MWPM-Threshold 0.103 (Dennis et al. 2002).
        abs_error: |p_threshold - p_literature|.
        p_ml_literature: optimaler/ML-Threshold 0.1094 (zur Einordnung, NICHT Orakel).
        ps: die abgetasteten p-Werte.
        pL_small, pL_large: logische Fehlerraten je Distanz an den ps.
    """

    d_small: int
    d_large: int
    crossing_found: bool
    """False, wenn diff nie das Vorzeichen wechselt (kein Threshold im p-Fenster);
    p_threshold ist dann NaN und abs_error inf -- LAUT geflaggt, nicht still."""
    p_threshold: float
    p_literature: float
    abs_error: float
    p_ml_literature: float
    ps: tuple[float, ...]
    pL_small: tuple[float, ...]
    pL_large: tuple[float, ...]


def estimate_mwpm_threshold(
    d_small: int,
    d_large: int,
    ps: tuple[float, ...] = (0.095, 0.10, 0.105, 0.11),
    shots: int = 40_000,
    seed: int = 11,
) -> ThresholdEstimate:
    """Schaetze den Surface-Code-MWPM-Threshold via Distanz-Kurven-Kreuzung.

    Fuer p < p_th draengt groessere Distanz die logische Fehlerrate nach unten, fuer
    p > p_th nach oben; der Schnittpunkt der beiden p_L-vs-p-Kurven schaetzt p_th. Wir
    interpolieren linear in der Differenz pL(d_large) - pL(d_small) auf ihre Nullstelle.
    Vergleich gegen den publizierten MWPM-Wert 0.103 (NICHT gegen den optimalen 0.1094).

    Args:
        d_small, d_large: verschiedene ungerade Distanzen >= 3, d_small < d_large.
        ps: streng aufsteigende p-Werte, die den Threshold klammern.
        shots: Shots pro (d, p)-Zelle.
        seed: Stim-Sampler-Seed.

    Returns:
        ThresholdEstimate mit Kreuzungs-p, Literatur-Wert und |err|.

    Raises:
        ImportError: ohne das [surface]-Extra.
        ValueError: invalide Distanzen/ps (Silent-Failure-Gate).
    """
    _require_surface()
    for dd in (d_small, d_large):
        if not isinstance(dd, (int, np.integer)) or dd < 3 or dd % 2 == 0:
            raise ValueError(f"distances must be odd ints >= 3, got {d_small}, {d_large}")
    if d_small >= d_large:
        raise ValueError(f"need d_small < d_large, got {d_small} >= {d_large}")
    if len(ps) < 2:
        raise ValueError("ps must contain at least 2 points to bracket the threshold")
    for pp in ps:
        if not math.isfinite(pp) or pp <= 0.0 or pp >= 0.5:
            raise ValueError(f"ps values must be in (0, 0.5), got {pp}")
    if list(ps) != sorted(ps) or len(set(ps)) != len(ps):
        raise ValueError("ps must be strictly increasing (distinct, ascending)")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    # Zell-eigene Seeds (cell_seed): vorher teilten ALLE (d, p)-Zellen denselben
    # Sampler-Seed -> Kurvenpunkte rangkorreliert statt unabhaengige Evidenz.
    pl_small = [
        surface_logical_error_rate(d_small, p, shots, cell_seed(seed, d_small, p)) for p in ps
    ]
    pl_large = [
        surface_logical_error_rate(d_large, p, shots, cell_seed(seed, d_large, p)) for p in ps
    ]
    diff = [a - b for a, b in zip(pl_large, pl_small, strict=True)]

    # Erste Vorzeichen-Aenderung in diff -> lineare Interpolation auf die Nullstelle.
    p_threshold = float("nan")
    for i in range(len(ps) - 1):
        if (diff[i] <= 0.0 <= diff[i + 1]) or (diff[i] >= 0.0 >= diff[i + 1]):
            x0, x1 = ps[i], ps[i + 1]
            y0, y1 = diff[i], diff[i + 1]
            p_threshold = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else 0.5 * (x0 + x1)
            break

    abs_error = (
        abs(p_threshold - MWPM_THRESHOLD_LITERATURE) if math.isfinite(p_threshold) else float("inf")
    )
    return ThresholdEstimate(
        d_small=int(d_small),
        d_large=int(d_large),
        crossing_found=bool(math.isfinite(p_threshold)),
        p_threshold=p_threshold,
        p_literature=MWPM_THRESHOLD_LITERATURE,
        abs_error=abs_error,
        p_ml_literature=ML_THRESHOLD_LITERATURE,
        ps=tuple(float(p) for p in ps),
        pL_small=tuple(pl_small),
        pL_large=tuple(pl_large),
    )


def run_surface_diagnostics(
    rep_distances: tuple[int, ...] = (3, 5, 7),
    rep_p: float = 0.10,
    rep_shots: int = 200_000,
    threshold_pairs: tuple[tuple[int, int], ...] = ((7, 9), (9, 11), (7, 11)),
    threshold_ps: tuple[float, ...] = (0.090, 0.095, 0.10, 0.105, 0.11, 0.115),
    threshold_shots: int = 80_000,
    seed: int = 11,
) -> dict:
    """Voller Inkrement-3-Lauf: Repetition-MWPM-vs-Orakel + Surface-Threshold-Schaetzer.

    Gibt ein JSON-serialisierbares dict zurueck (der `__main__`-Block persistiert es).
    Jede Zahl traegt ihr unabhaengiges Orakel (Binomial bzw. publizierter Threshold).
    """
    _require_surface()

    rep_rows = []
    worst_n_sigma = 0.0
    for d in rep_distances:
        # Zell-eigener Seed je Distanz (dekorreliert, reproduzierbar).
        est = repetition_mwpm_vs_oracle(d, rep_p, rep_shots, cell_seed(seed, d, rep_p))
        worst_n_sigma = max(worst_n_sigma, est.n_sigma)
        rep_rows.append(
            {
                "d": est.d,
                "p": est.p,
                "p_L_mwpm": est.p_L_mwpm,
                "std_err": est.std_err,
                "p_L_exact": est.p_L_exact,
                "n_sigma": est.n_sigma,
            }
        )

    thr_rows = []
    best_abs_error = float("inf")
    best_pair = None
    for d_small, d_large in threshold_pairs:
        thr = estimate_mwpm_threshold(
            d_small, d_large, ps=threshold_ps, shots=threshold_shots, seed=seed
        )
        if thr.abs_error < best_abs_error:
            best_abs_error = thr.abs_error
            best_pair = (thr.d_small, thr.d_large)
        thr_rows.append(
            {
                "d_small": thr.d_small,
                "d_large": thr.d_large,
                "crossing_found": thr.crossing_found,
                # None statt NaN/inf: das Artefakt bleibt strikt gueltiges JSON.
                "p_threshold": thr.p_threshold if thr.crossing_found else None,
                "p_literature": thr.p_literature,
                "abs_error": thr.abs_error if thr.crossing_found else None,
            }
        )

    return {
        "tool": "adaptiverg_qec.surface_decoder",
        "decoder": "PyMatching 2 (MWPM); Stim sampling",
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
        "oracle_A": {
            "model": "repetition code, code-capacity bit-flip, MWPM == ML on 1D matching graph",
            "oracle": "exact binomial p_L(d,p) (Inkrement 1)",
            "p": rep_p,
            "shots": rep_shots,
            "rows": rep_rows,
            "worst_n_sigma": worst_n_sigma,
        },
        "oracle_B": {
            "model": "rotated surface code, code-capacity pure-X-flip, MWPM threshold",
            "literature_mwpm_threshold": MWPM_THRESHOLD_LITERATURE,
            "literature_ml_threshold_for_context": ML_THRESHOLD_LITERATURE,
            "literature_source": (
                "Dennis, Kitaev, Landahl & Preskill, J. Math. Phys. 43, 4452 (2002)"
            ),
            "shots_per_cell": threshold_shots,
            "ps": list(threshold_ps),
            "rows": thr_rows,
            "best_pair": best_pair,
            "best_abs_error": best_abs_error,
        },
        "seed": seed,
    }


def _main() -> int:
    """Schreibe results/qec-surface-mwpm.json + Konsolen-Tabelle (Evidenz)."""
    import json
    from pathlib import Path

    _require_surface()
    payload = run_surface_diagnostics()
    out = Path("results") / "qec-surface-mwpm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    a = payload["oracle_A"]
    b = payload["oracle_B"]
    print("QEC-Diagnostik (Inkrement 3): ECHTES MWPM via Stim + PyMatching")
    print(f"  stim {payload['stim_version']} | pymatching {payload['pymatching_version']}")
    print("-" * 78)
    print(f"Orakel A: Repetition-Code MWPM vs exaktes Binomial-Orakel (p={a['p']}):")
    print(f"  {'d':>3} {'p_L_mwpm':>10} {'p_L_exact':>11} {'n_sigma':>8}")
    for r in a["rows"]:
        print(f"  {r['d']:>3} {r['p_L_mwpm']:>10.5f} {r['p_L_exact']:>11.5f} {r['n_sigma']:>8.2f}")
    print(f"  worst_n_sigma = {a['worst_n_sigma']:.2f}  (shots={a['shots']})")
    print(
        f"Orakel B: Surface-Code MWPM-Threshold vs publiziert "
        f"{b['literature_mwpm_threshold']} (Dennis 2002):"
    )
    print(f"  {'d_small':>7} {'d_large':>7} {'p_threshold':>12} {'abs_err':>10}")
    for r in b["rows"]:
        if r["crossing_found"]:
            print(
                f"  {r['d_small']:>7} {r['d_large']:>7} "
                f"{r['p_threshold']:>12.4f} {r['abs_error']:>10.4f}"
            )
        else:
            print(f"  {r['d_small']:>7} {r['d_large']:>7} {'no crossing':>12} {'--':>10}")
    print(
        f"  best_pair={b['best_pair']} best_abs_error={b['best_abs_error']:.4f} "
        f"(MWPM 0.103, NICHT ML/Nishimori {b['literature_ml_threshold_for_context']})"
    )
    print("-" * 78)
    print(f"evidence -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
