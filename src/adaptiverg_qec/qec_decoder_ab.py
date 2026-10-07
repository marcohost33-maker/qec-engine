"""Inkrement 3.3: gepaarter Decoder-A/B-Vergleich auf IDENTISCHEN Shots.

ZWECK:
Zwei Decoder werden nicht ueber zwei unabhaengige Stichproben verglichen, sondern ueber
dieselben Detection-Event-Shots. Pro Shot entsteht ein Paar (Baseline-Fehler?,
Kandidat-Fehler?) und damit eine 2x2-Tafel

                     Kandidat ok   Kandidat Fehler
    Baseline ok          n00            c = n01
    Baseline Fehler    b = n10            n11

Nur die diskordanten Paare b, c tragen Information ueber den Unterschied.

STATISTIK (Quellen in SOURCES.md):
- Exakter bedingter McNemar-Test: unter H0 (gleiche Fehlerrate) gilt
  b | (b+c) ~ Binomial(b+c, 1/2); zweiseitiger p-Wert = min(1, 2*P[X <= min(b,c)]).
  Das ist der konservative Gate-Test. Die mid-p-Variante (Fagerland, Lydersen & Laake
  2013, BMC Med. Res. Methodol. 13:91, doi:10.1186/1471-2288-13-91) wird zusaetzlich als
  Diagnose berichtet, entscheidet aber kein Gate.
- Effekt Delta = p_kandidat - p_baseline = (c - b)/n mit gepaartem Bootstrap. Ein
  Resampling der n Shot-Indizes ist in Verteilung identisch zu einer Multinomial-Ziehung
  der vier Zellzaehler; deshalb wird die Multinomial-Form verwendet (exakt dieselbe
  Bootstrap-Verteilung, O(n_boot) statt O(n_boot*n)).
- Pro Arm Clopper-Pearson-Intervalle (exakt, auch bei k=0 nicht entartet).

DECODER:
- ``pymatching-mwpm-dem``: PyMatching-MWPM aus dem Stim-DEM (decompose_errors=True),
  derselbe Pfad wie ``surface_decoder.run_experiment_manifest``.
- ``pymatching-correlated-mwpm-dem``: PyMatching Two-Pass-Correlated-Matching
  (``enable_correlations=True``; PyMatching >= 2.3). Nutzt zerlegbare Hyperkanten
  (z.B. Y-Fehler) zum Umgewichten im zweiten Pass.

STRUKTUR-INVARIANTE (Negativkontrolle ohne Statistik):
Enthaelt der DEM KEINE zerlegten Hyperkanten, hat Correlated Matching nichts zum
Umgewichten; die Entscheidungen muessen dann BITGLEICH zur Baseline sein. Jede Zelle
berichtet die Zahl der Hyperkanten und ob die Entscheidungen identisch sind.

CLAIM CEILING: bounded simulation, Decoder-GENAUIGKEIT auf einem deklarierten
Stim-Noise-Vertrag. Kein Latenz-/Real-Time-Claim (Python-Wallclock ist kein Benchmark),
kein Threshold-Claim, keine Uebertragung auf andere Noise-Konventionen.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from .qec_manifest_v2 import QECExperimentManifestV2

BASELINE_DECODER = "pymatching-mwpm-dem"
CORRELATED_DECODER = "pymatching-correlated-mwpm-dem"
DECODERS = (BASELINE_DECODER, CORRELATED_DECODER)
_MIN_PYMATCHING_FOR_CORRELATIONS = (2, 3)


# =============================================================================
# Statistik (reines numpy/scipy, ohne Surface-Extras)
# =============================================================================


def _count(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer count, got {type(value).__name__}")
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _alpha(alpha) -> float:
    a = float(alpha)
    if not math.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return a


def mcnemar_exact(b: int, c: int) -> dict:
    """Exakter bedingter McNemar-Test (zweiseitig) plus mid-p-Diagnose.

    ``b`` = nur Baseline falsch, ``c`` = nur Kandidat falsch. Ohne diskordante Paare
    (b+c=0) gibt es keine Evidenz gegen H0: p = 1.
    """
    b = _count("b", b)
    c = _count("c", c)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0, "p_value_mid": 1.0}
    k = min(b, c)
    cdf = float(stats.binom.cdf(k, n, 0.5))
    pmf = float(stats.binom.pmf(k, n, 0.5))
    p_value = min(1.0, 2.0 * cdf)
    # mid-p (Fagerland et al. 2013, Gl. 3): 2*[P(X<=k) - P(X=k)/2], erst dann gekappt.
    p_mid = min(1.0, 2.0 * cdf - pmf)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value, "p_value_mid": p_mid}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exaktes (konservatives) Binomial-Intervall; bei k=0 bzw. k=n einseitig offen."""
    k = _count("k", k)
    n = _count("n", n)
    if n < 1 or k > n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    a = _alpha(alpha)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi


def paired_bootstrap_delta(
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    *,
    n_boot: int = 20_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Perzentil-Bootstrap fuer Delta = p_kandidat - p_baseline = (n01 - n10)/n.

    Multinomial-Resampling der vier Zellen == Resampling der Shot-Indizes.
    Daneben steht das (falsche) UNGEPAARTE Wald-Intervall derselben Daten, damit der
    Varianzgewinn der Paarung sichtbar und testbar ist.
    """
    cells = [
        _count(name, v) for name, v in (("n00", n00), ("n01", n01), ("n10", n10), ("n11", n11))
    ]
    n = sum(cells)
    if n < 1:
        raise ValueError("need at least one shot")
    n_boot = _count("n_boot", n_boot)
    if n_boot < 100:
        raise ValueError(f"n_boot must be >= 100, got {n_boot}")
    seed = _count("seed", seed)
    a = _alpha(alpha)

    probs = np.asarray(cells, dtype=float) / n
    rng = np.random.Generator(np.random.Philox(seed))
    draws = rng.multinomial(n, probs, size=n_boot)
    deltas = (draws[:, 1] - draws[:, 2]) / n
    lo, hi = np.quantile(deltas, [a / 2.0, 1.0 - a / 2.0])

    delta = (cells[1] - cells[2]) / n
    p_base = (cells[2] + cells[3]) / n
    p_cand = (cells[1] + cells[3]) / n
    z = float(stats.norm.ppf(1.0 - a / 2.0))
    se_unpaired = math.sqrt((p_base * (1 - p_base) + p_cand * (1 - p_cand)) / n)
    return {
        "delta": delta,
        "ci": [float(lo), float(hi)],
        "ci_width": float(hi - lo),
        "unpaired_wald_ci": [delta - z * se_unpaired, delta + z * se_unpaired],
        "unpaired_wald_ci_width": 2.0 * z * se_unpaired,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": a,
        "method": "percentile bootstrap, multinomial over the paired 2x2 table",
    }


@dataclass(frozen=True)
class PairedCounts:
    """2x2-Tafel aus zwei Fehlervektoren gleicher Laenge (Baseline, Kandidat)."""

    n00: int
    n01: int
    n10: int
    n11: int

    @classmethod
    def from_failures(cls, base: np.ndarray, cand: np.ndarray) -> PairedCounts:
        base = np.asarray(base, dtype=bool)
        cand = np.asarray(cand, dtype=bool)
        if base.ndim != 1 or base.shape != cand.shape:
            raise ValueError("failure vectors must be 1-D and of equal length")
        return cls(
            n00=int(np.sum(~base & ~cand)),
            n01=int(np.sum(~base & cand)),
            n10=int(np.sum(base & ~cand)),
            n11=int(np.sum(base & cand)),
        )

    @property
    def shots(self) -> int:
        return self.n00 + self.n01 + self.n10 + self.n11

    @property
    def baseline_failures(self) -> int:
        return self.n10 + self.n11

    @property
    def candidate_failures(self) -> int:
        return self.n01 + self.n11


# =============================================================================
# Decoding (braucht das [surface]-Extra)
# =============================================================================


def _require_decoders():
    from . import surface_decoder as sd

    sd._require_surface()
    import pymatching  # type: ignore

    parts = tuple(int(x) for x in pymatching.__version__.split(".")[:2] if x.isdigit())
    if parts < _MIN_PYMATCHING_FOR_CORRELATIONS:
        raise ImportError(
            "correlated matching needs pymatching>=2.3 (enable_correlations), "
            f"found {pymatching.__version__}"
        )
    return sd.stim, pymatching


def count_hyperedges(dem) -> int:
    """Fehlermechanismen, die Stim in mehrere graphlike Teile ('^') zerlegt hat."""
    n = 0
    for inst in dem.flattened():
        if inst.type == "error" and any(t.is_separator() for t in inst.targets_copy()):
            n += 1
    return n


def _decode(matchings: dict, decoder: str, detectors: np.ndarray) -> np.ndarray:
    if decoder == BASELINE_DECODER:
        return matchings["plain"].decode_batch(detectors)
    if decoder == CORRELATED_DECODER:
        return matchings["corr"].decode_batch(detectors, enable_correlations=True)
    raise ValueError(f"unknown decoder {decoder!r}; known: {DECODERS}")


def paired_decoder_ab(
    manifest: QECExperimentManifestV2,
    *,
    baseline: str = BASELINE_DECODER,
    candidate: str = CORRELATED_DECODER,
    n_boot: int = 20_000,
    alpha: float = 0.05,
) -> dict:
    """Dekodiere jede Manifest-Zelle EINMAL gesampelt mit beiden Decodern.

    Der Sampling-Vertrag (Noise, Distanzen, Runden, Shots, Zell-Seeds) kommt
    unveraendert aus dem Manifest; ``manifest.decoder`` ist die Baseline und muss mit
    ``baseline`` uebereinstimmen. Damit sind die Baseline-Fehler einer Zelle bitgleich zu
    ``surface_decoder.run_experiment_manifest`` (gleicher Seed, gleicher Pfad).
    """
    if not isinstance(manifest, QECExperimentManifestV2):
        raise TypeError("manifest must be QECExperimentManifestV2")
    for name, dec in (("baseline", baseline), ("candidate", candidate)):
        if dec not in DECODERS:
            raise ValueError(f"{name} decoder {dec!r} unknown; known: {DECODERS}")
    if baseline != manifest.decoder:
        raise ValueError(
            f"baseline {baseline!r} must equal the manifest decoder {manifest.decoder!r}"
        )
    stim, pymatching = _require_decoders()

    cells = []
    for d in manifest.distances:
        rounds = manifest.resolved_rounds(d)
        seed = manifest.cell_seed(d)
        circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{manifest.memory_basis}",
            distance=d,
            rounds=rounds,
            **manifest.noise.to_stim_kwargs(),
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        matchings = {
            "plain": pymatching.Matching.from_detector_error_model(dem),
            "corr": pymatching.Matching.from_detector_error_model(dem, enable_correlations=True),
        }
        sampler = circuit.compile_detector_sampler(seed=seed)
        detectors, observables = sampler.sample(manifest.shots_per_cell, separate_observables=True)
        pred_base = _decode(matchings, baseline, detectors)
        pred_cand = _decode(matchings, candidate, detectors)
        fail_base = np.any(pred_base != observables, axis=1)
        fail_cand = np.any(pred_cand != observables, axis=1)
        counts = PairedCounts.from_failures(fail_base, fail_cand)
        n = counts.shots
        cells.append(
            {
                "d": d,
                "rounds": rounds,
                "seed": seed,
                "shots": n,
                "hyperedges": count_hyperedges(dem),
                "decisions_identical": bool(np.array_equal(pred_base, pred_cand)),
                "table": asdict(counts),
                "baseline": {
                    "failures": counts.baseline_failures,
                    "p_logical": counts.baseline_failures / n,
                    "clopper_pearson": list(clopper_pearson(counts.baseline_failures, n, alpha)),
                },
                "candidate": {
                    "failures": counts.candidate_failures,
                    "p_logical": counts.candidate_failures / n,
                    "clopper_pearson": list(clopper_pearson(counts.candidate_failures, n, alpha)),
                },
                "mcnemar": mcnemar_exact(counts.n10, counts.n01),
                # Bootstrap-Seed aus dem Zell-Seed: reproduzierbar und je Zelle eigen.
                "delta": paired_bootstrap_delta(
                    counts.n00,
                    counts.n01,
                    counts.n10,
                    counts.n11,
                    n_boot=n_boot,
                    seed=seed,
                    alpha=alpha,
                ),
            }
        )

    return {
        "tool": "adaptiverg_qec.qec_decoder_ab.paired_decoder_ab",
        "baseline": baseline,
        "candidate": candidate,
        "manifest": manifest.to_dict(),
        "manifest_fingerprint": manifest.fingerprint(),
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
        "pairing": "identical detector samples per cell (one sampler call, two decoders)",
        "cells": cells,
        "claim_ceiling": (
            "bounded simulation; decoder accuracy on the declared Stim noise contract only; "
            "no latency, real-time or threshold claim"
        ),
    }
