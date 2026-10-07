"""Gate-Log fuer die Surface-Inkremente 3.1, 3.2 und 3.3.

Inkrement 3.1 (Multi-Round-Baseline) und 3.2 (NoiseProfile + Manifest v2) hatten Code und
Tests, aber kein committetes Gate-Log; AGENTS.md (Working agreement 1) zaehlt sie deshalb
nicht als erledigt. Dieses Modul faehrt die echten Pfade und schreibt EIN Gate-Log:

    python -m adaptiverg_qec.qec_evidence --json results/qec-multiround-evidence.json

Exit 0 nur, wenn alle Gates PASS sind; ohne [surface]-Extra Exit 2 (NOT_RUN, weder PASS
noch FAIL). CI faehrt das im surface-Job und laedt das Log hoch.

Gates (Q1-Q4, Q6, Q8, Q9 mit expliziter Gegenrichtung; Q5 ist ein Orakelvergleich, Q7
und Q10 sind Konsistenz-/Wiederholungschecks):
  Q1  Null-Rausch-Orakel: p=0 -> exakt 0 logische Fehler (d=3,5,7; X- und Z-Memory).
  Q2  Unter der Schwelle: p_L faellt streng mit d, Clopper-Pearson-Intervalle disjunkt.
  Q3  Ueber der Schwelle: p_L STEIGT mit d (Gegenrichtung zu Q2), Intervalle disjunkt.
  Q4  Manifest-Replay: JSON-Round-Trip -> bitgleiche Zeilen; anderer base_seed -> andere
      ERGEBNISSE (verglichen werden nur p_L je d, nicht das Seed-Feld selbst).
  Q5  McNemar-Implementierung == scipy.stats.binomtest (exakt) auf einem (b,c)-Gitter.
  Q6  Struktur-Negativkontrolle: DEM ohne Hyperkanten -> Correlated == Baseline bitgleich.
  Q7  Selbstvergleich: Baseline gegen Baseline -> b=c=0, p=1, Delta=0.
  Q8  Positivkontrolle: Circuit-Level-Noise mit Hyperkanten -> Correlated besser,
      exakter McNemar p<1e-3 und Bootstrap-CI von Delta komplett < 0.
  Q9  Paarungsgewinn: gepaartes CI deutlich schmaler als das ungepaarte Wald-CI; Gegen-
      kontrolle: gleiche Raender mit unabhaengiger Paarung -> KEIN Gewinn.
  Q10 A/B-Baseline == run_experiment_manifest (gleicher Seed, gleiche Fehlerzahl) und
      A/B-Wiederholung shot-genau identisch (SHA-256 beider Fehlvektoren je Zelle).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .qec_decoder_ab import (
    BASELINE_DECODER,
    CORRELATED_DECODER,
    PairedCounts,
    clopper_pearson,
    independent_pairing_table,
    mcnemar_exact,
    paired_bootstrap_delta,
    paired_decoder_ab,
)
from .qec_manifest_v2 import QECExperimentManifestV2, StimNoiseProfile

ALPHA_ORDER = 0.01  # 99%-Intervalle fuer die Ordnungs-Gates Q2/Q3.


def _rows_hash(rows: list[dict]) -> str:
    # Nur Ergebnisfelder: das Seed-Feld aendert sich mit base_seed IMMER und wuerde die
    # Gegenrichtung von Q4 vakuos machen (Review PR #54).
    outcome = [(r["d"], r["rounds"], r["shots"], r["p_logical"]) for r in rows]
    blob = json.dumps(outcome, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _phenom(p: float) -> StimNoiseProfile:
    return StimNoiseProfile(
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


def _circuit_level(p: float) -> StimNoiseProfile:
    return StimNoiseProfile(
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
    )


def _ordered(rows: list[dict], *, increasing: bool) -> tuple[bool, list]:
    """Streng monoton in d mit disjunkten Clopper-Pearson-Intervallen."""
    ivs = []
    for r in rows:
        k = round(r["p_logical"] * r["shots"])
        ivs.append(clopper_pearson(k, r["shots"], ALPHA_ORDER))
    ok = True
    for (lo_a, hi_a), (lo_b, hi_b) in zip(ivs, ivs[1:], strict=False):
        ok &= (hi_a < lo_b) if increasing else (lo_a > hi_b)
    return bool(ok), [list(iv) for iv in ivs]


def build_evidence(*, quick: bool = False) -> dict:
    from . import surface_decoder as sd

    sd._require_surface()
    scale = 0.25 if quick else 1.0
    gates: list[dict] = []
    sections: dict = {}

    def gate(gid: str, name: str, passed: bool, detail: dict) -> None:
        gates.append({"id": gid, "name": name, "status": "PASS" if passed else "FAIL", **detail})

    # --- Q1 Null-Rausch-Orakel ---------------------------------------------------
    zero = {}
    for basis in ("x", "z"):
        m = QECExperimentManifestV2(
            memory_basis=basis, distances=(3, 5, 7), shots_per_cell=2_000, base_seed=1
        )
        zero[basis] = sd.run_experiment_manifest(m)["rows"]
    q1 = all(r["p_logical"] == 0.0 for rows in zero.values() for r in rows)
    gate("Q1", "zero-noise oracle p_L == 0 (d=3,5,7; x,z)", q1, {"rows": zero})

    # --- Q2/Q3 Unterdrueckung vs. Umkehr ------------------------------------------
    shots_sub = int(60_000 * scale)
    sub = QECExperimentManifestV2(
        distances=(3, 5, 7), shots_per_cell=shots_sub, base_seed=31, noise=_phenom(0.01)
    )
    sub_rows = sd.run_experiment_manifest(sub)["rows"]
    q2, q2_iv = _ordered(sub_rows, increasing=False)
    gate(
        "Q2",
        "below threshold (phenom p=0.01): p_L strictly decreasing in d, 99% CP disjoint",
        q2,
        {"rows": sub_rows, "clopper_pearson_99": q2_iv},
    )
    above = QECExperimentManifestV2(
        distances=(3, 5, 7), shots_per_cell=int(20_000 * scale), base_seed=32, noise=_phenom(0.06)
    )
    above_rows = sd.run_experiment_manifest(above)["rows"]
    q3, q3_iv = _ordered(above_rows, increasing=True)
    gate(
        "Q3",
        "above threshold (phenom p=0.06): p_L strictly increasing in d, 99% CP disjoint",
        q3,
        {"rows": above_rows, "clopper_pearson_99": q3_iv},
    )

    # --- Q4 Manifest-Replay -------------------------------------------------------
    replay = QECExperimentManifestV2.from_dict(json.loads(json.dumps(sub.to_dict())))
    replay_rows = sd.run_experiment_manifest(replay)["rows"]
    moved = QECExperimentManifestV2.from_dict({**sub.to_dict(), "base_seed": sub.base_seed + 1})
    moved_rows = sd.run_experiment_manifest(moved)["rows"]
    h0, h1, h2 = _rows_hash(sub_rows), _rows_hash(replay_rows), _rows_hash(moved_rows)
    q4 = h0 == h1 and h0 != h2 and replay.fingerprint() == sub.fingerprint()
    gate(
        "Q4",
        "manifest JSON round-trip replays bit-identical outcomes; changed base_seed changes them",
        q4,
        {"rows_hash": h0, "replay_rows_hash": h1, "moved_seed_rows_hash": h2},
    )

    # --- Q5 McNemar gegen scipy ----------------------------------------------------
    from scipy import stats

    worst = 0.0
    grid = [(b, c) for b in range(0, 41, 3) for c in range(0, 41, 4) if b + c > 0]
    grid += [(187, 107), (142, 51), (1000, 1000), (0, 25)]
    for b, c in grid:
        ref = stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        worst = max(worst, abs(mcnemar_exact(b, c)["p_value"] - ref))
    q5 = worst < 1e-12 and mcnemar_exact(0, 0)["p_value"] == 1.0
    gate(
        "Q5",
        "exact McNemar equals scipy.stats.binomtest",
        q5,
        {"grid_size": len(grid), "max_abs_diff": worst},
    )

    # --- Q6/Q7 Negativkontrollen ---------------------------------------------------
    no_hyper = QECExperimentManifestV2(
        distances=(3, 5),
        shots_per_cell=int(20_000 * scale),
        base_seed=61,
        noise=StimNoiseProfile(
            before_measure_flip_probability=0.02, after_reset_flip_probability=0.02
        ),
    )
    ab_nh = paired_decoder_ab(no_hyper, n_boot=2_000)
    q6 = all(
        c["hyperedges"] == 0
        and c["decisions_identical"]
        and c["mcnemar"]["n_discordant"] == 0
        and c["baseline"]["failures"] > 0  # sonst waere "identisch" trivial
        for c in ab_nh["cells"]
    )
    gate(
        "Q6",
        "structural negative control: no hyperedges -> correlated decisions bit-identical",
        q6,
        {"cells": _cell_summary(ab_nh)},
    )

    circuit = QECExperimentManifestV2(
        distances=(3, 5),
        shots_per_cell=int(40_000 * scale),
        base_seed=81,
        noise=_circuit_level(0.006),
    )
    ab_self = paired_decoder_ab(circuit, candidate=BASELINE_DECODER, n_boot=2_000)
    q7 = all(
        c["mcnemar"]["n_discordant"] == 0
        and c["mcnemar"]["p_value"] == 1.0
        and c["delta"]["delta"] == 0.0
        for c in ab_self["cells"]
    )
    gate(
        "Q7",
        "self-comparison baseline vs baseline: b=c=0, p=1, delta=0",
        q7,
        {"cells": _cell_summary(ab_self)},
    )

    # --- Q8/Q9/Q10 Positivkontrolle + Paarung + Replay -------------------------------
    ab = paired_decoder_ab(circuit, candidate=CORRELATED_DECODER)
    big = next(c for c in ab["cells"] if c["d"] == 5)
    q8 = (
        big["hyperedges"] > 0
        and big["candidate"]["failures"] < big["baseline"]["failures"]
        and big["mcnemar"]["p_value"] < 1e-3
        and big["delta"]["ci"][1] < 0.0
    )
    gate(
        "Q8",
        "positive control (circuit-level p=0.006, d=5): correlated MWPM better, "
        "exact McNemar p<1e-3, paired bootstrap CI(delta) < 0",
        q8,
        {"cell_d5": _cell_summary({"cells": [big]})[0]},
    )
    widths = []
    for c in ab["cells"]:
        indep = independent_pairing_table(PairedCounts(**c["table"]))
        ind = paired_bootstrap_delta(
            indep.n00, indep.n01, indep.n10, indep.n11, n_boot=20_000, seed=c["seed"]
        )
        widths.append(
            {
                "d": c["d"],
                "paired_ratio": c["delta"]["ci_width"] / c["delta"]["unpaired_wald_ci_width"],
                "independent_pairing_ratio": ind["ci_width"] / ind["unpaired_wald_ci_width"],
                "independent_table": indep.__dict__,
            }
        )
    q9 = all(
        w["paired_ratio"] < 0.8 and 0.9 <= w["independent_pairing_ratio"] <= 1.1 for w in widths
    )
    gate(
        "Q9",
        "pairing gain: paired/unpaired CI width < 0.8; counter-control with the same margins "
        "but independent pairing gives no gain (ratio in [0.9, 1.1])",
        q9,
        {"widths": widths},
    )
    direct = sd.run_experiment_manifest(circuit)["rows"]
    again = paired_decoder_ab(circuit, candidate=CORRELATED_DECODER)
    base_match = all(
        round(r["p_logical"] * r["shots"]) == c["baseline"]["failures"] and r["seed"] == c["seed"]
        for r, c in zip(direct, ab["cells"], strict=True)
    )
    shot_identical = all(
        a["baseline_failures_sha256"] == b["baseline_failures_sha256"]
        and a["candidate_failures_sha256"] == b["candidate_failures_sha256"]
        for a, b in zip(ab["cells"], again["cells"], strict=True)
    )
    q10 = (
        base_match
        and shot_identical
        and json.dumps(ab["cells"], sort_keys=True) == json.dumps(again["cells"], sort_keys=True)
    )
    gate(
        "Q10",
        "A/B baseline: same seed + failure count as run_experiment_manifest; "
        "A/B rerun shot-identical (failure-vector SHA-256)",
        q10,
        {"baseline_matches_manifest_run": base_match, "rerun_shot_identical": shot_identical},
    )

    sections["decoder_ab_circuit_level"] = ab
    sections["decoder_ab_no_hyperedges"] = ab_nh
    sections["manifest_rows"] = {
        "below_threshold": sub.to_dict(),
        "above_threshold": above.to_dict(),
    }
    any_payload = sd.run_experiment_manifest(
        QECExperimentManifestV2(distances=(3,), shots_per_cell=1)
    )
    return {
        "tool": "adaptiverg_qec.qec_evidence",
        "quick": quick,
        "stim_version": any_payload["stim_version"],
        "pymatching_version": any_payload["pymatching_version"],
        "runtime_environment": any_payload["runtime_environment"],
        "reproducibility": any_payload["reproducibility"],
        "gates": gates,
        "n_pass": sum(g["status"] == "PASS" for g in gates),
        "n_gates": len(gates),
        "all_pass": all(g["status"] == "PASS" for g in gates),
        "sections": sections,
        "claim_ceiling": (
            "bounded simulation (T1-candidate): orderings, replay and paired decoder accuracy "
            "on declared Stim noise contracts; no threshold value, no FSS, no latency claim"
        ),
    }


def _cell_summary(payload: dict) -> list[dict]:
    return [
        {
            "d": c["d"],
            "hyperedges": c["hyperedges"],
            "decisions_identical": c["decisions_identical"],
            "baseline_failures": c["baseline"]["failures"],
            "candidate_failures": c["candidate"]["failures"],
            "b": c["mcnemar"]["b"],
            "c": c["mcnemar"]["c"],
            "p_value": c["mcnemar"]["p_value"],
            "delta": c["delta"]["delta"],
            "delta_ci": c["delta"]["ci"],
            "shots": c["shots"],
        }
        for c in payload["cells"]
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=Path("results/qec-multiround-evidence.json"))
    ap.add_argument("--quick", action="store_true", help="kleineres Shot-Budget (nur lokal)")
    args = ap.parse_args(argv)

    from . import surface_decoder as sd

    if not sd.HAVE_SURFACE:
        print("NOT_RUN: [surface]-Extras fehlen (pip install 'adaptiverg-qec[surface]')")
        return 2
    from .qec_decoder_ab import _require_decoders

    try:
        _require_decoders()
    except ImportError as exc:
        print(f"NOT_RUN: {exc}")
        return 2
    t0 = time.perf_counter()
    payload = build_evidence(quick=args.quick)
    payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for g in payload["gates"]:
        print(f"[{g['status']}] {g['id']:>3}  {g['name']}")
    print(f"{payload['n_pass']}/{payload['n_gates']} PASS  -> {args.json}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
