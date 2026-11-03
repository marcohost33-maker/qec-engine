"""Inkrement 3.4: vorregistrierte p x d-Sweep-Engine + Bootstrap-FSS (phenomenological).

Gate-Log:  python -m adaptiverg_qec.qec_sweep --json results/qec-sweep-fss.json

WARUM EIN EIGENER PLAN-VERTRAG
``QECExperimentManifestV2`` beschreibt EINE Noise-Zeile ueber mehrere Distanzen. Ein
Threshold-Schaetzer braucht ein GITTER, und das Gitter muss VOR dem Lauf feststehen:
sonst kann man p-Punkte, Distanzen oder Shots nachtraeglich so waehlen, dass die Kreuzung
dort liegt, wo man sie haben will. ``SweepPlan`` friert deshalb Gitter, Shot-Budget
(fixed shots, KEIN datenabhaengiges Stoppen), FSS-Modell, Bootstrap-Groesse und Seed
ein; sein SHA-256-Fingerprint steht im Ergebnis und ``verify_result`` prueft, dass die
Zellen genau dieses Gitter abdecken.

NOISE-KONVENTION (die Falle, die dieser Vertrag schliesst)
Die Literatur-Schwelle ~2.9 % fuer MWPM unter phenomenological noise (Wang, Harrington &
Preskill 2003, doi:10.1016/S0003-4916(02)00019-2; zitiert u.a. in Bonilla Ataides et al.,
The XZZX surface code, Nat. Commun. 12:2172 (2021)) gilt fuer UNABHAENGIGE Bit-Flips mit
Wahrscheinlichkeit p je Daten-Qubit und Runde plus Messfehler q = p. Stims
``before_round_data_depolarization=p`` ist dagegen DEPOLARIZE1(p): X, Y, Z je p/3, also
X-Marginale 2p/3. Wer Stim-Depolarisierung mit 2.9 % vergleicht, vergleicht zwei
verschiedene Kanaele (eine oeffentliche Stim-Reproduktion landet so bei ~3.9 %).

``noise_model="phenomenological-bitflip"`` ersetzt deshalb jede DEPOLARIZE1(p)-Instruktion
des Stim-Memory-Circuits durch X_ERROR(p) (rekursiv, auch in REPEAT-Bloecken) und laesst
den Messfehler X_ERROR(p) vor jeder Messung stehen -- das ist der WHP-Kanal. Abweichung
zum WHP-Modell, offen benannt: Stim flippt auch die FINALE Daten-Messung mit p (WHP:
letzte Runde perfekt); das ist eine zusaetzliche Datenfehler-Schicht am Zeitrand und eine
Finite-Size-, keine Kanal-Abweichung.

``noise_model="stim-depolarizing"`` ist der bisherige Stim-Kanal, nur als Gegenkontrolle.

ZELL-IDENTITAET: jede (d, p)-Zelle zieht ihren Seed per SHA-256 aus dem kompletten Plan
plus (d, p); kein geteilter Strom, kein stilles CRN -> Zellen unabhaengig -> der
Zell-Bootstrap in ``qec_fss`` ist gueltig.

CLAIM CEILING: FSS-gestuetzte bounded simulation (Claim-Tier "FSS-supported simulation"
aus docs/SOTA_QEC_2026_NEXT.md) fuer DIESEN Kanal, Rotated-Surface-Memory, rounds = d,
PyMatching-MWPM. Endliche Distanzen (d <= 11): kein L->oo-Beweis, keine Latenz, keine
Hardware-Aussage, keine Uebertragung auf andere Noise-Konventionen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from . import __version__
from .qec_decoder_ab import clopper_pearson
from .qec_fss import (
    Cell,
    exact_coverage,
    fss_bootstrap,
    intervals,
    jeffreys_interval,
    repetition_fss_dataset,
    repetition_nu_oracle,
    wald_interval,
    wilson_interval,
)
from .qec_manifest_v2 import _require_exact_keys

SWEEP_PLAN_SCHEMA = "adaptiverg_qec.qec_sweep_plan/v1"
NOISE_MODELS = ("phenomenological-bitflip", "stim-depolarizing")
PHENOM_MWPM_THRESHOLD_LITERATURE = 0.029  # WHP 2003, MWPM, X-flip p == meas-flip q.
PHENOM_OPTIMAL_THRESHOLD_LITERATURE = 0.033  # RPGM/Nishimori (Ohno et al. 2004), NICHT MWPM.


def _int(name: str, v, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")
    return int(v)


@dataclass(frozen=True)
class SweepPlan:
    """Vorregistrierter Threshold-Sweep. Alles Laufbestimmende steht hier."""

    schema: str = SWEEP_PLAN_SCHEMA
    code_family: str = "surface_code:rotated_memory"
    memory_basis: str = "z"
    noise_model: str = "phenomenological-bitflip"
    distances: tuple[int, ...] = (5, 7, 9, 11)
    p_grid: tuple[float, ...] = (0.024, 0.026, 0.028, 0.030, 0.032, 0.034)
    rounds_policy: str = "distance"
    shots_per_cell: int = 50_000
    stopping_rule: str = "fixed-shots"
    decoder: str = "pymatching-mwpm-dem"
    base_seed: int = 20261103
    seed_policy: str = "sweep-plan-sha256-v1"
    fss_model: str = "whp-quadratic-v1"
    n_boot: int = 400
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.schema != SWEEP_PLAN_SCHEMA:
            raise ValueError(f"plan schema mismatch: {self.schema!r}")
        if self.code_family != "surface_code:rotated_memory":
            raise ValueError(f"unsupported code_family {self.code_family!r}")
        if self.memory_basis not in {"x", "z"}:
            raise ValueError("memory_basis must be 'x' or 'z'")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
        ds = tuple(self.distances)
        if len(ds) < 2 or tuple(sorted(set(ds))) != ds:
            raise ValueError("distances: >= 2, unique, strictly increasing")
        for d in ds:
            if isinstance(d, bool) or not isinstance(d, int) or d < 3 or d % 2 == 0:
                raise ValueError(f"distances must be odd ints >= 3, got {ds}")
        object.__setattr__(self, "distances", ds)
        ps = tuple(self.p_grid)
        if len(ps) < 3:
            raise ValueError("p_grid needs >= 3 points")
        for p in ps:
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise TypeError("p_grid entries must be real")
            if not math.isfinite(p) or not 0.0 < p < 0.5:
                raise ValueError(f"p_grid entries must be in (0, 0.5), got {p}")
        ps = tuple(float(p) for p in ps)
        if tuple(sorted(set(ps))) != ps:
            raise ValueError("p_grid must be unique and strictly increasing")
        object.__setattr__(self, "p_grid", ps)
        if len(ds) * len(ps) < 6:
            raise ValueError("grid needs >= 6 cells for a 5-parameter FSS fit")
        if self.rounds_policy != "distance":
            raise ValueError("rounds_policy must be 'distance' (rounds = d)")
        _int("shots_per_cell", self.shots_per_cell, 1)
        if self.stopping_rule != "fixed-shots":
            raise ValueError("only the pre-registered 'fixed-shots' rule is supported")
        if self.decoder != "pymatching-mwpm-dem":
            raise ValueError(f"unsupported decoder {self.decoder!r}")
        _int("base_seed", self.base_seed, 0)
        if self.seed_policy != "sweep-plan-sha256-v1":
            raise ValueError(f"unsupported seed_policy {self.seed_policy!r}")
        if self.fss_model != "whp-quadratic-v1":
            raise ValueError(f"unsupported fss_model {self.fss_model!r}")
        _int("n_boot", self.n_boot, 50)
        if isinstance(self.alpha, bool) or not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        object.__setattr__(self, "alpha", float(self.alpha))

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["distances"] = list(self.distances)
        d["p_grid"] = list(self.p_grid)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepPlan:
        if not isinstance(data, dict):
            raise TypeError("sweep plan must be a JSON object")
        _require_exact_keys("sweep plan", data, set(cls.__dataclass_fields__))
        clean = dict(data)
        clean["distances"] = tuple(clean["distances"])
        clean["p_grid"] = tuple(clean["p_grid"])
        return cls(**clean)

    def _canonical(self, extra: dict | None = None) -> bytes:
        payload: dict[str, Any] = {"plan": self.to_dict()}
        if extra is not None:
            payload.update(extra)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def fingerprint(self) -> str:
        return hashlib.sha256(self._canonical()).hexdigest()

    def cell_seed(self, d: int, p: float) -> int:
        if d not in self.distances or float(p) not in self.p_grid:
            raise ValueError(f"cell (d={d}, p={p}) is not in the registered grid")
        digest = hashlib.sha256(self._canonical({"cell": [int(d), float(p)]})).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


# =============================================================================
# Stim-Pfad (braucht [surface])
# =============================================================================


def _require():
    from . import surface_decoder as sd

    sd._require_surface()
    return sd.stim, sd.pymatching


def _depol_to_xflip(circuit, stim):
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(inst.repeat_count, _depol_to_xflip(inst.body_copy(), stim))
            )
        elif inst.name == "DEPOLARIZE1":
            args = inst.gate_args_copy()
            out.append(stim.CircuitInstruction("X_ERROR", inst.targets_copy(), args))
        else:
            out.append(inst)
    return out


def build_circuit(
    noise_model: str, memory_basis: str, d: int, p: float, *, p_meas: float | None = None
):
    """Rotated-Surface-Memory, rounds = d, Datenfehler p, Messfehler p_meas (Default p)."""
    stim, _ = _require()
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
    c = stim.Circuit.generated(
        f"surface_code:rotated_memory_{memory_basis}",
        distance=int(d),
        rounds=int(d),
        before_round_data_depolarization=float(p),
        before_measure_flip_probability=float(p if p_meas is None else p_meas),
    )
    if noise_model == "phenomenological-bitflip":
        c = _depol_to_xflip(c, stim)
        if "DEPOLARIZE" in str(c):  # fail-closed: Konvertierung muss vollstaendig sein
            raise RuntimeError("depolarizing noise survived the bit-flip conversion")
    return c


def decode_cell(
    noise_model: str,
    memory_basis: str,
    d: int,
    p: float,
    shots: int,
    seed: int,
    *,
    p_meas: float | None = None,
) -> int:
    """Ein Sampler-Aufruf, PyMatching-MWPM auf dem graphlike DEM. Gibt k (Fehler) zurueck."""
    _, pymatching = _require()
    circuit = build_circuit(noise_model, memory_basis, d, p, p_meas=p_meas)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=int(seed))
    det, obs = sampler.sample(int(shots), separate_observables=True)
    failures = np.any(matching.decode_batch(det) != obs, axis=1)
    return int(failures.sum())


def run_sweep(plan: SweepPlan) -> dict:
    """Fuehre den Plan aus (fixed shots), dann Bootstrap-FSS. Keine Datenabhaengigkeit."""
    if not isinstance(plan, SweepPlan):
        raise TypeError("plan must be a SweepPlan")
    stim, pymatching = _require()
    from . import surface_decoder as sd

    t0 = time.time()
    rows = []
    for d in plan.distances:
        for p in plan.p_grid:
            seed = plan.cell_seed(d, p)
            k = decode_cell(plan.noise_model, plan.memory_basis, d, p, plan.shots_per_cell, seed)
            n = plan.shots_per_cell
            rows.append(
                {
                    "d": d,
                    "rounds": d,
                    "p": p,
                    "seed": seed,
                    "k": k,
                    "n": n,
                    "p_logical": k / n,
                    "intervals": intervals(k, n, plan.alpha),
                }
            )
    cells = [Cell(r["d"], r["p"], r["k"], r["n"]) for r in rows]
    fss = fss_bootstrap(
        cells,
        n_boot=plan.n_boot,
        seed=plan.cell_seed(plan.distances[0], plan.p_grid[0]) % 2**32,
        alpha=plan.alpha,
    )
    return {
        "tool": "adaptiverg_qec.qec_sweep.run_sweep",
        "plan": plan.to_dict(),
        "plan_fingerprint": plan.fingerprint(),
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
        "runtime_environment": {
            "package_version": __version__,
            "distribution_version": sd._distribution_version(),
            "git_sha": sd._git_sha(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "reproducibility_tier": "SEEDED_SAME_STIM_VERSION_AND_ARCHITECTURE",
        "rows": rows,
        "fss": fss,
        "wallclock_s": round(time.time() - t0, 2),
    }


def rows_hash(rows: list[dict]) -> str:
    """Nur Ergebnisfelder (d, p, k, n) -- Seeds aendern sich mit base_seed immer."""
    blob = json.dumps([(r["d"], r["p"], r["k"], r["n"]) for r in rows], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def verify_result(result: dict) -> list[str]:
    """Fail-closed-Pruefung eines Sweep-Ergebnisses gegen seinen eigenen Plan.

    Leere Liste = konsistent. Fehler: Plan nicht rekonstruierbar, Fingerprint passt nicht,
    Zellmenge != registriertes Gitter, Shot-Zahl abweichend, Seed nicht der Plan-Seed.
    """
    problems: list[str] = []
    try:
        plan = SweepPlan.from_dict(result["plan"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"plan not reconstructible: {exc}"]
    if plan.fingerprint() != result.get("plan_fingerprint"):
        problems.append("plan fingerprint mismatch (plan edited after registration?)")
    want = {(d, p) for d in plan.distances for p in plan.p_grid}
    got = [(r["d"], r["p"]) for r in result.get("rows", [])]
    if len(got) != len(set(got)):
        problems.append("duplicate cells")
    if set(got) != want:
        problems.append(
            f"cells differ from registered grid: missing={sorted(want - set(got))}, "
            f"extra={sorted(set(got) - want)}"
        )
    for r in result.get("rows", []):
        if (r["d"], r["p"]) not in want:
            continue
        if r["n"] != plan.shots_per_cell:
            problems.append(f"cell {(r['d'], r['p'])}: n={r['n']} != {plan.shots_per_cell}")
        if r["seed"] != plan.cell_seed(r["d"], r["p"]):
            problems.append(f"cell {(r['d'], r['p'])}: seed is not the plan seed")
    return problems


# =============================================================================
# Gate-Log
# =============================================================================


def _two_proportion_z(k1: int, k2: int, n: int) -> float:
    p = (k1 + k2) / (2 * n)
    se = math.sqrt(2 * p * (1 - p) / n)
    return (k1 - k2) / (n * se) if se > 0 else 0.0


def _brackets(rows, d_list, p, *, increasing: bool, alpha: float) -> tuple[bool, dict]:
    """Punktschaetzer streng monoton in d UND 99%-CP von d_min/d_max disjunkt."""
    by = {(r["d"], r["p"]): r for r in rows}
    ph = [by[(d, p)]["k"] / by[(d, p)]["n"] for d in d_list]
    mono = all((b > a) if increasing else (b < a) for a, b in zip(ph, ph[1:], strict=False))
    lo_iv = clopper_pearson(by[(d_list[0], p)]["k"], by[(d_list[0], p)]["n"], alpha)
    hi_iv = clopper_pearson(by[(d_list[-1], p)]["k"], by[(d_list[-1], p)]["n"], alpha)
    disjoint = (lo_iv[1] < hi_iv[0]) if increasing else (lo_iv[0] > hi_iv[1])
    return bool(mono and disjoint), {
        "p_logical_by_d": ph,
        "cp99_d_min": list(lo_iv),
        "cp99_d_max": list(hi_iv),
    }


def build_evidence(*, quick: bool = False, replicates: int = 100) -> dict:
    gates: list[dict] = []

    def gate(gid: str, name: str, passed: bool, detail: dict) -> None:
        gates.append({"id": gid, "name": name, "status": "PASS" if passed else "FAIL", **detail})

    # ---- F1 Intervall-Formeln gegen unabhaengige Referenzen ----------------------
    diffs = []
    for k, n in [(0, 10), (3, 10), (10, 10), (0, 50_000), (17, 50_000), (2500, 50_000)]:
        z = float(stats.norm.ppf(0.975))
        # Wilson als Inversion des Score-Tests: Wurzeln von (ph-p)^2 = z^2 p(1-p)/n.
        ph = k / n
        a2, a1, a0 = 1 + z * z / n, -(2 * ph + z * z / n), ph * ph
        roots = np.sort(np.roots([a2, a1, a0]).real)
        w = wilson_interval(k, n)
        diffs.append(abs(w[0] - max(roots[0], 0.0)) + abs(w[1] - min(roots[1], 1.0)))
        j = jeffreys_interval(k, n)
        ref_lo = 0.0 if k == 0 else stats.beta(k + 0.5, n - k + 0.5).ppf(0.025)
        ref_hi = 1.0 if k == n else stats.beta(k + 0.5, n - k + 0.5).ppf(0.975)
        diffs.append(abs(j[0] - ref_lo) + abs(j[1] - ref_hi))
    gate(
        "F1",
        "Wilson == score-test roots, Jeffreys == Beta(k+1/2, n-k+1/2) quantiles",
        max(diffs) < 1e-10,
        {"max_abs_diff": max(diffs)},
    )

    # ---- F2 exakte Coverage bei seltenen Fehlern (beidseitig) ---------------------
    n_cov = 20_000
    grid = np.geomspace(2e-5, 2e-3, 25)
    cov = {
        m: [exact_coverage(m, n_cov, float(p)) for p in grid]
        for m in ("wilson", "jeffreys", "clopper_pearson", "wald")
    }
    mean = {m: float(np.mean(v)) for m, v in cov.items()}
    mins = {m: float(np.min(v)) for m, v in cov.items()}
    f2 = (
        0.94 <= mean["wilson"] <= 0.96
        and 0.94 <= mean["jeffreys"] <= 0.97
        and mins["clopper_pearson"] >= 0.95
        and mean["wald"] < 0.85
        and wald_interval(0, n_cov) == (0.0, 0.0)
    )
    gate(
        "F2",
        "exact coverage, n=20000, p in [2e-5,2e-3]: Wilson/Jeffreys ~0.95, CP >= 0.95; "
        "Wald collapses (k=0 -> width 0)",
        f2,
        {
            "mean_coverage": mean,
            "min_coverage": mins,
            "n": n_cov,
            "p_grid": [float(p) for p in grid],
        },
    )

    # ---- F3/F4 FSS gegen das geschlossene Repetition-Orakel -----------------------
    ds_rep = (9, 15, 21, 27)
    pg_rep = tuple(float(x) for x in np.linspace(0.47, 0.53, 7))
    nu_or = repetition_nu_oracle(ds_rep)
    rep = fss_bootstrap(repetition_fss_dataset(ds_rep, pg_rep, 50_000, 11), n_boot=400, seed=11)
    f3 = (
        rep["p_th_ci"][0] <= 0.5 <= rep["p_th_ci"][1]
        and rep["nu_ci"][0] <= nu_or["nu_effective"] <= rep["nu_ci"][1]
        and rep["point"]["chi2_p_value"] > 0.01
    )
    gate(
        "F3",
        "repetition oracle: 95% CI covers exact p_th=1/2 and exact effective nu; chi2 p > 0.01",
        f3,
        {"fss": rep, "nu_oracle": nu_or, "distances": ds_rep, "p_grid": pg_rep},
    )

    R = max(10, replicates // (4 if quick else 1))
    hit_p, hit_nu = 0, 0
    for r in range(R):
        b = fss_bootstrap(
            repetition_fss_dataset(ds_rep, pg_rep, 50_000, 1000 + r), n_boot=200, seed=1000 + r
        )
        hit_p += b["p_th_ci"][0] <= 0.5 <= b["p_th_ci"][1]
        hit_nu += b["nu_ci"][0] <= nu_or["nu_effective"] <= b["nu_ci"][1]
    cov_p, cov_nu = hit_p / R, hit_nu / R
    # 95%-Nominal; Toleranz = 3 Binomial-SD bei R Replikaten, nach unten.
    tol = 3.0 * math.sqrt(0.95 * 0.05 / R)
    gate(
        "F4",
        f"bootstrap CI coverage over {R} replicates >= 0.95 - 3 SD (p_th and nu)",
        cov_p >= 0.95 - tol and cov_nu >= 0.95 - tol,
        {"replicates": R, "coverage_p_th": cov_p, "coverage_nu": cov_nu, "tolerance": tol},
    )

    # ---- F5 Negativkontrolle: keine Kreuzung im Fenster ---------------------------
    rng = np.random.Generator(np.random.Philox(5))
    no_cross = [
        Cell(d, p, int(rng.binomial(50_000, min(0.9, p * (1 + 0.05 * d)))), 50_000)
        for d in (5, 7, 9)
        for p in (0.1, 0.12, 0.14, 0.16)
    ]
    nc = fss_bootstrap(no_cross, n_boot=100, seed=5)
    f5 = (
        not nc["point"]["p_th_in_window"]
        or nc["point"]["at_parameter_bound"]
        or nc["point"]["chi2_p_value"] < 1e-3
    ) and all(c["crossing"] is None for c in nc["crossings"])
    gate(
        "F5",
        "negative control (curves ordered everywhere): no crossing, fit flagged",
        f5,
        {"point": nc["point"], "crossings": nc["crossings"]},
    )

    sections: dict = {}
    try:
        _require()
    except ImportError as exc:
        return {
            "tool": "adaptiverg_qec.qec_sweep.build_evidence",
            "surface": "NOT_RUN",
            "reason": str(exc),
            "gates": gates,
        }

    # ---- S1 Konventions-Aequivalenz + Gegenrichtung -------------------------------
    n_conv = 40_000 if not quick else 10_000
    q, d_conv = 0.03, 5
    k_bit = decode_cell("phenomenological-bitflip", "z", d_conv, q, n_conv, 101)
    k_dep15 = decode_cell("stim-depolarizing", "z", d_conv, 1.5 * q, n_conv, 102, p_meas=q)
    k_dep = decode_cell("stim-depolarizing", "z", d_conv, q, n_conv, 103)
    z_same = _two_proportion_z(k_bit, k_dep15, n_conv)
    z_trap = _two_proportion_z(k_bit, k_dep, n_conv)
    c = build_circuit("phenomenological-bitflip", "z", 3, q)
    s1 = abs(z_same) < 3.0 and z_trap > 5.0 and "DEPOLARIZE" not in str(c)
    gate(
        "S1",
        "bit-flip(q) ~ Stim-depolarizing(data 3q/2, meas q): |z|<3; "
        "Stim-depolarizing(q) is a different, weaker channel (z>5)",
        s1,
        {
            "d": d_conv,
            "q": q,
            "shots": n_conv,
            "k_bitflip_q": k_bit,
            "k_depol_1.5q": k_dep15,
            "k_depol_q": k_dep,
            "z_same": z_same,
            "z_trap": z_trap,
            "why": "DEPOLARIZE1(p) has X-marginal 2p/3; Z-memory sees only X/Y components",
        },
    )

    # ---- S2..S5 vorregistrierter Sweep --------------------------------------------
    plan = SweepPlan(shots_per_cell=12_500 if quick else 50_000, n_boot=200 if quick else 400)
    registered = plan.fingerprint()
    res = run_sweep(plan)
    sections["sweep"] = res
    rows, fss = res["rows"], res["fss"]
    lo_ok, lo_iv = _brackets(rows, plan.distances, plan.p_grid[0], increasing=False, alpha=0.01)
    hi_ok, hi_iv = _brackets(rows, plan.distances, plan.p_grid[-1], increasing=True, alpha=0.01)
    gate(
        "S2",
        "grid brackets the threshold: at p_min p_L strictly decreasing in d, at p_max "
        "strictly increasing (point estimates; 99% CP of d_min vs d_max disjoint)",
        lo_ok and hi_ok,
        {"p_min": plan.p_grid[0], "at_p_min": lo_iv, "p_max": plan.p_grid[-1], "at_p_max": hi_iv},
    )

    pt = fss["point"]
    s3 = (
        pt["p_th_in_window"]
        and not pt["at_parameter_bound"]
        and pt["chi2_p_value"] > 0.01
        and fss["fit_success_fraction"] >= 0.99
        and fss["p_th_ci"] is not None
        and plan.p_grid[0] < fss["p_th_ci"][0]
        and fss["p_th_ci"][1] < plan.p_grid[-1]
    )
    gate(
        "S3",
        "FSS fit adequate: p_th and its 95% CI inside the window, chi2 p > 0.01, "
        ">= 99% bootstrap refits succeed",
        s3,
        {
            "p_th": pt["p_th"],
            "p_th_ci": fss["p_th_ci"],
            "nu": pt["nu"],
            "nu_ci": fss["nu_ci"],
            "chi2_per_dof": pt["chi2_per_dof"],
            "chi2_p_value": pt["chi2_p_value"],
            "crossings": fss["crossings"],
        },
    )

    lit_tol = 0.002  # VORAB festgelegt; WHP geben 2.9 % ohne engere Fehlerangabe.
    s4 = abs(pt["p_th"] - PHENOM_MWPM_THRESHOLD_LITERATURE) <= lit_tol and (
        fss["p_th_ci"][1] < PHENOM_OPTIMAL_THRESHOLD_LITERATURE
    )
    gate(
        "S4",
        "external anchor: |p_th - 0.029| <= 0.002 (MWPM, WHP 2003) and CI below the "
        "optimal-decoder value 0.033 (MWPM must not reach it)",
        s4,
        {
            "p_th": pt["p_th"],
            "literature_mwpm": PHENOM_MWPM_THRESHOLD_LITERATURE,
            "literature_optimal": PHENOM_OPTIMAL_THRESHOLD_LITERATURE,
            "tolerance": lit_tol,
            "abs_err": abs(pt["p_th"] - PHENOM_MWPM_THRESHOLD_LITERATURE),
        },
    )

    # ---- S5 Vorregistrierung + Replay ----------------------------------------------
    replay_plan = SweepPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    small = SweepPlan.from_dict(
        {**plan.to_dict(), "distances": [5, 7], "shots_per_cell": 2_000, "n_boot": 50}
    )
    a = run_sweep(small)
    b = run_sweep(SweepPlan.from_dict(json.loads(json.dumps(small.to_dict()))))
    moved = run_sweep(SweepPlan.from_dict({**small.to_dict(), "base_seed": small.base_seed + 1}))
    tampered = json.loads(json.dumps(res))
    tampered["plan"]["p_grid"] = tampered["plan"]["p_grid"][1:]  # nachtraeglich Gitter kuerzen
    dropped = json.loads(json.dumps(res))
    dropped["rows"] = dropped["rows"][1:]  # unbequeme Zelle weglassen
    s5 = (
        replay_plan.fingerprint() == registered == res["plan_fingerprint"]
        and verify_result(res) == []
        and rows_hash(a["rows"]) == rows_hash(b["rows"])
        and rows_hash(a["rows"]) != rows_hash(moved["rows"])
        and verify_result(tampered) != []
        and verify_result(dropped) != []
    )
    gate(
        "S5",
        "pre-registration: fingerprint fixed before run; replay bit-identical; "
        "base_seed+1 changes results; edited plan / dropped cell rejected",
        s5,
        {
            "registered_fingerprint": registered,
            "verify_ok": verify_result(res),
            "verify_tampered": verify_result(tampered),
            "verify_dropped": verify_result(dropped),
        },
    )

    return {
        "tool": "adaptiverg_qec.qec_sweep.build_evidence",
        "surface": "RUN",
        "quick": quick,
        "gates": gates,
        **sections,
        "claim_ceiling": (
            "FSS-supported bounded simulation for the declared phenomenological bit-flip "
            "channel (X-flip p, measurement flip p, rounds=d, d<=11, PyMatching MWPM); no "
            "L->oo proof, no latency, no hardware claim, no other noise convention"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="smaller budgets (CI smoke)")
    ap.add_argument("--replicates", type=int, default=100)
    args = ap.parse_args(argv)
    t0 = time.time()
    ev = build_evidence(quick=args.quick, replicates=args.replicates)
    ev["wallclock_s"] = round(time.time() - t0, 1)
    for g in ev["gates"]:
        print(f"[{g['status']}] {g['id']} {g['name']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(ev, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if ev.get("surface") == "NOT_RUN":
        print(f"surface gates NOT_RUN: {ev['reason']}")
        return 2 if all(g["status"] == "PASS" for g in ev["gates"]) else 1
    return 0 if all(g["status"] == "PASS" for g in ev["gates"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
