"""CLI + Selftest-Gates (u6-Stil): jedes Gate druckt [PASS]/[FAIL], exit 0/1.

Aufruf:
    python -m adaptiverg_qec.cli --selftest        # alle Gates, exit 0 gdw alle PASS
    python -m adaptiverg_qec.cli --selftest --json results/selftest.json
    python -m adaptiverg_qec.cli demo              # kurze Demonstration

Gates (jedes gegen ein UNABHAENGIGES Orakel, Codie-Disziplin):
  G1 Analytic-Oracle:    MCMC-<H> vs Transfer-Matrix-<H> (Ising-Orakel).
  G2 Drift-Guard-holds:  equilib-Kette -> lambda_hat < 1 (Guard greift NICHT faelschlich).
  G3 Drift-Guard-fires:  non-contracting Kette -> Guard FEUERT (lambda_hat >= 1).
  G4 Jacobian-Consistency: Complex-Step == FD (bis Toleranz) == analytisch.
  G5 RG-Fixpoint:        R(K*=0)=0; R-Iteration von kleinem K -> 0 (Stabilitaet).
  G6 Reproducibility:    gleicher Seed -> bit-identische Trajektorie.
  G7 Diminishing-Adapt:  sum_t a_t < inf (summierbarer Schedule).
  G8 Negative/Edge-Input: invalide Eingaben werfen sauber (Silent-Failure-Gate).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable

import numpy as np

from . import a_kernel, drift, ising1d, rg_map
from .mvp_instance import MVPConfig

_CFG = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)


def _g1_analytic_oracle() -> tuple[bool, str]:
    """MCMC-Mittel von H reproduziert das Transfer-Matrix-Orakel."""
    beta = 0.8
    r = a_kernel.run_adaptive_mcmc(
        _CFG,
        beta_target=beta,
        n_steps=6000,
        burn_in=1500,
        seed=2024,
        beta_start=beta,
    )
    exact = ising1d.mean_energy(beta, _CFG.L)
    # MC-Fehlerbalken grob: std(H)/sqrt(n_eff); tolerantes, aber echtes Gate.
    tol = 0.15
    err = abs(r.mean_H - exact)
    ok = err < tol
    return ok, f"<H>mcmc={r.mean_H:.4f} <H>exact={exact:.4f} |err|={err:.4f} tol={tol}"


def _g2_drift_holds() -> tuple[bool, str]:
    """Equilibrium-Kette: bedingte Drift kontrahiert (lambda_hat < 1)."""
    r = a_kernel.run_adaptive_mcmc(
        _CFG,
        beta_target=1.5,
        n_steps=8000,
        burn_in=0,
        seed=7,
        beta_start=1.5,
    )
    d = max(1.0 + ising1d.mean_energy(1.5, _CFG.L), 2.0)
    rep = drift.estimate_drift(r.H_traj, d=d)
    ok = rep.holds and rep.lambda_hat < 1.0
    return ok, f"lambda_hat={rep.lambda_hat:.4f} d={rep.d:.2f} n_out={rep.n_outside}"


def _g3_drift_fires() -> tuple[bool, str]:
    """Non-contracting Trajektorie: Guard FEUERT (lambda_hat >= 1)."""
    rng = np.random.default_rng(0)
    walk = np.abs(np.cumsum(rng.integers(-1, 2, size=6000))).astype(float) + 1.0
    rep = drift.estimate_drift(walk, d=5.0)
    ok = (not rep.holds) and (math.isnan(rep.lambda_hat) or rep.lambda_hat >= 1.0)
    return ok, f"lambda_hat={rep.lambda_hat:.4f} holds={rep.holds} (expected fire)"


def _g4_jacobian_consistency() -> tuple[bool, str]:
    """Complex-Step == FD == analytisch fuer R'(K)=tanh(2K)."""
    worst_cs_an = 0.0
    worst_cs_fd = 0.0
    for K in (0.2, 0.5, 1.0):
        g = np.array([K])
        cs = rg_map.jacobian_complex_step(rg_map.rg_map, g)[0, 0]
        fd = rg_map.jacobian_finite_difference(rg_map.rg_map, g)[0, 0]
        an = rg_map.rg_derivative_analytic(K)
        worst_cs_an = max(worst_cs_an, abs(cs - an))
        worst_cs_fd = max(worst_cs_fd, abs(cs - fd))
    ok = worst_cs_an < 1e-12 and worst_cs_fd < 1e-7
    return ok, f"max|CS-analytic|={worst_cs_an:.2e} max|CS-FD|={worst_cs_fd:.2e}"


def _g5_rg_fixpoint() -> tuple[bool, str]:
    """R(0)=0 exakt; RG-Iteration von kleinem K kontrahiert -> 0."""
    r0 = abs(rg_map.rg_map_scalar(0.0))
    K = 0.3
    for _ in range(60):
        K = float(rg_map.rg_map_scalar(K))
    ok = r0 < 1e-15 and K < 1e-6
    return ok, f"R(0)={r0:.2e} K_after_60_iter={K:.2e} (-> trivial fixpoint 0)"


def _g6_reproducibility() -> tuple[bool, str]:
    """Gleicher Seed -> bit-identische Trajektorie (Spec 10.3b bit-exact)."""
    kw = dict(beta_target=0.9, n_steps=2000, burn_in=500, seed=99, beta_start=0.3)
    a = a_kernel.run_adaptive_mcmc(_CFG, **kw)
    b = a_kernel.run_adaptive_mcmc(_CFG, **kw)
    ok = (
        np.array_equal(a.H_traj, b.H_traj)
        and np.array_equal(a.final_state, b.final_state)
        and a.mean_H == b.mean_H
    )
    same = np.array_equal(a.H_traj, b.H_traj)
    return ok, f"H_traj identical={same} mean_H eq={a.mean_H == b.mean_H}"


def _g7_diminishing_adaptation() -> tuple[bool, str]:
    """Schedule a_t = c/(1+t/T0)^2 ist summierbar (sum < inf)."""
    a = a_kernel.diminishing_step_sizes(200000, c=0.5, T0=100.0)
    s = float(np.sum(a))
    # Vergleich gegen lange Partialsumme bei doppelter Laenge -> Konvergenz.
    a2 = a_kernel.diminishing_step_sizes(400000, c=0.5, T0=100.0)
    s2 = float(np.sum(a2))
    converging = abs(s2 - s) < 0.01 * s
    ok = math.isfinite(s) and converging
    return ok, f"sum_200k={s:.4f} sum_400k={s2:.4f} converging={converging}"


def _g8_negative_edge_input() -> tuple[bool, str]:
    """Silent-Failure-Gate: invalide Eingaben werfen sauber (kein stiller rc=0)."""
    checks: list[tuple[str, Callable[[], object]]] = [
        ("L<2", lambda: MVPConfig(L=1)),
        ("beta_min<=0", lambda: MVPConfig(beta_min=0.0)),
        ("beta_max<=beta_min", lambda: MVPConfig(beta_min=1.0, beta_max=0.5)),
        (
            "beta_target outside Theta",
            lambda: a_kernel.run_adaptive_mcmc(
                _CFG, beta_target=99.0, n_steps=10, burn_in=0, seed=1
            ),
        ),
        (
            "burn_in>=n_steps",
            lambda: a_kernel.run_adaptive_mcmc(
                _CFG, beta_target=0.5, n_steps=10, burn_in=10, seed=1
            ),
        ),
        ("ising beta<=0", lambda: ising1d.mean_energy(-1.0, 8)),
        (
            "FD eta out of range",
            lambda: rg_map.jacobian_finite_difference(rg_map.rg_map, np.array([0.5]), eta=1.0),
        ),
        ("drift d<1", lambda: drift.estimate_drift(np.array([1.0, 2.0, 3.0]), d=0.5)),
        ("neg H in V", lambda: drift.lyapunov_V(np.array([-1.0, 0.0]))),
    ]
    failures = []
    for name, fn in checks:
        try:
            fn()
            failures.append(name)  # should have raised, did not
        except (ValueError, TypeError, OverflowError):
            pass  # expected: invalid input rejected loudly
    ok = not failures
    detail = "all invalid inputs rejected" if ok else f"NOT rejected: {failures}"
    return ok, f"{len(checks)} edge checks; {detail}"


_GATES: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("G1 Analytic-Oracle (MCMC vs Transfer-Matrix)", _g1_analytic_oracle),
    ("G2 Drift-Guard holds (equilib lambda<1)", _g2_drift_holds),
    ("G3 Drift-Guard fires (non-contract lambda>=1)", _g3_drift_fires),
    ("G4 Jacobian-Consistency (CS==FD==analytic)", _g4_jacobian_consistency),
    ("G5 RG-Fixpoint (R(0)=0, iter->0)", _g5_rg_fixpoint),
    ("G6 Reproducibility (seed->bit-identical)", _g6_reproducibility),
    ("G7 Diminishing-Adaptation (sum a_t<inf)", _g7_diminishing_adaptation),
    ("G8 Negative/Edge-Input rejection", _g8_negative_edge_input),
]


def run_selftest(json_path: str | None = None) -> int:
    """Fuehre alle Gates aus; gib 0 zurueck gdw alle PASS, sonst 1."""
    print("AdaptiveRG-QEC Phase-1 MVP -- selftest")
    print(
        f"MVP-Instanz: 1D-Repetition-Code Ring L={_CFG.L}, "
        f"Theta=[{_CFG.beta_min},{_CFG.beta_max}], V=1+H, R(K)=0.5*ln cosh(2K)"
    )
    print("-" * 70)
    results = []
    n_pass = 0
    t0 = time.time()
    for name, gate in _GATES:
        try:
            ok, detail = gate()
        except Exception as exc:  # noqa: BLE001 - report, never silently pass
            ok, detail = False, f"EXCEPTION: {exc!r}"
        ok = bool(ok)  # coerce np.bool_ -> native bool (JSON-serializable)
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {name}: {detail}")
        results.append({"gate": name, "pass": ok, "detail": detail})
        n_pass += int(ok)
    elapsed = time.time() - t0
    print("-" * 70)
    print(f"{n_pass}/{len(_GATES)} [PASS]  ({elapsed:.1f}s)")

    if json_path:
        payload = {
            "tool": "adaptiverg_qec",
            "version": __import__("adaptiverg_qec").__version__,
            "n_pass": n_pass,
            "n_total": len(_GATES),
            "all_pass": n_pass == len(_GATES),
            "elapsed_s": round(elapsed, 3),
            "gates": results,
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"gate-log -> {json_path}")

    return 0 if n_pass == len(_GATES) else 1


def run_demo() -> int:
    """Kurze Demonstration der MVP-Pipeline (A-Kernel + C-Kernel)."""
    print("Demo: A-Kernel sampling + C-Kernel RG-Jacobian")
    r = a_kernel.run_adaptive_mcmc(
        _CFG,
        beta_target=1.0,
        n_steps=4000,
        burn_in=1000,
        seed=42,
        beta_start=0.2,
    )
    exact = ising1d.mean_energy(1.0, _CFG.L)
    print(f"  A-Kernel: <H>={r.mean_H:.3f} (exact {exact:.3f}), acc={r.acceptance:.3f}")
    M = rg_map.jacobian_complex_step(rg_map.rg_map, np.array([0.5]))
    rep = rg_map.exponents_from_jacobian(M)
    print(
        f"  C-Kernel: DR(0.5)={M[0, 0]:.6f}, eig={rep.eigenvalues}, "
        f"y={rep.exponents}, hyperbolic={rep.hyperbolic}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptiverg_qec",
        description="AdaptiveRG-QEC Phase-1 MVP (Diagnostik-/Verifikations-Harness).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=["demo", "selftest"],
        help="demo (default) or selftest",
    )
    parser.add_argument(
        "--selftest", action="store_true", help="run selftest gates (alias for 'selftest' command)"
    )
    parser.add_argument("--json", metavar="PATH", default=None, help="write gate-log JSON to PATH")
    args = parser.parse_args(argv)

    if args.selftest or args.command == "selftest":
        return run_selftest(args.json)
    return run_demo()


if __name__ == "__main__":
    sys.exit(main())
