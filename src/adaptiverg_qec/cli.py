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
from pathlib import Path

import numpy as np

from . import a_kernel, autocorr, drift, ising1d, mcrg, rg_map
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
        ("mcrg K<=0", lambda: mcrg.sample_ising_open_chain(0.0, 16, 100, seed=1)),
        ("mcrg n_samples<2", lambda: mcrg.sample_ising_open_chain(0.5, 16, 1, seed=1)),
        ("mcrg L<2 sample", lambda: mcrg.sample_ising_open_chain(0.5, 1, 100, seed=1)),
        (
            "mcrg T L<4",
            lambda: mcrg.swendsen_T_scalar(np.ones((10, 2))),
        ),
        ("mcrg spins bad x", lambda: mcrg.spins_from_bits(np.array([0, 2, 1]))),
        ("mcrg validate <2 seeds", lambda: mcrg.validate_swendsen(seeds=(0,))),
        # Phase-3a autocorr edge cases (Silent-Failure-Gate).
        ("autocorr rho N<2", lambda: autocorr.autocorr_function_fft(np.array([1.0]))),
        (
            "autocorr rho constant series",
            lambda: autocorr.autocorr_function_fft(np.full(100, 3.0)),
        ),
        (
            "autocorr c_window<=0",
            lambda: autocorr.integrated_autocorr_time(np.arange(100.0), c_window=0.0),
        ),
        ("autocorr binning N<64", lambda: autocorr.binning_error(np.arange(10.0))),
        (
            "autocorr binning max_block<1",
            lambda: autocorr.binning_error(np.arange(100.0), max_block=0),
        ),
        (
            "jackknife block_size<1",
            lambda: autocorr.jackknife_ratio(
                np.ones((100, 1)), np.ones((100, 1)), block_size=0, combine=lambda a, b: 1.0
            ),
        ),
        (
            "jackknife <2 blocks",
            lambda: autocorr.jackknife_ratio(
                np.ones((100, 1)), np.ones((100, 1)), block_size=100, combine=lambda a, b: 1.0
            ),
        ),
        (
            "jackknife num/den len mismatch",
            lambda: autocorr.jackknife_ratio(
                np.ones((100, 1)), np.ones((50, 1)), block_size=5, combine=lambda a, b: 1.0
            ),
        ),
        (
            "swendsen_chain N<64",
            lambda: mcrg.swendsen_T_from_chain(np.arange(10.0), np.arange(10.0), K=0.5),
        ),
        (
            "swendsen_chain S/Sp len mismatch",
            lambda: mcrg.swendsen_T_from_chain(np.arange(100.0), np.arange(80.0), K=0.5),
        ),
        (
            "op_timeseries L<4",
            lambda: mcrg.operator_timeseries_from_configs(np.zeros((100, 2), dtype=np.int8)),
        ),
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


def _g9_swendsen_vs_oracle() -> tuple[bool, str]:
    """Phase-2: Swendsen-T-hat(K) trifft tanh(2K) innerhalb 3 sigma (Multi-Seed)."""
    rows = mcrg.validate_swendsen(
        K_values=(0.3, 0.5, 0.7, 0.9), L=64, n_samples=3000, seeds=(0, 1, 2, 3, 4, 5, 6, 7)
    )
    worst = max(rows, key=lambda r: r.n_sigma)
    ok = all(r.n_sigma <= 3.0 for r in rows)
    parts = " ".join(f"K={r.K}:{r.T_hat_mean:.4f}vs{r.oracle:.4f}({r.n_sigma:.1f}s)" for r in rows)
    return ok, f"{parts} | worst={worst.n_sigma:.2f}sigma (<=3) L=64 N=3000 seeds=8"


def _g10_connected_corr_exact() -> tuple[bool, str]:
    """Connected-corr-Schaetzer == exakte Enumeration (kleines L, unabh. Orakel).

    Swendsen-Identitaet im EXAKTEN Ensemble: T-hat = tanh(2K) bis Maschinen-eps.
    """
    L = 8
    K = 0.6
    states = np.arange(1 << L, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(L)[None, :]) & 1).astype(np.int8)
    s = mcrg.spins_from_bits(bits)
    Sfull = mcrg.nn_operator(s, periodic=False)
    Sp = mcrg.nn_operator(mcrg.decimate_b2(s), periodic=False)
    w = np.exp(K * Sfull)
    w /= w.sum()
    # exakte ensemble-gewichtete connected correlations (kein Sampling).
    e_s, e_sp = float((w * Sfull).sum()), float((w * Sp).sum())
    cov_sps = float((w * Sp * Sfull).sum()) - e_sp * e_s
    cov_spsp = float((w * Sp * Sp).sum()) - e_sp * e_sp
    T_exact = cov_sps / cov_spsp
    oracle = math.tanh(2.0 * K)
    err = abs(T_exact - oracle)
    ok = err < 1e-10
    return ok, f"L={L} K={K} T_exact={T_exact:.10f} tanh(2K)={oracle:.10f} |err|={err:.2e}"


def _g11_swendsen_reproducible() -> tuple[bool, str]:
    """Reproduzierbarkeit: gleicher Seed -> bit-identische Spin-Stichprobe + T-hat."""
    a = mcrg.sample_ising_open_chain(0.7, 32, 1000, seed=12345)
    b = mcrg.sample_ising_open_chain(0.7, 32, 1000, seed=12345)
    c = mcrg.sample_ising_open_chain(0.7, 32, 1000, seed=54321)
    same = np.array_equal(a, b)
    diff = not np.array_equal(a, c)
    ta = mcrg.swendsen_T_scalar(a).T_hat
    tb = mcrg.swendsen_T_scalar(b).T_hat
    ok = same and diff and ta == tb
    return ok, f"seed-eq identical={same} diff-seed differs={diff} T_hat_eq={ta == tb}"


def _g12_swendsen_bias_decreases() -> tuple[bool, str]:
    """Bias/Streuung sinkt mit N: Multi-Seed-std(T-hat) faellt ~1/sqrt(N)."""
    K, L, seeds = 0.6, 64, tuple(range(8))

    def spread(n: int) -> float:
        vals = np.array(
            [
                mcrg.swendsen_T_scalar(
                    mcrg.sample_ising_open_chain(K, L, n, seed=mcrg.K_seed(K, sd))
                ).T_hat
                for sd in seeds
            ]
        )
        return float(vals.std(ddof=1))

    s_small = spread(1000)
    s_large = spread(8000)  # 8x N -> erwartet ~ /sqrt(8) ~ 0.354x
    ratio = s_large / s_small if s_small > 0 else float("inf")
    ok = s_large < s_small and ratio < 0.7  # klar fallend, in Naehe von 1/sqrt(8)
    return ok, f"std(N=1k)={s_small:.4f} std(N=8k)={s_large:.4f} ratio={ratio:.3f} (~0.354 ideal)"


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    """AR(1)-Prozess x_t = phi x_{t-1} + eps mit bekanntem tau_int (Test-Orakel).

    rho(t) = phi^|t| => tau_int = 1/2 + phi/(1-phi) (geschlossen, unabhaengig).
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _g13_tau_int_ar1_oracle() -> tuple[bool, str]:
    """tau_int (Gamma+Wolff-Windowing) trifft das analytische AR(1)-Orakel.

    AR(1): tau_int = 1/2 + phi/(1-phi). Mittel ueber Seeds; Toleranz auf den
    relativen Fehler (Wolff-Windowing hat endliche-N-Streuung, aber kleinen Bias).
    """
    worst_rel = 0.0
    detail_parts = []
    for phi in (0.0, 0.5, 0.8):
        tau_true = 0.5 + phi / (1.0 - phi)
        taus = [autocorr.integrated_autocorr_time(_ar1(20000, phi, sd)).tau_int for sd in range(8)]
        tau_hat = float(np.mean(taus))
        rel = abs(tau_hat - tau_true) / tau_true
        worst_rel = max(worst_rel, rel)
        detail_parts.append(f"phi={phi}:{tau_hat:.2f}/{tau_true:.2f}")
    ok = worst_rel < 0.10  # <10% rel. Fehler gegen geschlossenes Orakel
    return ok, f"{' '.join(detail_parts)} worst_rel={worst_rel:.3f} (<0.10)"


def _g14_gamma_vs_binning() -> tuple[bool, str]:
    """Gamma-Methode (tau_int -> sem) == Binning-Plateau (unabh. Cross-Check).

    Rigorose Validierung (Wolff): beide Fehler-Schaetzer muessen im Plateau
    uebereinstimmen. Mittel ueber Seeds eines AR(1)-Prozesses (phi=0.8).
    """
    ratios = []
    for sd in range(6):
        x = _ar1(20000, 0.8, sd)
        g = autocorr.integrated_autocorr_time(x)
        b = autocorr.binning_error(x)
        ratios.append(b.sem_plateau / g.sem)
    r = float(np.mean(ratios))
    ok = 0.85 <= r <= 1.15  # Plateau-Uebereinstimmung +/-15%
    return ok, f"sem_binning/sem_gamma={r:.3f} (in [0.85,1.15]) over 6 AR(1) seeds phi=0.8"


def _g15_akernel_T_within_correlated_error() -> tuple[bool, str]:
    """T_hat(A-Kernel) trifft tanh(2K) innerhalb k*sigma (KORRELIERTER Fehler).

    Gegated auf K in {0.3,0.5,0.7}: dort ist die finite-L-Systematik (L=64) klein
    gegen den statistischen Fehler, sodass das L->oo-Orakel tanh(2K) eine saubere
    Referenz ist. K=0.9 wird BEWUSST NICHT gegated (siehe G16): dort waechst die
    Korrelationslaenge, die endliche-L-Abweichung wird vergleichbar mit dem
    Fehlerbalken (kein Ueber-Claim eines sauberen 3sigma-Treffers am Rand von
    Theta). Mehr-Seed-Median entschaerft Einzel-Seed-Ausreisser.
    """
    K_values = (0.3, 0.5, 0.7)
    worst_sigma = 0.0
    parts = []
    for K in K_values:
        # 3 Seeds, Median der n_sigma -> robust gegen Einzel-Seed-Glueck/Pech.
        ests = [
            mcrg.validate_swendsen_akernel(
                K_values=(K,), L=64, n_steps=14000, burn_in=2000, seed=sd
            )[0]
            for sd in range(3)
        ]
        med_sigma = float(np.median([e.n_sigma for e in ests]))
        worst_sigma = max(worst_sigma, med_sigma)
        e0 = ests[0]
        parts.append(f"K={K}:{e0.T_hat:.3f}vs{e0.oracle:.3f}(med{med_sigma:.1f}s)")
    ok = worst_sigma <= 3.0
    return ok, f"{' '.join(parts)} worst_median={worst_sigma:.2f}sigma(<=3) L=64 3seeds"


def _g16_neff_less_than_n_and_inflation() -> tuple[bool, str]:
    """N_eff < N (korrelierte Kette) UND korrelierter Fehler > i.i.d.-Fehler.

    Wissenschaftlicher Kernpunkt von Phase-3a: korrelierte Samples tragen weniger
    Information (tau_int>0.5 => N_eff<N), die korrekten Fehlerbalken sind GROESSER.
    """
    rows = mcrg.validate_swendsen_akernel(
        K_values=(0.5, 0.7, 0.9), L=64, n_steps=14000, burn_in=2000, seed=1
    )
    # N_eff<N und tau>0.5: muss fuer ALLE korrelierten Ketten gelten (Kernpunkt).
    all_neff_lt_n = all(r.n_eff < r.n_samples for r in rows)
    all_tau_gt_half = all(max(r.tau_int_S, r.tau_int_Sp, r.tau_int_product) > 0.5 for r in rows)
    # Fehler-Inflation des RATIOS: bei kleinem K (tau~0.6) liegt sie bei ~1.00
    # (Zaehler/Nenner-Fluktuationen kuerzen sich, s. AKernelSwendsenEstimate-Doc);
    # gegated auf K>=0.7, wo der Effekt eindeutig > 1 ist (sonst Ueber-Claim).
    strong = [r for r in rows if r.K >= 0.7]
    strong_inflated = all(r.error_inflation > 1.0 for r in strong)
    monotone = all(r.error_correlated >= r.error_iid_naive for r in rows)  # nie kleiner
    ok = all_neff_lt_n and all_tau_gt_half and strong_inflated and monotone
    parts = " ".join(
        f"K={r.K}:tau={max(r.tau_int_S, r.tau_int_Sp, r.tau_int_product):.2f}"
        f",Neff/N={r.n_eff / r.n_samples:.2f},infl={r.error_inflation:.3f}"
        for r in rows
    )
    return ok, (
        f"{parts} | Neff<N={all_neff_lt_n} tau>0.5={all_tau_gt_half} "
        f"infl(K>=0.7)>1={strong_inflated} err_c>=err_iid(all)={monotone}"
    )


def _g17_autocorr_reproducible() -> tuple[bool, str]:
    """Reproduzierbarkeit: gleicher Seed -> bit-identische A-Kernel-T_hat + Fehler."""
    a = mcrg.validate_swendsen_akernel(K_values=(0.6,), L=64, n_steps=8000, burn_in=1000, seed=7)
    b = mcrg.validate_swendsen_akernel(K_values=(0.6,), L=64, n_steps=8000, burn_in=1000, seed=7)
    c = mcrg.validate_swendsen_akernel(K_values=(0.6,), L=64, n_steps=8000, burn_in=1000, seed=8)
    same = a[0].T_hat == b[0].T_hat and a[0].error_correlated == b[0].error_correlated
    diff = a[0].T_hat != c[0].T_hat
    ok = same and diff
    return ok, f"seed-eq T_hat&err identical={same} diff-seed differs={diff}"


def _g18_iid_limit_no_inflation() -> tuple[bool, str]:
    """Grenzfall-Sanity: auf i.i.d.-Daten ergibt die Gamma-Methode tau~0.5, infl~1.

    Schuetzt vor einem stillen Bug, der IMMER aufbloeht (false positive in G16).
    """
    rng = np.random.default_rng(123)
    x = rng.standard_normal(20000)
    r = autocorr.integrated_autocorr_time(x)
    ok = abs(r.tau_int - 0.5) < 0.05 and abs(r.inflation - 1.0) < 0.05 and r.n_eff > 0.95 * r.n
    return ok, (
        f"iid: tau_int={r.tau_int:.3f}(~0.5) inflation={r.inflation:.3f}(~1) "
        f"Neff/N={r.n_eff / r.n:.3f}"
    )


_GATES: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("G1 Analytic-Oracle (MCMC vs Transfer-Matrix)", _g1_analytic_oracle),
    ("G2 Drift-Guard holds (equilib lambda<1)", _g2_drift_holds),
    ("G3 Drift-Guard fires (non-contract lambda>=1)", _g3_drift_fires),
    ("G4 Jacobian-Consistency (CS==FD==analytic)", _g4_jacobian_consistency),
    ("G5 RG-Fixpoint (R(0)=0, iter->0)", _g5_rg_fixpoint),
    ("G6 Reproducibility (seed->bit-identical)", _g6_reproducibility),
    ("G7 Diminishing-Adaptation (sum a_t<inf)", _g7_diminishing_adaptation),
    ("G8 Negative/Edge-Input rejection", _g8_negative_edge_input),
    ("G9 Swendsen-MCRG T-hat vs tanh(2K) (<=3sigma)", _g9_swendsen_vs_oracle),
    ("G10 Connected-corr vs exact enumeration", _g10_connected_corr_exact),
    ("G11 Swendsen reproducibility (seed)", _g11_swendsen_reproducible),
    ("G12 Swendsen bias decreases with N", _g12_swendsen_bias_decreases),
    ("G13 tau_int (Gamma+Wolff) vs AR(1) oracle", _g13_tau_int_ar1_oracle),
    ("G14 Gamma-method == Binning plateau", _g14_gamma_vs_binning),
    (
        "G15 A-Kernel T-hat vs tanh(2K) (correlated <=3sigma)",
        _g15_akernel_T_within_correlated_error,
    ),
    ("G16 N_eff<N AND correlated err > i.i.d. err", _g16_neff_less_than_n_and_inflation),
    ("G17 A-Kernel autocorr-T reproducibility (seed)", _g17_autocorr_reproducible),
    ("G18 i.i.d.-limit sanity (tau~0.5, no inflation)", _g18_iid_limit_no_inflation),
]


def run_selftest(json_path: str | None = None) -> int:
    """Fuehre alle Gates aus; gib 0 zurueck gdw alle PASS, sonst 1."""
    print("AdaptiveRG-QEC Phase-1/2 MVP -- selftest")
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
        out_path = _resolve_json_path(json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"gate-log -> {out_path}")

    return 0 if n_pass == len(_GATES) else 1


def _resolve_json_path(json_path: str) -> Path:
    """Normalisiere den --json-Pfad (Aegis-P3 Pfad-Traversal-Defense).

    Politik (fail-closed nur fuer den Angriffsvektor, nicht fuer legitime
    Operator-Pfade):
      - Nackter Dateiname (z.B. 'gate.json') -> cwd/results/gate.json.
      - RELATIVER Pfad -> gegen cwd aufgeloest; er MUSS innerhalb von cwd bleiben,
        '..'-Ausbruch wird abgewiesen (das ist der eigentliche Traversal-Vektor).
      - ABSOLUTER Pfad -> als explizite Operator-Wahl akzeptiert, aber via
        resolve() normalisiert (entfernt '..'/Symlink-Tricks im Pfad selbst).
    """
    cwd = Path.cwd().resolve()
    p = Path(json_path)
    if p.is_absolute():
        return p.resolve()
    if p.parent == Path("."):
        p = Path("results") / p  # nackter Dateiname -> results/
    candidate = (cwd / p).resolve()
    try:
        candidate.relative_to(cwd)
    except ValueError as exc:
        raise ValueError(
            f"--json path {json_path!r} resolves outside cwd ({candidate}); "
            "refusing path traversal (Aegis-P3 fail-closed)"
        ) from exc
    return candidate


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
        description="AdaptiveRG-QEC Phase-1/2 MVP (Diagnostik-/Verifikations-Harness).",
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
