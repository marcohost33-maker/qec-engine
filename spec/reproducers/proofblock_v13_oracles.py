"""Deterministischer Reproducer der numerischen ProofBlock-v1.3-Orakel (Defekt 3 & 4).

ZWECK / POSITIONIERUNG (AGENTS.md Working-Agreement 1):
Der Errata-Companion `spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md` belegt zwei
Defekte gegen ein *unabhaengiges numerisches Orakel*. AGENTS.md verlangt:
„Keine Komponente als 'validiert' behaupten ohne lauffaehigen Code + Gate-Log in results/."
Dieses Skript IST dieser lauffaehige Code: es erzeugt die zwei Orakel-Tabellen
deterministisch (fixer Seed) und schreibt sie als Gate-Log nach
`results/proofblock-v13-oracles.json`. Die Zahlen im Errata-Doc stammen aus GENAU
diesem Lauf (Seed unten verankert); der Doc-Text verlinkt dieses Artefakt.

Keine neuen Dependencies (nur numpy, bereits Repo-Dependency).

--- Defekt 3 (Knoten P3.3): Eigenwert-Empfindlichkeit skaliert mit kappa(lambda), NICHT sep^-1 ---
Matrix A = [[1, t], [0, 2]]: Eigenwerte {1, 2}, konstante Separation sep == 1 fuer alle t,
aber mit wachsendem t zunehmend parallele Eigenvektoren.
- sep^-1 (docx-Bound-Faktor) = 1 (konstant, t-unabhaengig).
- kappa(lambda_i) = ||u_i|| ||v_i|| / |u_i^H v_i| = sqrt(1 + t^2) (analytisch; s. Herleitung unten).
- gemessen: mittleres |delta lambda| / ||E||_2 ueber n_trials Zufalls-E mit Spektralnorm eps.
ORAKEL-GATE: gemessen skaliert mit kappa (measured <= kappa*(1+tol)) und ueberschreitet
sep^-1 fuer t>0 deutlich -> der docx-sep^-1-Bound wird verletzt.

Analytische kappa-Herleitung (unabhaengiges Orakel, nicht aus dem Messcode):
  Eigenwert 1: rechter EV v=[1,0], linker EV u ~ [1,-t]  -> kappa = sqrt(1+t^2).
  Eigenwert 2: rechter EV v ~ [t,1], linker EV u=[0,1]   -> kappa = sqrt(1+t^2).

--- PRAEZISIERUNG: DREI VERSCHIEDENE Sensitivitaeten (haeufig verwechselt) ---
Defekt 3 betraf genau EINE dieser drei; der docx nutzte faelschlich (b). Trennung:

  (a) EIGENWERT-Sensitivitaet (relevant hier, P3.3):
      Bauer-Fike / Wilkinson-Konditionszahl  kappa(lambda) = 1 / |u^H v|
      mit auf Einheitslaenge normierten linkem/rechtem Eigenvektor u,v
      (aequivalent: ||u|| ||v|| / |u^H v|, wie im Code fuer beliebige Skalierung).
      Klein, wenn u,v (fast) orthogonal -> nicht-normale Matrix -> kappa >> 1.
      Regiert |delta lambda| ~< kappa(lambda) * ||E||. NICHT sep, NICHT gap.
      Ref: Bauer & Fike 1960; Wilkinson, Algebraic Eigenvalue Problem.

  (b) INVARIANTER-UNTERRAUM-Sensitivitaet (Stewart, allgemein/nicht-normal):
      geregelt von sep^-1 mit  sep(A11,A22) = min_{||X||=1} ||A11 X - X A22||.
      Fuer 1x1-Bloecke degeneriert sep zwar zum Eigenwert-Abstand min|li-lj|,
      aber sep MISST die Unterraum-Drehung, NICHT die Eigenwert-Verschiebung
      -> Anwendung auf einzelne Eigenwerte (wie im docx) ist der Kategorienfehler.
      Ref: Stewart & Sun, Matrix Perturbation Theory, Kap. V.

  (c) HERMITESCHER-EIGENRAUM-Sensitivitaet (Davis-Kahan sin(Theta)):
      NUR fuer hermitesche/normale A: Unterraum-Drehung ~< ||E|| / gap,
      gap = spektraler Abstand des Ziel-Clusters zum Rest.
      Hier NICHT anwendbar (A ist nicht hermitesch/normal).
      Ref: Davis & Kahan 1970; moderne Fassung arXiv:2203.00068 (Xia/Yu, sin-Theta).

Merksatz: Eigenwert -> kappa(lambda)=1/|u^H v| (a);  Unterraum(allg.) -> sep^-1 (b);
          Unterraum(hermitesch) -> gap^-1 (c).  Der docx-Bound war (b) auf ein (a)-Problem.

--- Defekt 4 (Knoten P2.1/P2.2): Kugel-Bedingung ignoriert instabile Mannigfaltigkeit ---
Lineare stochastische Rekursion  x_{k+1} = DR x_k + eta_k,  DR = diag(0.5, 1.5),
eta_k ~ N(0, sigma^2 I). Start bei Abstand start_dist << r_lin.
- untuned: volle Iteration. Die instabile Richtung (lambda_u=1.5) waechst geometrisch
  -> Blow-up. Zusaetzlich gemessen: mittlerer Schritt, an dem ||x_k|| erstmals r_lin
  verlaesst (belegt: das Gegenbeispiel VERLAESST die Kugel -> es widerlegt die
  P2.2-BESCHRAENKTHEIT, und die P2.1-HYPOTHESE 'bleibt in B' ist off-W^s gar nicht erfuellbar).
- tuned: instabile Komponente je Schritt auf 0 projiziert (Iteration auf W^s(g*)).
  -> bleibt in einer O(sigma)-Umgebung.
ORAKEL-GATE: mean_end_norm(untuned) >> r_lin  UND  mean_end_norm(tuned) = O(sigma).
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

# --- Verankerte Parameter (Reproduzierbarkeit; Herkunft aller Doc-Zahlen) ---
SEED = 2026
DEFECT3_T_VALUES = (0.0, 5.0, 50.0)
DEFECT3_N_TRIALS = 400
DEFECT3_EPS = 1e-6  # ||E||_2 (Spektralnorm) der Stoerung
DEFECT4_DR_DIAG = (0.5, 1.5)  # stabil / instabil
DEFECT4_SIGMA = 1e-3
DEFECT4_R_LIN = 1e-2
DEFECT4_START_DIST = 1.4e-4  # << r_lin
DEFECT4_N_ITERS = 60
DEFECT4_N_RUNS = 200


def eigenvalue_condition_numbers(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-Eigenwert-Konditionszahl kappa(lambda_i) = ||u_i|| ||v_i|| / |u_i^H v_i|.

    u_i, v_i = linker bzw. rechter Eigenvektor. Fuer normale Matrizen ist kappa=1;
    fuer nicht-normale (wie hier) kann kappa >> 1. Dies ist die KORREKTE
    Eigenwert-Empfindlichkeit (Kato 1995; Golub-Van Loan Sec. 7.2), NICHT sep^-1.

    Returns:
        (eigenwerte, kappas) aufsteigend nach Realteil sortiert.
    """
    lam_r, V = np.linalg.eig(A)  # A v = lambda v  (rechte EV)
    lam_l, U = np.linalg.eig(A.T.conj())  # A^H u = conj(lambda) u  (linke EV)
    order_r = np.argsort(lam_r.real)
    order_l = np.argsort(lam_l.real)
    lam_r, V = lam_r[order_r], V[:, order_r]
    U = U[:, order_l]
    kappas = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        u, v = U[:, i], V[:, i]
        denom = abs(np.vdot(u, v))
        kappas[i] = np.linalg.norm(u) * np.linalg.norm(v) / denom if denom > 0 else np.inf
    return lam_r.real, kappas


def _random_perturbation(rng: np.random.Generator, dim: int, eps: float) -> np.ndarray:
    """Zufalls-Stoerung E mit exakter Spektralnorm ||E||_2 = eps (reell, dicht)."""
    G = rng.standard_normal((dim, dim))
    s = np.linalg.norm(G, ord=2)
    return eps * G / s


def defect3_oracle(
    t_values=DEFECT3_T_VALUES,
    n_trials: int = DEFECT3_N_TRIALS,
    eps: float = DEFECT3_EPS,
    seed: int = SEED,
) -> dict:
    """Defekt-3-Orakel: gemessene |delta lambda|/||E|| vs. kappa(lambda) vs. sep^-1."""
    rng = np.random.default_rng(seed)
    rows = []
    all_ok = True
    for t in t_values:
        A = np.array([[1.0, t], [0.0, 2.0]])
        lam, kappas = eigenvalue_condition_numbers(A)  # lam ~ [1, 2]
        sep = 1.0  # min_{i!=j} |lambda_i - lambda_j| = |2-1| = 1 fuer alle t
        sep_inv = 1.0 / sep
        # gemessen: mittlere Eigenwert-Verschiebung pro Einheit ||E||_2.
        ratios_per_eig = np.zeros(len(lam))
        for _ in range(n_trials):
            E = _random_perturbation(rng, 2, eps)
            lam_pert = np.sort(np.linalg.eig(A + E)[0].real)  # nahe {1,2}, sep=1 -> matcht
            ratios_per_eig += np.abs(lam_pert - lam) / eps
        ratios_per_eig /= n_trials
        measured = float(ratios_per_eig.mean())  # aggregiert ueber beide Eigenwerte
        kappa = float(kappas.mean())  # beide Eigenwerte teilen kappa = sqrt(1+t^2)
        # Gate: gemessen skaliert mit kappa (<= kappa*(1+tol)); Bound sep^-1 verletzt fuer t>0.
        kappa_holds = measured <= kappa * 1.05 + 1e-9
        # t=0: sep=kappa=1, kein Verstoss noetig
        sep_violated = measured > sep_inv * 1.05 if t > 0 else True
        ok = kappa_holds and (sep_violated or t == 0.0)
        all_ok = all_ok and ok
        rows.append(
            {
                "t": t,
                "sep_inv_docx_bound": sep_inv,
                "kappa_lambda_analytic": kappa,
                "kappa_sqrt_1_plus_t2": float(np.sqrt(1.0 + t * t)),
                "measured_dlambda_over_E": measured,
                "measured_over_kappa": measured / kappa,
                "bound_kappa_holds": bool(kappa_holds),
                "bound_sep_inv_violated": bool(measured > sep_inv * 1.05),
            }
        )
    return {
        "defect": 3,
        "node": "P3.3 (Exponenten-Konvergenz)",
        "claim": "Eigenwert-Empfindlichkeit skaliert mit kappa(lambda), nicht sep^-1",
        "matrix": "A = [[1, t], [0, 2]]",
        "eps_spectral_norm_E": eps,
        "n_trials": n_trials,
        "oracle_kappa": "kappa(lambda_i) = sqrt(1+t^2) (analytisch, Kato 1995 / GVL Sec. 7.2)",
        "gate_pass": bool(all_ok),
        "rows": rows,
    }


def defect4_oracle(
    dr_diag=DEFECT4_DR_DIAG,
    sigma: float = DEFECT4_SIGMA,
    r_lin: float = DEFECT4_R_LIN,
    start_dist: float = DEFECT4_START_DIST,
    n_iters: int = DEFECT4_N_ITERS,
    n_runs: int = DEFECT4_N_RUNS,
    seed: int = SEED,
) -> dict:
    """Defekt-4-Orakel: untuned Blow-up (verlaesst Kugel) vs. getunt O(sigma) auf W^s."""
    rng = np.random.default_rng(seed + 1)  # eigener Stream (unabhaengig von Defekt 3)
    dr = np.asarray(dr_diag, dtype=float)
    dim = dr.size
    unstable = np.abs(dr) > 1.0  # instabile Richtungen (relevante Skalenfelder)

    end_norm_untuned = np.zeros(n_runs)
    end_norm_tuned = np.zeros(n_runs)
    exit_step = np.full(n_runs, n_iters, dtype=float)  # Schritt, an dem Kugel verlassen wird
    for r in range(n_runs):
        # gleicher Startpunkt + gleiche Rausch-Realisierung fuer untuned/tuned (faire Paarung)
        x0 = np.zeros(dim)
        x0[0] = start_dist / np.sqrt(2.0)
        x0[1] = start_dist / np.sqrt(2.0)  # Startabstand start_dist, beide Richtungen belegt
        noise = rng.standard_normal((n_iters, dim)) * sigma

        x_u = x0.copy()
        x_t = x0.copy()
        left_ball = False
        for k in range(n_iters):
            x_u = dr * x_u + noise[k]
            if not left_ball and np.linalg.norm(x_u) > r_lin:
                exit_step[r] = k + 1
                left_ball = True
            # getunt: Iteration auf W^s(g*) halten -> instabile Komponente je Schritt = 0
            x_t = dr * x_t + noise[k]
            x_t[unstable] = 0.0
        end_norm_untuned[r] = np.linalg.norm(x_u)
        end_norm_tuned[r] = np.linalg.norm(x_t)

    mean_untuned = float(end_norm_untuned.mean())
    mean_tuned = float(end_norm_tuned.mean())
    mean_exit = float(exit_step.mean())
    # Gate: untuned Blow-up (>> r_lin) UND getunt O(sigma) (<= 10*sigma) UND Kugel wird verlassen.
    gate_untuned_blowup = mean_untuned > 100.0 * r_lin
    gate_tuned_bounded = mean_tuned <= 10.0 * sigma
    gate_ball_left = mean_exit < n_iters
    gate_pass = gate_untuned_blowup and gate_tuned_bounded and gate_ball_left
    return {
        "defect": 4,
        "node": "P2.1/P2.2 (Fixpunkt-Konsistenz)",
        "claim": (
            "Kugel-Bedingung B(g*,r_lin) ist als Konvergenz-Hinreichung unzureichend; "
            "untuned Iterate verlassen die Kugel (Blow-up entlang lambda_u); "
            "nur auf W^s(g*) getunt bleibt es O(sigma)."
        ),
        "DR_diag": list(dr_diag),
        "sigma": sigma,
        "r_lin": r_lin,
        "start_dist": start_dist,
        "n_iters": n_iters,
        "n_runs": n_runs,
        "mean_end_norm_untuned": mean_untuned,
        "mean_end_norm_tuned": mean_tuned,
        "mean_exit_step_untuned": mean_exit,
        "o_sigma_reference": sigma * np.sqrt(2.0 / np.pi) / np.sqrt(1.0 - dr_diag[0] ** 2),
        "gate_untuned_blowup": bool(gate_untuned_blowup),
        "gate_tuned_bounded_O_sigma": bool(gate_tuned_bounded),
        "gate_untuned_leaves_ball": bool(gate_ball_left),
        "gate_pass": bool(gate_pass),
    }


def run() -> dict:
    """Erzeuge beide Orakel + Gesamt-Gate-Log (JSON-serialisierbar)."""
    d3 = defect3_oracle()
    d4 = defect4_oracle()
    return {
        "tool": "spec.reproducers.proofblock_v13_oracles",
        "purpose": "Numerische Orakel des Errata-Companion ProofBlock v1.3 (Defekt 3 & 4)",
        "spec_doc": "spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md",
        "seed": SEED,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "gate_pass": bool(d3["gate_pass"] and d4["gate_pass"]),
        "defect3": d3,
        "defect4": d4,
    }


def _main() -> int:
    """Schreibe results/proofblock-v13-oracles.json + Konsolen-Tabellen (Evidenz, fail-closed)."""
    payload = run()
    out = Path("results") / "proofblock-v13-oracles.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    d3, d4 = payload["defect3"], payload["defect4"]
    print(f"ProofBlock v1.3 Orakel-Reproducer  seed={payload['seed']}  numpy={np.__version__}")
    print("=" * 78)
    print(
        "Defekt 3 (P3.3): Eigenwert-Empfindlichkeit  A=[[1,t],[0,2]]  "
        f"||E||_2={d3['eps_spectral_norm_E']:.0e}"
    )
    header = (
        f"{'t':>5} {'sep^-1':>8} {'kappa(l)':>10} {'gemessen':>10} "
        f"{'meas/kappa':>11} {'sep-verletzt':>13}"
    )
    print(header)
    for r in d3["rows"]:
        print(
            f"{r['t']:>5.0f} {r['sep_inv_docx_bound']:>8.2f} {r['kappa_lambda_analytic']:>10.2f} "
            f"{r['measured_dlambda_over_E']:>10.2f} {r['measured_over_kappa']:>11.3f} "
            f"{str(r['bound_sep_inv_violated']):>13}"
        )
    print(f"  gate_pass = {d3['gate_pass']}")
    print("-" * 78)
    print("Defekt 4 (P2.1/P2.2): Kugel-Bedingung ignoriert instabile Mannigfaltigkeit")
    print(
        f"  DR=diag{tuple(d4['DR_diag'])}  sigma={d4['sigma']:.0e}  r_lin={d4['r_lin']:.0e}  "
        f"start={d4['start_dist']:.1e}  iters={d4['n_iters']}  runs={d4['n_runs']}"
    )
    print(
        f"  mean end-norm UNTUNED (nur Kugel) = {d4['mean_end_norm_untuned']:.3e}  "
        "(Blow-up entlang lambda_u)"
    )
    print(f"  mean end-norm TUNED   (auf W^s)   = {d4['mean_end_norm_tuned']:.3e}  ~ O(sigma)")
    print(
        f"  mean exit-step untuned (verlaesst Kugel) = "
        f"{d4['mean_exit_step_untuned']:.1f} / {d4['n_iters']}"
    )
    print(f"  gate_pass = {d4['gate_pass']}")
    print("=" * 78)
    print(f"GESAMT gate_pass = {payload['gate_pass']}")
    print(f"evidence -> {out}")
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
