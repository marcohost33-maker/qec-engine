"""Deterministischer Reproducer der numerischen ProofBlock-v1.3-Orakel (Defekt 2, 3 & 4).

ZWECK / POSITIONIERUNG (AGENTS.md Working-Agreement 1):
Der Errata-Companion `spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md` belegt zwei
Defekte gegen ein *unabhaengiges numerisches Orakel*. AGENTS.md verlangt:
„Keine Komponente als 'validiert' behaupten ohne lauffaehigen Code + Gate-Log in results/."
Dieses Skript IST dieser lauffaehige Code: es erzeugt die zwei Orakel-Tabellen
deterministisch (fixer Seed) und schreibt sie als Gate-Log nach
`results/proofblock-v13-oracles.json`. Die Zahlen im Errata-Doc stammen aus GENAU
diesem Lauf (Seed unten verankert); der Doc-Text verlinkt dieses Artefakt.

Keine neuen Dependencies (nur numpy, bereits Repo-Dependency).

--- Defekt 2 (Knoten P3.1): Jacobian-Konvergenzrate hat 1/eps_diff-Verstaerkung + 1/sqrt(n) ---
Der docx behauptet  ||M_hat - DR(g*)||_F = O_P(1/sqrt(N_rep)) + O(eps^2) + O(n^-beta), also OHNE
1/eps_diff und OHNE 1/sqrt(n). Korrekt (SoT v1.3 Z.87):
  ||M_hat - DR(g*)||_F = O_P( 1/(eps_diff*sqrt(n*N_rep)) ) + O(eps_diff^2) + O(n^-beta/eps_diff).
Zentraler pruefbarer Anker (SoT Z.81-83):  MSE(eps) ~ eps^4 + c/(n*N_rep*eps^2)
  => d/deps = 0 => eps_opt ~ (n*N_rep)^(-1/6). Der endliche Kompromiss existiert NUR, weil der
  Varianzterm proportional 1/eps^2 ist (die 1/eps_diff-Verstaerkung, die der docx fallen liess).

TEST-MAP (glatt, analytisch bekannter Jacobian als Ground-Truth):
  R_i(g) = sum_j A0_ij * sin(g_j)   =>   DR(g*)_ij = A0_ij * cos(g*_j)   (EXAKT, analytisch).
  Gewaehlt, weil sin nicht-triviale 3. Ableitung hat (sin''' = -cos != 0), sodass die zentrale
  Differenz einen echten O(eps^2)-Trunkierungsbias traegt (bei rein linearer Map waere er 0):
    [R(g*+eps e_j) - R(g*-eps e_j)]/(2 eps) = DR_ij * (1 - eps^2/6) + O(eps^4) + Rausch/(2 eps).
  A0 (dim=3), g* fix aus Seed. Verrauschte Samples: additives Gauss mit Var = c0/(n*N_rep) je
  Funktionsauswertung -> im zentralen Quotienten Var(stoch) = Var/(2 eps^2) -> std proportional
  1/(eps*sqrt(n*N_rep)) (exakt der SoT-Term).

  Gate A (VARIANZ-VERSTAERKUNG): log-log-Fit des reinen Stochastik-Terms von ||M_hat - DR||_F
    ueber eps_diff (mittel-subtrahiert, isoliert die Rausch-Komponente) -> Steigung soll ~ -1
    sein (docx behauptete implizit 0). Zusatz-Beleg: rauschfreie Trunkierung -> Steigung ~ +2.
  Gate B (eps_opt-SKALIERUNG): pro n*N_rep wird MSE(eps) = mean ||M_hat - DR||_F^2 gemessen und
    an das Modell  a*eps^4 + b*eps^-2  gefittet -> eps_opt = (b/(2a))^(1/6); der log-log-Exponent
    von eps_opt ueber n*N_rep soll ~ -1/6 (-0.1667) sein.
  ORAKEL-GATE (PROPOSED, Equalita-Kalibrierung ausstehend): slope_A in [-1.15,-0.85],
    exp_B in [-0.20,-0.13].

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
# --- Defekt 2 (Jacobian-Konvergenzrate) ---
DEFECT2_DIM = 3
DEFECT2_C0 = 1.0  # Rausch-Varianz je Funktionsauswertung = c0 / (n*N_rep)
DEFECT2_GATE_A_NN = 1e6  # fixes n*N_rep fuer die Varianz-Amplifikations-Kurve (Gate A)
DEFECT2_GATE_A_EPS = (1e-3, 1e-1, 8)  # (lo, hi, npts) log-Gitter eps_diff
DEFECT2_GATE_A_K = 500  # Realisierungen je eps (Stochastik-Term)
DEFECT2_GATE_B_NN = (3e4, 1e5, 3e5, 1e6, 3e6, 1e7)  # n*N_rep-Gitter fuer eps_opt-Skalierung
DEFECT2_GATE_B_EPS = (1e-2, 1.0, 12)  # eps_diff-Gitter je n*N_rep (MSE-Kurve)
DEFECT2_GATE_B_K = 300  # Realisierungen je (n*N_rep, eps)
DEFECT2_BIAS_EPS = (1e-2, 3e-1, 8)  # eps-Gitter fuer rauschfreien Trunkierungsbias (Zusatz-Beleg)
# Gate-Toleranzen: PROPOSED (Equalita-Kalibrierung ausstehend, aus SoT-Theorie nicht literal belegt)
DEFECT2_SLOPE_A_RANGE = (-1.15, -0.85)  # PROPOSED: Varianz-Term std ~ eps^-1
DEFECT2_EXP_B_RANGE = (-0.20, -0.13)  # PROPOSED: eps_opt ~ (n*N_rep)^(-1/6) = -0.1667
DEFECT2_BIAS_SLOPE_RANGE = (1.8, 2.2)  # PROPOSED: rauschfreier Bias ~ eps^2
DEFECT2_R2_A_MIN = 0.99  # PROPOSED: Guete des Varianz-Log-log-Fits
DEFECT2_R2_B_MIN = 0.95  # PROPOSED: Guete des eps_opt-Log-log-Fits


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


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Steigung + R^2 einer log-log-Regression log(y) = slope*log(x) + b."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    # ss_tot == 0 => konstante log(y)-Serie: R^2 undefiniert (0/0). NICHT still 1.0
    # zurueckgeben (das liesse einen degenerierten Fit als perfekt gelten und ein
    # r2>=threshold-Gate faelschlich bestehen). nan ist fail-closed: nan>=t == False.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def _d2_A0(dim: int = DEFECT2_DIM, seed: int = SEED) -> np.ndarray:
    """Feste Koeffizientenmatrix A0 der Test-Map R_i(g)=sum_j A0_ij sin(g_j) (deterministisch)."""
    return np.random.default_rng(seed + 3).standard_normal((dim, dim))


def _d2_gstar(dim: int = DEFECT2_DIM, seed: int = SEED) -> np.ndarray:
    """Fester Auswertepunkt g* (deterministisch); cos(g*_j) != 0 => DR nicht entartet."""
    return np.random.default_rng(seed + 7).uniform(-1.0, 1.0, dim)


def defect2_analytic_jacobian(g_star: np.ndarray, A0: np.ndarray) -> np.ndarray:
    """ANALYTISCHER Ground-Truth-Jacobian der Test-Map.

    R_i(g) = sum_j A0_ij sin(g_j)  =>  DR(g*)_ij = A0_ij cos(g*_j).
    """
    return A0 * np.cos(g_star)[None, :]


def _d2_test_map(g: np.ndarray, A0: np.ndarray) -> np.ndarray:
    """Glatte Test-Map R(g) = A0 @ sin(g) (nicht-lineare 3. Ableitung -> echter O(eps^2)-Bias)."""
    return A0 @ np.sin(g)


def _d2_estimate_jacobian(
    A0: np.ndarray, g_star: np.ndarray, eps: float, noise_var: float, rng: np.random.Generator
) -> np.ndarray:
    """M_hat via spaltenweiser zentraler Differenz aus verrauschten Samples.

    Jede Funktionsauswertung erhaelt additives Gauss mit Varianz noise_var (= c0/(n*N_rep)).
    Spalte j: [R(g*+eps e_j) - R(g*-eps e_j)]/(2 eps).
    """
    dim = g_star.size
    s = float(np.sqrt(noise_var))
    M = np.empty((dim, dim))
    for j in range(dim):
        step = np.zeros(dim)
        step[j] = eps
        rp = _d2_test_map(g_star + step, A0) + rng.standard_normal(dim) * s
        rm = _d2_test_map(g_star - step, A0) + rng.standard_normal(dim) * s
        M[:, j] = (rp - rm) / (2.0 * eps)
    return M


def defect2_oracle(
    dim: int = DEFECT2_DIM,
    gate_a_nn: float = DEFECT2_GATE_A_NN,
    gate_a_eps=DEFECT2_GATE_A_EPS,
    gate_a_k: int = DEFECT2_GATE_A_K,
    gate_b_nn=DEFECT2_GATE_B_NN,
    gate_b_eps=DEFECT2_GATE_B_EPS,
    gate_b_k: int = DEFECT2_GATE_B_K,
    bias_eps=DEFECT2_BIAS_EPS,
    seed: int = SEED,
) -> dict:
    """Defekt-2-Orakel: 1/eps_diff-Varianzverstaerkung (Gate A) + eps_opt~(n*N_rep)^-1/6 (Gate B).

    Belegt die SoT-v1.3-Korrektur numerisch gegen einen analytisch bekannten Jacobian:
    der Varianz-Term von ||M_hat-DR||_F skaliert ~ 1/eps_diff (docx liess das fallen), und der
    daraus folgende Bias-Varianz-Kompromiss gibt eps_opt ~ (n*N_rep)^(-1/6).
    """
    A0 = _d2_A0(dim, seed)
    g_star = _d2_gstar(dim, seed)
    DR = defect2_analytic_jacobian(g_star, A0)

    # --- Gate A: Varianz-Verstaerkung des Stochastik-Terms ueber eps_diff ---
    rng_a = np.random.default_rng(seed + 11)
    lo_a, hi_a, npts_a = gate_a_eps
    eps_a = np.geomspace(lo_a, hi_a, npts_a)
    v_a = DEFECT2_C0 / gate_a_nn
    stoch = np.empty(len(eps_a))
    for i, eps in enumerate(eps_a):
        Ms = np.array([_d2_estimate_jacobian(A0, g_star, eps, v_a, rng_a) for _ in range(gate_a_k)])
        resid = Ms - Ms.mean(axis=0)  # deterministischer Bias faellt raus -> reiner Stochastik-Term
        stoch[i] = float(np.sqrt(np.mean(np.sum(resid**2, axis=(1, 2)))))
    slope_a, r2_a = _loglog_fit(eps_a, stoch)

    # --- Zusatz-Beleg: rauschfreier Trunkierungsbias ~ eps^2 ---
    lo_b, hi_b, npts_b = bias_eps
    eps_bias = np.geomspace(lo_b, hi_b, npts_b)
    rng_nf = np.random.default_rng(0)  # noise_var=0 -> Stream unbenutzt, nur Signatur
    bias_nf = np.array(
        [
            float(np.linalg.norm(_d2_estimate_jacobian(A0, g_star, e, 0.0, rng_nf) - DR))
            for e in eps_bias
        ]
    )
    bias_slope, bias_r2 = _loglog_fit(eps_bias, bias_nf)

    # --- Gate B: eps_opt-Skalierung ueber n*N_rep ---
    rng_b = np.random.default_rng(seed + 13)
    lo_e, hi_e, npts_e = gate_b_eps
    eps_e = np.geomspace(lo_e, hi_e, npts_e)
    design = np.column_stack([eps_e**4, eps_e**-2])  # MSE-Modell a*eps^4 + b*eps^-2
    nn_grid = np.asarray(gate_b_nn, float)
    eps_opt = np.empty(len(nn_grid))
    mse_rows = []
    for i, nn in enumerate(nn_grid):
        v = DEFECT2_C0 / nn
        mse = np.empty(len(eps_e))
        for jx, eps in enumerate(eps_e):
            Ms = np.array(
                [_d2_estimate_jacobian(A0, g_star, eps, v, rng_b) for _ in range(gate_b_k)]
            )
            mse[jx] = float(np.mean(np.sum((Ms - DR) ** 2, axis=(1, 2))))
        (a_coef, b_coef), *_ = np.linalg.lstsq(design, mse, rcond=None)
        eps_opt[i] = float((b_coef / (2.0 * a_coef)) ** (1.0 / 6.0))
        mse_rows.append(
            {
                "n_times_Nrep": float(nn),
                "a_eps4": float(a_coef),
                "b_eps_m2": float(b_coef),
                "eps_opt": eps_opt[i],
            }
        )
    exp_b, r2_b = _loglog_fit(nn_grid, eps_opt)

    # --- Gates (PROPOSED-Toleranzen) ---
    gate_a_pass = (DEFECT2_SLOPE_A_RANGE[0] <= slope_a <= DEFECT2_SLOPE_A_RANGE[1]) and (
        r2_a >= DEFECT2_R2_A_MIN
    )
    gate_bias_pass = DEFECT2_BIAS_SLOPE_RANGE[0] <= bias_slope <= DEFECT2_BIAS_SLOPE_RANGE[1]
    gate_b_pass = (DEFECT2_EXP_B_RANGE[0] <= exp_b <= DEFECT2_EXP_B_RANGE[1]) and (
        r2_b >= DEFECT2_R2_B_MIN
    )
    gate_pass = bool(gate_a_pass and gate_bias_pass and gate_b_pass)
    return {
        "defect": 2,
        "node": "P3.1 (Jacobian-Konvergenz)",
        "claim": (
            "Rate O_P(1/(eps_diff*sqrt(n*N_rep))) + O(eps_diff^2) + O(n^-beta/eps_diff); "
            "Varianz-Term ~ 1/eps_diff (docx liess 1/eps_diff und 1/sqrt(n) fallen), "
            "=> eps_opt ~ (n*N_rep)^(-1/6)."
        ),
        "test_map": "R_i(g) = sum_j A0_ij sin(g_j); DR(g*)_ij = A0_ij cos(g*_j) (analytisch)",
        "dim": dim,
        "DR_frobenius_norm": float(np.linalg.norm(DR)),
        "gate_a_variance_amplification": {
            "n_times_Nrep_fixed": float(gate_a_nn),
            "eps_grid": [float(e) for e in eps_a],
            "stoch_std": [float(s) for s in stoch],
            "measured_slope": slope_a,
            "theory_slope": -1.0,
            "r2": r2_a,
            "tol_slope_PROPOSED": list(DEFECT2_SLOPE_A_RANGE),
            "gate_pass": bool(gate_a_pass),
        },
        "truncation_bias_check": {
            "eps_grid": [float(e) for e in eps_bias],
            "bias_frobenius_noisefree": [float(b) for b in bias_nf],
            "measured_slope": bias_slope,
            "theory_slope": 2.0,
            "r2": bias_r2,
            "tol_slope_PROPOSED": list(DEFECT2_BIAS_SLOPE_RANGE),
            "gate_pass": bool(gate_bias_pass),
        },
        "gate_b_eps_opt_scaling": {
            "n_times_Nrep_grid": [float(n) for n in nn_grid],
            "eps_opt_measured": [float(e) for e in eps_opt],
            "measured_exponent": exp_b,
            "theory_exponent": -1.0 / 6.0,
            "r2": r2_b,
            "tol_exp_PROPOSED": list(DEFECT2_EXP_B_RANGE),
            "rows": mse_rows,
            "gate_pass": bool(gate_b_pass),
        },
        "gate_pass": gate_pass,
    }


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
    """Erzeuge alle Orakel + Gesamt-Gate-Log (JSON-serialisierbar)."""
    d2 = defect2_oracle()
    d3 = defect3_oracle()
    d4 = defect4_oracle()
    return {
        "tool": "spec.reproducers.proofblock_v13_oracles",
        "purpose": "Numerische Orakel des Errata-Companion ProofBlock v1.3 (Defekt 2, 3 & 4)",
        "spec_doc": "spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md",
        "seed": SEED,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "gate_pass": bool(d2["gate_pass"] and d3["gate_pass"] and d4["gate_pass"]),
        "defect2": d2,
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

    d2, d3, d4 = payload["defect2"], payload["defect3"], payload["defect4"]
    print(f"ProofBlock v1.3 Orakel-Reproducer  seed={payload['seed']}  numpy={np.__version__}")
    print("=" * 78)
    ga = d2["gate_a_variance_amplification"]
    gb = d2["gate_b_eps_opt_scaling"]
    tb = d2["truncation_bias_check"]
    print(f"Defekt 2 (P3.1): Jacobian-Konvergenzrate  R_i(g)=sum_j A0_ij sin(g_j)  dim={d2['dim']}")
    print(
        f"  Gate A Varianz-Term std(eps) ueber eps_diff (n*N_rep={ga['n_times_Nrep_fixed']:.0e}): "
        f"slope={ga['measured_slope']:+.4f} (Theorie {ga['theory_slope']:+.1f})  R2={ga['r2']:.5f}"
    )
    print(
        f"  Trunkierungsbias (rauschfrei) ~ eps^2: slope={tb['measured_slope']:+.4f} "
        f"(Theorie +2)  R2={tb['r2']:.5f}"
    )
    print(
        f"  Gate B eps_opt ueber n*N_rep: exponent={gb['measured_exponent']:+.4f} "
        f"(Theorie {gb['theory_exponent']:+.4f} = -1/6)  R2={gb['r2']:.5f}"
    )
    print(f"  gate_pass = {d2['gate_pass']}   [Toleranzen PROPOSED, Equalita-Kalibrierung offen]")
    print("-" * 78)
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
