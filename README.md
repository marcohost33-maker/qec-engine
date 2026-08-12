# AdaptiveRG-QEC Engine

**Spec + Phase-1-MVP.** Eine *Adaptive Renormalization-Group + Quantum-Error-Correction* Simulation-Engine:
integriert stochastische Dynamik, adaptive Steuerung, Sampling, RG-Analyse, Surrogate-Beschleunigung und
Stabilitäts-Guards. Coworker-Research Säule 3 (Physik/Methodik).

> **Ehrlicher Status (2026-06-18):** Die gehärtete Theorie (`spec/`) liegt vor; ein **bounded Phase-1-MVP**
> (`src/adaptiverg_qec/`, Version `0.1.0.dev0`) implementiert den MCMC-A-Kernel mit Foster-Lyapunov-Guard
> einen MCRG-C-Kernel und einen Swendsen-MCRG-Schätzer (skalare sample-geschätzte R̂). Reifegrad **dev/Prototyp** — nicht release-fertig, nicht selbst-zertifiziert.
> Was MVP-real vs. offen ist, steht unten und in `src/adaptiverg_qec/mvp_instance.py`.

## Was dieses Tool IST — und was NICHT (Positionierung)

Dieses Tool ist ein **Diagnostik- und Verifikations-Harness mit rigorosen Konvergenz-Guards**, KEIN
Frontier-Threshold-Tool. Die Einordnung ehrlich:

- **"Adaptive RG-QEC" ist eine interne Neu-Prägung** dieses Projekts — **kein etabliertes Forschungsfeld.**
  "RG" meint hier *Threshold-Bestimmung über kritische Exponenten* (MCRG-artig), **NICHT** den
  *RG-Decoder* à la Duclos-Cianci & Poulin (2010), der ein eigenständiges, unverwandtes Konzept ist.
- **SOTA für Präzisions-Thresholds liegt anderswo:** Tensor-Network/MPS-Kontraktion des statistical-mechanics
  mapping (Bravyi, Suchara & Vargo, *Phys. Rev. A* 90, 032326, 2014) liefert die genauesten Schwellen.
  Der hier verfolgte MCMC + MCRG-Ansatz ist **inkrementell**, nicht SOTA für Schwellen-Präzision.
- **Stärke des Harness:** mathematisch sauber verifizierte *Maschinerie* — Foster-Lyapunov-Drift als
  Laufzeit-Guard, Complex-Step-Jacobian gegen analytische Orakel, bit-exakte Reproduzierbarkeit,
  Negativ-/Edge-Input-Gates. Jede Zahl wird gegen ein **unabhängiges** Orakel geprüft, nicht gegen sich selbst.

### Architektur-Entscheid: MCRG + QEC-Diagnostik bleiben EIN Repo (2026-06-19)

Die zwei Hälften des Repos — (A) die **MCRG-Maschinerie für kritische Exponenten**
(Ising/Onsager, Wolff/Swendsen, `mcrg_*`) und (B) die **QEC-Code-Diagnostik**
(`qec_diagnostics` / `qec_fit_diagnostics`: p_L-vs-p, Distanz-Exponent, Pseudo-Threshold,
Lambda) — sind **kein thematischer Bruch**, sondern ein kohärentes Programm: der
**QEC-Threshold ist über das RBIM-Nishimori-Mapping** (Dennis, Kitaev, Landahl, Preskill,
*J. Math. Phys.* **43**, 4452, 2002) ein **Phasenübergang im random-bond Ising-Modell** —
genau die Art Phasenübergang, die die MCRG/kritische-Exponenten-Maschinerie misst. **Wichtig
(Status):** die bestehende `mcrg_*`-Maschinerie misst bislang das *uniforme* 2D-Ising
(`ising2d.py`); die RBIM-Nishimori-Brücke wird durch **Inkr.4 (`rbim_nishimori.py`)** konkret
realisiert — sie lokalisiert den Übergang bei `p*≈0,12` (L=12) vs publiziertem `p_c≈0,1094`
und verschweisst damit beide Hälften empirisch.

**Entscheid (Kohärenz/Korrektheit + Effizienz/Produktivität): NICHT splitten.** Ein Split
würde die physikalische Brücke zerschneiden und CI/Spec/Provenienz unnötig duplizieren;
die gemeinsame `spec/`-Schicht ist die Wurzel beider Hälften. Sollte Inkr.4 wider Erwarten
zeigen, dass die Brücke nicht trägt, wird der Split neu bewertet (Pre-Mortem dokumentiert).

### Gültigkeitsgrenzen

- Die Phase-1-**MVP-Instanz** ist der **1D-Repetition-Code-Ring** (= 1D-Ising). Der hat `T_c = 0`, also
  **keinen endlichen Threshold** und keinen nicht-trivialen 2D-Exponenten. Das MVP demonstriert die
  **Korrektheit der Maschinerie** gegen geschlossene analytische Orakel — es liefert **keinen** physikalischen
  Frontier-Wert.
- **Phase-3b (NEU) bringt das 2D-Ising-Modell** (`T_c = 2/ln(1+√2) ≈ 2.269`, nicht-trivialer Fixpunkt) und
  die volle **Multi-Operator-Swendsen-MATRIX**. Der thermische Exponent `y_t` wird gegen das **exakte
  Onsager-Orakel `y_t = 1`** geprüft — aber **auf Plausibilitäts-Niveau, NICHT als Frontier-Wert**:
  Single-Spin-Metropolis nahe T_c hat kritisches Slowing-Down, kleines L + 1 RG-Stufe → `y_t` ist GROB
  (`y_t ≈ 0.97 ± 0.01`, |err| ≈ 0.035).
- **Phase-4 (NEU) adressiert die drei vom Cross-Family-Review benannten Präzisions-Schwächen:**
  (1) **Wolff-Single-Cluster-Sampler** (`wolff2d.py`, `P_add = 1−e^{−2K}`) schlägt das kritische
  Slowing-Down — `τ_int(Wolff) ≪ τ_int(Metropolis)` (gemessen `×12–16` bei L=32);
  (2) **mehrere iterierte RG-Stufen** (`mcrg_multirg.py`) → `y_t` rückt näher an den Fixpunkt:
  bester Iterationswert `|y_t−1| ≈ 0.006`, tiefste-Iter `≈ 0.020` — **beide** schlagen Phase-3b `0.035`;
  (3) **ungerader (magnetischer) Sektor** → `y_h` gegen das exakte Onsager-Orakel `y_h = 15/8 = 1.875`:
  bester Iterationswert `|y_h − 15/8| ≈ 0.002`, tiefste-Iter `≈ 0.003`. **Ehrlich:** „best" = Minimum
  über Iterationen (proximity-selektiert, optimistisch; NICHT die tiefste Stufe — bei kleinem L kein klarer
  Plateau, alle Iterationen geloggt); Rest-finite-Size-Systematik bleibt, **keine** `L→∞`-FSS, **kein** Frontier-Wert.
- **ROADMAP-Inkr.4 (NEU) — die RBIM-Nishimori ↔ MCRG/QEC-Brücke (Architektur-Schlussstein).**
  `rbim_nishimori.py` implementiert das **±J random-bond Ising-Modell (RBIM) auf der Nishimori-Linie**
  (`p = 1/(1+e^{2β})`) mit DERSELBEN Maschinerie (Wolff-Cluster aus `wolff2d.py`-Idee, RBIM-generalisiert:
  `P_add = 1−e^{−2β·J_ij·s_i·s_j}` auf befriedigten Bonds, hybridisiert mit gewichteten Metropolis-Sweeps
  + aligned-Start-Ordnungsparameter-Protokoll). Das ist **kein Folklore, sondern ein exaktes Mapping**:
  der 2D-±J-RBIM-**Nishimori-Multikritische-Punkt** = Code-Capacity-**Toric/Surface-Threshold**
  `p_c ≈ 0.1094` (Dennis/Kitaev/Landahl/Preskill, *J. Math. Phys.* 43, 2002; Honecker/Picco/Pujol PRL 87,
  2001: `0.1094(2)`; Merz/Chalker PRB 65, 2002: `0.1093(2)`). **Gemessen — regenerierbares Artefakt
  `results/inkr4-rbim-nishimori.json` (L=8, 24 Disorder-Realisierungen):** disorder-gemittelte `[<|m|>]`
  fällt monoton von `0.98` (FM, p=0.04) auf `0.36` (PM, p=0.20); steilster Abfall lokalisiert bei
  `p* ≈ 0.145` (|err| ≈ 0.036 vs `p_c`). **Feinere Auflösung — regenerierbares Artefakt `results/inkr4-rbim-nishimori-L12.json` (L=12, 30 Realisierungen, via `python -m adaptiverg_qec.rbim_nishimori --L 12 --n-disorder 30`):**
  `[<|m|>] 0.99 → 0.22`, `p* = 0.120` (|err| = 0.011) — der erwartete Finite-Size-Shift zu kleinerem `p*` mit
  wachsendem L. Beide **konsistent mit `p_c ≈ 0.109`** auf der groben Auflösung. **Ehrlich:** kleines L +
  endliches Disorder-Sampling → **Plausibilitäts-Niveau**, KEINE `L→∞`-FSS, KEIN Frontier-Wert.
  **Bedeutung:** misst das RBIM-Tooling (gebaut aus `ising2d`/`wolff2d`/MCRG) `p* ≈ p_c`, dann misst die
  MCRG-Exponenten-Maschinerie GENAU dasselbe Objekt wie der QEC-Threshold → der **„nicht-splitten"-
  Architektur-Entscheid (MCRG + QEC in EINEM Repo) ist empirisch gestützt** (im Rahmen dieser Auflösung).
- **Phase-6 (NEU) schließt die drei dokumentierten Phase-4/5-Lücken:** SNIS mit geschlossenem
  χ²-Orakel (`snis.py`), Surrogate-Beschleunigung als Delayed-Acceptance (`surrogate.py`,
  Christen & Fox 2005) und Checkpoint/Restart mit Lockfile (`checkpoint.py`) — alle in **bounded,
  orakel-validierter** Form (s. Abschnitt Phase-6). **Weiterhin offen:** Defensive Mixture,
  SNIS auf 2D/RBIM-Targets, Kombination DA×Diminishing-Adaptation, MMD-Drift.

## Phase-1-MVP: real implementiert vs. offen

| Komponente | Status |
|---|---|
| A-Kernel: adaptiver Single-Spin-Metropolis (`a_kernel.py`) | **real** (detailed balance, Philox-Seed) |
| Foster-Lyapunov-Drift-Guard, conditional-mean (`drift.py`) | **real** (greift + feuert, getestet) |
| Diminishing-Adaptation-Schedule `sum a_t < inf` + Containment-Clip | **real** |
| C-Kernel: 1D-Ising-Decimation-RG-Map (`rg_map.py`) | **real** (lehrbuchexakt, b=2) |
| Jacobian: Complex-Step + zentrale Differenzen + Exponenten/Hyperbolizität | **real** (CS==FD==analytisch) |
| Analytisches Transfer-Matrix-Orakel (`ising1d.py`) | **real** (machine-precision gegen Brute-Force) |
| Selftest-Gates (`cli.py --selftest`, 45 Gates, JSON-Log) | **real** (45/45 [PASS], exit 0) |
| **SNIS mit χ²-Varianz-Bound (`snis.py`, Phase-6)** | **real** — offene 1D-Ising-Kette: χ² GESCHLOSSEN `[cosh(2K_t−K_p)cosh(K_p)/cosh²(K_t)]^{L−1}−1`; ESS/N trifft `1/(1+χ²)` (\|diff\|<0.005); gemessener Bias trifft den geschlossenen O(1/N)-Koeffizienten `(1+χ²)(tanh K_t−tanh(2K_t−K_p))` (bias·N/c ≈ 0.97 @ N=100); MSE ≤ `4(1+χ²)/N` (Agapiou et al. 2017); ESS-Kollaps-Guard beidseitig. **Offen:** Defensive Mixture, 2D/RBIM-Targets |
| Swendsen-MCRG-Schätzer (skalare stochastische R̂ aus Samples, Spec §6) | **real** — `T̂=⟨S'S⟩_c/⟨S'S'⟩_c` vs `tanh(2K)`. Phase-2: exakt-i.i.d.-Sampler (≤0.54σ, Bias↓ mit 1/√N). **Phase-3a (NEU): aus dem korrelierten A-Kernel-MCMC mit autokorrelations-bewussten Fehlerbalken** (s.u.) |
| **Autokorr-bewusste Fehler (`autocorr.py`, Phase-3a)** | **real** — FFT-ρ (Wiener-Khinchin), τ_int + Wolff-g-Windowing, Binning-Cross-Check, Block-Jackknife für das Verhältnis. Validiert gegen AR(1)-Orakel + Γ==Binning-Plateau |
| **Multi-Operator-Swendsen-MATRIX (`ising2d.py` + `mcrg_matrix.py`, Phase-3b)** | **real** — 2D-Ising-Checkerboard-Metropolis (vektorisiert) + Majority-Rule-Blocking b=2; ≥2 gerade Operatoren (NN/NNN/Plaquette); `T = A·B⁻¹` via `np.linalg.solve`; Eigenwerte → `y_t = ln λ_max/ln 2`; Block-Jackknife-Fehler. Gegen Onsager `y_t=1` (**grob**: `y_t≈0.97±0.01`, kritisches Slowing-Down) |
| **Wolff-Single-Cluster-Sampler (`wolff2d.py`, Phase-4)** | **real** — `P_add = 1−e^{−2K}`, vektorisierter BFS-Cluster, rejection-free. Energie gegen exakte L=4-Enumeration (\|err\|<0.008); `τ_int(Wolff) ≪ τ_int(Metropolis)` belegt (×12–16 @ L=32) |
| **Multi-RG-Iteration + ungerader Sektor `y_h` (`mcrg_multirg.py`, Phase-4)** | **real** — iterierte Majority-Stufen (L=32→16→8→4); gerader `y_t` konvergiert (bester \|y_t−1\|≈0.006 vs 3b 0.035); 3-Spin-ungerade Operatoren → `y_h` vs Onsager `15/8` (bester \|y_h−15/8\|≈0.002). **Ehrlich:** Rest-finite-Size, keine L→∞-FSS |
| **RBIM-Nishimori ↔ QEC-Brücke (`rbim_nishimori.py`, Inkr.4)** | **real** — ±J-RBIM auf der Nishimori-Linie (`p=1/(1+e^{2β})`); RBIM-generalisierter Wolff (`P_add=1−e^{−2β·J_ij·s_i·s_j}`) + gewichtete Metropolis-Sweeps + aligned-Start. Verifiziert gegen **exakte L=4-Enumeration** (auch frustriert, E/N + ⟨\|m\|⟩) + **Gauge-Invarianz** + Stationaritäts-Sektor (aligned-Start → korrekter FM/PM-Sektor, G-N5). Lokalisiert `p* = 0.145` (L=8) bzw. `0.120` (L=12, |err|=0.011) vs publiziertem `p_c≈0.1094` (Toric-Threshold) — beide Werte committet/regenerierbar. **Ehrlich: grob/Plausibilität, keine L→∞-FSS** |
| **CLT-Varianz σ²_g (`clt.py`, Phase-5)** | **real** — MCMC-CLT `σ²_g = 2·τ_int·Var(g)` (Γ-Methode) + **unabhängiger OBM-Schätzer** (Flegal-Jones 2010). Gegen geschlossene AR(1)-Form `σ²_g=σ²_ε/(1−φ)²` (φ=0.8, Orakel 25.0 → Γ 25.26 rel.0.010, OBM 24.49 rel.0.020). **Coverage beidseitig:** korrektes σ²_g deckt 95%-CI mit 0.937; iid-Annahme UNTERdeckt mit 0.490 |
| **R̂-Multichain (`rhat.py`, Phase-5)** | **real** — rank-normalized split-R̂ + folded-R̂ + bulk/tail-ESS (Vehtari et al. 2021, R̂<1.01). A-Kernel M=4 → R̂=1.0003 (converged, ESS_bulk≈5149). **Beidseitig:** Mittel-Drift → R̂=1.62 (bulk); Skalen-Drift → R̂=1.27 (folded>bulk) |
| **Run-Manifest (`manifest.py` + CLI `phase5`, Phase-5)** | **real** — JSON mit Seeds/Parametern/Versionen/git-SHA/Plattform; `--from-manifest` reproduziert **byte-identisch** (SHA-256). **Beidseitig:** Round-trip == identischer Hash; geänderter Seed → anderer Hash |
| **Surrogate-DA + Drift-Guard (`surrogate.py`, Phase-6)** | **real** — Delayed-Acceptance-Metropolis (Christen & Fox 2005), Surrogat `β̃=β(1+γ)`: γ=0 **bit-identisch** zum Metropolis-A-Kernel; absichtlich miskalibriertes Surrogat (γ=±0.25/0.3) bleibt exakt vs Transfer-Matrix-Orakel (\|err\|<0.05); 34–42 % weniger Stufe-2-Auswertungen (Accounting-Größe, kein gemessener Speedup im 1D-Toy); Drift-Guard hält (γ=0) und feuert (γ groß) — beidseitig |
| **Checkpoint/Restart + Lockfile (`checkpoint.py`, Phase-6)** | **real** — Philox-State-Serialisierung + gemeinsamer Sweep-/Postprocess-Code-Pfad: Interrupt (auch mehrfach) + Resume ⇒ **byte-identischer** `result_hash` wie der ununterbrochene Lauf; O_EXCL-Lockfile gegen konkurrierende Writer (über Laden+Lauf gehalten); SHA-256-Integritäts-Hash weist korrumpierte Checkpoints LAUT ab (unkeyed — Korruptions-Erkennung, keine krypt. Authentifizierung) |
| MMD-Drift + Defensive Mixture (Spec §8/§5) | **offen** (NICHT erledigt) |

### Phase-3a: korrelierter A-Kernel als Sample-Quelle + autokorr-Fehler (NEU)

Phase-2 nutzte einen exakten **i.i.d.**-Sampler (sauberer 1/√N-Test). Phase-3a verdrahtet die **echte**
Sample-Quelle — die korrelierte Markov-Kette des adaptiven A-Kernels — und ersetzt die naiven Fehlerbalken
durch **autokorrelations-bewusste** (Wolff Γ-Methode + g-Funktions-Windowing, hep-lat/0306017; Binning als
Cross-Check; Block-Jackknife für das Verhältnis T̂). Alles in numpy (keine neuen Runtime-Deps).

**Wissenschaftlicher Kernpunkt (ehrlich belegt, `results/phase3a-akernel-autocorr.json`):** korrelierte
Samples tragen weniger Information, also ist N_eff = N/(2·τ_int) < N und die **korrekten Fehler sind GRÖSSER
als die naiven i.i.d.-Fehler**. Gemessen (A-Kernel, L=64, N=18 000 nach Burn-in):

| K | T̂ ± σ_korr | tanh(2K) | σ_iid (naiv) | Inflation | τ_int(max) | N_eff |
|---|---|---|---|---|---|---|
| 0.3 | 0.537 ± 0.0096 | 0.537 | 0.0095 | 1.006 | 0.65 | 13 745 |
| 0.5 | 0.768 ± 0.0078 | 0.762 | 0.0076 | 1.026 | 0.90 |  9 951 |
| 0.7 | 0.878 ± 0.0065 | 0.885 | 0.0060 | 1.073 | 1.66 |  5 438 |
| 0.9 | 0.924 ± 0.0054 | 0.947 | 0.0048 | 1.129 | 2.81 |  3 198 |

τ_int wächst mit K (langsameres Mischen näher am T=0-Fixpunkt); Inflation > 1 durchgehend; N_eff < N immer.
**Ehrliche Nuance (kein Über-Claim):** das *Verhältnis* T̂ erbt die τ_int der Rohfelder nicht voll (Zähler/
Nenner-Fluktuationen kürzen sich teilweise), daher ist die Inflation moderat; der Block-Jackknife misst die
korrekte Verhältnis-Inflation direkt. K≤0.7 wird gegen das L→∞-Orakel gegated (finite-L-Bias ≪ Fehler);
**K=0.9 ist ein Autokorrelations-Diagnostik-Punkt** — am Rand von Θ wird die endliche-L-Systematik
vergleichbar mit dem Fehlerbalken (kein sauberer 3σ-Treffer behauptet).

### Phase-3b: 2D-Ising + Multi-Operator-Swendsen-MATRIX (NEU)

Phase-2/3a war **skalar** (1D-Ising, `T_c=0` → kein nicht-trivialer Exponent, nur die Schätz-Maschinerie
prüfbar). Phase-3b bringt das **2D-Ising-Modell** (echter Fixpunkt bei `T_c ≈ 2.269`) und die volle
**Multi-Operator-Matrix**:

- `ising2d.py` — **vektorisierter** Checkerboard-(Schachbrett-)Metropolis (numpy, kein Python-Spin-Loop) +
  Majority-Rule-Blocking `b=2` (Kadanoff) mit **unbiased, deterministischem** Tie-Break (50/50, getestet).
- `mcrg_matrix.py` — gerade Operatoren `S₁=NN`, `S₂=NNN(Diag)`, `S₃=Plaquette`; connected-correlation-Matrizen
  `A_ab=⟨S′_a S_b⟩_c`, `B_ac=⟨S′_a S′_c⟩_c`; RG-Matrix **`T = A·B⁻¹` via `np.linalg.solve`** (keine explizite
  Inverse); Exponenten `y_i = ln|λ_i|/ln 2`; Block-Jackknife-Fehler über die Matrix-Pipeline.

**Resultat (ehrlich, `results/phase3b-swendsen-matrix.json`, L=16, N_op=2, K=K_c, 3 Seeds):**

| Größe | Wert | Orakel (Onsager, web-verifiziert) |
|---|---|---|
| `y_t` (multi-seed) | **0.965 ± 0.009** | `y_t = 1/ν = 1` |
| |y_t − 1| | ≈ 0.035 | — |
| λ_max | ≈ 1.95 | `b^{y_t} = 2` |
| `T = A·B⁻¹` Residuum `max|T·B−A|` | ≈ 2e-13 | 0 (linear-solve) |
| `cond(B)` | ≈ 87 | gut konditioniert |

**Ehrliche Scope-Grenze (kein Über-Claim):** Single-Spin-Metropolis nahe T_c leidet unter **kritischem
Slowing-Down**; kleines L + 1 RG-Stufe → `y_t` ist nur **grob** (Plausibilität, KEIN Frontier-Wert).
Cluster-Algorithmus (Wolff/Swendsen-Wang) + mehrere RG-Iterationen wären für Präzision nötig — das ist
**Phase-4**. Validiert wird die **MATRIX-MASCHINERIE** (≥2 Operatoren, `A·B⁻¹`, Eigenwert-Exponenten) gegen
das exakte Orakel, nicht ein Hochpräzisions-Wert. `y_h = 15/8` (ungerader Sektor) ist nicht implementiert.

## QEC-Diagnostik-Harness (Inkrement 1 + 2 + 3)

Additiver, eigenständiger Strang (`qec_diagnostics.py`, `qec_fit_diagnostics.py`): die
**eigentliche Fehlerkorrektur-Diagnostik**, die dem MCRG-Teil fehlte — logische Fehlerrate
`p_L` vs. physikalische `p` am exakt orakelbaren Repetition-Code. **Kein Decoder erfunden,
keine neuen Dependencies.** Positionierung + Roadmap: `docs/QEC_DIAGNOSTICS_ROADMAP.md`.

- **Inkr. 1** (`qec_diagnostics.py`): exaktes Binomial-Orakel `p_L(n,p)` + vektorisierte
  Monte-Carlo (Majority-Vote) mit `n_sigma`-Beleg gegen das Orakel; Pseudo-Threshold `p*=1/2`.
- **Inkr. 2** (`qec_fit_diagnostics.py`): die drei Standard-Vergleichskennzahlen der QEC-
  Literatur, jede gegen ein **geschlossenes** Orakel (kein Monte-Carlo, kein Decoder):

| Kennzahl | Methode | Orakel | Real gemessen |
|---|---|---|---|
| Code-Distanz-Exponent | sub-threshold log-log-Steigung `p_L`-vs-`p` | `(d+1)/2` (= `C(d,(d+1)/2)`-Leitterm) | `d∈{3,5,7,9}` → worst `abs_err = 7.2e-4` |
| Pseudo-Threshold | Kurven-Kreuzungs-Bisektion | analytisch `p* = 1/2` (Symmetrie `p_L(d,½)=½`) | worst `abs_err = 2.2e-16` |
| Lambda-Suppression `p_L(d)/p_L(d+2)` | Distanz-Sprung `d→d+2` @ p=0.1 | small-p `~(A_d/A_{d+2})/p` | `Λ ≈ 3.27 / 3.14 / 3.06` (alle > 1) |

Evidenz: `results/qec-fit-diagnostics-rep-code.json`. Jede Kennzahl ist gegen ihr
geschlossenes Orakel geprüft (s. Tabelle). Ein zusätzlicher Cross-Anchor gegen einen
ML-Brute-Force-Decoder ist für Inkr.3 (PyMatching) vorgesehen; der eigene Prototyp wurde
verworfen (verließ für `d=5` den Codespace) — daher wird hier **keine** Brute-Force-
Vergleichszahl behauptet.

- **Inkr. 3** (`surface_decoder.py`): **ECHTES MWPM-Decoding** via **Stim** (Sampling) +
  **PyMatching 2** (Minimum-Weight-Perfect-Matching) — kein Eigen-Decoder, hinter dem
  optional-dependency-Gate `pip install ".[surface]"`. Ohne das Extra SKIPPEN die Tests
  (sie failen nicht hart). Zwei unabhängige Orakel:

| Orakel | Modell | Vergleich | Real gemessen (regenerierbar) |
|---|---|---|---|
| A | Repetition-Code, code-capacity bit-flip, MWPM (= ML auf 1D-Matching-Graph) | exaktes Binomial-Orakel (Inkr.1) | `d∈{3,5,7}` @ p=0.1, 200k shots → worst `n_sigma = 1.46` |
| B | Rotated-Surface-Code, code-capacity pure-X-flip, MWPM-Threshold (Distanz-Kurven-Kreuzung) | publizierter MWPM-Threshold **0.103** (Dennis et al. 2002) | Paare (7,9)/(9,11)/(7,11) → `p_th ∈ [0.096, 0.098]`, best `abs_err = 0.0050` (80k shots/Zelle, zell-eigene Seeds) |

Evidenz: `results/qec-surface-mwpm.json` (`python -m adaptiverg_qec.surface_decoder`).

> **Ehrliche Korrektheits-Grenze (kein Overclaim):** Validiert wird gegen den MWPM-Wert
> 0.103 (Nulltemperatur-RBIM), **NICHT** gegen den optimalen ML/Tensor-Network-Threshold
> 0.1094 (Nishimori-Punkt) — MWPM ist near-optimal, aber sub-optimal; ein Schätzer, der
> 0.109 „erreicht", wäre verdächtig. Endliche Distanzen liefern **keine** `L→∞`-FSS; der
> Kreuzungs-Schätzer liegt systematisch unterhalb (gemessen mit zell-eigenen Seeds:
> (7,9)→0.0980, (9,11)→0.0962, (7,11)→0.0970 — der Distanz-Drift ist bei 80k Shots/Zelle
> nicht aufgelöst; eine frühere Messung mit geteiltem Seed über alle Zellen war
> rangkorreliert und wurde ersetzt). Phenomenological-noise (Mess-Fehler, mehrere Runden;
> publizierter MWPM-Threshold ~2.9%) ist noch nicht implementiert (nächster Schritt).
> Der in Inkr.2 verworfene hand-gerollte Spacetime-Matcher ist hiermit durch PyMatching ersetzt.

## Phase-6: SNIS + Surrogate-DA + Checkpoint/Restart-Lockfile (NEU)

Schließt die drei seit Phase-4/5 dokumentierten Lücken — jede Komponente gegen ein
**unabhängiges, geschlossenes** Orakel (Evidenz: `results/phase6-snis-surrogate-checkpoint.json`,
regenerierbar via `adaptiverg-qec phase6 --json ...`; Gates G39–G45):

1. **SNIS (`snis.py`):** Reweighting `K_p → K_t` auf der offenen 1D-Ising-Kette. Dort ist die
   χ²-Divergenz **exponential-family-exakt**: `1+χ² = [cosh(2K_t−K_p)cosh(K_p)/cosh²(K_t)]^{L−1}`.
   Validiert: ESS/N trifft `1/(1+χ²)`; der gemessene Bias trifft den **geschlossenen führenden
   Koeffizienten** `bias ≈ (1+χ²)(tanh K_t − tanh(2K_t−K_p))/N` (Delta-Methode; gemessen
   `bias·N/c ≈ 0.97` bei N=100 über 6000 Replikate, korrektes Vorzeichen); die MSE respektiert
   den Bound `4(1+χ²)/N` (Agapiou et al. 2017, Thm 2.1); Delta-Methoden-Fehler kalibriert
   (Schätzung/empirische MSE ∈ [0.88, 0.97]). **ESS-Kollaps-Guard beidseitig:** extremes
   Reweighting (χ²≈7·10⁵) wird geflaggt, gesundes nicht.
2. **Surrogate via Delayed Acceptance (`surrogate.py`):** zweistufiger Metropolis (Christen &
   Fox 2005) — Stufe 1 befragt nur das Surrogat `β̃=β(1+γ)`, Stufe 2 korrigiert exakt. Detailed
   balance gegen das EXAKTE π für **jedes** Surrogat: γ=0 ist **bit-identisch** zum
   Metropolis-A-Kernel (gleicher Philox-Stream), miskalibrierte Surrogate (γ=−0.25/+0.3)
   treffen das Transfer-Matrix-Orakel trotzdem. Die Ersparnis-Zähler (34–42 % weniger
   Stufe-2-Auswertungen) sind eine **Accounting-Größe** — sie zählen, wie oft ein *reales*
   teures Target ausgewertet werden müsste; in diesem 1D-Toy wird ΔH ohnehin für jeden
   Proposal exakt berechnet (kein gemessener Speedup). **Surrogate-Drift-Guard** (Spec
   Phase-4) überwacht die Log-Diskrepanz je Stufe-2-Auswertung: hält bei γ=0, feuert bei
   γ=0.4. **Ehrlich:** 1D-Toy — Korrektheits-Maschinerie, kein Speedup-Claim; DA×Adaptation offen.
3. **Checkpoint/Restart + Lockfile (`checkpoint.py`):** vollständige Philox-State-Serialisierung
   (JSON, verlustfrei) + gemeinsamer `advance_chain`/`postprocess_multichain`-Code-Pfad mit
   `manifest.run()`. **Determinismus-Vertrag:** Interrupt (auch mehrfach) + Resume ⇒
   `result_hash` **byte-identisch** zum ununterbrochenen Lauf (Gate G44). Fail-closed:
   O_EXCL-Lockfile gegen konkurrierende Writer/Resumer (über Laden+Lauf gehalten);
   SHA-256-Integritäts-Hash weist korrumpierte/inkonsistent editierte Checkpoints laut ab
   (Gate G45) — **unkeyed**, d. h. Korruptions-Erkennung, *keine* kryptographische
   Authentifizierung gegen Akteure mit Schreibzugriff; atomare Writes (`os.replace`);
   Checkpoints auch an Ketten-Grenzen.

Zusätzlich wurden im selben Inkrement Audit-Befunde gefixt (Details im PR/Commit): zell-eigene
Seeds in den QEC-Sweeps (vorher teilten alle Zellen einer Kurve denselben RNG-Stream),
Jeffreys-regularisierte Standardfehler (Null-Ereignis-Zellen sind jetzt falsifizierbar),
`n_sigma` der Multi-Seed-Validierung auf SEM statt Einzel-Seed-std (√8 strenger), R̂ erkennt
konstante Ketten mit verschiedenen Mitteln als nicht-konvergiert, Manifest-Validierung an der
Vertrauensgrenze, `rg_map`-dtype-Härtung.

## Schnellstart

```bash
pip install -e ".[dev]"
adaptiverg-qec selftest          # 45 Gates, exit 0 gdw alle [PASS]
adaptiverg-qec demo              # A-Kernel + C-Kernel-Demo
adaptiverg-qec phase5 --json results/phase5-clt-rhat-manifest.json --manifest-out results/phase5-manifest.json
adaptiverg-qec phase5 --from-manifest results/phase5-manifest.json   # byte-identische Reproduktion (CLT+R-hat, Phase-5)
adaptiverg-qec phase6 --json results/phase6-snis-surrogate-checkpoint.json  # SNIS+DA+Checkpoint (Phase-6)
pytest                           # Test-Suite
python -m adaptiverg_qec.qec_diagnostics      # results/qec-diagnostics-rep-code.json (Inkr.1)
python -m adaptiverg_qec.qec_fit_diagnostics  # results/qec-fit-diagnostics-rep-code.json (Inkr.2)
pip install ".[surface]"                       # optionales MWPM-Extra (stim + pymatching)
python -m adaptiverg_qec.surface_decoder      # results/qec-surface-mwpm.json (Inkr.3, braucht [surface])
python -m adaptiverg_qec.mcrg_matrix    # schreibt results/phase3b-swendsen-matrix.json
python -m adaptiverg_qec.mcrg_multirg   # schreibt results/phase4-wolff-multirg.json (Wolff+Multi-RG+y_h)
```

## Inhalt
- `src/adaptiverg_qec/` — Phase-1-MVP-Code (A-Kernel, C-Kernel, Drift-Guard, Orakel, CLI/Selftests).
- `tests/` — Test-Suite (Orakel-Vergleiche, Reproduzierbarkeit, Negativ-/Edge-Gates).
- `results/` — Gate-Logs (Evidenz, AGENTS.md-Pflicht).
- `spec/AdaptiveRG_QEC_Engine_Spec_v1_0_hardened.pdf` — gehärtete Kernel-Spec v1.0: augmentierter Zustand,
  Foster-Lyapunov-Drift (Meyn & Tweedie), Petite-Set/Minorization, Diminishing-Adaptation + Containment
  (Roberts & Rosenthal 2007), SNIS-Sampling-Bounds, *Unified Stability–RG Convergence Theorem*.
- `spec/AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx` — Beweis-Block: P1 (Ergodizität A-Kern),
  P2 (RG-Fixpunkt-Konsistenz), CLT (Andrieu & Moulines).
- `spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md` — **Errata-Companion v1.3** zum obigen ProofBlock:
  korrigiert 4 mathematisch defekte Knoten (P4.1 `n_eff`-Doppel-Deflation, P3.1 Jacobian-Rate, P3.3
  Eigenwert-Bound `sep⁻¹`→κ(λ), P2.1/P2.2 Kugel-Bedingung→W^s∩B). Numerische Orakel (Defekt 3 & 4) als
  lauffähiger Reproducer `spec/reproducers/proofblock_v13_oracles.py` → `results/proofblock-v13-oracles.json`.
- `docs/ROADMAP.md` — die 5-Phasen-Implementierungs-Roadmap (Phasen 1–5 in bounded MVP-Form
  abgearbeitet; Phase-6-Inkrement schließt die SNIS/Surrogate/Checkpoint-Restlücken; verbleibende
  offene Punkte dort ehrlich gelistet).
- `requirements-lock.txt` — eingefrorene Referenz-Umgebung (Reproduzierbarkeits-Hilfe; die
  Bibliothek selbst pinnt NICHT, s. pyproject.toml).

## Provenance
Quellen + Drive-IDs: siehe `SOURCES.md`. Code-frei verifiziert 2026-06-02 (G:-weit kein `.py` zu QEC/AdaptiveRG).

## Lizenz
Apache-2.0. Attribution: Coworker Research / Coworkerz (keine Einzelnamen).
