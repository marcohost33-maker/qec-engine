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
- Surrogate-Layer, SNIS, MMD-Drift, Checkpoint/Restart, R-hat-Multichain-Diagnostik (Spec-Phasen 2–5)
  sind **noch nicht** implementiert.

## Phase-1-MVP: real implementiert vs. offen

| Komponente | Status |
|---|---|
| A-Kernel: adaptiver Single-Spin-Metropolis (`a_kernel.py`) | **real** (detailed balance, Philox-Seed) |
| Foster-Lyapunov-Drift-Guard, conditional-mean (`drift.py`) | **real** (greift + feuert, getestet) |
| Diminishing-Adaptation-Schedule `sum a_t < inf` + Containment-Clip | **real** |
| C-Kernel: 1D-Ising-Decimation-RG-Map (`rg_map.py`) | **real** (lehrbuchexakt, b=2) |
| Jacobian: Complex-Step + zentrale Differenzen + Exponenten/Hyperbolizität | **real** (CS==FD==analytisch) |
| Analytisches Transfer-Matrix-Orakel (`ising1d.py`) | **real** (machine-precision gegen Brute-Force) |
| Selftest-Gates (`cli.py --selftest`, 29 Gates, JSON-Log) | **real** (29/29 [PASS], exit 0) |
| SNIS / Defensive Mixture / ESS-per-Observable (Spec §5) | **offen** (Phase 4) |
| Swendsen-MCRG-Schätzer (skalare stochastische R̂ aus Samples, Spec §6) | **real** — `T̂=⟨S'S⟩_c/⟨S'S'⟩_c` vs `tanh(2K)`. Phase-2: exakt-i.i.d.-Sampler (≤0.54σ, Bias↓ mit 1/√N). **Phase-3a (NEU): aus dem korrelierten A-Kernel-MCMC mit autokorrelations-bewussten Fehlerbalken** (s.u.) |
| **Autokorr-bewusste Fehler (`autocorr.py`, Phase-3a)** | **real** — FFT-ρ (Wiener-Khinchin), τ_int + Wolff-g-Windowing, Binning-Cross-Check, Block-Jackknife für das Verhältnis. Validiert gegen AR(1)-Orakel + Γ==Binning-Plateau |
| **Multi-Operator-Swendsen-MATRIX (`ising2d.py` + `mcrg_matrix.py`, Phase-3b)** | **real** — 2D-Ising-Checkerboard-Metropolis (vektorisiert) + Majority-Rule-Blocking b=2; ≥2 gerade Operatoren (NN/NNN/Plaquette); `T = A·B⁻¹` via `np.linalg.solve`; Eigenwerte → `y_t = ln λ_max/ln 2`; Block-Jackknife-Fehler. Gegen Onsager `y_t=1` (**grob**: `y_t≈0.97±0.01`, kritisches Slowing-Down) |
| **Wolff-Single-Cluster-Sampler (`wolff2d.py`, Phase-4)** | **real** — `P_add = 1−e^{−2K}`, vektorisierter BFS-Cluster, rejection-free. Energie gegen exakte L=4-Enumeration (\|err\|<0.008); `τ_int(Wolff) ≪ τ_int(Metropolis)` belegt (×12–16 @ L=32) |
| **Multi-RG-Iteration + ungerader Sektor `y_h` (`mcrg_multirg.py`, Phase-4)** | **real** — iterierte Majority-Stufen (L=32→16→8→4); gerader `y_t` konvergiert (bester \|y_t−1\|≈0.006 vs 3b 0.035); 3-Spin-ungerade Operatoren → `y_h` vs Onsager `15/8` (bester \|y_h−15/8\|≈0.002). **Ehrlich:** Rest-finite-Size, keine L→∞-FSS |
| Surrogate + MMD-Drift + Checkpoint/Restart + R-hat-Multichain (Spec §8/§10) | **offen** (Phase 5) |

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

## QEC-Diagnostik-Harness (Inkrement 1 + 2)

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

Evidenz: `results/qec-fit-diagnostics-rep-code.json`. **Cross-Anchor:** das ML-Brute-Force-
Decoding stimmt im perfekt-Messung-Limit maschinengenau mit dem Inkr.1-Binomial-Orakel
überein (Diff < 3e-17 über `d∈{3,5,7}`) — die beiden Orakel stützen sich gegenseitig.

> **Ehrliche Abgrenzung:** Surface-Code, MWPM und phenomenological-noise-Spacetime-Decoder
> sind bewusst NICHT hier. Sie erfordern eine etablierte Matching-Bibliothek (PyMatching)
> hinter optional-dependency-Gate — das ist **Inkrement 3** der Roadmap. Ein hand-gerollter
> Spacetime-Matcher wurde im Prototyp gebaut und verworfen (verließ für `d=5` den Codespace
> in ~2–6 % der Fälle = Decoder-Bug), genau weil etablierte Tools existieren.

## Schnellstart

```bash
pip install -e ".[dev]"
adaptiverg-qec selftest          # 29 Gates, exit 0 gdw alle [PASS]
adaptiverg-qec demo              # A-Kernel + C-Kernel-Demo
pytest                           # Test-Suite
python -m adaptiverg_qec.qec_diagnostics      # results/qec-diagnostics-rep-code.json (Inkr.1)
python -m adaptiverg_qec.qec_fit_diagnostics  # results/qec-fit-diagnostics-rep-code.json (Inkr.2)
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
- `docs/ROADMAP.md` — die 5-Phasen-Implementierungs-Roadmap (Phase 1 angefangen, 2–5 offen).

## Provenance
Quellen + Drive-IDs: siehe `SOURCES.md`. Code-frei verifiziert 2026-06-02 (G:-weit kein `.py` zu QEC/AdaptiveRG).

## Lizenz
Apache-2.0. Attribution: Coworker Research / Coworkerz (keine Einzelnamen).
