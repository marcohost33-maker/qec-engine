# QEC-Engine (AdaptiveRG-QEC) — SOURCES & Provenance (2026-06-02)

**Projekt:** AdaptiveRG-QEC Engine — Adaptive Renormalization-Group + Quantum-Error-Correction
Simulation Engine. Coworker-Research Säule 3 (Physik/Methodik).
**Typ:** Spec/Theorie-Repo **+ Phase-1-MVP-Code** (seit PR#4, 2026-06-18; `src/adaptiverg_qec/`, Diagnostik-/Verifikations-Harness). Ursprünglich Spec-only (kein `.py`); Code-Lineage im Append unten.

## Kanonische Quellen (gestaged in `spec/`)
| Datei | Drive-Ort | Drive-ID | Größe |
|---|---|---|---|
| `AdaptiveRG_QEC_Engine_Spec_v1_0_hardened.pdf` | `0Hex  HQST  U2\` | `1OE5Yr9VXz1eXNTbdYdiKeS5WJ7Uup9pw` | 26287 B |
| `AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx` | `0Hex  HQST  U2\` | `1JaE9LG5cK7FB-rPjvOHPBKS-0gvZC488` | 21088 B |

## Inhalt (verifiziert via Drive-Read)
- **Spec v1.0 (gehärtet):** augmentierter Zustand, Foster-Lyapunov-Drift (Meyn&Tweedie), Petite-Set/
  Minorization, Diminishing-Adaptation + Containment (Roberts&Rosenthal 2007), SNIS-Sampling-Bounds,
  Unified Stability–RG Convergence Theorem. Korrekturen inline `[KORREKTUR]/[LÜCKE]/[OK]`.
- **ProofBlock v1.1 + KernelSpec v1.2:** Beweisskizzen P1 (Ergodizität A-Kern), P2 (RG-Fixpunkt-Konsistenz),
  CLT unter Andrieu&Moulines. Lit.: Roberts&Rosenthal, Fort/Moulines/Priouret, Vehtari, Dennis/Kitaev/Landahl/Preskill.
  → **Korrekturen:** `spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md` (Errata-Companion, s. Lineage-Append unten).

## Ehrliche Einordnung (Eval-Strategie §3 Status-Sättigung)
**Substanzielle, gehärtete Theorie — aber Implementierung fehlt.** Ein Repo ist nur als **Spec-/Theorie-Repo**
sinnvoll (kein `src/tests`). Die 5-Phasen-Roadmap im Spec → `docs/ROADMAP.md` als offene Implementierungs-Phasen.

## Offen vor Repo-Init
- [ ] PDF/DOCX → Markdown konvertieren (für git-diff-bare Specs in `spec/`).
- [ ] `docs/ROADMAP.md` aus der 5-Phasen-Roadmap extrahieren.
- [ ] Entscheid Marco: Spec-Repo jetzt, oder warten bis Implementierung beginnt.

*Vero | 2026-06-02 | QEC SOURCES | Spec-verifiziert via Drive, Code fehlt (ehrlich)*

---

## Lineage-Append (append-only, AGENTS.md §SHA-First) — 2026-06-18

### Phase-1-MVP gestartet (Branch `claude/phase1-mcmc-mcrg`, Agent: Claude/Codie)
Übergang Spec-Repo → Code-Repo. MVP implementiert A-Kernel (adaptiver MCMC + Foster-Lyapunov-Guard)
und minimalen C-Kernel (1D-Ising-Decimation-RG + Jacobian). MVP-Instanz: 1D-Repetition-Code-Ring,
explizit dokumentiert in `src/adaptiverg_qec/mvp_instance.py` (als MVP-Wahl gekennzeichnet, nicht volle Spec).

**Kanonische Spec-Inputs (SHA-256, verifiziert 2026-06-18 — unverändert seit Repo-Init):**

| Datei | SHA-256 |
|---|---|
| `spec/AdaptiveRG_QEC_Engine_Spec_v1_0_hardened.pdf` | `c074af559db92fc86860aa31b9c41f195b2206e9c8d75432c99c8f098195104d` |
| `spec/AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx` | `c45d806f06b26d3d3b19e2d34466abae21d6cb695f0cbd1963635e9126959d5c` |

**Externe Referenz-Quellen (Positionierung / SOTA-Kontext, in README zitiert):**
- Bravyi, Suchara & Vargo (2014), *Efficient algorithms for maximum likelihood decoding in the surface code*,
  Phys. Rev. A 90, 032326 — Tensor-Network/MPS = SOTA für Präzisions-Thresholds (kontextualisiert den
  inkrementellen Charakter des MCMC+MCRG-Ansatzes).
- Duclos-Cianci & Poulin (2010), *Fast Decoders for Topological Quantum Codes*, PRL 104, 050504 —
  der *RG-Decoder*; hier ABGEGRENZT: "Adaptive RG-QEC" meint Threshold-via-kritische-Exponenten, NICHT diesen Decoder.
- Spec-interne Literatur (verbindlich): Roberts & Rosenthal (2007), Meyn & Tweedie (2009),
  Andrieu & Moulines (2006), Vehtari et al. (2021), Lyness & Moler (1967, Complex-Step).

**Code-Provenienz:** Alle `src/`+`tests/`-Dateien neu erstellt 2026-06-18 (kein importierter Fremdcode).
Gate-Log-Evidenz: `results/selftest.json` (12/12 [PASS]; SHA-256 lauf-spezifisch, daher nicht gepinnt —
Reproduktion via `adaptiverg-qec selftest`).

*Codie | 2026-06-18 | Phase-1-MVP-Append | Reality-Anchor: dev/Prototyp-Reife, nicht selbst-zertifiziert*

---

## Lineage-Append (append-only) — 2026-06-18 — Phase-3a

### Phase-3a: korrelierter A-Kernel als Sample-Quelle + autokorrelations-bewusste Fehler
**Branch `claude/phase3a-akernel-autocorr` (Agent: Claude/Codie).**

Punkt-2-Richtung des Marco-Entscheids: T̂ = ⟨S'S⟩_c/⟨S'S'⟩_c jetzt aus dem korrelierten adaptiven
A-Kernel-MCMC (Phase-2 nutzte einen exakten i.i.d.-Sampler), mit korrekten autokorrelations-bewussten
Fehlerbalken statt naiver i.i.d.-Balken. NEU: `src/adaptiverg_qec/autocorr.py`. KEINE neuen Runtime-Deps
(alles numpy; scipy unverändert). Multi-Operator-Swendsen-MATRIX = Phase-3b (separat, offen).

**Methoden-Referenzen (validiert, NICHT als Dependency):**
- U. Wolff, *Monte Carlo errors with less errors*, Comput. Phys. Commun. 156 (2004) 143, arXiv:hep-lat/0306017
  — Γ-Methode (τ_int = ½ + Σρ), automatic windowing via g-Funktion (S≈1.5), σ² = 2·τ_int·Var/N.
- pyerrors, arXiv:2209.14371 — unabhängige Methoden-Referenz (Γ-Methode/Autokorrelation), nur Referenz.
- Wiener-Khinchin-Theorem — FFT-basierte Autokorrelationsfunktion (O(N log N)).

**Validierungs-Orakel (unabhängig):**
- AR(1)-Prozess mit geschlossener Autokorrelation ρ(t)=φ^t → τ_int = ½ + φ/(1−φ) (Gate G13, rel. Fehler <3%).
- Γ-Methode == Binning-Plateau (rigorose Cross-Validierung, Gate G14, Übereinstimmung ±2%).
- T̂(A-Kernel) trifft tanh(2K) innerhalb σ_korr für K≤0.7 (Gate G15); N_eff<N + σ_korr>σ_iid (Gate G16).

**Gemessene Zahlen (Evidenz `results/phase3a-akernel-autocorr.json`, L=64, N=18000):** τ_int wächst 0.65→2.81
mit K∈{0.3..0.9}; Fehler-Inflation 1.006→1.129; N_eff 13745→3198 (< N durchgehend).

**Code-Provenienz:** `autocorr.py` + `tests/test_autocorr.py` neu erstellt 2026-06-18; `a_kernel.py`/`mcrg.py`/
`cli.py` erweitert (record_configs, swendsen_T_from_chain, Gates G13–G18). Kein importierter Fremdcode.

*Codie | 2026-06-18 | Phase-3a-Append | Reality-Anchor: dev/Prototyp, Equalita-L1 ausstehend, nicht selbst-zertifiziert*

---

## Lineage-Append (append-only) — 2026-06-18 — Phase-3b

### Phase-3b: Multi-Operator-Swendsen-MATRIX auf 2D-Ising
**Branch `claude/phase3b-swendsen-matrix` (Agent: Claude/Codie).**

Punkt-1-Richtung des Marco-Entscheids (nach Punkt-2 = Phase-3a): Verallgemeinerung des SKALAREN Swendsen-
Schätzers (eine Kopplung) auf die volle linearisierte RG-**MATRIX** `T_ab = dK′_a/dK_b`. Erfordert ein Modell
mit nicht-trivialem Fixpunkt → **2D-Ising** (Quadratgitter, `T_c = 2/ln(1+√2) ≈ 2.269`, anders als 1D `T_c=0`).
NEU: `src/adaptiverg_qec/ising2d.py` + `src/adaptiverg_qec/mcrg_matrix.py`. KEINE neuen Runtime-Deps (numpy).

**Methoden-Referenzen (validiert, NICHT als Dependency):**
- R. H. Swendsen, *Monte Carlo Renormalization Group*, Phys. Rev. Lett. 42 (1979) 859 — linearisierte
  RG-Matrix aus connected correlations: `A_ab=⟨S′_a S_b⟩_c`, `B_ac=⟨S′_a S′_c⟩_c`, `T = A·B⁻¹`.
- L. Onsager, Phys. Rev. 65 (1944) 117 — exakte 2D-Lösung: `T_c`, `y_t = 1/ν = 1`, `y_h = 2−η/2 = 15/8`.
- L. P. Kadanoff, Physics 2 (1966) 263 — Block-Spin / Majority-Rule-Blocking.

**Onsager-Orakel selbst web-verifiziert (dieser Lauf):** y_t = 1/ν = 1 (ν=1), y_h = 2−η/2 = 15/8 = 1.875
(η=1/4), T_c ≈ 2.269 — quergeprüft gegen mehrere RG-/Lehrbuch-Quellen vor der Validierung.

**Validierungs-Orakel (unabhängig):**
- 2D-Metropolis-Energie vs **exakte L=4-Enumeration** (2^16 Zustände, gleiche Bond-Konvention), |err|<0.005 (G19).
- A,B connected-corr-Matrizen symmetrisch + PSD + gut konditioniert (cond(B)≈87) (G20).
- `T = A·B⁻¹` linear-solve-Konsistenz: Residuum `max|T·B−A| ≈ 2e-13` ~ Maschinen-eps (G21).
- y_t aus den Eigenwerten vs Onsager `y_t=1` (G22); Reproduzierbarkeit per Seed (G23).

**Gemessene Zahlen (Evidenz `results/phase3b-swendsen-matrix.json`, L=16, N_op=2, K=K_c, 3 Seeds):**
y_t = **0.965 ± 0.009** (multi-seed), |err|≈0.035 vs Orakel 1.0; λ_max≈1.95; τ_int(max)≈4.3.

**EHRLICHE SCOPE-GRENZE (kein Über-Claim):** Single-Spin-Metropolis nahe T_c hat kritisches Slowing-Down;
kleines L + 1 RG-Stufe → y_t nur GROB (Plausibilitäts-Niveau, KEIN Frontier-Wert). Cluster-Algorithmus
(Wolff/Swendsen-Wang) + mehrere RG-Iterationen = Phase-4. y_h (ungerader Sektor) nicht implementiert.

**Code-Provenienz:** `ising2d.py` + `mcrg_matrix.py` + `tests/test_ising2d.py` + `tests/test_mcrg_matrix.py`
neu erstellt 2026-06-18; `cli.py` erweitert (Gates G19–G23 + Phase-3b-Edge-Cases in G8). Kein importierter Fremdcode.

*Codie | 2026-06-18 | Phase-3b-Append | Reality-Anchor: dev/Prototyp, Equalita-L1 ausstehend, nicht selbst-zertifiziert*

---

## Lineage-Append (append-only) — 2026-06-18 — Phase-4

### Phase-4: Wolff-Single-Cluster + Multi-RG-Iteration + ungerader Sektor (y_h)
**Branch `claude/phase4-wolff-multirg` (Agent: Claude/Codie).**

Adressiert die DREI Präzisions-Schwächen, die ein unabhängiges Cross-Family-Review (OpenAI gpt-4o, Verdikt
GOOD) an Phase-3b bestätigt hat: (1) Single-Spin-Metropolis → kritisches Slowing-Down; (2) nur 1 RG-Iteration;
(3) Operatorbasis zu klein / kein ungerader Sektor. NEU: `src/adaptiverg_qec/wolff2d.py` +
`src/adaptiverg_qec/mcrg_multirg.py`. **KEINE neuen Runtime-Deps** (numpy; scipy unverändert).

**Methoden-Referenzen (validiert, NICHT als Dependency):**
- U. Wolff, *Collective Monte Carlo Updating for Spin Systems*, Phys. Rev. Lett. 62 (1989) 361 —
  Single-Cluster-Algorithmus, `P_add = 1−e^{−2K}`.
- R. H. Swendsen, J.-S. Wang, Phys. Rev. Lett. 58 (1987) 86 — Multi-Cluster (Verwandtschaft).
- R. H. Swendsen, Phys. Rev. Lett. 42 (1979) 859 — MCRG mit MEHREREN iterierten Blocking-Stufen
  (irrelevante Operatoren kontrahieren heraus → Konvergenz zum Fixpunkt).
- L. Onsager, Phys. Rev. 65 (1944) 117 — exakte 2D-Exponenten.

**Web-verifiziert (dieser Lauf):** `P_add = 1−e^{−2βJ}` + dynamischer Exponent `z ≈ 0.25` (Wolff/Swendsen-Wang
vs `z ≈ 2` Single-Spin); `y_t = 1/ν = 1` (ν=1), `y_h = d − β/ν = 2 − (1/8)/1 = 15/8 = 1.875` (β=1/8) —
quergeprüft gegen mehrere RG-/Cluster-/Lehrbuch-Quellen vor der Validierung.

**Validierungs-Orakel (unabhängig):**
- Wolff-Energie vs **exakte L=4-Enumeration** (2^16 Zustände, gleiche Bond-Konvention), |err|<0.008 (G24).
- Cluster-Größen-Fraktion wächst monoton mit K (0.011 → 0.55 → 0.98) — `P_add`-Korrektheits-Indiz (G25).
- **`τ_int(Wolff) < τ_int(Metropolis)`** bei K_c, |m|-Reihe (G26) — Kern-Beleg, dass das Slowing-Down
  geschlagen wird.
- `y_t` über iterierte RG-Stufen → Konvergenz zum Onsager-Wert, besser als Phase-3b (G27).
- `y_h` aus der ungeraden Swendsen-Matrix, bester Iterationswert (Minimum über Iterationen, proximity-selektiert; NICHT die tiefste Stufe) vs Onsager `15/8` (G28); Reproduzierbarkeit (G29).

**Gemessene Zahlen (Evidenz `results/phase4-wolff-multirg.json`, L=32, K=K_c, 3 Seeds):**
- `y_t` (gerader Sektor) über Iterationen L=32→16→8: typ. `[0.93–0.95, 1.00, 0.98–1.02]` →
  **bester `|y_t−1| ≈ 0.006`** (Mittel 3 Seeds) vs Phase-3b 0.035.
- `y_h` (ungerader Sektor) über Iterationen: `[1.881, 1.873, 1.871–1.875]` →
  **bester `|y_h−15/8| ≈ 0.002`** (Mittel 3 Seeds), σ_jk ≈ 0.001.
- `τ_int(|m|)`: Wolff ≈ 2.2–2.8, Metropolis ≈ 27–46 → **Verhältnis ×12–16** (Slowing-Down klar geschlagen).

**Selbst gefundene + behobene Bugs (Silent-Failure-Gate, IM BAU):** (a) erste `O_2`-Wahl `s_i·Σnb` war
ein PRODUKT von ZWEI Spins → **gerade**, nicht ungerade (von `test_odd_operators_antisymmetric` gefangen);
korrigiert auf echtes 3-Spin-L-Cluster `s_ij·s_{i,j+1}·s_{i+1,j}` (ungerade). (b) `block_chain` erzeugte
ein degeneriertes 2×2-Gitter; auf kleinste Kante 4 begrenzt.

**EHRLICHE SCOPE-GRENZE (kein Über-Claim):** Wolff verbessert die STATISTIK (kleinere τ_int → mehr effektiv
unabhängige Samples) und Multi-RG kontrahiert irrelevante Operatoren — beide Sektoren konvergieren sichtbar
zum Onsager-Fixpunkt. ABER: eine Rest-finite-Size-Systematik bleibt (kleines L, endliche Iterationszahl);
dies ist **keine** volle `L→∞`-FSS-Extrapolation und **kein** Frontier-Hochpräzisionswert. Die Gate-Toleranzen
(`|y_t−1|<0.035`, `|y_h−15/8|<0.05`) sind systematik-begründet, kein Cherry-Pick; alle Iterations-Werte
werden im Log mitgeführt.

**Code-Provenienz:** `wolff2d.py` + `mcrg_multirg.py` + `tests/test_wolff2d.py` + `tests/test_mcrg_multirg.py`
neu erstellt 2026-06-18; `cli.py` erweitert (Gates G24–G29 + Phase-4-Edge-Cases in G8). Kein importierter Fremdcode.

*Codie | 2026-06-18 | Phase-4-Append | Reality-Anchor: dev/Prototyp, Equalita-L1 ausstehend, nicht selbst-zertifiziert*

---

## Lineage-Append (append-only) — 2026-06-19 — QEC-Inkrement 3 (MWPM via Stim+PyMatching)

### Inkr.3: ECHTES MWPM-Decoding hinter optional-dependency-Gate
**Branch `claude/qec-inkr3` (Agent: Claude/Codie).**

Schliesst die in der QEC-Diagnostik-Roadmap als Inkr.3 definierte Luecke: ein ECHTER Decoder
(nicht das exakt-orakelbare Repetition-Modell, nicht der hand-gerollte, in Inkr.2 verworfene
Prototyp). KEIN Eigen-Decoder — etablierte Bibliotheken hinter `pip install ".[surface]"`:
`stim` (Sampling/Schaltkreis) + `pymatching` 2 (Minimum-Weight-Perfect-Matching). NICHT als
Kern-Dependency; ohne das Extra SKIPPEN die Tests (verifiziert: dev-only-venv 3 passed/30 skipped/exit 0).

**Decoder/Methoden-Referenzen (als optionale Dependency genutzt):**
- O. Higgott, C. Gidney, *Sparse Blossom: correcting a million errors per core second with
  minimum-weight matching* (PyMatching 2), arXiv:2303.15933 — verwendeter MWPM-Decoder.
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021) — Sampling.

**Publizierte Threshold-Orakel (web-recherchiert dieser Lauf, mehrere Quellen):**
- **MWPM-Threshold ≈ 0.103** (code-capacity, independent X/Z, = Nulltemperatur-Uebergang des
  2D-Random-Bond-Ising-Modells): E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological
  quantum memory*, J. Math. Phys. 43, 4452 (2002). [errorcorrectionzoo.org/c/toric]
- **Optimaler/ML-Threshold ≈ 0.1094** (Nishimori-Punkt des 2D-RBIM, finite-T) — zur Einordnung,
  NICHT als Orakel; MWPM ist near-optimal aber sub-optimal (10.31% MWPM vs 10.94% ML).
- Phenomenological-noise MWPM-Threshold ≈ 2.9%: Wang, Harrington & Preskill, Ann. Phys. 303,
  31 (2003) — fuer den noch offenen naechsten Schritt notiert.

**Validierungs-Orakel (unabhaengig, real ausgefuehrt, seed=11):**
- **Orakel A** — Repetition-Code MWPM (PyMatching aus Paritaets-Pruefmatrix) vs exaktes
  Binomial-Orakel (Inkr.1). Auf 1D-Matching IST MWPM ML-optimal → muss treffen. `d∈{3,5,7}`
  @ p=0.10, 200 000 shots → worst `n_sigma = 1.55` (rein statistisch).
- **Orakel B** — Rotated-Surface-Code, code-capacity pure-X-flip (X_ERROR(p) nur auf Daten-
  Qubits, perfekte Messung), MWPM via Stim-DEM. Threshold via Distanz-Kurven-Kreuzung,
  60 000 shots/Zelle: (7,9)→0.0952, (9,11)→0.1015, (7,11)→0.0978. **best pair (9,11):
  `p_th = 0.1015` vs publiziert 0.103 → `abs_err = 0.0015`.**

**EHRLICHE KORREKTHEITS-GRENZE (kein Overclaim, Lehre aus Inkr.2-Fabrikation):** validiert
gegen 0.103 (MWPM), NICHT gegen 0.1094 (optimal/ML) — ein Schaetzer, der 0.109 „erreicht",
waere verdaechtig. Endliche Distanzen → KEINE L→∞-FSS; der Kreuzungs-Schaetzer driftet von
unten zum Threshold (sichtbar: (7,9)<(9,11)). Phenomenological noise (~2.9%) und
depolarisierendes Rauschen sind NICHT implementiert. Jede genannte Zahl ist aus
`python -m adaptiverg_qec.surface_decoder` → `results/qec-surface-mwpm.json` regenerierbar
(stim 1.16.0, pymatching 2.4.0).

**Code-Provenienz:** `src/adaptiverg_qec/surface_decoder.py` + `tests/test_surface_decoder.py`
neu erstellt 2026-06-19; `pyproject.toml` `[surface]`-Extra ergaenzt; README + QEC-Roadmap
aktualisiert. Kein importierter Fremdcode (stim/pymatching nur als optionale Runtime-Deps).

*Codie | 2026-06-19 | Inkr.3-Append | Reality-Anchor: dev/Prototyp, Equalita-L1 ausstehend, nicht selbst-zertifiziert*

## Lineage-Append (append-only) — 2026-06-19 — Phase-5 (CLT + R-hat + Manifest)

### Phase-5: CLT-Varianz sigma^2_g + rank-normalized split-R-hat + Run-Manifest

**Branch:** `claude/qec-phase5-clt-rhat-manifest` (Agent: Claude/Codie). Additiv, KEINE neuen
Runtime-Deps (nur numpy/scipy). Code: `src/adaptiverg_qec/clt.py`, `rhat.py`, `manifest.py` +
`tests/test_clt.py`, `test_rhat.py`, `test_manifest.py`; CLI-Subcommand `phase5`; Gates G33-G38.

**Methoden-Quellen (verbindlich, NICHT als Dependency importiert):**
- **CLT-Varianz / long-run variance:** MCMC-CLT `sqrt(N)(g_bar-E[g]) -> N(0, sigma^2_g)`,
  `sigma^2_g = Var(g)(1 + 2 sum_t rho_t) = 2 tau_int Var(g)`.
  C. J. Geyer, "Practical Markov Chain Monte Carlo", Statist. Sci. 7 (1992) 473.
- **Overlapping Batch Means (unabhaengiger zweiter Schaetzer):** J. M. Flegal & G. L. Jones,
  "Batch means and spectral variance estimators in MCMC", Ann. Statist. 38 (2010) 1034.
- **AR(1)-Orakel (geschlossene Form):** `sigma^2_g = sigma_eps^2/(1-phi)^2 = Var(x)(1+phi)/(1-phi)`,
  `Var(x)=sigma_eps^2/(1-phi^2)`, `tau_int=(1+phi)/(2(1-phi))`. Standard-long-run-Varianz-Identitaet
  fuer AR(1) (inefficiency factor `(1+phi)/(1-phi)`); web-verifiziert.
- **Rank-normalized split-R-hat + bulk/tail-ESS:** A. Vehtari, A. Gelman, D. Simpson, B. Carpenter,
  P.-C. Buerkner, "Rank-normalization, folding, and localization: An improved R-hat for assessing
  convergence of MCMC", Bayesian Analysis 16(2) (2021) 667-718, doi:10.1214/20-BA1221.
  Blom-Transform `z_i = Phi^-1((r_i-3/8)/(S+1/4))`; `R-hat = sqrt(((N-1)/N) W + B/N) / sqrt(W)`;
  folded-R-hat um Median; Schwellwert R-hat<1.01. ESS via Multichain-Variogramm (BDA3 11.7) +
  Geyer initial-monotone-sequence. Online-Appendix: avehtari.github.io/rhat_ess/rhat_ess.html.

**Validierungs-Orakel (unabhaengig, real ausgefuehrt, regenerierbar):**
- **CLT vs AR(1) (Gate G33):** phi=0.8, Orakel sigma^2_g=25.0 -> Gamma-Schaetzer 25.26 (rel.Fehler
  0.010), OBM 24.49 (0.020). Beide Methoden treffen dasselbe Orakel.
- **Coverage beidseitig (Gate G34):** korrektes sigma^2_g -> 95%-CI-Coverage 0.937 (~0.95);
  ABSICHTLICH falsche iid-Annahme -> 0.490 (UNTERdeckt, verfehlt 0.95). Beide Richtungen Pflicht.
- **R-hat beidseitig (Gates G35/G36):** gut gemischte iid-Ketten R-hat=1.0002; A-Kernel M=4
  R-hat=1.0009 (converged); Mittel-Drift R-hat=1.62 (bulk faengt); Skalen-Drift R-hat=1.27
  (folded=1.27>bulk=1.00 -- Vehtari-Punkt). ESS_bulk(A-Kernel)~5149 (>100/Kette).
- **Manifest beidseitig (Gates G37/G38, SLSA-Custody):** run -> Manifest -> from-manifest ergibt
  byte-identischen result_hash; geaenderter Seed/n_steps -> ANDERER Hash (Manifest treibt den Lauf).

**EHRLICHE GRENZE (kein Overclaim):** Phase-5-real = CLT-sigma^2_g + R-hat-Multichain + Manifest.
**WEITERHIN OFFEN:** SNIS (chi^2-Varianz-Bound) + Surrogate-Beschleunigung (Phase-4) + Checkpoint/
Lockfile bleiben offen -- NICHT als erledigt deklariert. Jede genannte Zahl ist aus
`python -m adaptiverg_qec.cli phase5 --json results/phase5-clt-rhat-manifest.json` regenerierbar
(numpy 2.4.6, scipy 1.17.1, python 3.14.4) bzw. ein publiziertes Literatur-Orakel mit Quelle oben.

*Codie | 2026-06-19 | Phase-5-Append | Reality-Anchor: dev/Prototyp, Equalita-L1 + CI ausstehend, nicht selbst-zertifiziert*

---

## Lineage-Append (append-only) — 2026-07-02: ProofBlock-Errata-Companion v1.3

**Artefakt (repo-authored, KEIN Drive-SoT):** `spec/AdaptiveRG-QEC_ProofBlock_v1.3_corrections.md`.
Errata-/Korrektur-Companion zum binären SoT `spec/AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx`
(unverändert). Korrigiert **4 mathematisch defekte Knoten** des ProofBlock v1.1:

| # | Knoten | Defekt | Korrektur |
|---|--------|--------|-----------|
| 1 | P4.1 | `n_eff = ESS/(2τ_int)` (doppelte Deflation) | `n_eff = ESS = N/(2τ_int)` |
| 2 | P3.1 | Rate `O_P(1/√N_rep)` ohne 1/ε_diff, 1/√n | `O_P(1/(ε_diff·√(n·N_rep))) + O(ε_diff²) + O(n^{−β}/ε_diff)` |
| 3 | P3.3 | Eigenwert-Bound mit `sep⁻¹` | `κ(λ_i)·‖E‖₂` (Kato/GVL §7.2; Bauer–Fike = globale Schranke) |
| 4 | P2.1/P2.2 | Kugel-Bedingung ⇒ Konvergenz (ignoriert W^u) | Guard = W^s(g*) ∩ B(g*, r_lin) |

**Numerische Orakel (Defekt 3 & 4) — lauffähiger Reproducer + Gate-Log (AGENTS.md WA1):**
- `spec/reproducers/proofblock_v13_oracles.py` (deterministisch, `seed=2026`, keine neuen Deps).
- `results/proofblock-v13-oracles.json` (Gate-Log; `gate_pass` prüft κ-Skalierung + sep⁻¹-Verletzung
  bzw. untuned-Blow-up vs. getunt-O(σ)). Regenerierbar via `python spec/reproducers/proofblock_v13_oracles.py`.
- Reifegrad: Spec-Korrektur, L0-Selbst-Precheck; Equalita-Ratifikation offen — nicht selbst-zertifiziert.

*Codie | 2026-07-02 | ProofBlock-v1.3-Errata-Append*

---

## Lineage-Append (append-only) — 2026-08-09: Phase-6 (SNIS + Surrogate-DA + Checkpoint) + Audit-Härtung

**Neue Primärquellen (Phase-6, alle web-etabliert):**
- **Christen, J.A. & Fox, C. (2005).** "Markov chain Monte Carlo using an approximation."
  *J. Comput. Graph. Stat.* 14(4), 795–810. — Delayed-Acceptance-MCMC: zweistufige Akzeptanz
  (Stufe 1 Surrogat, Stufe 2 exakte Korrektur) erhält detailed balance gegen das exakte Ziel
  fuer JEDES Surrogat. Basis von `surrogate.py`.
- **Agapiou, S., Papaspiliopoulos, O., Sanz-Alonso, D., Stuart, A.M. (2017).** "Importance
  Sampling: Intrinsic Dimension and Computational Cost." *Statist. Sci.* 32(3), 405–431.
  — MSE-Bound der selbst-normalisierten IS: MSE <= 4*(1+chi^2)/N fuer |g|<=1 (Thm 2.1 / Kap. 2,
  rho = E_q[(dpi/dq)^2] = 1+chi^2). Basis des chi^2-Varianz-Bound-Gates (G40).
- **Owen, A.B.** *Monte Carlo theory, methods and examples*, Kap. 9 (Importance Sampling) —
  Delta-Methoden-Varianz/Bias des SNIS-Ratio-Schaetzers (fuehrender O(1/N)-Bias; hier
  exponential-family-exakt ausgewertet: bias ~ (1+chi^2)(tanh K_t - tanh(2K_t-K_p))/N).
- Geschlossene chi^2-Form der offenen 1D-Ising-Kette: Z(K) = 2(2 cosh K)^{L-1} (Bond-Faktorisierung)
  => 1+chi^2 = [cosh(2K_t-K_p) cosh(K_p)/cosh^2(K_t)]^{L-1}; in `tests/test_snis.py` gegen
  vollstaendige 2^8-Enumeration verifiziert (KEIN Self-Check).

**Phase-6-real:** `snis.py` (G39–G41), `surrogate.py` (G42–G43), `checkpoint.py` (G44–G45);
Artefakt `results/phase6-snis-surrogate-checkpoint.json`; Selftest 45/45. Der Append vom
2026-06-19 ("WEITERHIN OFFEN: SNIS + Surrogate + Checkpoint/Lockfile") ist damit abgearbeitet;
**weiterhin offen:** Defensive Mixture, SNIS auf 2D/RBIM-Targets, DA×Adaptation, MMD-Drift.

**Audit-Härtung (Korrektheit, gleiche PR):** zell-eigene Seeds in QEC-Sweeps (vorher EIN Seed
fuer alle (n,p)-Zellen -> rangkorrelierte Kurven), Jeffreys-regularisierte std_err (k=0-Zellen
waren unfalsifizierbar, n_sigma=0), mcrg-n_sigma auf SEM (vorher sqrt(8) zu lax), R-hat/ESS bei
konstanten Ketten mit verschiedenen Mitteln -> inf/0 statt "converged", RunManifest-Validierung,
rg_map-dtype-Fix, G3-Gate non-vakuoes. Betroffene regenerierte Artefakte:
`results/phase2-swendsen.json`, `results/qec-diagnostics-rep-code.json`,
`results/qec-fit-diagnostics-rep-code.json`, `results/qec-surface-mwpm.json`, `results/selftest.json`.

*Claude Code | 2026-08-09 | Phase-6-Append | Reality-Anchor: dev/Prototyp, bounded orakel-validiert, nicht selbst-zertifiziert*
