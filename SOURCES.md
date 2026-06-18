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
- `y_h` aus der ungeraden Swendsen-Matrix, tiefste RG-Iteration vs Onsager `15/8` (G28); Reproduzierbarkeit (G29).

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
