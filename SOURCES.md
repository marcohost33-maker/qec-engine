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
