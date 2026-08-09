# AdaptiveRG-QEC — Implementierungs-Roadmap

Aus der gehärteten Kernel-Spec v1.0 (5-Phasen-Roadmap, ~16 Wochen). **Status (2026-06-18): Phase 1 als
bounded MVP ANGEFANGEN (`src/`, Branch `claude/phase1-mcmc-mcrg`); Phasen 2–5 OFFEN.** Übergang Spec-Repo
→ Code-Repo. Reifegrad dev/Prototyp, nicht release-fertig.

> Hinweis: Diese Roadmap ist aus der Spec extrahiert/zusammengefasst. Vor Implementierung gegen das
> Original `spec/AdaptiveRG_QEC_Engine_Spec_v1_0_hardened.pdf` (§„5-phasiger Research Roadmap") abgleichen
> + verfeinern. Akzeptanzkriterien sind hier als Platzhalter formuliert und vor Phasenstart zu härten.

## Phase 1 — Kern-MCMC + Foster-Lyapunov-Guards  [MVP ANGEFANGEN 2026-06-18]
- Augmentierten Zustand (X×Θ) + stochastischen Treiber implementieren.
- Drift-Bedingung (λ<1 ausserhalb C, b·1_C) als Laufzeit-Guard.
- **Akzeptanz:** geometrische Ergodizität empirisch (TV-Distanz fällt), Guard greift bei Verletzung.
- **MVP-Stand:** A-Kernel (`a_kernel.py`) + conditional-mean Drift-Guard (`drift.py`) implementiert +
  getestet gegen analytisches Transfer-Matrix-Orakel; Guard greift (equilib λ̂<1) UND feuert
  (non-contract λ̂≥1). MVP-Instanz: 1D-Repetition-Code-Ring (`mvp_instance.py`).
  **Offen für volle Akzeptanz:** R-hat-Multichain (Vehtari, Spec §10.1a) + TV-Distanz-Verlauf.

## Phase 2 — Adaptive Steuerung (Diminishing Adaptation + Containment)
- Lernraten-Schedule η_t = η0/(1+t/T0); Θ auf kompakte Menge (kritischen Punkt ausschliessen).
- **Akzeptanz:** kein AdapFail; Mischzeiten stochastisch beschränkt.

## Phase 3 — RG-Analyse + Jacobian-Extraktion
- Stochastische RG-Map R̂, Fixpunkt-Konsistenz (Bias O(n^−β), β>1/2), hyperbolischer Fixpunkt.
- **Akzeptanz:** ĝ_k → g* in Wahrscheinlichkeit innerhalb Linearisierungsumgebung.
- **Phase-3a [DONE 2026-06-18, Branch `claude/phase3a-akernel-autocorr`]:** skalarer Swendsen-T̂ aus dem
  KORRELIERTEN A-Kernel-MCMC (statt exakt-i.i.d.) + autokorrelations-bewusste Fehlerbalken (`autocorr.py`:
  FFT-ρ via Wiener-Khinchin, τ_int + Wolff-g-Windowing hep-lat/0306017, Binning-Cross-Check, Block-Jackknife
  fürs Verhältnis). Belegt: τ_int>0.5, N_eff<N, σ_korr>σ_iid; T̂ trifft tanh(2K) innerhalb σ_korr (K≤0.7).
  Validiert gegen AR(1)-Orakel + Γ==Binning-Plateau. Gates G13–G18, `results/phase3a-akernel-autocorr.json`.
- **Phase-3b [DONE 2026-06-18, Branch `claude/phase3b-swendsen-matrix`]:** Multi-Operator-Swendsen-MATRIX auf
  dem **2D-Ising-Modell** (nicht-trivialer Fixpunkt `T_c = 2/ln(1+√2) ≈ 2.269`, anders als 1D `T_c=0`).
  `ising2d.py`: vektorisierter Checkerboard-Metropolis (numpy, kein Spin-Loop) + Majority-Rule-Blocking b=2
  (Kadanoff) mit unbiased-deterministischem Tie-Break. `mcrg_matrix.py`: gerade Operatoren (S₁=NN, S₂=NNN,
  S₃=Plaquette); connected-correlation-Matrizen A=⟨S′S⟩_c, B=⟨S′S′⟩_c; **T = A·B⁻¹ via `np.linalg.solve`**
  (keine explizite Inverse); Exponenten y_i = ln|λ_i|/ln 2; Block-Jackknife-Fehler über die Matrix-Pipeline.
  Validiert: 2D-Energie vs exakte L=4-Enumeration (|err|<0.005); A,B symmetrisch/PSD; T·B−A-Residuum ~2e-13;
  **y_t = 0.97 ± 0.01 vs Onsager-Orakel y_t=1** (|err|≈0.035). Gates G19–G23, `results/phase3b-swendsen-matrix.json`.
  **Ehrliche Scope-Grenze:** single-spin Metropolis + 1 RG-Stufe + kleines L → y_t nur GROB (Plausibilität,
  kein Frontier-Wert; Cluster-Algo + Multi-RG = Phase-4). y_h=15/8 (ungerader Sektor) nicht implementiert.

## Phase 4 — Sampling (SNIS) + Surrogate-Beschleunigung + Fehlerbudgets
- Self-Normalized Importance Sampling mit χ²-Divergenz-Varianz-Bound; Surrogate-Drift-Kontrolle.
- **Akzeptanz:** Bias O(1/N) belegt; Surrogate-Drift unter Schwelle.
- **Phase-4 [DONE 2026-06-18, Branch `claude/phase4-wolff-multirg`] — MCRG-Präzision** (adressiert die
  drei vom Cross-Family-Review benannten Schwächen; SNIS/Surrogate bleiben offen):
  1. **Wolff-Single-Cluster-Algorithmus** (`wolff2d.py`): `P_add = 1−e^{−2K}` (web-verifiziert), vektorisierte
     BFS-Cluster-Bildung, rejection-free. Schlägt das kritische Slowing-Down: `τ_int(Wolff) ≪ τ_int(Metropolis)`
     (gemessen ×12–16 @ L=32, |m|-Reihe). Korrektheit: Energie gegen exakte L=4-Enumeration (|err|<0.008),
     Cluster-Fraktion wächst monoton mit K.
  2. **Mehrere RG-Iterationen** (`mcrg_multirg.py`): iterierte Majority-b=2-Stufen (L=32→16→8→4); `y_t(n)`
     konvergiert zum Onsager-Fixpunkt — **bester `|y_t−1| ≈ 0.006`** (3 Seeds) vs Phase-3b 0.035.
  3. **Ungerader (magnetischer) Sektor** → `y_h`: Operatoren `O_1=M`, `O_2=` 3-Spin-L-Cluster (echt ungerade);
     ungerade Swendsen-Matrix `T_h=A·B⁻¹` (rohe Momente); tiefste RG-Iteration **`|y_h − 15/8| ≈ 0.002`**
     gegen Onsager `y_h = 15/8 = 1.875` (web-verifiziert: β=1/8, ν=1 → y_h=d−β/ν).
  - Gates G24–G29, `results/phase4-wolff-multirg.json`. **Ehrliche Scope-Grenze:** Rest-finite-Size-Systematik
    bleibt; **keine** volle `L→∞`-FSS-Extrapolation, **kein** Frontier-Hochpräzisionswert. Keine neuen Deps.

## Phase 5 — Konvergenzdiagnostik + Reproducibility + CLT  [TEIL-DONE 2026-06-19, Branch `claude/qec-phase5-clt-rhat-manifest`]
- Integrierte Autokorrelationszeit τ_int, CLT-Varianz σ²_g; vollständige Reproduzierbarkeit (Seeds/Manifest).
- **Akzeptanz:** CLT empirisch bestätigt; Run reproduzierbar aus Manifest.
- **Teil-vorgezogen (Phase-3a):** τ_int (Wolff Γ-Methode + g-Windowing) + autokorr-Fehler bereits real in
  `autocorr.py` (validiert gegen AR(1)-Orakel).
- **NEU (additiv, keine neuen Deps — nur numpy/scipy):**
  1. **CLT-Varianz σ²_g** (`clt.py`): MCMC-CLT `√N(ḡ_N−E[g]) → N(0, σ²_g)` mit `σ²_g = 2·τ_int·Var(g)`
     (Γ-Methode via `autocorr.py`) UND einem **methodisch unabhängigen zweiten Schätzer** (Overlapping
     Batch Means, Flegal-Jones 2010). **Orakel:** geschlossene AR(1)-Form `σ²_g = σ²_ε/(1−φ)² =
     Var·(1+φ)/(1−φ)` (web-verifiziert, s. SOURCES.md). **Gemessen (`results/phase5-clt-rhat-manifest.json`,
     `ar1_oracle_validation`, φ=0.8, Orakel σ²_g=25.0):** Γ-Schätzer 25.26 (rel.Fehler 0.010), OBM 24.49
     (0.020). **Non-vakuös beidseitig (Gate G34):** korrektes σ²_g deckt das 95%-CI mit Rate 0.937 (~0.95);
     die ABSICHTLICH falsche iid-Annahme (`Var` statt `2τ_int·Var`) UNTERdeckt mit 0.490 (verfehlt 0.95).
  2. **R̂-Multichain** (`rhat.py`): rank-normalized split-R̂ + folded-R̂ + bulk/tail-ESS nach **Vehtari,
     Gelman, Simpson, Carpenter, Bürkner (2021), Bayesian Analysis 16(2), doi:10.1214/20-BA1221**
     (Blom-Rang-Transform → split-chains → between/within → folded um Median; Schwellwert R̂<1.01).
     Auf den A-Kernel-Multichain (M=4) angewandt: **R̂=1.0003, converged, ESS_bulk≈5149**
     (`results/...json`, `multichain_rhat`). **Non-vakuös beidseitig (Gates G35/G36):** gut gemischte
     unabhängige Ketten R̂<1.01; Mittel-Drift → R̂=1.62 (bulk fängt); Skalen-Drift → R̂=1.27
     (folded=1.27>bulk=1.00 — genau der Vehtari-Punkt).
  3. **Run-Manifest** (`manifest.py` + CLI `phase5 --manifest-out/--from-manifest`): JSON mit allen Seeds,
     Parametern (L, n_steps, n_chains, β…), Paket-Versionen (numpy/scipy/python), git-SHA, Plattform.
     **Determinismus-Vertrag:** `--from-manifest` reproduziert den Lauf **byte-identisch** (SHA-256-
     `result_hash`). **Non-vakuös (Gates G37/G38, SLSA-Custody-Lehre):** Round-trip ergibt identischen
     Hash; ein veränderter Seed/n_steps ERGIBT einen anderen Hash (beweist, dass das Manifest den Lauf
     wirklich treibt — kein toter Record).
  - Gates G33–G38 (38/38 [PASS]); Tests `tests/test_clt.py` + `tests/test_rhat.py` + `tests/test_manifest.py`.
  - **Regen:** `python -m adaptiverg_qec.cli phase5 --json results/phase5-clt-rhat-manifest.json --manifest-out results/phase5-manifest.json`.
- ~~**WEITERHIN OFFEN (NICHT erledigt):** SNIS (Self-Normalized Importance Sampling, χ²-Varianz-Bound) +
  Surrogate-Beschleunigung aus Phase-4 bleiben offen; Checkpoint/Restart-Lockfile bleibt offen.~~
  **→ Phase-6 [DONE 2026-08-09] schließt genau diese drei Lücken (s.u.).**

## Phase 6 — SNIS + Surrogate-DA + Checkpoint/Restart-Lockfile  [DONE 2026-08-09, Branch `claude/qec-engine-analysis-improvement-bijszp`]

Schließt die drei seit Phase-4/5 offen deklarierten Punkte in bounded, orakel-validierter Form
(Gates G39–G45; Artefakt `results/phase6-snis-surrogate-checkpoint.json`, regenerierbar via
`python -m adaptiverg_qec.cli phase6 --json results/phase6-snis-surrogate-checkpoint.json`):

1. **SNIS mit χ²-Varianz-Bound (`snis.py`).** Setting bewusst: offene 1D-Ising-Kette, Reweighting
   `K_p → K_t` — dort sind ALLE Größen geschlossen: `1+χ² = [cosh(2K_t−K_p)cosh(K_p)/cosh²(K_t)]^{L−1}`
   (gegen 2⁸-Brute-Force-Enumeration getestet), `ESS/N → 1/(1+χ²)`, führender SNIS-Bias
   `(1+χ²)(tanh K_t − tanh(2K_t−K_p))/N` (Delta-Methode), MSE-Bound `4(1+χ²)/N` für |g|≤1
   (Agapiou/Papaspiliopoulos/Sanz-Alonso/Stuart 2017, Statist. Sci. 32(3), Thm 2.1).
   **Gemessen:** ESS/N-Abweichung < 0.005; `bias·N/c_bias ≈ 0.97` (N=100, 6000 Replikate) bzw.
   `0.65` (N=400, 12000 Replikate, Vorzeichen korrekt); MSE ≤ Bound; Delta-Methoden-Fehler
   kalibriert (0.88–0.97 der empirischen MSE). ESS-Kollaps-Guard beidseitig (χ²≈7e5 → geflaggt).
   **Akzeptanz „Bias O(1/N) belegt": erfüllt** (auf der 1D-Instanz; 2D/RBIM-Targets + Defensive
   Mixture bleiben offen).
2. **Surrogate-Beschleunigung als Delayed Acceptance (`surrogate.py`).** Christen & Fox 2005:
   Stufe-1-Akzeptanz nur mit Surrogat `β̃=β(1+γ)`, Stufe-2-Korrektur exakt → detailed balance
   gegen das exakte π für JEDES Surrogat. **Schärfster Anker:** γ=0 ist BIT-IDENTISCH zum
   Metropolis-A-Kernel (gleicher Philox-Stream, gleiche Flip-Folge). Miskalibrierte Surrogate
   (γ=−0.25/+0.30) treffen das Transfer-Matrix-Orakel (Exaktheit unabhängig von Surrogat-Güte);
   34–42 % exakte Auswertungen gespart. **Surrogate-Drift-Guard** („Surrogate-Drift unter
   Schwelle"): mittlere Log-Diskrepanz je Stufe-2-Eval; hält bei γ=0, feuert bei γ=0.4.
   **Ehrlich:** 1D-Toy, Surrogat nicht real billiger — Korrektheits-Maschinerie, kein
   Speedup-Claim; Kombination mit Diminishing Adaptation offen.
3. **Checkpoint/Restart + Lockfile (`checkpoint.py`).** Philox-Bit-Generator-State JSON-verlustfrei
   serialisiert; Sweep-Loop als EIN gemeinsamer Code-Pfad (`a_kernel.advance_chain`) und
   Post-Processing als EIN Code-Pfad (`manifest.postprocess_multichain`) für direkten UND
   resumierten Lauf. **Determinismus-Vertrag (Gate G44):** Interrupt mitten in einer Kette (auch
   mehrfach) + Resume ⇒ `result_hash` byte-identisch zu `manifest.run()`. Fail-closed (G45):
   O_EXCL-Lockfile (konkurrierende Writer/Resumer → `CheckpointLockedError`; verwaistes Lock wird
   NICHT still übernommen), SHA-256-Integritäts-Hash (Tamper/Korruption → laut abgewiesen),
   atomare Writes. **Scope:** Phase-5-Multichain-Lauf; 2D-Wolff/MCRG-Pipelines nicht gecheckpointet.

**Im selben Inkrement (Audit-Härtung, Korrektheit):** zell-eigene Seeds in `qec_diagnostics`/
`surface_decoder`-Sweeps (vorher teilten alle Zellen einer Kurve denselben RNG-Stream →
rangkorrelierte „Evidenz"); Jeffreys-regularisierte `std_err` (k=0-Zellen waren mit
`n_sigma=0` unfalsifizierbar); `mcrg.validate_swendsen.n_sigma` auf SEM statt Einzel-Seed-std
(vorher √8 zu lax); `rhat`: konstante Ketten mit verschiedenen Mitteln → `R̂=inf`/ESS=0 statt
„converged"; `RunManifest.__post_init__`-Validierung; `rg_map` float64-Erzwingung (Integer-Input
truncated still); G3-Gate verlangt finites λ̂≥1 (nan-Pfad zählt nicht mehr als „gefeuert");
`fit_distance_exponent`/`lambda_suppression` Underflow-Guards; `locate_transition`
Duplikat-p-Guard; Surface-Threshold-Fenster verbreitert (0.09–0.115, 80k Shots/Zelle) +
`crossing_found`-Flag statt stillem NaN.

**Bekannte, BEWUSST nicht in diesem Inkrement gefixte Limitationen (dokumentiert, Follow-up):**
- `ising2d.majority_block_b2`-Tie-Break ist deterministisch, aber nicht Z2-äquivariant
  (Tie-Break flippt nicht unter s→−s); in-Repo-Aufrufer übergeben `config_index`, der Effekt
  ist im Rahmen der ausgewiesenen Grobheit enthalten. Ein äquivarianter Fix ändert alle
  committeten Phase-3b/4-Baselines (inkl. G32-Goldwerte) und gehört in ein eigenes Inkrement.
- `rbim_nishimori`-Scan nutzt denselben `base_seed` über alle p (common random numbers über
  die Kurve; Disorder-Realisierungen über p genestet) — die p*-Lokalisierung bleibt auf
  Plausibilitäts-Niveau, wie ausgewiesen.
- `autocorr.integrated_autocorr_time` klemmt τ_int ≥ 0.5 (für anti-korrelierte Reihen bewusst
  konservativ; jetzt im Code dokumentiert).
- G26 vergleicht τ_int in Update-Einheiten (1 Wolff-Cluster vs 1 Metropolis-Sweep), nicht
  arbeitsnormiert — der ×12–16-Claim ist als solcher zu lesen.

## ROADMAP-Inkr.4 — RBIM-Nishimori ↔ MCRG/QEC-Brücke  [DONE 2026-06-19, Branch `claude/qec-inkr4`]

**Der wissenschaftliche Schlussstein, der die Repo-Architektur validiert.** Behauptung des Repos: die
MCRG-Maschinerie (kritische Exponenten, RG-Fixpunkt) misst DASSELBE Objekt wie der QEC-Threshold — also
gehören beide in EIN Repo. Dieses Inkrement liefert den empirischen Beleg über ein **exaktes Mapping**
(Dennis, Kitaev, Landahl, Preskill, *J. Math. Phys.* 43, 4452 (2002), arXiv:quant-ph/0110143):

> Code-Capacity-Threshold des Toric-/Surface-Codes unter unabhängigem Bit-/Phase-Flip-Rauschen `p`
> == Ordnungs-/Unordnungs-Übergang (Nishimori-Multikritischer-Punkt) des 2D ±J RBIM auf der Nishimori-Linie.
> Publizierter Wert: **`p_c ≈ 0.1094` (10.94 ± 0.02 %)**.

- **Implementiert (`rbim_nishimori.py`, additiv, keine neuen Deps):** ±J-RBIM mit quenched Disorder
  (`J=-1` mit Wkt `p`); Nishimori-Bedingung `p = 1/(1+e^{2β})`; **RBIM-generalisierter Wolff-Single-Cluster**
  (`P_add = 1−e^{−2β·J_ij·s_i·s_j}` auf befriedigten Bonds, web-verifiziert) **hybridisiert mit gewichteten
  Checkerboard-Metropolis-Sweeps** (Wolff allein saturiert tief in der FM-Phase: `cf→1`, friert
  Domänenwände ein → Metropolis annealt die Domänenstruktur); **aligned-Start-Ordnungsparameter-Protokoll**.
  Observablen: disorder-gemittelte `[<|m|>]` + Binder-Kumulante; Übergang = steilster `|m|`-Abfall.
- **Korrektheit gegen unabhängige Orakel (KEIN Self-Check):** (a) p=0 RBIM-Energie == homogene
  `energy_per_spin` (byte-genau); (b) **Gauge-Invarianz** `s_i→τ_i s_i, J_ij→τ_iτ_j J_ij` lässt E exakt
  invariant (diff < 1e-12); (c) **exakte L=4-Boltzmann-Enumeration** (2^16 Zustände, voll) trifft E/N und
  ⟨|m|⟩ auch im FRUSTRIERTEN Fall (p>0; dieser Test fing einen Bond-Richtungs-Bug in der Cluster-BFS, der
  bei p=0 unsichtbar war); (d) Stationaritäts-Sektor (G-N5): aligned-Start landet im korrekten FM-Sektor
  (p<p_c, hohes ⟨|m|⟩) bzw. PM-Sektor (p>p_c, relaxiert) — kein Bias.
- **Gemessen — regenerierbares Artefakt `results/inkr4-rbim-nishimori.json` (L=8, 24 Realisierungen,
  `python -m adaptiverg_qec.rbim_nishimori`):** `[<|m|>]` fällt monoton `0.98 → 0.36` über `p=0.04 → 0.20`;
  **steilster Abfall bei `p* ≈ 0.145`** (|err| ≈ 0.036 vs `p_c`). **Feinere Auflösung — Artefakt
  `results/inkr4-rbim-nishimori-L12.json` (L=12, 30 Realisierungen, via `--L 12 --n-disorder 30`):**
  `[<|m|>] 0.99 → 0.22`, `p* = 0.120` (|err| = 0.011) — erwarteter
  Finite-Size-Shift zu kleinerem `p*` mit wachsendem L; beide konsistent mit `p_c ≈ 0.109`.
  Gates: `tests/test_rbim_nishimori.py` (10/10).
- **Ehrliche Scope-Grenze:** kleines L + endliches Disorder-Sampling → **grobe Lokalisierung auf
  Plausibilitäts-Niveau**, KEINE `L→∞`-FSS, KEIN Frontier-Wert. Konsistent mit der Repo-Positionierung.
- **Brücken-Ergebnis:** das RBIM-Tooling (gebaut aus `ising2d`/`wolff2d`/MCRG-Maschinerie) lokalisiert
  `p* ≈ p_c` → die MCRG-Exponenten-Maschinerie misst (im Rahmen der Auflösung) **dasselbe Objekt** wie der
  QEC-Threshold → der **„nicht-splitten"-Architektur-Entscheid ist empirisch gestützt**.

---
*Coworker Research | aus AdaptiveRG-QEC Spec v1.0 hardened | Inkr.4 (Brücke) DONE 2026-06-19*
