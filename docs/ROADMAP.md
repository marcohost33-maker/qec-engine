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

## Phase 5 — Konvergenzdiagnostik + Reproducibility + CLT
- Integrierte Autokorrelationszeit τ_int, CLT-Varianz σ²_g; vollständige Reproduzierbarkeit (Seeds/Manifest).
- **Akzeptanz:** CLT empirisch bestätigt; Run reproduzierbar aus Manifest.
- **Teil-vorgezogen (Phase-3a):** τ_int (Wolff Γ-Methode + g-Windowing) + autokorr-Fehler bereits real in
  `autocorr.py` (validiert gegen AR(1)-Orakel). Offen für Phase-5: CLT-Varianz σ²_g, Run-Manifest.

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
