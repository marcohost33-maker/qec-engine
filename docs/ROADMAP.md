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
- **Phase-3b [OFFEN]:** Multi-Operator-Swendsen-MATRIX (mehrere S_a, lineares Gleichungssystem dK'_a/dK_b),
  Bias-Ordnung O(n^−β) empirisch, hyperbolischer Fixpunkt im Mehr-Kopplungs-Fall.

## Phase 4 — Sampling (SNIS) + Surrogate-Beschleunigung + Fehlerbudgets
- Self-Normalized Importance Sampling mit χ²-Divergenz-Varianz-Bound; Surrogate-Drift-Kontrolle.
- **Akzeptanz:** Bias O(1/N) belegt; Surrogate-Drift unter Schwelle.

## Phase 5 — Konvergenzdiagnostik + Reproducibility + CLT
- Integrierte Autokorrelationszeit τ_int, CLT-Varianz σ²_g; vollständige Reproduzierbarkeit (Seeds/Manifest).
- **Akzeptanz:** CLT empirisch bestätigt; Run reproduzierbar aus Manifest.
- **Teil-vorgezogen (Phase-3a):** τ_int (Wolff Γ-Methode + g-Windowing) + autokorr-Fehler bereits real in
  `autocorr.py` (validiert gegen AR(1)-Orakel). Offen für Phase-5: CLT-Varianz σ²_g, Run-Manifest.

---
*Coworker Research | aus AdaptiveRG-QEC Spec v1.0 hardened | alle Phasen offen 2026-06-02*
