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

## Phase 4 — Sampling (SNIS) + Surrogate-Beschleunigung + Fehlerbudgets
- Self-Normalized Importance Sampling mit χ²-Divergenz-Varianz-Bound; Surrogate-Drift-Kontrolle.
- **Akzeptanz:** Bias O(1/N) belegt; Surrogate-Drift unter Schwelle.

## Phase 5 — Konvergenzdiagnostik + Reproducibility + CLT
- Integrierte Autokorrelationszeit τ_int, CLT-Varianz σ²_g; vollständige Reproduzierbarkeit (Seeds/Manifest).
- **Akzeptanz:** CLT empirisch bestätigt; Run reproduzierbar aus Manifest.

---
*Coworker Research | aus AdaptiveRG-QEC Spec v1.0 hardened | alle Phasen offen 2026-06-02*
