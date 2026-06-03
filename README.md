# AdaptiveRG-QEC Engine

**Spec / Theorie-Repo.** Eine *Adaptive Renormalization-Group + Quantum-Error-Correction* Simulation-Engine:
integriert stochastische Dynamik, adaptive Steuerung, Sampling, RG-Analyse, Surrogate-Beschleunigung und
Stabilitäts-Guards. Coworker-Research Säule 3 (Physik/Methodik).

> **Ehrlicher Status:** Dies ist ein **Spec-/Theorie-Repo — es gibt (noch) KEINEN Implementierungs-Code.**
> Vorhanden sind zwei gehärtete mathematische Dokumente. Ein Code-Repo entsteht erst, wenn die Roadmap
> (`docs/ROADMAP.md`) implementiert wird.

## Inhalt
- `spec/AdaptiveRG_QEC_Engine_Spec_v1_0_hardened.pdf` — gehärtete Kernel-Spec v1.0: augmentierter Zustand,
  Foster-Lyapunov-Drift (Meyn & Tweedie), Petite-Set/Minorization, Diminishing-Adaptation + Containment
  (Roberts & Rosenthal 2007), SNIS-Sampling-Bounds, *Unified Stability–RG Convergence Theorem*.
- `spec/AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx` — Beweis-Block: P1 (Ergodizität A-Kern),
  P2 (RG-Fixpunkt-Konsistenz), CLT (Andrieu & Moulines).
- `docs/ROADMAP.md` — die 5-Phasen-Implementierungs-Roadmap (offen).

## Provenance
Quellen + Drive-IDs: siehe `SOURCES.md`. Code-frei verifiziert 2026-06-02 (G:-weit kein `.py` zu QEC/AdaptiveRG).

## Lizenz
Apache-2.0. Attribution: Coworker Research / Coworkerz (keine Einzelnamen).
