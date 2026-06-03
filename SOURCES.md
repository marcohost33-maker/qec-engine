# QEC-Engine (AdaptiveRG-QEC) — SOURCES & Provenance (2026-06-02)

**Projekt:** AdaptiveRG-QEC Engine — Adaptive Renormalization-Group + Quantum-Error-Correction
Simulation Engine. Coworker-Research Säule 3 (Physik/Methodik).
**Typ:** ⚠ **Spec/Theorie-Repo — KEIN Code vorhanden** (G:-weit kein `.py` zu QEC/AdaptiveRG).

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
