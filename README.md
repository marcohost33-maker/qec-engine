# AdaptiveRG-QEC Engine

**Spec + Phase-1-MVP.** Eine *Adaptive Renormalization-Group + Quantum-Error-Correction* Simulation-Engine:
integriert stochastische Dynamik, adaptive Steuerung, Sampling, RG-Analyse, Surrogate-Beschleunigung und
Stabilitäts-Guards. Coworker-Research Säule 3 (Physik/Methodik).

> **Ehrlicher Status (2026-06-18):** Die gehärtete Theorie (`spec/`) liegt vor; ein **bounded Phase-1-MVP**
> (`src/adaptiverg_qec/`, Version `0.1.0.dev0`) implementiert den MCMC-A-Kernel mit Foster-Lyapunov-Guard
> und einen minimalen MCRG-C-Kernel. Reifegrad **dev/Prototyp** — nicht release-fertig, nicht selbst-zertifiziert.
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
  Frontier-Wert und behauptet die Spec-Phase-2-Ziele (2D-Ising `y_t=1`, `y_h=15/8`) **nicht**.
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
| Selftest-Gates (`cli.py --selftest`, 8 Gates, JSON-Log) | **real** (8/8 [PASS], exit 0) |
| SNIS / Defensive Mixture / ESS-per-Observable (Spec §5) | **offen** (Phase 4) |
| MCRG aus echten MCMC-Samples (stochastische R̂, Spec §6 Kopplung A↔C) | **offen** — MVP nutzt die *exakte* RG-Map, nicht die sample-geschätzte |
| Surrogate + MMD-Drift + Checkpoint/Restart + R-hat-Multichain (Spec §8/§10) | **offen** (Phasen 2–5) |

## Schnellstart

```bash
pip install -e ".[dev]"
adaptiverg-qec selftest          # 8 Gates, exit 0 gdw alle [PASS]
adaptiverg-qec demo              # A-Kernel + C-Kernel-Demo
pytest                           # Test-Suite
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
