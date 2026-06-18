# results/ — Gate-Logs (Evidenz)

AGENTS.md verlangt Evidenz hier (keine Physik-/Status-Claims ohne Gate-Log).

- `selftest.json` — JSON-Gate-Log des letzten lokalen `adaptiverg-qec selftest`-Laufs
  (23 Gates, je gegen ein unabhängiges Orakel; G9–G12 = Phase-2 Swendsen-MCRG,
  G13–G18 = Phase-3a autokorr-Fehler, G19–G23 = Phase-3b 2D-Ising-Swendsen-MATRIX).
  Reproduzierbar via:

  ```bash
  adaptiverg-qec selftest --json results/selftest.json
  ```

- `phase3b-swendsen-matrix.json` — Phase-3b-Validierung: Multi-Operator-Swendsen-MATRIX
  `T = A·B⁻¹` auf dem 2D-Ising-Modell bei `K_c`; thermischer Exponent `y_t = ln λ_max/ln 2`
  vs. exaktes Onsager-Orakel `y_t = 1` (3 Seeds, Block-Jackknife-Fehler). **Ehrlich grob**
  (single-spin Metropolis + 1 RG-Stufe → kritisches Slowing-Down). Reproduzierbar via:

  ```bash
  python -m adaptiverg_qec.mcrg_matrix   # schreibt results/phase3b-swendsen-matrix.json
  ```

- `phase2-swendsen.json` — Phase-2-Validierungstabelle: Swendsen-MCRG-Schätzer
  T̂(K) vs. analytisches Orakel tanh(2K) über 4 K-Werte, 8 Seeds, mit
  Multi-Seed-Fehlerbalken (std/SEM) und n_sigma. Reproduzierbar via:

  ```bash
  python -m adaptiverg_qec.mcrg   # schreibt results/phase2-swendsen.json
  ```

Die SHA-256 dieser Logs ist lauf-spezifisch (Zeitstempel/Elapsed) und wird daher nicht
in SOURCES.md gepinnt — die Reproduktion erfolgt durch erneuten Lauf. CI lädt zusätzlich
`results/selftest-ci.json` als Artefakt hoch.
