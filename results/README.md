# results/ — Gate-Logs (Evidenz)

AGENTS.md verlangt Evidenz hier (keine Physik-/Status-Claims ohne Gate-Log).

- `selftest.json` — JSON-Gate-Log des letzten lokalen `adaptiverg-qec selftest`-Laufs
  (12 Gates, je gegen ein unabhängiges Orakel; G9–G12 = Phase-2 Swendsen-MCRG).
  Reproduzierbar via:

  ```bash
  adaptiverg-qec selftest --json results/selftest.json
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
