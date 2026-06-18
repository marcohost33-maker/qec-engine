# results/ — Gate-Logs (Evidenz)

AGENTS.md verlangt Evidenz hier (keine Physik-/Status-Claims ohne Gate-Log).

- `selftest.json` — JSON-Gate-Log des letzten lokalen `adaptiverg-qec selftest`-Laufs
  (8 Gates, je gegen ein unabhängiges Orakel). Reproduzierbar via:

  ```bash
  adaptiverg-qec selftest --json results/selftest.json
  ```

Die SHA-256 dieser Logs ist lauf-spezifisch (Zeitstempel/Elapsed) und wird daher nicht
in SOURCES.md gepinnt — die Reproduktion erfolgt durch erneuten Lauf. CI lädt zusätzlich
`results/selftest-ci.json` als Artefakt hoch.
