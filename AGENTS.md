---
format: agents.md
version: "1.0"
project: AdaptiveRG-QEC
---

# AGENTS.md — AdaptiveRG-QEC Engine (Spec-Repo)

Reihenfolge: §1 Working agreements > §2 Conventions > §3 Don't > §4 When stuck.

## Project
- **Was:** SPEC/Theorie eines adaptiven RG-QEC-Simulators. **Kein Code** (Spec-Repo).
- **Inhalt:** gehärtete Kernel-Spec v1.0 + Proof-Block v1.1 / KernelSpec v1.2.

## Working agreements
1. **Kein Overclaim „implementiert".** Solange `src/` leer ist, ist dies ein Spec-Repo — README/Status sagen das ehrlich.
2. **Mathematische Strenge ist tragend.** Korrekturen in der Spec sind inline markiert (`[KORREKTUR]/[LÜCKE]/[OK]`);
   diese Disziplin bei Spec-Änderungen beibehalten.
3. **Roadmap-getrieben.** Implementierung folgt `docs/ROADMAP.md` Phase für Phase; jede Phase mit Akzeptanzkriterium.

## Conventions
- Specs als versionierte Dokumente in `spec/`; bei Implementierung: `src/` + `tests/` ergänzen, AGENTS.md auf Code-Repo umstellen.
- Literatur-Referenzen in der Spec sind verbindlich (Roberts&Rosenthal, Meyn&Tweedie, Andrieu&Moulines, …).

## Branch & PR conventions (agents)

- **One agent = one branch prefix:** `claude/<task>` (Claude Code), `codex/<task>`
  (OpenAI Codex), `bot/<task>` (CI/automation). Human-led work: `feat|fix|docs/<task>`.
- **Agent output opens as a Draft PR** and stays draft until Definition-of-Done is
  verified; then mark ready.
- **Label agent PRs:** `agent:claude` / `agent:codex` / `agent:bot`.
- **Auto-merge over manual merge:** enable `gh pr merge --auto --squash` once required
  checks exist; a second concurrent PR must rebase on the updated main.
- **No concurrent agent pushes** to the same repo: serialize, or split work by branch
  namespace and let auto-merge order the merges.

## Don't
- Don't behaupten, eine Komponente sei implementiert/validiert, ohne lauffähigen Code + Test.
- Don't use `--no-verify`, `--no-gpg-sign`, `--force` ohne explizites User1-OK.
- Don't commit or leak secrets, API keys, `.env`/credential files, `GITHUB_TOKEN`, or signing keys (logs included).

## When stuck
- `spec/` (die 2 gehärteten Docs) + `docs/ROADMAP.md`. Bei offener math. Frage: Spec-Referenzen prüfen, nicht raten.

---
*Coworker Research. Format: <https://agents.md/>.*
