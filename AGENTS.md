---
format: agents.md
version: "1.0"
project: AdaptiveRG-QEC
---

# AGENTS.md — AdaptiveRG-QEC Engine (Spec + Phase-1/2-MVP)

Reihenfolge: §1 Working agreements > §2 Conventions > §3 Don't > §4 When stuck.

## Project
- **Was:** Adaptiver RG-QEC-Simulator als **Diagnostik-/Verifikations-Harness mit Konvergenz-Guards** (KEIN Frontier-Threshold-Tool). Spec/Theorie + lauffähiger Phase-1-MVP (seit PR#4).
- **Inhalt:** gehärtete Kernel-Spec v1.0 + Proof-Block v1.1 / KernelSpec v1.2 (`spec/`) + Phase-1-MVP-Code (`src/adaptiverg_qec/`, `tests/`, CI).
- **MVP-Stand:** real = A-Kernel-MCMC + Foster-Lyapunov-Drift-Guard + MCRG-Map/Jacobian + **skalarer Swendsen-MCRG-Schätzer** (sample-geschätzte R̂ `T̂=⟨S'S⟩_c/⟨S'S'⟩_c` vs `tanh(2K)` validiert; 1D-Ising-Instanz, exakt-i.i.d.-Sampler). Offen/gestubbt (ehrlich): **Multi-Operator-Swendsen-Matrix** + A-Kernel-Autokorrelations-Fehlerbalken (Phase-3), SNIS, Multichain/R-hat, Surrogate, Checkpoint, Lockfile — Phasen 3-5 der ROADMAP.

## Working agreements
1. **Kein Overclaim „implementiert".** `src/` enthält den Phase-1-MVP; README/SOURCES/Status nennen ehrlich, was MVP-real vs. gestubbt ist (stochastische R̂, SNIS, Multichain offen). Keine Komponente als „validiert" behaupten ohne lauffähigen Code + Gate-Log in `results/`.
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
