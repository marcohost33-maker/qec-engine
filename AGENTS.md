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
- **MVP-Stand:** real = A-Kernel-MCMC + Foster-Lyapunov-Drift-Guard + MCRG-Map/Jacobian + **skalarer Swendsen-MCRG-Schätzer** (sample-geschätzte R̂ `T̂=⟨S'S⟩_c/⟨S'S'⟩_c` vs `tanh(2K)` validiert; 1D-Ising-Instanz). **Phase-3a (NEU): T̂ aus dem korrelierten A-Kernel-MCMC (statt exakt-i.i.d.) mit autokorrelations-bewussten Fehlerbalken** — `autocorr.py`: FFT-ρ (Wiener-Khinchin), τ_int + Wolff-g-Windowing, Binning-Cross-Check, Block-Jackknife fürs Verhältnis; belegt N_eff<N und σ_korr>σ_iid (`results/phase3a-akernel-autocorr.json`). **Phase-3b (NEU): Multi-Operator-Swendsen-MATRIX auf 2D-Ising** — `ising2d.py` (vektorisierter Checkerboard-Metropolis + Majority-Rule-Blocking b=2) + `mcrg_matrix.py` (gerade Operatoren S₁/S₂/S₃, connected-corr-Matrizen A,B, `T=A·B⁻¹` via `np.linalg.solve`, Eigenwert-Exponenten, Block-Jackknife). y_t=0.97±0.01 vs Onsager-Orakel y_t=1 — **ehrlich GROB** (single-spin + 1 RG-Stufe → kritisches Slowing-Down; Plausibilität, KEIN Frontier-Wert), `results/phase3b-swendsen-matrix.json`. **Phase-4 (NEU): Wolff-Cluster + Multi-RG + y_h** — `wolff2d.py` (Wolff-Single-Cluster, `P_add=1−e^{−2K}`, vektorisierter BFS, rejection-free; Energie vs exakte L=4-Enum |err|<0.008; `τ_int(Wolff)≪τ_int(Metropolis)`, ×12–16 @ L=32) + `mcrg_multirg.py` (iterierte Majority-Stufen L=32→16→8→4: gerader y_t konvergiert, bester `|y_t−1|≈0.006` vs 3b 0.035; ungerade Operatoren O₁=M, O₂=3-Spin-L-Cluster → ungerade Swendsen-Matrix `T_h=A·B⁻¹` → y_h, bester Iterationswert `|y_h−15/8|≈0.002` (Minimum über Iterationen, NICHT tiefste Stufe; tiefste-Iter ≈0.003) vs Onsager y_h=15/8=1.875). Gates G24–G29 (29/29 [PASS]), `results/phase4-wolff-multirg.json`. **Ehrlich:** Rest-finite-Size bleibt, KEINE L→∞-FSS, KEIN Frontier-Wert. Keine neuen Deps. Offen/gestubbt (ehrlich): SNIS, Multichain/R-hat, Surrogate, Checkpoint, Lockfile — Phase-5 der ROADMAP.

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
