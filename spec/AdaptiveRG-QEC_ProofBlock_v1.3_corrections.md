# AdaptiveRG-QEC — ProofBlock v1.3 (Mathematische Korrekturen)

> **Status:** Errata-/Korrektur-Companion zum binären Quell-Dokument
> `spec/AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx` (SoT, unverändert).
> v1.3 korrigiert **4 mathematisch defekte Knoten** (P2, P3.1, P3.3, P4) des ProofBlock v1.1.
> Marker-Konvention gemäß `AGENTS.md`: `[KORREKTUR]`.
> **Reifegrad:** Spec-Korrektur, jeder Defekt gegen ein unabhängiges Orakel belegt
> (numerisch und/oder gegen die eigene, oracle-validierte Repo-Implementierung).
> Selbst-Precheck (L0) — NICHT selbst-zertifiziert; Equalita-Ratifikation offen.

## Verortung
Der ProofBlock liegt binär als `.docx` vor (kein Zeilen-Diff möglich). Locations werden
referenziert per **Abschnitts-ID** (stabil) plus **Absatz-Nr. der Klartext-Extraktion**
(`word/document.xml` → Absätze in Dokumentreihenfolge; reproduzierbar via unten stehendem
Extraktions-Snippet). Der defekte Originaltext ist jeweils wörtlich zitiert.

---

## Defekt 1 — Knoten P4.1 (Fehlerbudget): `n_eff` doppelt deflationiert

**Ort:** §P4.1 „Die Terme im Einzelnen", Absatz 106.

**Defekter Originaltext:**
> „c₁ σ²_MCMC/n_eff: MCMC-Sampling-Fehler, wobei **n_eff = ESS/(2τ_int)** die effektive Samplezahl ist."

**Root-Cause (warum falsch):**
Die *effective sample size* ESS **ist bereits** die autokorrelations-korrigierte Samplezahl.
In der Standard-Konvention τ_int = ½ + Σ_{k≥1} ρ_k gilt

    ESS = N / (2 τ_int)   (= N / (1 + 2 Σ_{k≥1} ρ_k)).

Der Ausdruck `n_eff = ESS/(2τ_int)` teilt **ein zweites Mal** durch die
Autokorrelationszeit und liefert die sinnlose Größe

    n_eff = ESS/(2τ_int) = N / (2τ_int)²,

d.h. eine doppelte Deflation. Für τ_int = 5 unterschätzt das die effektive Stichprobe um
den Faktor 2τ_int = 10 und **überschätzt** den MCMC-Fehler c₁σ²/n_eff entsprechend um Faktor 10.

**Beleg (eigenes Repo als Orakel):** Die implementierte, gegen ein AR(1)-Orakel validierte
Definition im selben Projekt ist
`src/adaptiverg_qec/autocorr.py:173` → `n_eff = n / (2.0 * tau_int)` (Kommentar Z. 22/108:
„N_eff = N / (2 tau_int)"), ebenso `README.md:117`. Der ProofBlock widerspricht seinem
eigenen validierten Code.

**[KORREKTUR] Korrekte Fassung:**

    n_eff = ESS = N / (2 τ_int) = N / (1 + 2 Σ_{k≥1} ρ_k),

mit τ_int in der ½-Konvention. `n_eff` ist synonym zu ESS, **nicht** ESS nochmals durch 2τ_int.

---

## Defekt 2 — Knoten P3.1 (Jacobian-Konvergenz): stochastische Rate ohne 1/ε_diff und 1/√n

**Ort:** §P3.1 Theorem (Jacobian-Konvergenz), Absatz 85.

**Defekter Originaltext:**
> „‖M̂ − DR(g*)‖_F = **O_P(1/√N_rep)** + O(ε_diff²) + O(n^{−β})"

**Root-Cause (warum falsch):**
Die zentrale Differenz **dividiert den Schätzfehler durch die Schrittweite ε_diff** — genau
das führt der Knoten in seiner *eigenen* Beweisskizze P3.2 (Absatz 92) aus:

    M̂_ij = ∂R_i/∂g_j(g*) + O(ε²) + O(b/ε) + ζ_ij/ε,   Var(ζ_ij) = O(1/(n·N_rep)).

Daraus folgt für den stochastischen (Varianz-)Term die Standardabweichung

    std(ζ/ε) = O( 1 / (ε_diff · √(n · N_rep)) ),

also mit **1/ε_diff-Verstärkung** und **1/√n-Faktor**. Die in P3.1 angegebene Rate
`O_P(1/√N_rep)` lässt **beide** fallen (kein ε_diff, kein n). Ebenso ist der Bias-Term
in Wahrheit `O(b/ε) = O(n^{−β}/ε_diff)`, nicht `O(n^{−β})`. Das Operative Theorem v1.2
(Absatz 249) schreibt zudem eine **dritte, abweichende** Rate
`O_P(n_eff^{−1/2} + ε_diff² + n^{−β})` — die drei Stellen sind untereinander inkonsistent.
Das Weglassen von 1/ε_diff macht insbesondere den zwei Absätze später hergeleiteten
Bias-Varianz-Schrittweiten-Kompromiss (ε_opt) inkohärent, da genau die Verstärkung des
Rauschens bei kleinem ε_diff der Grund für ein endliches ε_opt ist.

**Beleg (Herleitung ε_opt, Konsistenz-Check):**
MSE(ε) ≈ ε⁴ (Trunkierung, zentrale Diff. O(ε²) → quadriert) + 1/(n·N_rep·ε²) (Varianz).
d/dε [ε⁴ + c/ε²] = 0 ⇒ ε_opt ~ (n·N_rep)^{−1/6}. Dieser endliche Kompromiss existiert **nur**,
weil der Varianzterm ∝ 1/ε² ist — konsistent mit P3.2, inkonsistent mit der Rate `O_P(1/√N_rep)`.

**[KORREKTUR] Korrekte Fassung:**

    ‖M̂ − DR(g*)‖_F = O_P( 1 / (ε_diff · √(n · N_rep)) ) + O(ε_diff²) + O(n^{−β} / ε_diff).

(Bei fixem ε_diff = Θ(1) absorbieren sich die 1/ε_diff-Faktoren in Konstanten; die Aussage
`O_P(1/√N_rep)` gilt dann nur unter der zusätzlichen, in P3.1 **nicht** genannten Annahme
ε_diff = Θ(1) — und widerspricht dann dem ε_opt ~ (n·N_rep)^{−1/6} aus P3.2.)

---

## Defekt 3 — Knoten P3.3 (Exponenten-Konvergenz): Eigenwert-Schranke via sep⁻¹ statt Konditionszahl

**Ort:** §P3.3, Absatz 97.

**Defekter Originaltext:**
> „|λ̂_i − λ_i| ≤ ‖M̂ − DR(g*)‖_F · **sep(λ_i)^{−1}**, wobei sep(λ_i) = min_{j≠i} |λ_i − λ_j|"

**Root-Cause (warum falsch):**
`sep(λ_i)` steuert die Perturbation des **Eigenvektors / invarianten Unterraums**
(Davis–Kahan sinΘ, Stewart–Sun), **nicht** die des **Eigenwerts**. Für einen einfachen
Eigenwert einer (generisch nicht-normalen) Matrix DR(g*) lautet die Erste-Ordnung-Perturbation

    δλ_i = (u_iᴴ E v_i) / (u_iᴴ v_i)   ⇒   |λ̂_i − λ_i| ≤ κ(λ_i) · ‖E‖₂ + O(‖E‖²),

mit der **Eigenwert-Konditionszahl** κ(λ_i) = ‖u_i‖‖v_i‖ / |u_iᴴ v_i| = 1/|cos∠(u_i, v_i)|
(Bauer–Fike; Kato 1995; Golub–Van Loan §7.2). `sep` kommt in der Eigenwert-Schranke **nicht**
vor (für normale Matrizen ist κ = 1 → Weyl: |λ̂−λ| ≤ ‖E‖). Da DR(g*) i.A. nicht-normal ist,
kann `sep⁻¹` die tatsächliche Eigenwert-Empfindlichkeit **beliebig unterschätzen**.

**Beleg (unabhängiges numerisches Orakel):** A = [[1, t], [0, 2]] hat konstante
Eigenwert-Separation sep ≡ 1, aber mit t wachsend zunehmend parallele Eigenvektoren.
Gemessenes |δλ|/‖E‖ (Mittel über 400 Zufalls-E, ‖E‖=1e−6):

| t | sep⁻¹ (docx-Bound-Faktor) | κ(λ) (korrekt) | gemessen |δλ|/‖E‖ |
|---|---|---|---|
| 0  | 1.0 | 1.0  | 0.79 |
| 5  | 1.0 | 5.10 | 3.95 |
| 50 | 1.0 | 50.0 | 39.0 |

Die reale Perturbation skaliert mit **κ(λ)**, nicht mit sep⁻¹; der docx-Bound wird bei t=50
um ~40× verletzt.

**[KORREKTUR] Korrekte Fassung:**

    |λ̂_i − λ_i| ≤ κ(λ_i) · ‖M̂ − DR(g*)‖₂ + O(‖·‖²),   κ(λ_i) = 1/|cos∠(u_i, v_i)|,

wobei u_i, v_i der linke bzw. rechte Eigenvektor sind. `sep(λ_i)` steuert die
**Eigenvektor-/Unterraum**-Schranke ‖δv_i‖ ≲ ‖E‖/sep(λ_i) — das ist im **Folgesatz** von P3.3
(Unterraum-Tracking bei sep ≈ 0) korrekt verwendet; nur die Eigen**wert**-Zeile mislabelt sep.

---

## Defekt 4 — Knoten P2.1/P2.2 (Fixpunkt-Konsistenz): Kugel-Bedingung ignoriert instabile Mannigfaltigkeit

**Ort:** §P2.1 Theorem (Absätze 62–66) + §P2.2 Schritt 2 (Absätze 76–77).

**Defekter Originaltext:**
> P2.1: „Dann konvergiert die iterierte Schätzung ĝ_k gegen g* in Wahrscheinlichkeit,
> **sofern die Iteration in der Linearisierungsumgebung B(g*, r_lin) verbleibt**."
> P2.2 Schritt 2: „… folgt …, dass **ĝ_k in einer O(n^{−1/2})-Umgebung von g* bleibt**
> (sofern der Startpunkt hinreichend nah an g* liegt)."

**Root-Cause (warum falsch):**
Der Fixpunkt ist **hyperbolisch** (C3, in P2.2 Schritt 1 explizit): DR(g*) besitzt neben
stabilen Eigenwerten (|λ_s| < 1) auch **instabile** (|λ_u| > 1) — die *relevanten* Skalenfelder.
Schritt 1 beweist Kontraktion **nur entlang der stabilen** Richtungen (ρ_s = max|λ_s| < 1).
Projiziert man die Linearisierung ĝ_{k+1} − g* = DR(g*)(ĝ_k − g*) + η_k + r_k auf die
instabile Eigenrichtung u_u, so gilt

    u_uᴴ(ĝ_k − g*) ≈ λ_u^k · u_uᴴ(ĝ_0 − g*) + Σ_{j<k} λ_u^{k−1−j} · u_uᴴ η_j,

dessen deterministischer **und** stochastischer Anteil **geometrisch wachsen**
(Var ~ λ_u^{2k} · Var(η)/(λ_u² − 1)). ĝ_k bleibt also **nicht** in einer O(n^{−1/2})-Umgebung —
es entweicht entlang der instabilen Mannigfaltigkeit. Eine **Kugel** B(g*, r_lin) beschränkt
nur den Abstand, nicht die Richtung; der Guard „‖ĝ_k − g*‖ < r_lin" ist als
Konvergenz-Hinreichung **unzureichend**. Das ist der physikalisch tragende Punkt der MCRG:
die relevanten Kopplungen müssen auf die **kritische Fläche** (stabile Mannigfaltigkeit
W^s(g*)) getunt werden; die Exponenten y_i liest man dann am **linearisierten** Jacobian
DR(g*) am getunten Punkt ab — ohne globale Konvergenz von ĝ_k vorauszusetzen.

**Beleg (unabhängiges numerisches Orakel):** DR = diag(0.5, 1.5), σ = 1e−3, Start bei Abstand
1.4e−4 ⟪ r_lin = 1e−2, 60 Iterationen, Mittel über 200 Läufe:
- **Nur Kugel-Bedingung (untuned):** mittleres End-‖ĝ−g*‖ = **2.45e+07** (Blow-up entlang λ_u).
- **Getunt auf W^s (u_u-Feld = 0):** mittleres End-‖ĝ−g*‖ = **9.18e−4 ≈ O(σ)**.
Start *innerhalb* der Kugel ⇒ trotzdem Divergenz. Die Theorem-Aussage
„B(g*, r_lin) ⇒ Konvergenz" ist damit widerlegt.

**[KORREKTUR] Korrekte Fassung (Bedingung):**
Ersetze die Guard-/Konvergenz-Bedingung durch:

> „… sofern die **relevanten (instabilen) Skalenfelder auf die kritische Fläche getunt** sind,
> d.h. die Iteration auf der **stabilen Mannigfaltigkeit** W^s(g*) verbleibt
> (u_uᴴ(ĝ_k − g*) = 0 bis O(‖·‖²)) **und** ‖ĝ_k − g*‖ < r_lin. Nur dann konvergiert
> ĝ_k → g* in Wahrscheinlichkeit. Andernfalls konvergieren lediglich die **stabilen**
> Komponenten; die Exponenten y_i = log_b|λ_i| sind aus dem linearisierten Jacobian DR(g*)
> am getunten kritischen Punkt zu extrahieren, ohne globale Konvergenz von ĝ_k vorauszusetzen."

Die zulässige Guard-Menge ist **W^s(g*) ∩ B(g*, r_lin)**, nicht B(g*, r_lin) allein.

---

## Zusammenfassung der 4 Knoten

| # | Knoten | Defekt | Korrektur |
|---|--------|--------|-----------|
| 1 | P4.1 | `n_eff = ESS/(2τ_int)` (doppelte Deflation) | `n_eff = ESS = N/(2τ_int)` |
| 2 | P3.1 | Rate `O_P(1/√N_rep)` ohne 1/ε_diff, 1/√n | `O_P(1/(ε_diff·√(n·N_rep))) + O(ε_diff²) + O(n^{−β}/ε_diff)` |
| 3 | P3.3 | Eigenwert-Bound mit `sep⁻¹` | `κ(λ_i)·‖E‖₂`, κ = 1/|cos∠(u_i,v_i)| |
| 4 | P2.1/P2.2 | Kugel-Bedingung ⇒ Konvergenz (ignoriert W^u) | Guard = W^s(g*) ∩ B(g*, r_lin) |

## Reproduktion der Klartext-Extraktion (Absatz-Nummern)

```python
import zipfile, re
z = zipfile.ZipFile("spec/AdaptiveRG-QEC_ProofBlock_v1.1_KernelSpec_v1.2.docx")
xml = z.read("word/document.xml").decode("utf-8")
paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
for i, p in enumerate(paras, 1):
    print(i, "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)))
```

## Belege / Orakel (Herkunft: eigener-lauf, 2026-07-02)
- Defekt 1: eigene validierte Repo-Definition `src/adaptiverg_qec/autocorr.py:173`.
- Defekt 2: analytische ε_opt-Herleitung; Konsistenz mit P3.2 (Absatz 92).
- Defekt 3: numerisches Orakel A=[[1,t],[0,2]], 400 Zufalls-E je t; κ vs sep vs Messung.
- Defekt 4: numerisches Orakel DR=diag(0.5,1.5), σ=1e−3, 200 Läufe; untuned 2.45e7 vs getunt 9.2e−4.

## Referenzen
- Bauer, F.L. & Fike, C.T. (1960). Norms and exclusion theorems. *Numer. Math.* 2:137–141.
- Kato, T. (1995). *Perturbation Theory for Linear Operators*, 2nd ed. Springer. (Kap. II).
- Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations*, 4th ed. §7.2 (eigenvalue condition),
  §8.1 (Davis–Kahan / sep).
- Stewart, G.W. & Sun, J. (1990). *Matrix Perturbation Theory*. Academic Press.
- Guckenheimer, J. & Holmes, P. (1983). *Nonlinear Oscillations, Dynamical Systems, and
  Bifurcations of Vector Fields*. Springer. (stabile/instabile Mannigfaltigkeit).
- Geyer, C.J. (1992). Practical Markov Chain Monte Carlo. *Statist. Sci.* 7:473–483 (ESS/τ_int).
- Vehtari, A. et al. (2021). *Bayesian Anal.* 16(2):667–718 (ESS, R̂).
