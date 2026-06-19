# QEC-Diagnostik-Harness — Roadmap (Repositioning 2026-06-18)

> Ehrliche Positionierung. Dieser Harness ist KEIN SOTA-Threshold-Tool. SOTA fuer
> Praezisions-Schwellen ist Tensor-Network/MPS-Kontraktion des statistical-mechanics
> mapping (Bravyi, Suchara & Vargo, *Phys. Rev. A* 90, 032326, 2014). Ziel hier ist
> ein **ehrliches Diagnose-/Vergleichs-Geruest**: jede Decoder-/Code-Variante wird
> gegen ein unabhaengiges Orakel und gegen eine Baseline gemessen, reproduzierbar.

## Warum dieser Harness (Gap-Analyse)

Das bestehende Repo (`a_kernel`, `mcrg*`, `ising2d`, `wolff2d`) ist ein MCMC/MCRG-
Harness fuer **kritische Exponenten** (statistische Mechanik). Es enthielt **keine**
eigentliche Fehlerkorrektur-Diagnostik: keinen Decoder, keine logische Fehlerrate,
keinen Sweep ueber die physikalische Fehlerrate. Dieser Harness fuellt genau diese
Luecke, additiv und ohne Bestehendes anzufassen.

## Inkrement 1 (DIESE PR) — Repetition-Code, exakt orakelbar  [DONE]

- Modul `src/adaptiverg_qec/qec_diagnostics.py`:
  - `logical_error_rate_exact(n, p)` — geschlossene Binomialsumme (Orakel).
  - `simulate_logical_error_rate(n, p, shots, seed)` — vektorisierte Monte-Carlo,
    Majority-Vote-Decoder, Binomial-Standardfehler + `n_sigma` gegen das Orakel.
  - `pseudo_threshold(n_small, n_large)` — exaktes `p* = 1/2` (Kurven-Schnittpunkt).
  - `run_diagnostic_sweep(...)` + `python -m adaptiverg_qec.qec_diagnostics`
    schreibt `results/qec-diagnostics-rep-code.json`.
- **Metrik (real ausgefuehrt):** logische Fehlerrate `p_L` vs. physikalische `p`
  fuer `n in {1,3,5,7}`, `p in {0.02..0.6}`, 200 000 shots/Zelle, seed=2026.
  MC trifft das exakte Binomial-Orakel ueber alle 32 Zellen (worst `n_sigma = 2.71`).
  Threshold-Verhalten sichtbar: fuer `p<1/2` faellt `p_L` mit groesserem `n`,
  fuer `p>1/2` steigt es; alle Kurven schneiden sich bei `p*=1/2`.
- Tests `tests/test_qec_diagnostics.py` (Orakel == Brute-Force, Reproduzierbarkeit,
  Silent-Failure-Gate fuer gerade/negative `n`, `p` ausserhalb [0,1], `inf/nan`,
  nicht-ganzes `n`, `shots<1`).
- **Korrektheits-Beleg (web-Falle gefangen):** die kursierende Formel
  `p_L = p^3 + 2 p^2 (1-p)` fuer `n=3` ist FALSCH (ergibt 0.019 bei p=0.1); korrekt
  `3 p^2 - 2 p^3 = 0.028`. Genau deshalb wird gegen das unabhaengig hergeleitete
  Orakel geprueft, nicht gegen eine Literatur-Formel. Test `test_n3_specific_value_p_0_1`.

## Inkrement 2 (DIESE PR) — Fit-Diagnostik gegen geschlossene Orakel  [DONE]

> **Scope-Schaerfung gegenueber der urspruenglichen Skizze (ehrlich, mit Grund).**
> Die Skizze plante zusaetzlich einen **hand-gerollten MWPM-Spacetime-Decoder** fuer
> phenomenological noise. Ein Prototyp davon wurde gebaut (scipy `linear_sum_assignment`
> als exaktes Matching) und **verworfen**: er verliess fuer `d=5` in ~2–6 % der Shots den
> Codespace (Residuum mit nicht-trivialem Syndrom) = Decoder-Bug in der Matching->
> Korrektur-Rekonstruktion. Das Codie-Verbot „KEIN Eigen-Decoder neu erfinden, wo
> etablierte Tools existieren" greift genau hier: ein korrekter Spacetime-Decoder ist
> Aufgabe von **PyMatching** (Inkr. 3, optional-Gate). Inkr. 2 liefert daher den
> vollstaendig **orakelbaren** Teil — die Fit-Diagnostik — ohne Decoder-Erfindung.

- Modul `src/adaptiverg_qec/qec_fit_diagnostics.py` (baut auf Inkr.1-Orakel auf):
  - `fit_distance_exponent(d)` — sub-threshold log-log-Steigung `p_L`-vs-`p` →
    empirischer Code-Distanz-Exponent, Orakel `(d+1)/2`.
  - `pseudo_threshold_bisection(d1,d2)` — Kurven-Kreuzungs-Bisektion → empirischer
    Pseudo-Threshold, Orakel analytisch `p* = 1/2` (Symmetrie `p_L(d,1/2)=1/2`).
  - `lambda_suppression(d,p)` — Fehler-Unterdrueckungs-Faktor `p_L(d)/p_L(d+2)`
    (Google-„below threshold"-Kennzahl), small-p-Orakel `(A_d/A_{d+2})/p`,
    `A_k = C(k,(k+1)/2)`.
  - `run_fit_diagnostics()` + `python -m adaptiverg_qec.qec_fit_diagnostics` →
    `results/qec-fit-diagnostics-rep-code.json`.
- **Metrik (real ausgefuehrt):** Exponent `d∈{3,5,7,9}` worst `abs_err=7.2e-4`;
  Pseudo-Threshold worst `abs_err=2.2e-16`; Lambda `≈3.27/3.14/3.06` (alle >1).
- **Cross-Anchor-Orakel (fuer Inkr.3 vorgesehen):** ein ML-Brute-Force-Decoder als
  unabhaengiger Gegen-Check zum Binomial-Orakel. Der eigene Prototyp wurde verworfen
  (verliess fuer `d=5` den Codespace) und kommt mit PyMatching in Inkr.3 zurueck —
  bis dahin wird **keine** Brute-Force-Vergleichszahl behauptet.
- Tests `tests/test_qec_fit_diagnostics.py` (Orakel-Vergleiche je Kennzahl +
  Silent-Failure-Gate: gerade/negative/nicht-int `d`, p_probe nicht aufsteigend/
  nicht-sub-threshold, Bisektions-Klammer ohne 0.5, Lambda-p ausserhalb (0,0.5), NaN).

## Inkrement 3 (DIESE PR) — ECHTES MWPM via Stim + PyMatching  [DONE]

- Modul `src/adaptiverg_qec/surface_decoder.py`. Decoder = **PyMatching 2** (MWPM),
  Sampling/Schaltkreis = **Stim** — KEIN Eigen-Decoder. HINTER optional-dependency-Gate
  `pip install ".[surface]"` (NICHT Kern-Dependency). Ohne das Extra importiert das Modul
  nicht und seine Tests SKIPPEN (`pytest.mark.skipif`), sie failen NICHT hart — lokal in
  einem dev-only-venv verifiziert (3 passed, 30 skipped, exit 0).
- **Absorbiert den aus Inkr.2 ausgelagerten Decoder:** der hand-gerollte Prototyp
  (Decoder-Bug, verließ den Codespace) ist durch PyMatching 2 ersetzt.
- **Orakel A (real ausgefuehrt):** Repetition-Code, code-capacity bit-flip, MWPM aus der
  Paritaets-Pruefmatrix. Auf dem 1D-Matching-Graphen IST MWPM ML-optimal -> MWPM-Monte-Carlo
  MUSS das exakte Binomial-Orakel (Inkr.1) treffen. `d∈{3,5,7}` @ p=0.10, 200 000 shots,
  seed=11 -> worst `n_sigma = 1.55` (rein statistische Uebereinstimmung). Das ist der von
  Inkr.2 zurueckgestellte unabhaengige Cross-Anchor-Decoder.
- **Orakel B (real ausgefuehrt):** Rotated-Surface-Code, code-capacity pure-X-flip
  (X_ERROR(p) nur auf Daten-Qubits, perfekte Messung, 1 Runde), MWPM via Stim-DEM +
  PyMatching. Threshold-Schaetzer = Distanz-Kurven-Kreuzung. 60 000 shots/Zelle, seed=11:
  `(7,9)->0.0952`, `(9,11)->0.1015`, `(7,11)->0.0978`; **best pair (9,11): `p_th = 0.1015`
  vs publiziert 0.103 -> `abs_err = 0.0015`**.
- **Korrektheits-Disziplin (ehrlich, kein Overclaim):** validiert gegen den MWPM-Wert
  **0.103** (Dennis, Kitaev, Landahl & Preskill, J. Math. Phys. 43, 4452 (2002) =
  Nulltemperatur-RBIM), BEWUSST NICHT gegen den optimalen ML/Tensor-Network-Threshold
  **0.1094** (Nishimori-Punkt 2D-RBIM). MWPM ist near-optimal aber sub-optimal; ein
  Schaetzer, der 0.109 „erreicht", waere verdaechtig. Endliche Distanzen -> keine L->oo-FSS;
  der Kreuzungs-Schaetzer driftet von unten zum Threshold (sichtbar: (7,9)<(9,11)).
- `results/qec-surface-mwpm.json` (`python -m adaptiverg_qec.surface_decoder`). Tests
  `tests/test_surface_decoder.py` (Orakel A/B + Threshold-Verhalten + Silent-Failure-Gate
  + optional-dep-Gate-Verhalten).
- **Offen (naechster Schritt):** phenomenological noise (Mess-Fehler, mehrere Runden;
  publizierter MWPM-Threshold ~2.9%, Wang, Harrington & Preskill 2003) ist NICHT
  implementiert. Depolarisierendes Rauschen (statt pure-X) waere ein weiterer Schritt
  (anderer Threshold). CI laeuft weiterhin nur mit `[dev]` -> Surface-Tests skippen dort;
  ein separater optionaler CI-Job mit `[surface]` ist vorbereitet (s. PR).

## Inkrement 4 — Brueckenschlag zum MCRG-Teil (warum beide im selben Repo)

- Statistical-mechanics mapping: der Surface-Code-Threshold == Phasenuebergang des
  Random-Bond-Ising-Modells (Nishimori-Linie). Die bestehende MCRG-Maschinerie
  (`ising2d`, `mcrg_matrix`, `wolff2d`) kann den **kritischen Punkt** liefern, den
  der Diagnostik-Harness als unabhaengiges Orakel fuer den Decoder-Threshold nutzt.
  Das ist die ehrliche Kohaerenz-Geschichte, die die zwei Repo-Haelften verbindet.

## Akzeptanz / Definition-of-Done je Inkrement

- Lauffaehiger Code + `results/`-Evidenz-Log (AGENTS.md-Pflicht).
- Jede Zahl gegen ein **unabhaengiges** Orakel (geschlossen, Brute-Force oder Literatur).
- Negativ-/Edge-Input-Gates (Silent-Failure-Gate) IM Bau, nicht nachgereicht.
- Reproduzierbar (fixe Seeds). Keine neuen Kern-Dependencies ohne optional-Gate.
- Equalita verifiziert; Marco entscheidet Promotion. Kein Self-Cert.

---
*Coworker Research | QEC-Diagnostik-Harness | Inkrement 1+2 von N | 2026-06-19*
