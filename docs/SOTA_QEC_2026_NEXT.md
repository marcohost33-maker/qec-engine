# QEC-Engine — SOTA-Roadmap 2026-09-19

Status: Arbeits-/Entwicklungsgrundlage, kein wissenschaftliches Promotion-Verdikt.

## 1. Zielbild

Die QEC-Engine soll sich von einem bounded Diagnose-/Verifikations-Harness zu einer reproduzierbaren
QEC-Forschungsplattform entwickeln. Der Kern bleibt: **jede Zahl braucht Modellvertrag,
unabhaengiges Orakel, Unsicherheit, Provenienz und Claim-Ceiling.**

Die Plattform wird bewusst NICHT als eigener Decoder positioniert. Decoder werden als austauschbare
Backends behandelt; die Eigenleistung liegt in Versuchsentwurf, Modellvergleich, statistischer
Validierung, Reproduzierbarkeit, RG/QEC-Bruecke und systematischer Erkennung falscher Claims.

## 2. Externe Referenzlage

### Surface Code / reale Systeme
- Google Quantum AI, *Quantum error correction below the surface code threshold*, Nature 638
  (2025; Author Correction 2026): below-threshold surface-code memories und integrierter
  real-time decoder; die Arbeit betont, dass Decoder mit dem QEC-Zyklus Schritt halten muessen.
  https://www.nature.com/articles/s41586-024-08449-y
- Google DeepMind/Quantum AI, *Learning high-accuracy error decoding for quantum processors*,
  Nature 635 (2024): AlphaQubit als lernender Surface-Code-Decoder; Genauigkeit allein genuegt
  nicht als Produktionskriterium, Skalierung/Training/Latenz bleiben eigene Achsen.
  https://www.nature.com/articles/s41586-024-08148-8
- Yang et al., *Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network
  Decoder* (2026): sub-microsecond closed-loop decoding als relevante Hardware-Referenz.
  https://arxiv.org/abs/2605.04892

### Decoder-Toolchain
- Stim: schneller stabilizer-/detector-orientierter Simulator; Circuit.generated kann
  Daten-, Mess-, Reset- und Clifford-Noise getrennt parametrisieren.
  https://github.com/quantumlib/Stim
- PyMatching 2: Sparse-Blossom-MWPM, direkte Uebernahme von Stim DetectorErrorModels;
  aktuelle Linie unterstuetzt correlated matching.
  https://github.com/oscarhiggott/PyMatching

### qLDPC / neue Decoder
- Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum memory*, Nature 627
  (2024): qLDPC als relevante Low-Overhead-Alternative zum Surface Code.
  https://www.nature.com/articles/s41586-024-07107-7
- Hillmann et al., *Localized statistics decoding for quantum low-density parity-check codes*,
  Nature Communications 16 (2025): LSD als parallelisierbarer, hardwarefreundlicher Decoder.
  https://www.nature.com/articles/s41467-025-63214-7
- Maan et al., *Decoding correlated errors in quantum LDPC codes*, Nature Communications 17
  (2026): GARI fuer korrelierte Fehler unter circuit-level noise.
  https://www.nature.com/articles/s41467-026-70556-3

## 3. Entwicklungsphasen

### Phase 7A — Research-grade Multi-Round / Phenomenological Thresholds
**Status:** begonnen mit Inkrement 3.1.

Pflicht:
1. Versioniertes `NoiseProfile`-Schema statt impliziter Parameter.
2. Direkter Stim-Pfad fuer geseedete bounded Runs; Sinter fuer skalierte p x d x rounds-Sweeps erst nach dem Reproduzierbarkeitsgate aus Issue #42.
3. Pro Zelle eigene RNG-Identitaet, kein stilles CRN; Sampling-Backend und Reproduzierbarkeits-Tier explizit ausweisen.
4. Seltene Fehler: Wilson- oder Beta-Posterior-Intervalle statt Null-Varianz bei k=0.
5. Sequential stopping: min shots + min logical failures + max shots.
6. FSS-/Crossing-Fit mit Bootstrap ueber Zellen statt Einzel-Crossing als Promotion.
7. Modellvergleich nur apples-to-apples: Kanal, Decoder, Rundenvertrag und Observable identisch.

Akzeptanz:
- NoiseProfile vollstaendig serialisiert.
- Resultat aus Manifest reproduzierbar innerhalb des deklarierten Tiers: Stim-Seed ist nur bei gleicher Stim-Version, Maschinenarchitektur und gleichem Sampling-Aufruf bitgleich; Sinter wird zunaechst als statistisch reproduzierbar behandelt.
- Nullmodell + Positivkontrolle + Literaturmodell separat.
- Threshold-Claim nur mit Finite-Size-Drift und Unsicherheitsintervall.

### Phase 7B — Circuit-Level Noise + Decoder-Linsen
1. Stim circuit-level profiles: Clifford-, Reset- und Measurement-Noise getrennt.
2. Decoder-Linsen: MWPM baseline, correlated MWPM, spaeter optional externe Learned/TN-Decoder.
3. Gleiche Samples oder explizit gepaarte Seeds fuer Decoder-A/B; absolute Kurven mit
   unabhaengigen Zellen.
4. Metriken: logical error rate, Lambda-Suppression, threshold/FSS, decoder failures,
   memory footprint und runtime.

### Phase 7C — Real-Time Decoder Benchmarking
Threshold allein reicht nicht fuer fault-tolerante Systeme.

Pflichtmetriken:
- latency p50 / p95 / p99 / max,
- throughput syndromes/s,
- batching policy,
- warm-up getrennt von steady state,
- CPU/GPU/FPGA/OS/Frequenz/Threadzahl als Environment-Fingerprint,
- Deadline-Miss-Rate gegen ein explizites QEC-cycle budget.

No-Go:
- kein real-time-Claim aus Python-Wallclock auf beliebiger Hardware;
- kein Mittelwert ohne Tail-Latenzen;
- kein Hardwarevergleich ohne Environment-Fingerprint.

### Phase 7D — qLDPC Track
Neue Plugin-Schnittstelle fuer Tanner-/Detector-Graph-basierte Codes:
- BP+OSD als etablierte Baseline,
- LSD als skalierbare/hardwarefreundliche Linse,
- korrelationsbewusste Decoder fuer circuit-level noise.

QEC-Engine bleibt primaer Benchmark-/Evidenz-Harness und erfindet nicht automatisch einen neuen Decoder.

### Phase 7E — Learned Decoder / Adaptive Calibration
Optionaler Forschungszweig:
- externer trainierter Decoder als Backend,
- Training/Test strikt getrennt,
- Cross-device/generalization split,
- calibration drift + OOD-Gates,
- accuracy-latency Pareto statt nur logical error rate.

## 4. PR — Positive Results / Best Practices

- DEM-first statt handgebauter Spacetime-Korrekturpfade.
- Decoder als austauschbares Backend.
- Exakter Noise-Vertrag als Teil jedes Resultat-Manifests.
- Zell-eigene RNG-Identitaet.
- Seltene Fehler: Jeffreys/Wilson/Beta statt Null-Varianz bei k=0.
- Full evidence lineage: source revision + dependency versions + seed policy + sampling backend + reproducibility tier + environment.
- Threshold, suppression factor und latency getrennt reporten.
- Cross-family / independent-oracle Review bei Theorie- und Threshold-Claims.
- Claim-Tiers: unit/oracle, bounded simulation, FSS-supported simulation, hardware experiment.

## 5. NR — Negative Results / verbotene Abkuerzungen

- keinen handgerollten MWPM-/Spacetime-Decoder bauen, wenn Stim/PyMatching oder etablierte
  qLDPC-Decoder die Funktion bereits korrekt abdecken;
- keine Literatur-Thresholds zwischen verschiedenen Noise-Konventionen vergleichen;
- keine einzelne Crossing-Interpolation als Threshold promoten;
- keine gemeinsam wiederverwendeten RNG-Streams als unabhaengige Evidenz behandeln;
- keine SOTA- oder real-time-Claims ohne direkte Vergleichs-/Latenzevidenz;
- keine train/test-Leakage bei Learned Decodern;
- keine qLDPC-Ausdehnung, bevor Backend-/NoiseProfile-Interface stabil ist.

## 6. Naechste konkrete Implementierungsreihenfolge

1. PR #39 gruener machen und integrieren: R-hat-Semantik + RBIM-Streams.
2. PR #40 gruener machen und integrieren: bounded Multi-Round-Phenomenological-Baseline.
3. `NoiseProfile` + `ExperimentManifest v2` spezifizieren und implementieren. **Umgesetzt in PR #41.**
4. Sinter-Reproduzierbarkeitsgate aus Issue #42 klaeren; danach Sweep-Engine mit sequential stopping.
5. Wilson/Beta-Intervalle + Bootstrap-FSS.
6. correlated-MWPM A/B-Linse.
7. Circuit-level profiles + Latency-Harness.
8. Erst danach qLDPC-Backend-Schnittstelle.

*Coworker Research / QEC-Engine | SOTA roadmap | 2026-09-19*