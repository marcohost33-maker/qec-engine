"""AdaptiveRG-QEC — Phase-1 MVP (Diagnostik-/Verifikations-Harness).

Bounded MVP-Implementierung der gehaerteten Kernel-Spec v1.0 (spec/):
- A-Kernel: adaptiver MCMC-Sampler auf augmentiertem Zustand z=(x, theta)
  mit Foster-Lyapunov-Drift-Guard.
- C-Kernel: minimale MCRG-/RG-Map + Jacobian (Complex-Step / FD) + Exponenten.
- Phase-3a: Swendsen-T_hat aus dem korrelierten A-Kernel + autokorrelations-bewusste
  Fehler (autocorr.py: FFT-rho, tau_int + Wolff-Windowing, Binning, Block-Jackknife).
- Phase-3b: Multi-Operator-Swendsen-MATRIX auf 2D-Ising (ising2d.py: vektorisierter
  Checkerboard-Metropolis + Majority-Rule-Blocking b=2; mcrg_matrix.py: gerade
  Operatoren, connected-corr-Matrizen A,B, T=A.B^-1, Eigenwert-Exponent y_t vs
  Onsager-Orakel y_t=1 -- ehrlich GROB, kein Frontier-Wert).

EHRLICHER STATUS: Diagnostik- und Verifikations-Harness mit rigorosen
Konvergenz-Guards, KEIN Frontier-Threshold-Tool. SOTA fuer Praezisions-
Thresholds = Tensor-Network/MPS (Bravyi-Suchara-Vargo 2014). Siehe README.

Die MVP-Instanz (Code/V/g) ist explizit in `mvp_instance.py` dokumentiert und
als MVP-WAHL gekennzeichnet, nicht als die volle Spec.
"""

# EINZIGE Versionsquelle (PEP 440). pyproject.toml liest dieses Literal statisch
# (`[tool.setuptools.dynamic] version = { attr = ... }`), die installierte
# Distribution traegt also dieselbe Nummer wie die Laufzeit-Evidenz
# (cli selftest "version", manifest/runtime "package_version"). Ein reines
# String-Literal bleiben lassen: setuptools liest es per AST, ohne zu importieren.
# Gepinnt durch tests/test_version_ssot.py. dev-Reife, NICHT release-fertig.
# 0.5.0.dev0 statt 0.1.0.dev2 (2026-10-05): ein Entwicklungsstand nach dem Tag
# v0.4.0 muss nach PEP 440 darueber liegen; 0.1.0.dev2 < 0.4.0 ordnete falsch.
# Aeltere results/*.json tragen weiterhin 0.1.0.dev2 (historische Evidenz).
__version__ = "0.5.0.dev0"

__all__ = ["__version__"]
