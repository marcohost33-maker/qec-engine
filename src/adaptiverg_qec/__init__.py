"""AdaptiveRG-QEC — Phase-1 MVP (Diagnostik-/Verifikations-Harness).

Bounded MVP-Implementierung der gehaerteten Kernel-Spec v1.0 (spec/):
- A-Kernel: adaptiver MCMC-Sampler auf augmentiertem Zustand z=(x, theta)
  mit Foster-Lyapunov-Drift-Guard.
- C-Kernel: minimale MCRG-/RG-Map + Jacobian (Complex-Step / FD) + Exponenten.
- Phase-3a: Swendsen-T_hat aus dem korrelierten A-Kernel + autokorrelations-bewusste
  Fehler (autocorr.py: FFT-rho, tau_int + Wolff-Windowing, Binning, Block-Jackknife).

EHRLICHER STATUS: Diagnostik- und Verifikations-Harness mit rigorosen
Konvergenz-Guards, KEIN Frontier-Threshold-Tool. SOTA fuer Praezisions-
Thresholds = Tensor-Network/MPS (Bravyi-Suchara-Vargo 2014). Siehe README.

Die MVP-Instanz (Code/V/g) ist explizit in `mvp_instance.py` dokumentiert und
als MVP-WAHL gekennzeichnet, nicht als die volle Spec.
"""

__version__ = "0.1.0.dev1"  # PEP 440; Phase-1/2-MVP + Phase-3a (dev-Reife, NICHT release-fertig)

__all__ = ["__version__"]
