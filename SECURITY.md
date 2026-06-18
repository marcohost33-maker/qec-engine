# Security Policy

## Supported versions

The AdaptiveRG-QEC Engine is research software in early development (Phase-1 MVP,
positioned as a diagnostics/verification harness). Only the most recent commit on
`main` receives fixes. There is no LTS branch and no backport policy.

| Version    | Supported          |
| ---------- | ------------------ |
| 0.1.x.dev  | :white_check_mark: |
| < 0.1      | :x:                |

## Reporting a vulnerability

If you believe you have found a security-relevant issue, **please do not open a public issue**.

Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/marcohost33-maker/qec-engine/security) of this repo
2. Click **Report a vulnerability**
3. Provide:
   - A description of the issue
   - Steps to reproduce
   - The affected commit SHA
   - Suggested mitigation if you have one

We aim to acknowledge the report within **7 working days** and to either ship a patch or publish
an advisory within **30 days** of the initial report. If you do not receive a response, please
escalate through the GitHub profile contact information.

## Scope

In scope:

- Code under `src/adaptiverg_qec/`
- CI workflows under `.github/workflows/`
- The selftest / gate-log writers under `results/`

Out of scope:

- Upstream NumPy / SciPy / Python interpreter vulnerabilities (report to those projects directly)
- Findings that require physical access to the machine running the engine
- Issues in third-party forks
- Numerical-stability or convergence bugs that do not constitute a security vulnerability
  (open a normal issue — these belong to the diagnostics layer, not the threat surface)

## Supply-chain hardening

This repo follows the OpenSSF SHA-pinning recommendation: every external GitHub Action in
`.github/workflows/` is pinned to a 40-character commit SHA, with a `# vX.Y.Z` comment indicating
the semantic version. A `zizmor` workflow audit gates CI against workflow-security regressions.

Python dependencies are intentionally minimal (`numpy`, `scipy`); transitive supply-chain risk is
audited via Dependabot updates (`.github/dependabot.yml`). The MCMC RNG is seeded deterministically
(NumPy Philox) for reproducibility; no cryptographic or security guarantee is claimed for the
sampler — it is a scientific Monte-Carlo engine, not a CSPRNG.
