"""Issue #51: Freeze-Vertrag des A-Kernels (Gates G7a-G7d + Regression der Fehlerklasse).

Pipeline: WARM-UP (adaptiv, Kalibration) -> FREEZE (beta := beta_star exakt)
-> FIXED-TARGET BURN-IN -> PRODUCTION. Orakel:
- geschlossenes Produkt prod_{n>=1} (1 - 1/(2 n^2)) = sin(pi/sqrt2)/(pi/sqrt2)
  (Euler-Produkt des Sinus, sin(pi x)/(pi x) = prod (1 - x^2/n^2) mit x = 1/sqrt2);
- Transfer-Matrix-<H> (ising1d.mean_energy) bei beta_star vs beim eingefrorenen
  falschen Ziel.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from adaptiverg_qec import a_kernel, checkpoint, ising1d, manifest
from adaptiverg_qec.mvp_instance import MVPConfig

CFG = MVPConfig(L=16, beta_min=0.1, beta_max=2.0)

# Issue-#51-Beispiel: c = 0.5, T0 = 1, beta_0 = 0.1, beta_star = 0.8.
_X = math.pi / math.sqrt(2.0)
PROD_CLOSED = math.sin(_X) / _X
BETA_INF = 0.8 - 0.7 * PROD_CLOSED


def _summable_run(n_warm: int, n_steps: int, burn_in: int, seed: int = 51):
    return a_kernel.run_adaptive_mcmc(
        CFG,
        beta_target=0.8,
        n_steps=n_steps,
        burn_in=burn_in,
        seed=seed,
        beta_start=0.1,
        adapt_c=0.5,
        adapt_T0=1.0,
        warmup_steps=n_warm,
    )


# --- Regression der Fehlerklasse (analytisch gepinnt) -------------------------


def test_closed_form_product_and_beta_inf_pinned() -> None:
    """prod(1 - a_t) des Plans konvergiert gegen sin(pi/sqrt2)/(pi/sqrt2) ~ 0.358."""
    n = 2_000_000
    a = a_kernel.diminishing_step_sizes(n, c=0.5, T0=1.0)
    np.testing.assert_array_equal(a[:3], [0.5, 0.125, 0.5 / 9.0])  # a_t = 1/(2 (t+1)^2)
    log_p = float(np.sum(np.log1p(-a)))
    p_n = math.exp(log_p)
    # Rest: P = P_N prod_{n>N}(1 - 1/(2n^2)) und 0 <= P_N - P <= P_N / (2N).
    assert 0.0 <= p_n - PROD_CLOSED <= p_n / (2 * n) + 1e-12
    assert pytest.approx(0.358187786, abs=1e-9) == PROD_CLOSED
    assert pytest.approx(0.549268550, abs=1e-9) == BETA_INF


def test_summable_warmup_freezes_at_the_wrong_target() -> None:
    """Ohne Freeze stuende die Kette bei ~0.549 statt 0.8 (Fehlerklasse belegt)."""
    n_warm = 4000
    r = _summable_run(n_warm, n_steps=n_warm + 200, burn_in=50)
    assert r.calibration is not None
    end = r.calibration.beta_end
    # |beta_N - beta_inf| = 0.7 (P_N - P) <= 0.7 P_N / (2N) < 0.7 / (2N)
    assert abs(end - BETA_INF) <= 0.7 / (2 * n_warm)
    assert abs(end - 0.8) > 0.25
    # Die Iteration selbst ist monoton und erreicht 0.8 nie.
    assert np.all(np.diff(r.calibration.beta) >= 0.0)
    assert np.all(r.calibration.beta < 0.8)


def test_frozen_pipeline_samples_beta_star_not_beta_inf() -> None:
    """Diskriminierend auf Verteilungsebene: <H> der Produktion trifft E_{0.8}, nicht E_{0.549}."""
    r = _summable_run(2000, n_steps=2000 + 500 + 5000, burn_in=500, seed=7)
    e_star = ising1d.mean_energy(0.8, CFG.L)
    e_wrong = ising1d.mean_energy(BETA_INF, CFG.L)
    assert e_wrong - e_star > 0.8  # Orakel-Abstand (Transfer-Matrix)
    assert abs(r.mean_H - e_star) < 0.15
    assert abs(r.mean_H - e_wrong) > 0.6


# --- G7b: Freeze bit-/float-exakt --------------------------------------------


@pytest.mark.parametrize(
    ("beta_start", "beta_star"), [(0.3, 0.9), (0.2, 1.0), (0.1, 0.8), (1.9, 0.35)]
)
def test_g7b_freeze_sets_target_exactly(beta_start: float, beta_star: float) -> None:
    n_warm = 600
    r = a_kernel.run_adaptive_mcmc(
        CFG,
        beta_target=beta_star,
        n_steps=n_warm + 300,
        burn_in=100,
        seed=3,
        beta_start=beta_start,
        warmup_steps=n_warm,
    )
    # Die Iteration allein kommt nicht bit-genau an (ulp-Stau bzw. falsches Ziel) ...
    assert r.beta_traj[n_warm - 1] != beta_star
    # ... der Freeze setzt es per Zuweisung: Gleichheit, keine Naehe.
    assert r.beta_traj[n_warm] == beta_star
    assert np.all(r.beta_traj[n_warm:] == beta_star)
    assert r.production is not None
    assert np.all(r.production.beta == beta_star)
    assert r.production.beta_star == beta_star


def test_without_warmup_beta_is_exact_from_the_first_sweep() -> None:
    r = a_kernel.run_adaptive_mcmc(CFG, beta_target=0.9, n_steps=50, burn_in=10, seed=1)
    assert np.all(r.beta_traj == 0.9)
    assert r.adaptation_sum == 0.0
    assert r.calibration is not None and r.calibration.H.size == 0


# --- G7a: Kalibration nie Produktion -----------------------------------------


def test_g7a_calibration_and_production_are_separate_records() -> None:
    w, b, n = 150, 100, 600
    r = a_kernel.run_adaptive_mcmc(
        CFG,
        beta_target=1.0,
        n_steps=n,
        burn_in=b,
        seed=11,
        beta_start=0.2,
        warmup_steps=w,
        record_configs=True,
    )
    cal, prod = r.calibration, r.production
    assert cal is not None and prod is not None
    assert prod.t_start == w + b
    assert prod.H.size == n - w - b and cal.H.size == w
    np.testing.assert_array_equal(cal.H, r.H_traj[:w])
    np.testing.assert_array_equal(prod.H, r.H_traj[w + b :])
    np.testing.assert_array_equal(prod.configs, r.configs[w + b :])
    for x, y in [
        (cal.H, prod.H),
        (prod.H, r.H_traj),
        (cal.H, r.H_traj),
        (prod.beta, r.beta_traj),
        (prod.configs, r.configs),
    ]:
        assert not np.shares_memory(x, y)
    assert r.mean_H == float(np.mean(r.H_traj[w + b :]))
    # Ein Schreibzugriff auf die Gesamt-Trajektorie erreicht den Produktions-Record nicht.
    r.H_traj[w + b] = -1.0
    assert prod.H[0] != -1.0


# --- G7c: Burn-in-Laenge im Manifest vor dem Lauf ------------------------------


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_g7c_manifest_must_declare_warmup_and_burn_in(tmp_path) -> None:
    mf = manifest.RunManifest(
        base_seed=3, n_chains=2, n_steps=400, burn_in=60, warmup_steps=40, beta_start=0.2, L=16
    )
    p = manifest.write_manifest(mf, tmp_path / "m.json")
    full = json.loads(p.read_text(encoding="utf-8"))
    assert manifest.load_manifest(p) == manifest.RunManifest(**full)
    for key in ("burn_in", "warmup_steps", "beta_start"):
        d = dict(full)
        del d[key]
        _write(tmp_path / f"no_{key}.json", d)
        with pytest.raises(ValueError, match="lacks run parameters"):
            manifest.load_manifest(tmp_path / f"no_{key}.json")
    d = dict(full, schema="adaptiverg_qec.phase5.run_manifest/v1")
    _write(tmp_path / "v1.json", d)
    with pytest.raises(ValueError, match="schema mismatch"):
        manifest.load_manifest(tmp_path / "v1.json")


def test_g7c_declared_lengths_drive_the_run() -> None:
    base = dict(base_seed=5, n_chains=2, n_steps=400, L=16, beta_start=0.2)
    mf = manifest.RunManifest(burn_in=60, warmup_steps=40, **base)
    assert mf.n_production == 300
    assert manifest._multichain_H(mf).shape == (2, 300)
    a = manifest.run(mf)
    b = manifest.run(manifest.RunManifest(burn_in=61, warmup_steps=40, **base))
    c = manifest.run(manifest.RunManifest(burn_in=60, warmup_steps=80, **base))
    # (burn_in=60, warmup=41) waere KEIN guter Gegenfall: gleiches Produktionsfenster
    # [101, 400) wie b, und die Metropolis-Entscheide eines einzelnen Sweeps bei
    # leicht anderem beta sind mit hoher Wahrscheinlichkeit dieselben (Kopplung).
    assert len({a.result_hash, b.result_hash, c.result_hash}) == 3


@pytest.mark.parametrize(
    "kw",
    [
        dict(warmup_steps=-1, beta_start=0.2),
        dict(warmup_steps=True, beta_start=0.2),
        dict(warmup_steps=1.0, beta_start=0.2),
        dict(warmup_steps=300, burn_in=100, beta_start=0.2),  # 300 + 100 >= n_steps
        dict(warmup_steps=10),  # Warm-up ohne Start-beta
        dict(beta_start=0.2),  # Start-beta ohne Warm-up (wuerde still ignoriert)
        dict(beta_start=float("nan"), warmup_steps=10),
        dict(schema="adaptiverg_qec.phase5.run_manifest/v1"),
    ],
)
def test_manifest_rejects_invalid_freeze_parameters(kw) -> None:
    base = dict(n_chains=2, n_steps=400, burn_in=50, L=16)
    base.update(kw)
    with pytest.raises(ValueError):
        manifest.RunManifest(**base)


# --- G7d: kein Pre-Freeze-Sample in einem Produktionsschaetzer -----------------


def _frozen_record():
    r = _summable_run(300, n_steps=600, burn_in=50, seed=2)
    assert r.calibration is not None and r.production is not None
    return r


def test_g7d_negative_control_injected_prefreeze_sample_fails() -> None:
    r = _frozen_record()
    a_kernel.production_mean(r.production)  # Kontrolle: der echte Record geht durch
    injected = a_kernel.ProductionRecord(
        beta_star=0.8,
        t_start=r.production.t_start - 1,
        H=np.concatenate(([r.calibration.H[-1]], r.production.H)),
        beta=np.concatenate(([r.calibration.beta_end], r.production.beta)),
    )
    with pytest.raises(a_kernel.FreezeContractError, match="not at the frozen target"):
        a_kernel.production_mean(injected)


def test_g7d_one_ulp_off_is_rejected_after_construction() -> None:
    """Mutation nach dem Bau wird erkannt (require_frozen prueft bei jedem Aufruf)."""
    r = _frozen_record()
    rec = r.production
    rec.beta[-1] = np.nextafter(0.8, 0.0)  # 1 ulp unter beta_star
    with pytest.raises(a_kernel.FreezeContractError):
        a_kernel.production_mean(rec)


@pytest.mark.parametrize(
    "bad",
    [
        np.arange(5.0),  # kein ProductionRecord
        a_kernel.ProductionRecord(beta_star=0.8, t_start=0, H=np.array([]), beta=np.array([])),
        a_kernel.ProductionRecord(
            beta_star=0.8, t_start=0, H=np.array([1.0, np.nan]), beta=np.array([0.8, 0.8])
        ),
        a_kernel.ProductionRecord(
            beta_star=float("nan"), t_start=0, H=np.array([1.0]), beta=np.array([np.nan])
        ),
        a_kernel.ProductionRecord(
            beta_star=0.8, t_start=0, H=np.array([1.0, 2.0]), beta=np.array([0.8])
        ),
        a_kernel.ProductionRecord(
            beta_star=0.8,
            t_start=0,
            H=np.array([1.0, 2.0]),
            beta=np.array([0.8, 0.8]),
            configs=np.zeros((3, 4), dtype=np.int8),
        ),
    ],
)
def test_g7d_estimator_rejects_malformed_records(bad) -> None:
    with pytest.raises(a_kernel.FreezeContractError):
        a_kernel.production_mean(bad)


def test_advance_chain_freezes_by_assignment() -> None:
    """Primitive ohne Record-Schicht: vor Sweep t=freeze_at ist beta exakt beta_star."""
    state, rng = a_kernel.new_chain_state(CFG, seed=1, beta_start=0.2)
    a_t = a_kernel.diminishing_step_sizes(60, 0.5, 100.0)
    H, B = np.empty(60), np.empty(60)
    a_kernel.advance_chain(
        state, rng, CFG, beta_target=1.0, a_t=a_t, t_stop=60, H_out=H, beta_out=B, freeze_at=25
    )
    assert np.all(B[:25] < 1.0)
    assert np.all(np.diff(B[:25]) > 0.0)
    assert B[25] == 1.0
    assert np.all(B[25:] == 1.0)
    assert state.beta == 1.0


def test_split_phases_window_is_warmup_plus_burn_in() -> None:
    """Produktion beginnt bei warmup_steps + burn_in, nicht bei burn_in (G7a/G7d)."""
    H = np.arange(50.0)
    beta = np.full(50, 0.8)
    beta[:10] = np.linspace(0.5, 0.79, 10)  # Warm-up
    cal, prod = a_kernel.split_phases(H, beta, warmup_steps=10, burn_in=10, beta_star=0.8)
    assert prod.t_start == 20
    np.testing.assert_array_equal(prod.H, H[20:])
    np.testing.assert_array_equal(cal.H, H[:10])


def test_split_phases_refuses_unfrozen_production() -> None:
    """Eine Trajektorie mit wanderndem Ziel ergibt keinen Produktions-Record."""
    beta = np.linspace(0.5, 0.8, 50)
    with pytest.raises(a_kernel.FreezeContractError):
        a_kernel.split_phases(np.ones(50), beta, warmup_steps=10, burn_in=10, beta_star=0.8)


def test_advance_chain_rejects_unfrozen_state_past_freeze() -> None:
    """Resume eines Zustands hinter dem Freeze mit falschem beta -> fail-closed."""
    state, rng = a_kernel.new_chain_state(CFG, seed=1, beta_start=0.5)
    state.t = 20
    a_t = a_kernel.diminishing_step_sizes(40, 0.5, 100.0)
    H, B = np.empty(40), np.empty(40)
    with pytest.raises(a_kernel.FreezeContractError):
        a_kernel.advance_chain(
            state, rng, CFG, beta_target=0.8, a_t=a_t, t_stop=40, H_out=H, beta_out=B, freeze_at=10
        )


@pytest.mark.parametrize("bad", [-1, True, 2.0])
def test_advance_chain_rejects_bad_freeze_at(bad) -> None:
    state, rng = a_kernel.new_chain_state(CFG, seed=1, beta_start=0.5)
    a_t = a_kernel.diminishing_step_sizes(10, 0.5, 100.0)
    with pytest.raises(ValueError):
        a_kernel.advance_chain(
            state,
            rng,
            CFG,
            beta_target=0.8,
            a_t=a_t,
            t_stop=10,
            H_out=np.empty(10),
            beta_out=np.empty(10),
            freeze_at=bad,
        )


@pytest.mark.parametrize(
    "kw",
    [
        dict(beta_start=0.2),  # Start-beta ohne Warm-up
        dict(warmup_steps=5),  # Warm-up ohne Start-beta
        dict(warmup_steps=-1, beta_start=0.2),
        dict(warmup_steps=True, beta_start=0.2),
        dict(warmup_steps=90, burn_in=10, beta_start=0.2),  # 90 + 10 >= 100
    ],
)
def test_run_rejects_invalid_freeze_parameters(kw) -> None:
    base = dict(beta_target=0.8, n_steps=100, burn_in=0, seed=1)
    base.update(kw)
    with pytest.raises(ValueError):
        a_kernel.run_adaptive_mcmc(CFG, **base)


# --- Checkpoint: Freeze ueber Interrupt/Resume bit-identisch --------------------


@pytest.mark.parametrize("interrupt_after", [25, 70, 330])
def test_resume_across_warmup_and_freeze_is_byte_identical(tmp_path, interrupt_after) -> None:
    mf = manifest.RunManifest(
        base_seed=4711, n_chains=2, n_steps=300, burn_in=40, warmup_steps=50, beta_start=0.3, L=16
    )
    direct = manifest.run(mf)
    p = tmp_path / "ck.json"
    r1 = checkpoint.run_resumable(mf, p, checkpoint_every=20, interrupt_after=interrupt_after)
    assert r1 is None
    r2 = checkpoint.resume(p, checkpoint_every=20)
    assert r2 is not None
    assert r2.result_hash == direct.result_hash
