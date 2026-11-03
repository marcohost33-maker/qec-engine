"""Inkrement 3.4: vorregistrierter Sweep-Plan (ohne Extras) + Stim-Pfad (mit [surface])."""

from __future__ import annotations

import json

import pytest

from adaptiverg_qec import qec_sweep as qs
from adaptiverg_qec import surface_decoder as sd

requires_surface = pytest.mark.skipif(
    not sd.HAVE_SURFACE, reason="needs the [surface] extras (stim + pymatching)"
)


def test_plan_roundtrip_and_fingerprint_stable():
    plan = qs.SweepPlan()
    again = qs.SweepPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again == plan and again.fingerprint() == plan.fingerprint()


def test_cell_seeds_unique_and_bound_to_plan():
    plan = qs.SweepPlan()
    seeds = {plan.cell_seed(d, p) for d in plan.distances for p in plan.p_grid}
    assert len(seeds) == len(plan.distances) * len(plan.p_grid)
    other = qs.SweepPlan(shots_per_cell=plan.shots_per_cell + 1)
    assert other.cell_seed(5, 0.024) != plan.cell_seed(5, 0.024)
    with pytest.raises(ValueError):
        plan.cell_seed(13, 0.024)  # nicht registriert
    with pytest.raises(ValueError):
        plan.cell_seed(5, 0.025)


@pytest.mark.parametrize(
    "kw",
    [
        {"distances": (5,)},
        {"distances": (7, 5)},
        {"distances": (4, 6)},
        {"p_grid": (0.01, 0.02)},
        {"p_grid": (0.03, 0.02, 0.04)},
        {"p_grid": (0.0, 0.02, 0.04)},
        {"p_grid": (0.01, 0.02, float("nan"))},
        {"shots_per_cell": 0},
        {"stopping_rule": "sequential"},
        {"noise_model": "circuit"},
        {"n_boot": 10},
        {"base_seed": -1},
        {"alpha": 1.0},
        {"distances": (5, 7), "p_grid": (0.01, 0.02)},
    ],
)
def test_plan_validation_fail_closed(kw):
    with pytest.raises((TypeError, ValueError)):
        qs.SweepPlan(**kw)


def test_from_dict_requires_exact_keys():
    d = qs.SweepPlan().to_dict()
    with pytest.raises(ValueError):
        qs.SweepPlan.from_dict({k: v for k, v in d.items() if k != "shots_per_cell"})
    with pytest.raises(ValueError):
        qs.SweepPlan.from_dict({**d, "sneaky": 1})


def _fake_result(plan: qs.SweepPlan) -> dict:
    rows = [
        {"d": d, "p": p, "seed": plan.cell_seed(d, p), "k": 1, "n": plan.shots_per_cell}
        for d in plan.distances
        for p in plan.p_grid
    ]
    return {"plan": plan.to_dict(), "plan_fingerprint": plan.fingerprint(), "rows": rows}


def test_verify_result_detects_post_hoc_edits():
    plan = qs.SweepPlan()
    good = _fake_result(plan)
    assert qs.verify_result(good) == []

    edited = json.loads(json.dumps(good))
    edited["plan"]["p_grid"] = edited["plan"]["p_grid"][1:]
    assert qs.verify_result(edited)  # Fingerprint + Zellmenge

    dropped = json.loads(json.dumps(good))
    dropped["rows"] = dropped["rows"][1:]
    assert any("missing" in m for m in qs.verify_result(dropped))

    shots = json.loads(json.dumps(good))
    shots["rows"][0]["n"] += 1
    assert any("n=" in m for m in qs.verify_result(shots))

    seed = json.loads(json.dumps(good))
    seed["rows"][0]["seed"] += 1
    assert any("seed" in m for m in qs.verify_result(seed))

    assert qs.verify_result({"plan": {}})  # nicht rekonstruierbar


def test_rows_hash_ignores_seed_field():
    plan = qs.SweepPlan()
    a = _fake_result(plan)["rows"]
    b = [{**r, "seed": 0} for r in a]
    assert qs.rows_hash(a) == qs.rows_hash(b)
    c = [{**a[0], "k": 2}, *a[1:]]
    assert qs.rows_hash(a) != qs.rows_hash(c)


@requires_surface
def test_bitflip_circuit_has_no_depolarizing_noise():
    c = qs.build_circuit("phenomenological-bitflip", "z", 5, 0.03)
    text = str(c)
    assert "DEPOLARIZE" not in text
    assert "X_ERROR(0.03)" in text
    assert "DEPOLARIZE1(0.03)" in str(qs.build_circuit("stim-depolarizing", "z", 5, 0.03))


@requires_surface
@pytest.mark.parametrize("basis", ["x", "z"])
def test_zero_noise_oracle(basis):
    assert qs.decode_cell("phenomenological-bitflip", basis, 5, 1e-12, 500, 1) == 0


@requires_surface
def test_small_sweep_reproducible_and_verifiable():
    plan = qs.SweepPlan(
        distances=(3, 5), p_grid=(0.02, 0.03, 0.04), shots_per_cell=1_000, n_boot=50
    )
    a = qs.run_sweep(plan)
    b = qs.run_sweep(qs.SweepPlan.from_dict(json.loads(json.dumps(plan.to_dict()))))
    assert qs.rows_hash(a["rows"]) == qs.rows_hash(b["rows"])
    assert qs.verify_result(a) == []
    assert a["fss"]["n_boot"] == 50
