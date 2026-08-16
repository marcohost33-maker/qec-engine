"""Tests for explicit RBIM random-stream policies."""

from __future__ import annotations

import numpy as np
import pytest

from adaptiverg_qec.rbim_scan import derive_stream_seeds, nishimori_scan_grid


def test_independent_policy_is_reproducible_and_p_specific() -> None:
    a = derive_stream_seeds(base_seed=2026, p=0.10, L=8, replicate=3)
    b = derive_stream_seeds(base_seed=2026, p=0.10, L=8, replicate=3)
    c = derive_stream_seeds(base_seed=2026, p=0.11, L=8, replicate=3)
    assert a == b
    assert (a.bond_seed, a.thermal_seed) != (c.bond_seed, c.thermal_seed)
    assert a.bond_seed != a.thermal_seed


def test_common_random_numbers_policy_is_explicitly_p_shared() -> None:
    a = derive_stream_seeds(
        base_seed=2026,
        p=0.10,
        L=8,
        replicate=3,
        policy="common_random_numbers",
    )
    b = derive_stream_seeds(
        base_seed=2026,
        p=0.11,
        L=8,
        replicate=3,
        policy="common_random_numbers",
    )
    assert (a.bond_seed, a.thermal_seed) == (b.bond_seed, b.thermal_seed)


def test_stream_identity_changes_with_lattice_and_replicate() -> None:
    base = derive_stream_seeds(base_seed=17, p=0.1, L=8, replicate=0)
    other_l = derive_stream_seeds(base_seed=17, p=0.1, L=10, replicate=0)
    other_rep = derive_stream_seeds(base_seed=17, p=0.1, L=8, replicate=1)
    assert base.bond_seed != other_l.bond_seed
    assert base.bond_seed != other_rep.bond_seed


def test_many_child_streams_have_no_deterministic_collisions() -> None:
    seeds = {
        derive_stream_seeds(base_seed=99, p=0.1, L=8, replicate=r).bond_seed
        for r in range(1000)
    }
    assert len(seeds) == 1000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_seed": -1, "p": 0.1, "L": 8, "replicate": 0},
        {"base_seed": 1, "p": 0.0, "L": 8, "replicate": 0},
        {"base_seed": 1, "p": np.nan, "L": 8, "replicate": 0},
        {"base_seed": 1, "p": 0.1, "L": 2, "replicate": 0},
        {"base_seed": 1, "p": 0.1, "L": 8, "replicate": -1},
        {"base_seed": 1, "p": 0.1, "L": 8, "replicate": 0, "policy": "implicit"},
    ],
)
def test_invalid_seed_identity_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        derive_stream_seeds(**kwargs)


def test_seeded_scan_grid_reproducible_smoke() -> None:
    kwargs = dict(
        n_disorder=2,
        n_records=8,
        burn_in=8,
        base_seed=123,
        n_skip=1,
        sweeps_per_step=1,
        aligned_start=True,
    )
    a = nishimori_scan_grid([0.08, 0.12], 4, **kwargs)
    b = nishimori_scan_grid([0.08, 0.12], 4, **kwargs)
    assert [(r.abs_m, r.binder) for r in a] == [(r.abs_m, r.binder) for r in b]


def test_scan_grid_requires_strictly_increasing_valid_ps() -> None:
    kwargs = dict(n_disorder=1, n_records=2, burn_in=0, base_seed=1)
    with pytest.raises(ValueError):
        nishimori_scan_grid([0.1, 0.1], 4, **kwargs)
    with pytest.raises(ValueError):
        nishimori_scan_grid([0.2, 0.1], 4, **kwargs)
    with pytest.raises(ValueError):
        nishimori_scan_grid([0.0, 0.1], 4, **kwargs)
