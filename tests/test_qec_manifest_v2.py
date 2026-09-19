"""Tests fuer den versionierten QEC-Noise-/Experimentvertrag."""

from __future__ import annotations

import pytest

from adaptiverg_qec.qec_manifest_v2 import QECExperimentManifestV2, StimNoiseProfile


def test_noise_profile_roundtrip_and_stim_mapping() -> None:
    p = StimNoiseProfile(
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.02,
        after_clifford_depolarization=0.003,
        after_reset_flip_probability=0.004,
    )
    q = StimNoiseProfile.from_dict(p.to_dict())
    assert q == p
    assert q.to_stim_kwargs()["before_measure_flip_probability"] == 0.02
    assert q.fingerprint() == p.fingerprint()


def test_noise_profile_fingerprint_changes_when_channel_changes() -> None:
    a = StimNoiseProfile(before_round_data_depolarization=0.01)
    b = StimNoiseProfile(before_round_data_depolarization=0.011)
    assert a.fingerprint() != b.fingerprint()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan")])
def test_noise_profile_rejects_invalid_probabilities(value: float) -> None:
    with pytest.raises(ValueError):
        StimNoiseProfile(before_measure_flip_probability=value)


def test_noise_profile_rejects_unknown_keys_and_schema() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        StimNoiseProfile.from_dict({"bogus": 1})
    with pytest.raises(ValueError, match="schema mismatch"):
        StimNoiseProfile(schema="wrong/v9")


def test_manifest_roundtrip_distance_rounds_and_hash() -> None:
    m = QECExperimentManifestV2(
        distances=(3, 5, 7),
        noise=StimNoiseProfile(
            before_round_data_depolarization=0.005,
            before_measure_flip_probability=0.005,
        ),
    )
    assert [m.resolved_rounds(d) for d in m.distances] == [3, 5, 7]
    restored = QECExperimentManifestV2.from_dict(m.to_dict())
    assert restored == m
    assert restored.fingerprint() == m.fingerprint()


def test_manifest_fixed_rounds_policy() -> None:
    m = QECExperimentManifestV2(distances=(3, 7), rounds_policy="fixed", rounds=9)
    assert m.resolved_rounds(3) == 9
    assert m.resolved_rounds(7) == 9
    with pytest.raises(ValueError, match="not declared"):
        m.resolved_rounds(5)


def test_manifest_fingerprint_tracks_run_driving_noise() -> None:
    a = QECExperimentManifestV2(noise=StimNoiseProfile(before_measure_flip_probability=0.01))
    b = QECExperimentManifestV2(noise=StimNoiseProfile(before_measure_flip_probability=0.02))
    assert a.fingerprint() != b.fingerprint()


def test_manifest_cell_seed_is_stable_and_tracks_noise_contract() -> None:
    a = QECExperimentManifestV2(noise=StimNoiseProfile(before_measure_flip_probability=0.01))
    a2 = QECExperimentManifestV2.from_dict(a.to_dict())
    b = QECExperimentManifestV2(noise=StimNoiseProfile(before_measure_flip_probability=0.02))
    assert a.cell_seed(3) == a2.cell_seed(3)
    assert a.cell_seed(3) != a.cell_seed(5)
    assert a.cell_seed(3) != b.cell_seed(3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distances": ()},
        {"distances": (3, 3)},
        {"distances": (5, 3)},
        {"distances": (3, 4)},
        {"rounds_policy": "fixed", "rounds": 0},
        {"rounds_policy": "distance", "rounds": 3},
        {"shots_per_cell": 0},
        {"base_seed": -1},
        {"memory_basis": "y"},
        {"decoder": "mystery"},
        {"seed_policy": "shared"},
        {"sampling_backend": "sinter"},
        {"reproducibility_tier": "BITWISE_FOREVER"},
    ],
)
def test_manifest_invalid_contracts_fail_closed(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        QECExperimentManifestV2(**kwargs)


def test_manifest_rejects_unknown_json_keys() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        QECExperimentManifestV2.from_dict({"surprise": True})
