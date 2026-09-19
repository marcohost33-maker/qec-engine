"""Versionierte QEC-Noise- und Experimentvertraege.

Ziel: Kein QEC-Ergebnis darf seinen Noise-Kanal, Decoder, Distanz-/Rundenvertrag
oder Seed-Policy nur implizit im aufrufenden Code tragen. Diese Datenklassen
machen die laufbestimmenden Parameter serialisierbar, validierbar und hashbar.

Die v1-Noise-Felder entsprechen direkt den Parametern von Stim Circuit.generated.
Das ExperimentManifest v2 ist bewusst klein: es beschreibt den bestehenden
rotated-surface-memory + PyMatching/DEM-Pfad, statt vorzeitig eine universelle
QEC-Spezifikation zu erfinden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

NOISE_PROFILE_SCHEMA = "adaptiverg_qec.stim_noise_profile/v1"
EXPERIMENT_MANIFEST_SCHEMA = "adaptiverg_qec.qec_experiment_manifest/v2"


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real probability, got {type(value).__name__}")
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class StimNoiseProfile:
    """Explizite Noise-Policy fuer Stim-generierte Memory-Circuits."""

    schema: str = NOISE_PROFILE_SCHEMA
    before_round_data_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_clifford_depolarization: float = 0.0
    after_reset_flip_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.schema != NOISE_PROFILE_SCHEMA:
            raise ValueError(
                f"noise schema mismatch: expected {NOISE_PROFILE_SCHEMA!r}, got {self.schema!r}"
            )
        for name in (
            "before_round_data_depolarization",
            "before_measure_flip_probability",
            "after_clifford_depolarization",
            "after_reset_flip_probability",
        ):
            object.__setattr__(self, name, _probability(name, getattr(self, name)))

    def to_stim_kwargs(self) -> dict[str, float]:
        """1:1-Abbildung auf Stim Circuit.generated Noise-Argumente."""
        return {
            "before_round_data_depolarization": self.before_round_data_depolarization,
            "before_measure_flip_probability": self.before_measure_flip_probability,
            "after_clifford_depolarization": self.after_clifford_depolarization,
            "after_reset_flip_probability": self.after_reset_flip_probability,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StimNoiseProfile:
        if not isinstance(data, dict):
            raise TypeError("noise profile must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"noise profile has unknown keys: {sorted(unknown)}")
        return cls(**data)

    def fingerprint(self) -> str:
        blob = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class QECExperimentManifestV2:
    """Reproduzierbarer Vertrag fuer einen Surface-Code-MWPM-Sweep."""

    schema: str = EXPERIMENT_MANIFEST_SCHEMA
    code_family: str = "surface_code:rotated_memory"
    memory_basis: str = "z"
    distances: tuple[int, ...] = (3, 5, 7)
    rounds_policy: str = "distance"
    rounds: int | None = None
    shots_per_cell: int = 20_000
    base_seed: int = 20260919
    seed_policy: str = "manifest-sha256-v1"
    decoder: str = "pymatching-mwpm-dem"
    noise: StimNoiseProfile = field(default_factory=StimNoiseProfile)
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != EXPERIMENT_MANIFEST_SCHEMA:
            raise ValueError(
                f"manifest schema mismatch: expected {EXPERIMENT_MANIFEST_SCHEMA!r}, "
                f"got {self.schema!r}"
            )
        if self.code_family != "surface_code:rotated_memory":
            raise ValueError(f"unsupported code_family {self.code_family!r}")
        if self.memory_basis not in {"x", "z"}:
            raise ValueError(f"memory_basis must be 'x' or 'z', got {self.memory_basis!r}")
        if not isinstance(self.distances, tuple):
            object.__setattr__(self, "distances", tuple(self.distances))
        if not self.distances:
            raise ValueError("distances must not be empty")
        if tuple(sorted(set(self.distances))) != self.distances:
            raise ValueError("distances must be unique and strictly increasing")
        for d in self.distances:
            if isinstance(d, bool) or not isinstance(d, int) or d < 3 or d % 2 == 0:
                raise ValueError(f"distances must contain odd ints >=3, got {self.distances}")
        if self.rounds_policy not in {"distance", "fixed"}:
            raise ValueError("rounds_policy must be 'distance' or 'fixed'")
        if self.rounds_policy == "distance":
            if self.rounds is not None:
                raise ValueError("rounds must be null when rounds_policy='distance'")
        elif isinstance(self.rounds, bool) or not isinstance(self.rounds, int) or self.rounds < 1:
            raise ValueError("rounds must be an int >=1 when rounds_policy='fixed'")
        if (
            isinstance(self.shots_per_cell, bool)
            or not isinstance(self.shots_per_cell, int)
            or self.shots_per_cell < 1
        ):
            raise ValueError(f"shots_per_cell must be an int >=1, got {self.shots_per_cell}")
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError(f"base_seed must be a non-negative int, got {self.base_seed!r}")
        if self.seed_policy != "manifest-sha256-v1":
            raise ValueError(f"unsupported seed_policy {self.seed_policy!r}")
        if self.decoder != "pymatching-mwpm-dem":
            raise ValueError(f"unsupported decoder {self.decoder!r}")
        if not isinstance(self.noise, StimNoiseProfile):
            if isinstance(self.noise, dict):
                object.__setattr__(self, "noise", StimNoiseProfile.from_dict(self.noise))
            else:
                raise TypeError("noise must be StimNoiseProfile or a compatible dict")
        if not isinstance(self.environment, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.environment.items()
        ):
            raise TypeError("environment must be dict[str, str]")

    def resolved_rounds(self, distance: int) -> int:
        if distance not in self.distances:
            raise ValueError(f"distance {distance} not declared in manifest")
        return distance if self.rounds_policy == "distance" else int(self.rounds)

    def cell_seed(self, distance: int) -> int:
        """Deterministische Zellidentitaet aus dem vollstaendigen Run-Vertrag."""
        rounds = self.resolved_rounds(distance)
        payload = {
            "base_seed": self.base_seed,
            "distance": distance,
            "rounds": rounds,
            "memory_basis": self.memory_basis,
            "decoder": self.decoder,
            "noise": self.noise.to_dict(),
        }
        blob = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest = hashlib.sha256(blob).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["distances"] = list(self.distances)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QECExperimentManifestV2:
        if not isinstance(data, dict):
            raise TypeError("experiment manifest must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"experiment manifest has unknown keys: {sorted(unknown)}")
        clean = dict(data)
        if "distances" in clean:
            clean["distances"] = tuple(clean["distances"])
        if "noise" in clean and isinstance(clean["noise"], dict):
            clean["noise"] = StimNoiseProfile.from_dict(clean["noise"])
        return cls(**clean)

    def fingerprint(self) -> str:
        blob = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
