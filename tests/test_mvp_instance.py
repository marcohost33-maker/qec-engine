"""MVPConfig-Vertrag: kompaktes Theta (PR #53 R2).

Bewusst OHNE modulweite MVPConfig-Instanz: ein Waechter, der faelschlich IMMER
feuert, soll hier an einer Zusicherung scheitern, nicht schon beim Sammeln.
"""

from __future__ import annotations

import pytest

from adaptiverg_qec.mvp_instance import MVPConfig


def test_valid_configs_are_accepted() -> None:
    for kw in (dict(), dict(beta_max=1e300), dict(beta_min=1e-300, beta_max=1e-299)):
        try:
            cfg = MVPConfig(**kw)
        except ValueError as exc:
            raise AssertionError(f"valid MVPConfig({kw}) rejected: {exc!r}") from exc
        assert cfg.beta_min < cfg.beta_max


@pytest.mark.parametrize(
    "kw",
    [dict(beta_max=float("inf")), dict(beta_min=float("inf"), beta_max=float("inf"))],
    ids=["max_inf", "both_inf"],
)
def test_infinite_bounds_are_rejected(kw) -> None:
    with pytest.raises(ValueError, match="beta_max must be finite"):
        MVPConfig(**kw)
