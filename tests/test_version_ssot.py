"""Eine Versionsquelle: pyproject.toml, Laufzeit und installierte Distribution.

Vorher deklarierte pyproject.toml ``0.1.0.dev0`` und ``adaptiverg_qec.__version__``
unabhaengig ``0.1.0.dev2``. Jede Evidenz, die ``__version__`` schreibt (CLI-Selftest
``version``, Run-Manifest ``package_version``), nannte damit eine andere Nummer als
die gebaute bzw. installierte Distribution (Codex-Befund auf PR #41).

Vertrag:
(a) pyproject.toml hat KEINE statische Version, sondern liest das Literal
    ``adaptiverg_qec.__version__`` (setuptools ``dynamic``) -- es gibt nur eine Zahl.
(b) Dieses Literal ist genau EIN reines String-Literal (setuptools liest es per AST),
    gleich der Laufzeitversion und kanonisches PEP 440.
(c) Ist die Distribution aus DIESEM Checkout installiert (CI: ``pip install -e .``),
    meldet ihre Metadaten-Version exakt die Laufzeitversion. Im CI ist ein Skip
    hier ein Fehler, damit der Pin nicht still entfaellt.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tomllib
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

import adaptiverg_qec

ROOT = Path(__file__).resolve().parents[1]

# PEP 440, Appendix B: kanonische oeffentliche Versionsform.
_CANONICAL_PEP440 = re.compile(
    r"([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?"
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _version_literals() -> list[object]:
    tree = ast.parse((ROOT / "src" / "adaptiverg_qec" / "__init__.py").read_text(encoding="utf-8"))
    found: list[object] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__version__" for t in targets):
            found.append(value.value if isinstance(value, ast.Constant) else ast.dump(value))
    return found


def test_pyproject_has_no_second_version_source() -> None:
    cfg = _pyproject()
    project = cfg["project"]
    assert "version" not in project, f"static version in pyproject.toml: {project.get('version')}"
    assert "version" in project.get("dynamic", []), project.get("dynamic")
    dynamic = cfg.get("tool", {}).get("setuptools", {}).get("dynamic", {})
    assert dynamic.get("version") == {"attr": "adaptiverg_qec.__version__"}, dynamic


def test_version_literal_is_single_pep440_and_equals_runtime() -> None:
    literals = _version_literals()
    assert len(literals) == 1, f"expected exactly one __version__ assignment, got {literals}"
    (literal,) = literals
    assert isinstance(literal, str), f"__version__ must be a plain string literal: {literal}"
    assert _CANONICAL_PEP440.fullmatch(literal), f"not canonical PEP 440: {literal!r}"
    assert adaptiverg_qec.__version__ == literal


def _checkout_install() -> tuple[metadata.Distribution | None, str]:
    """Distribution, falls sie aus DIESEM Checkout installiert ist, sonst Grund."""
    # Nicht metadata.distribution(name): mit pytest ``pythonpath = ["src"]`` findet sie
    # zuerst das Build-Artefakt src/<pkg>.egg-info (ohne direct_url.json), das setuptools
    # bei ``pip install -e .`` auch in CI anlegt. Massgeblich ist die installierte
    # Distribution, deren direct_url.json auf genau diesen Checkout zeigt.
    name = _pyproject()["project"]["name"]
    candidates = list(metadata.distributions(name=name))
    if not candidates:
        return None, f"{name} is not installed"
    reasons: list[str] = []
    for dist in candidates:
        raw = dist.read_text("direct_url.json")
        if not raw:
            reasons.append("entry without direct_url.json")
            continue
        url = json.loads(raw).get("url", "")
        if not url.startswith("file:"):
            reasons.append(f"installed from {url!r}")
            continue
        source = Path(url2pathname(urlparse(url).path))
        if os.path.normcase(source.resolve()) == os.path.normcase(ROOT):
            return dist, ""
        reasons.append(f"installed from another checkout: {source}")
    return None, f"{name} not installed from this checkout ({'; '.join(reasons)})"


def test_installed_distribution_reports_the_runtime_version() -> None:
    dist, reason = _checkout_install()
    if dist is None:
        if os.environ.get("CI"):
            raise AssertionError(f"CI must install this checkout (pip install -e .): {reason}")
        pytest.skip(reason)
    module_file = Path(adaptiverg_qec.__file__).resolve()
    assert module_file.is_relative_to(ROOT / "src"), f"imported from {module_file}"
    assert dist.version == adaptiverg_qec.__version__, (
        f"installed metadata {dist.version!r} != runtime {adaptiverg_qec.__version__!r}"
    )
