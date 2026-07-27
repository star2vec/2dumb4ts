"""Nothing in src/archive/ may reach live code.

The archive keeps retired modules so the numbers they produced stay traceable. That is only
safe while they are inert: a retired module that gets imported again is worse than a deleted
one, because its docstring says "retired" while it runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "src" / "archive"


def _live_python() -> list[Path]:
    return [p for p in (ROOT / "src").rglob("*.py")
            if "__pycache__" not in p.parts and ARCHIVE not in p.parents]


def test_no_live_module_imports_the_archive():
    offenders = []
    for path in _live_python():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "archive" in (node.module or ""):
                offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if "archive" in a.name:
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"live code imports the archive: {offenders}"

    # NEGATIVE CONTROL: the walk must have actually visited the live tree.
    assert len(_live_python()) >= 15, "the live-module walk found almost nothing"


def test_the_archive_is_not_empty_and_every_module_declares_itself_retired():
    modules = [p for p in ARCHIVE.glob("*.py") if p.name != "__init__.py"]
    assert modules, "the archive is empty -- this test would pass vacuously"
    for path in modules:
        head = path.read_text(encoding="utf-8")[:400]
        assert "RETIRED" in head, f"{path.name} does not declare itself retired"
        assert "src/archive/README.md" in head, f"{path.name} does not point at the README"


def test_the_readme_accounts_for_every_archived_module():
    readme = (ARCHIVE / "README.md").read_text(encoding="utf-8")
    for path in ARCHIVE.glob("*.py"):
        if path.name == "__init__.py":
            continue
        assert path.name in readme, f"{path.name} is archived but unexplained in the README"


@pytest.mark.parametrize("name,replacement", [
    ("template_dependence.py", "excess_slope_ppc_null"),
    ("operating_window.py", "pass_a_pairwise.operating_window"),
    ("probe_choice.py", "readout/choice.py"),
])
def test_each_retirement_names_its_replacement(name, replacement):
    """A retirement without a forwarding address invites someone to resurrect it."""
    readme = (ARCHIVE / "README.md").read_text(encoding="utf-8")
    assert replacement in readme, f"{name}'s replacement is not named"
