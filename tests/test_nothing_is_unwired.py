"""Every definition in src/ must be reachable from something.

WHY. Three times in this project a correct mechanism was built and the last connection left
out, and every time the run machine found it rather than a test:

  - `choice_messages` rendered the choice prompt correctly and nothing called it, while
    Pass C built that prompt by string-surgery and showed the model NEITHER option
  - the Pass B source digest was applied to the posterior and not to contrasts_/results_
  - `robustness_rows` was written to satisfy A4.1 commitment 4 and never called, so the
    non-centered fit reached no report

The shape is always the same: the helper is tested, and nothing tests that anything CALLS
the helper. A reference map catches it in under a second. It was run once as a one-off
sweep -- days before two of the three were written -- which is why it caught only the first.

So it runs here instead. The point is not the allowlist below; it is that this fails on the
NEXT one.

LIMITATION, stated rather than discovered later: references are matched by bare NAME, so a
function sharing a name with a live one elsewhere (`main`, `artifact_path`, `load_or_run`)
is treated as referenced. That under-reports and never false-positives. Conventional names
are entry points anyway; the helpers this exists to catch have unique names.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Defined, called by nothing, deliberately kept for now. Each needs a reason and a plan --
#: an allowlist without either is just a muted alarm.
ALLOWED = {
    # pass_b.py is inside _PASS_B_SOURCES, so editing it changes the Pass B artifact digest
    # and would invalidate the run machine's rebuilt pairs mid-transmit. Remove once the
    # Pass C snapshot is pushed. run.py inlines the cache check this duplicates.
    "load_or_run": "superseded by run.py's inline cache check; pass_b.py is cache-keyed",
    # No caller since provenance columns moved into write_parquet. Harmless, unreferenced.
    "payload_columns": "superseded by write_parquet; remove with the next provenance edit",
}


#: This file names every allowlisted symbol as a string literal, and the walk counts string
#: constants as references (they catch getattr and monkeypatch use). Walking itself would
#: therefore make every allowlist entry look reachable -- the allowlist defeating its own
#: staleness check. Excluded.
_SELF = Path(__file__).name


def _py(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py")
            if "__pycache__" not in p.parts and "archive" not in p.parts
            and p.name != _SELF]


def _definitions() -> dict[str, str]:
    out = {}
    for path in _py(ROOT / "src"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[node.name] = f"{path.relative_to(ROOT)}:{node.lineno}"
    return out


def _references() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in _py(ROOT / "src") + _py(ROOT / "tests") + _py(ROOT / "scripts"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
            if isinstance(name, str):
                out.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    return out


def _unwired() -> list[tuple[str, str]]:
    defs, refs = _definitions(), _references()
    notebook = (ROOT / "notebooks" / "stage0_analysis.ipynb").read_text(encoding="utf-8")
    out = []
    for name, where in sorted(defs.items()):
        module = where.split(":")[0]
        if refs.get(name, set()) - {module}:
            continue                      # referenced from somewhere else
        if name in notebook:
            continue                      # the notebook is a caller too
        body = (ROOT / module).read_text(encoding="utf-8")
        if body.count(name) > 1:
            continue                      # used within its own module
        out.append((name, where))
    return out


def test_no_definition_in_src_is_unreachable():
    """The tripwire. Fails on the next mechanism that gets built and not wired in."""
    found = {name: where for name, where in _unwired()}
    unexpected = {n: w for n, w in found.items() if n not in ALLOWED}
    assert not unexpected, (
        "defined but called by nothing -- wire it in, delete it, or allowlist it with a "
        f"reason: {unexpected}")


def test_the_allowlist_does_not_outlive_its_entries():
    """An allowlist entry for something now wired in is a muted alarm for the next one."""
    still = {name for name, _ in _unwired()}
    stale = sorted(set(ALLOWED) - still)
    assert not stale, (
        f"allowlisted but now reachable -- drop from ALLOWED: {stale}")


def test_the_walk_actually_resolves_the_codebase():
    """NEGATIVE CONTROL. A broken parse or a wrong root makes every check above pass by
    finding nothing, which is the failure class this whole file is about."""
    defs, refs = _definitions(), _references()
    assert len(defs) >= 100, f"only {len(defs)} definitions found in src/"
    assert len(refs) >= 500, f"only {len(refs)} referenced names found"
    # A function known to be wired must read as wired, and one known not to must not.
    assert refs.get("choice_messages", set()) - {"src/stimuli/build.py"}, (
        "choice_messages reads as unreferenced; it is called by pass_c and this walk is wrong")
    assert "robustness_rows" in refs, "the scripts/ tree is not being walked"


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_every_allowlist_entry_states_a_reason(name):
    assert len(ALLOWED[name]) > 30, f"{name} is allowlisted without a real reason"
