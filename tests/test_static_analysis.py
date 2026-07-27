"""Undefined names, across all of src/.

`run.py` referenced `fit.sigma_item` in the H1 stage. `fit` is a local inside
`_instrument_fit` and was never in scope there -- a NameError sitting in the ONE stage the
pipeline had never reached, because every run so far stopped at or before Pass B. It would
have fired AFTER Pass C's 20,000 forward passes per model: hours of GPU time, then a crash
before a single number was computed.

No test could have caught it, because reaching that line requires a full Pass C. A static
check does catch it, in under a second, which is why this file exists.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pyflakes(target: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(ROOT / target)],
        capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        pytest.skip(f"pyflakes unavailable: {proc.stderr.strip()[:120]}")
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_no_undefined_names_anywhere_in_src():
    """The check that would have caught it. Cheap, total, and independent of coverage."""
    findings = [ln for ln in _pyflakes("src") if "undefined name" in ln]
    assert not findings, "undefined names in src/:\n  " + "\n  ".join(findings)


def test_no_undefined_names_in_tests():
    findings = [ln for ln in _pyflakes("tests") if "undefined name" in ln]
    assert not findings, "undefined names in tests/:\n  " + "\n  ".join(findings)


def test_the_checker_actually_detects_an_undefined_name(tmp_path):
    """NEGATIVE CONTROL. If pyflakes were missing or misinvoked both checks above would
    return an empty list and pass -- the vacuous-match class, in the tool this time."""
    bad = tmp_path / "planted.py"
    bad.write_text("def f():\n    return not_a_real_name + 1\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(bad)], capture_output=True, text=True)
    assert "undefined name" in proc.stdout, (
        "pyflakes did not flag a planted undefined name; the checks above are vacuous")
