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


def test_every_slow_test_file_forces_single_core_sampling():
    """A slow test that hangs on the run machine is worse than no slow test.

    PyMC deadlocks on the Windows spawn start method unless cores=1 -- the same reason
    run.py carries --sampler-cores. Five slow tests were written without it and hung there,
    including all three covering the SOLE exclusion criterion, so the run machine had to
    skip exactly the checks it most needed to run.

    Checked per FILE rather than per test: any file with a slow marker must name
    sampler_cores somewhere, which is coarse but cannot be satisfied by accident.
    """
    offenders = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if "mark.slow" in text and "sampler_cores" not in text:
            offenders.append(path.name)
    assert not offenders, (
        f"slow tests that will deadlock on the run machine: {offenders}")

    # NEGATIVE CONTROL: the walk must find slow files at all, or this passes vacuously.
    slow_files = [p.name for p in (ROOT / "tests").glob("test_*.py")
                  if "mark.slow" in p.read_text(encoding="utf-8")]
    assert len(slow_files) >= 5, f"only {len(slow_files)} slow files found"


def test_status_write_emits_the_newest_result_per_model():
    """The --write loop iterated ALL results into a filename keyed on (stage, model), so
    last-write-wins emitted the OLDEST. With stale artifact dirs present the git-tracked
    results/ JSONs would have carried numbers predating every recent fix -- the "silently
    reported superseded numbers" failure STATUS.md is generated to prevent, reproduced
    inside the generator. Found by the run machine while staging a transmit.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("status", ROOT / "scripts" / "status.py")
    status = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(status)

    rows = [  # already newest-first, as _load_results guarantees
        {"_stage": "analysis", "model": "gemma", "_when": "2026-07-28T10:00:00", "tag": "new"},
        {"_stage": "analysis", "model": "gemma", "_when": "2026-07-26T10:00:00", "tag": "old"},
        {"_stage": "analysis", "model": "llama", "_when": "2026-07-28T11:00:00", "tag": "new"},
        {"_stage": "analysis", "model": "llama", "_when": "2026-07-26T09:00:00", "tag": "old"},
        {"_stage": "pass_b", "model": "gemma", "_when": "2026-07-28T09:00:00", "tag": "new"},
    ]
    kept = status._newest_per_model(rows)
    assert len(kept) == 3, "one per (stage, model)"
    assert all(r["tag"] == "new" for r in kept), (
        f"an older result survived de-duplication: {[r['tag'] for r in kept]}")

    # NEGATIVE CONTROL: without dedup, last-write-wins gives the OLD one -- which is the
    # bug. Demonstrated so this test cannot pass while the defect is reintroduced.
    naive = {}
    for r in rows:
        naive[(r["_stage"], r["model"])] = r
    assert naive[("analysis", "gemma")]["tag"] == "old"


def test_status_reports_pass_c_and_keeps_its_fields():
    """The keep-list predated Pass C, so the git-tracked summaries carried NO H1 result."""
    src = (ROOT / "scripts" / "status.py").read_text(encoding="utf-8")
    for field in ('"primary"', '"contrasts"', '"sesoi"', '"power"'):
        assert field in src, f"status.py drops {field} from the transmitted summaries"
    assert "def pass_c_table" in src, "STATUS.md has no Pass C table"
    assert "_newest_per_model(results)" in src, "the write loop is not de-duplicated"


def test_the_robustness_refit_cannot_displace_or_overwrite_the_primary():
    """A4.1 commitment 2: the CENTERED fit stays primary, the non-centered one is a check.

    Two ways that could have been broken, both now closed:
      - the refit writing over contrasts_/results_ (they are keyed on the model source
        digest, as the posterior already was), and
      - status.py's newest-wins de-duplication silently promoting the newer robustness fit
        over the fit the amendment named primary.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("status", ROOT / "scripts" / "status.py")
    status = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(status)

    rows = [  # the refit is NEWER, as it will be in reality
        {"_stage": "analysis", "model": "llama", "_when": "2026-07-29T10:00:00",
         "spread_parameterization": "non-centered", "primary": {"median": 0.44}},
        {"_stage": "analysis", "model": "llama", "_when": "2026-07-28T10:00:00",
         "spread_parameterization": "centered", "primary": {"median": 0.436}},
    ]
    kept = status._newest_per_model(rows)
    assert len(kept) == 1
    assert kept[0]["spread_parameterization"] == "centered", (
        "the robustness refit displaced the primary -- A4.1 commitment 2 is broken")
    assert status.robustness_rows(rows)[0]["spread_parameterization"] == "non-centered"

    # Artifacts predating A4.1 carry no field and ARE centered; they must stay primary.
    legacy = [{"_stage": "analysis", "model": "gemma", "_when": "2026-07-28T09:00:00"},
              {"_stage": "analysis", "model": "gemma", "_when": "2026-07-29T09:00:00",
               "spread_parameterization": "non-centered"}]
    assert "spread_parameterization" not in status._newest_per_model(legacy)[0]

    # NEGATIVE CONTROL: without the preference, newest-wins gives the refit -- the bug.
    assert sorted(rows, key=lambda r: r["_when"], reverse=True)[0][
        "spread_parameterization"] == "non-centered"


def test_the_three_clobberable_artifacts_are_all_keyed_on_the_model_source():
    """The posterior was keyed first; contrasts_ and results_ were not, so a refit would
    have destroyed the evidence for the fit A4.1 calls primary."""
    src = (ROOT / "src" / "experiments" / "run.py").read_text(encoding="utf-8")
    assert "spread_model.posterior_path(cfg, out_dir)" in src
    assert 'contrasts_{cfg.hash()}-{spread_model.source_digest()}' in src
    assert 'results_{cfg.hash()}-{digest}' in src
    assert 'results["spread_parameterization"]' in src, (
        "an artifact must say which parameterization produced it")


def _status():
    import importlib.util

    spec = importlib.util.spec_from_file_location("status", ROOT / "scripts" / "status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(model, param, median, when):
    return {"_stage": "analysis", "model": model, "_when": when,
            "spread_parameterization": param, "sesoi": 0.2355,
            "outcome": "primary-inconclusive", "provenance": {"device": "cuda"},
            "power": {"mde_over_sesoi": 1.37, "equivalence_reachable": True},
            "primary": {"median": median, "hdi_low": median - 0.2,
                        "hdi_high": median + 0.2, "p_negative": 0.5,
                        "decision": "inconclusive"}}


def test_both_fits_reach_the_report_not_just_the_artifacts():
    """A4.1 commitment 4: BOTH fits reported for all three models.

    `robustness_rows()` was written and never called, so the non-centered fit existed only
    in raw artifacts -- absent from STATUS.md and from the tracked results/. Present as data
    is not reported. Found by the run machine while staging the transmit; it is the third
    time a mechanism was built here and not wired in.
    """
    st = _status()
    rows = []
    for m, cen, non in [("gemma", -0.0015, 0.0003), ("llama", 0.4361, 0.4401)]:
        rows.append(_result(m, "centered", cen, "2026-07-28T10:00"))
        rows.append(_result(m, "non-centered", non, "2026-07-29T10:00"))

    primary = st.pass_c_table(rows).set_index("model")["lambda_median"]
    robust = st.pass_c_robustness_table(rows).set_index("model")["lambda_median"]
    assert primary["llama"] == pytest.approx(0.4361), "the primary table lost the centered fit"
    assert robust["llama"] == pytest.approx(0.4401), "the robustness table is not the refit"
    assert len(robust) == 2, "one robustness row per model"

    text = st.render(rows)
    assert "Pass C robustness" in text, "STATUS.md has no robustness section"
    assert "remains PRIMARY" in text, "the report does not say which fit is primary"

    # NEGATIVE CONTROL: with no refit present the section must vanish rather than render
    # empty, or its presence stops meaning anything.
    centered_only = [r for r in rows if r["spread_parameterization"] == "centered"]
    assert st.pass_c_robustness_table(centered_only).empty
    assert "Pass C robustness" not in st.render(centered_only)


def test_every_status_table_survives_having_no_matching_results():
    """`absolute_table` built a DataFrame from an empty list and then sorted it by a column
    that does not exist -- a KeyError crash whenever no result carries `gates`. The other
    builders guard it; two did not. Found by feeding status.py synthetic Pass C rows."""
    st = _status()
    rows = [_result("gemma", "centered", -0.0015, "2026-07-28T10:00")]
    for name in ("absolute_table", "pairwise_table", "readout_mass_table",
                 "pass_c_robustness_table"):
        got = getattr(st, name)(rows)
        assert got.empty, f"{name} should be empty for these rows"
    assert st.render(rows)  # must not raise


def test_every_expensive_artifact_is_keyed_on_the_code_that_writes_it():
    """Three artifacts have now failed this the same way, so it is asserted for all of them.

      Pass B  -- signed theta added, config hash unmoved, stale pairs would be reused
      posterior -- template effect non-centred, config hash unmoved, stale fit reloaded
      Pass C  -- p_item1 added, config hash unmoved, stale trials would be reused

    The config hash covers PARAMETERS. It cannot see a module changing what it writes. Every
    artifact whose contents depend on module source must carry a digest of that source.
    """
    from src.analysis import spread_model
    from src.experiments import pass_b, pass_c

    for mod, name in ((pass_b, "pass_b"), (pass_c, "pass_c"), (spread_model, "spread_model")):
        assert hasattr(mod, "source_digest"), f"{name} has no source digest"
        digest = mod.source_digest()
        assert len(digest) == 8 and digest.isalnum(), f"{name}: {digest!r}"

    from src.config import load_config

    cfg = load_config("configs/stage0_gemma-2-2b.yaml")
    # Both halves of the key must be present: the config hash AND the source digest.
    for stage, mod in (("pass_b", pass_b), ("pass_c", pass_c)):
        name = mod.artifact_path(cfg).name
        assert cfg.hash(stage) in name, f"{name} lost its config hash"
        assert mod.source_digest() in name, f"{name} lost its source digest"

    # NEGATIVE CONTROL: the digests must differ between modules, or one is being reused
    # for all of them and the check means nothing.
    assert len({pass_b.source_digest(), pass_c.source_digest(),
                spread_model.source_digest()}) == 3
