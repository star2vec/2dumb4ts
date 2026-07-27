"""Every amended constant, asserted against what the amendment actually specifies.

WHY THIS FILE EXISTS. `match_tolerance` was re-expressed by A2.9.2 as `0.26 x sigma_item`
and the code was never changed: a constant declared in *rating points* stayed pinned at 0.15
and was applied as a hard filter to theta-scale gaps in logits. Audit item T2.3 had already
named it. It was marked closed with nothing verifying it, and document and code diverged
unwatched for two amendments.

The damage was never strictness. A FIXED logit constant is a DIFFERENT effective strictness
per model, varying inversely with `sigma_item`, so each model's matched sets were built under
a different rule while the ladder comparison assumes they were not -- and that is invisible
within any single model, which is why no per-model check would have caught it.

So the rule is: an amendment that changes a number gets a test here that reads the rule and
asserts the config implements it. Amendments without enforcement tests are aspirational.

Each test names the amendment it enforces and carries a NEGATIVE CONTROL -- a demonstration
that it fails when the defect is present. A test that passes by matching nothing is the
failure class this file exists to close, and it would otherwise be free to reappear here.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from src.config import load_config

SRC = Path(__file__).resolve().parents[1] / "src"


def _sources_mentioning(needle: str) -> list[Path]:
    """Every .py under src/ containing `needle`. Never __pycache__.

    A `grep -r src/` walk picks up .pyc files and yields paths like `src//config.py`, so
    exclusion lists silently miss and the assertion fires on compiled bytecode. Both
    happened on the first run of this file.
    """
    return sorted(f for f in SRC.rglob("*.py")
                  if "__pycache__" not in f.parts and needle in f.read_text(encoding="utf-8"))


CONFIGS = [
    "configs/stage0_qwen2.5-0.5b.yaml",
    "configs/stage0_qwen2.5-1.5b.yaml",
    "configs/stage0_qwen2.5-3b.yaml",
    "configs/stage0_gemma-2-2b.yaml",
    "configs/stage0_llama-3.2-3b.yaml",
]


@pytest.mark.parametrize("config", CONFIGS)
def test_a2_9_2_match_tolerance_is_a_fraction_of_sigma_item(config):
    """A2.9.2: `match_tolerance` is `0.26 x sigma_item`, per model, from that model's fit."""
    cfg = load_config(config)
    assert cfg.pass_b.match_tolerance_sigma_fraction == 0.26

    # It must be a FUNCTION of sigma_item, not a stored number.
    assert callable(cfg.pass_b.match_tolerance)
    assert not hasattr(type(cfg.pass_b), "model_fields") or (
        "match_tolerance" not in type(cfg.pass_b).model_fields), (
        "match_tolerance must not survive as a plain field -- that is the bug")

    assert cfg.pass_b.match_tolerance(1.573) == pytest.approx(0.26 * 1.573)
    assert cfg.pass_b.match_tolerance(0.500) == pytest.approx(0.26 * 0.500)

    # NEGATIVE CONTROL: the defect was a constant that did NOT move with sigma_item.
    lo, hi = cfg.pass_b.match_tolerance(0.5), cfg.pass_b.match_tolerance(2.0)
    assert lo != hi, "a tolerance constant across models is exactly the A3.9 bug"
    assert hi / lo == pytest.approx(4.0)


def test_a2_9_2_the_old_rating_point_constant_is_gone_from_every_config():
    """The retired constant must not survive anywhere, including as an override."""
    from pathlib import Path

    import yaml

    for path in Path("configs").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pass_b = raw.get("pass_b") or {}
        assert "match_tolerance" not in pass_b, (
            f"{path.name} still sets the retired rating-point constant")

    # NEGATIVE CONTROL: the check must actually reject the old key.
    with pytest.raises(Exception):
        load_config(CONFIGS[0], overrides={"pass_b": {"match_tolerance": 0.15}})


@pytest.mark.parametrize("config", CONFIGS)
def test_a2_9_2_sesoi_is_a_fraction_of_sigma_item_and_the_raw_anchor_is_retired(config):
    """A2.9.2: SESOI = 0.15 x sigma_item; `sesoi_raw_secondary` has no logit meaning."""
    cfg = load_config(config)
    assert cfg.analysis.sesoi_sigma_fraction == 0.15
    # A3.3 quotes 0.2359 for gemma at sigma_item 1.573 -- the arithmetic that pins the scale.
    assert cfg.analysis.sesoi_sigma_fraction * 1.573 == pytest.approx(0.2359, abs=5e-5)

    # The retired anchor may linger as a FIELD (config.py) and inside the inert
    # absolute-Pass-A record (reliability.py). Anywhere else is live analysis code.
    live = [str(f) for f in _sources_mentioning("sesoi_raw_secondary")
            if f.name not in {"config.py", "reliability.py"}]
    assert not live, f"retired sesoi_raw_secondary reached live analysis code: {live}"
    # NEGATIVE CONTROL: the walk must find the field, or the check is vacuous.
    assert _sources_mentioning("sesoi_raw_secondary"), "the source walk found nothing"


def test_deferred_rating_point_constants_reach_nothing_that_gates():
    """A1.8 deferred re-expressing three rating-point quantities. Two are still deferred.

    A2.9.2 resolved `match_tolerance` and the SESOI. It did NOT resolve `sigma_between_min`
    or the +/-16 spread bound. That is survivable only because both now live exclusively in
    the absolute-Pass-A path, which A1.5 made an instrument-validation record that gates
    nothing -- `run.py` prints its verdict and records `gates_nothing: True`.

    This test is what keeps that true. If a deferred rating-point constant ever regains the
    power to exclude a model, it does so with this assertion failing first.
    """
    inert = {"config.py", "reliability.py"}
    for constant in ("sigma_between_min", "validity_rho_min",
                     "validity_min_surviving_templates", "icc_tripwire"):
        offenders = [f for f in _sources_mentioning(constant) if f.name not in inert]
        # run.py may PRINT them; it must not branch on them.
        if any(f.name == "run.py" for f in offenders):
            src = (SRC / "experiments" / "run.py").read_text(encoding="utf-8")
            assert 'gates_nothing": True' in src, (
                f"{constant} is referenced in run.py and the absolute pass no longer "
                "declares itself inert")
            offenders = [f for f in offenders if f.name != "run.py"]
        assert not offenders, f"{constant} is live outside the inert path: {offenders}"

    # NEGATIVE CONTROL: the walk must actually find things, or this passes vacuously --
    # which is the exact failure class this file exists to close.
    assert _sources_mentioning("sigma_between_min"), "the source walk found nothing"


def test_the_realized_match_tolerance_is_recorded_not_just_the_rule():
    """The config hash keys on the FRACTION; the applied value depends on `sigma_item`.

    `sigma_item` comes from the instrument fit, which is keyed on the estimator's SOURCE
    digest -- nothing the pass_b config hash can see. So two different pair sets can share
    one pass_b hash. That is the stimulus-digest failure again, and the guard has to be a
    recorded value rather than a recomputed one.
    """
    import numpy as np
    import pandas as pd

    from src.experiments import pass_b

    cfg = load_config(CONFIGS[3])
    frame = pd.DataFrame({"match_tolerance_realized": [0.26 * 1.573] * 3,
                          "sigma_item": [1.573] * 3})
    pass_b.assert_tolerance_matches(cfg, frame, 1.573)          # agrees -> silent

    # NEGATIVE CONTROL 1: a different sigma_item must be caught.
    with pytest.raises(ValueError, match="cannot see this difference"):
        pass_b.assert_tolerance_matches(cfg, frame, 1.900)

    # NEGATIVE CONTROL 2: a pre-A3.9 artifact has no column at all and must be refused
    # rather than assumed compatible -- those were built with the 0.15 constant.
    with pytest.raises(ValueError, match="predates A3.9"):
        pass_b.assert_tolerance_matches(cfg, pd.DataFrame({"pair_id": ["a"]}), 1.573)

    assert np.isfinite(cfg.pass_b.match_tolerance(1.573))


def test_a3_1_the_withdrawn_power_criterion_is_not_reinstated_in_config():
    """A3.1 withdrew 'power at the SESOI >= 0.80' and put nothing in its place."""
    cfg = load_config(CONFIGS[3])
    assert cfg.analysis.power_target == 0.80, (
        "power_target still defines the MDE's operating point; A3.1 withdrew the "
        "criterion applied AT the SESOI, not the 80% convention")

    from src.analysis import power

    # NEGATIVE CONTROL: the SESOI term must still bind, or A3.6's refusal was undone.
    import inspect
    src = inspect.getsource(power._decide_pass)
    assert "max(z * post_sd, sesoi)" in src, (
        "the SESOI floor is gone from _decide_pass -- that is the change A3.6 declined")


# ---------------------------------------------------------------------------
# the ledger


def _headers() -> set[str]:
    """Section and amendment identifiers declared by preregistration.md headers."""
    import re

    text = (Path(__file__).resolve().parents[1] / "preregistration.md").read_text(
        encoding="utf-8")
    out: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^#{2,3} (A?\d+(?:\.\d+)*)\.? ", line)
        if m:
            out.add(m.group(1))
    return out


def test_the_ledger_cites_only_things_that_exist():
    """PREREGISTRATION_LEDGER.md is a reviewer-facing count; a stale citation discredits it.

    Amendments without enforcement tests are aspirational (A3.9), and a ledger of amendments
    is no exception.
    """
    import re

    ledger = (Path(__file__).resolve().parents[1] / "PREREGISTRATION_LEDGER.md").read_text(
        encoding="utf-8")
    headers = _headers()

    sections = {m.group(1) for m in re.finditer(r"§(\d+(?:\.\d+)*)", ledger)}
    amendments = {m.group(1) for m in re.finditer(r"\b(A\d+\.\d+(?:\.\d+)*)\b", ledger)}

    # NEGATIVE CONTROL: a broken regex would make both sets empty and pass silently.
    assert len(sections) >= 25, f"only found {len(sections)} section citations"
    assert len(amendments) >= 10, f"only found {len(amendments)} amendment citations"
    assert "5.1" in sections and "A3.9" in amendments

    missing_s = sorted(s for s in sections if s not in headers)
    missing_a = sorted(a for a in amendments if a not in headers)
    assert not missing_s, f"ledger cites nonexistent sections: {missing_s}"
    assert not missing_a, f"ledger cites nonexistent amendments: {missing_a}"


def test_the_ledger_covers_every_top_level_preregistered_section():
    """A count is only trustworthy if nothing was left out of it."""
    import re

    ledger = (Path(__file__).resolve().parents[1] / "PREREGISTRATION_LEDGER.md").read_text(
        encoding="utf-8")
    cited = {m.group(1) for m in re.finditer(r"§(\d+)(?:\.\d+)*", ledger)}
    frozen = {h for h in _headers() if re.fullmatch(r"\d+", h)}
    assert frozen, "no top-level frozen sections found -- the header parse is broken"
    assert not (frozen - cited), f"ledger omits frozen sections: {sorted(frozen - cited)}"


@pytest.mark.parametrize("config", CONFIGS)
def test_a2_2_reliability_threshold_lives_in_config_and_is_0_70(config):
    """A2.2: the sole exclusion is empirical split-half Spearman of theta >= 0.70.

    It was `RELIABILITY_MIN = 0.70`, a module constant in pass_a_pairwise.py -- so the one
    threshold deciding which models enter the study sat outside the config, outside the
    config hash, and outside this audit, which exists to catch exactly that.
    """
    cfg = load_config(config)
    assert cfg.gates.reliability_min == 0.70

    # It must be READ from config, not re-hardcoded next to it.
    gate_src = (SRC / "experiments" / "pass_a_pairwise.py").read_text(encoding="utf-8")
    assert "cfg.gates.reliability_min" in gate_src
    assert "RELIABILITY_MIN" not in gate_src, "the module constant is back"

    # NEGATIVE CONTROL: the threshold must be live, not decorative.
    strict = cfg.model_copy(update={"gates": cfg.gates.model_copy(
        update={"reliability_min": 0.99})})
    assert strict.gates.reliability_min != cfg.gates.reliability_min
    # ...and it must not touch any cached forward pass.
    for stage in ("pass_a", "pass_b", "pass_c"):
        assert strict.hash(stage) == cfg.hash(stage)
    assert strict.hash() != cfg.hash(), "the run identity must record the gate threshold"
