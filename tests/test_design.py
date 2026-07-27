"""Stimuli, template invariants, and counterbalancing.

These assert the properties the causal interpretation depends on. If any of them
breaks, the contrasts stop meaning what the preregistration says they mean.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CONDITIONS, load_config
from src.experiments.pass_b import build_pairs
from src.stimuli.build import (
    VERBATIM_PAIRS,
    assert_template_invariants,
    audit_templates,
    balanced_designation,
    load_items,
    load_templates,
    pass_a_trials,
    post_messages,
    pre_messages,
)

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def templates(cfg):
    return load_templates(cfg)


# ---------------------------------------------------------------------------
# items


def test_item_pool_is_400_items_100_per_domain(cfg):
    items = load_items(cfg)
    assert len(items) == 400
    counts = pd.Series([i.domain for i in items]).value_counts()
    assert set(counts.index) == set(cfg.stimuli.domains)
    assert (counts == 100).all()


def test_item_ids_are_unique_and_slug_derived(cfg):
    items = load_items(cfg)
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))
    for i in items:
        assert i.id.startswith(f"{i.domain}/")
        assert re.fullmatch(r"[a-z0-9-]+", i.id.split("/", 1)[1])


def test_no_item_label_contains_a_digit_or_a_factual_marker(cfg):
    """Stage 0 stimuli must not carry anything correctness-bearing.

    Digits are the sharpest automatable proxy: a year, a price, a spec, or a
    quantity is the kind of verifiable property that could engage truth-tracking,
    and the two paradigms must not overlap.
    """
    banned = re.compile(r"\d|\bbest\b|\bworst\b|\bfastest\b|\bhealthiest\b", re.I)
    offenders = [i.label for i in load_items(cfg) if banned.search(i.label)]
    assert not offenders, f"items carry correctness-bearing content: {offenders}"


# ---------------------------------------------------------------------------
# template invariants


def test_template_invariants_hold(cfg, templates):
    assert len(templates) == 5
    assert_template_invariants(templates)  # raises on violation


def test_designation_only_conditions_are_verbatim_identical(templates):
    """yoked/random and 3p-yoked/3p-random may differ ONLY in which item is
    designated. Any wording difference would confound the selection-artifact
    estimate with a text difference -- and that estimate is what licenses the
    claim that the artifact cancels in the primary contrast."""
    for t in templates:
        for a, b in VERBATIM_PAIRS:
            assert t.antecedent[a] == t.antecedent[b], f"{t.id}: {a} != {b}"


def test_receipt_is_shared_across_all_five_conditions(templates):
    """Receipt matching by construction: without it, 3p-yoked - yoked would
    confound endorsement with ownership."""
    for t in templates:
        rendered = {
            c: t.receipt.format(designated="X", other="Y") for c in CONDITIONS
        }
        assert len(set(rendered.values())) == 1


def test_every_antecedent_names_each_item_exactly_once(templates):
    """Balanced mention counts: no contrast may be confounded with how many times
    an item is named, which is a repetition confound of exactly the kind the
    rebuttal exploits."""
    for t in templates:
        for cond, text in t.antecedent.items():
            assert text.count("{designated}") == 1, f"{t.id}/{cond}"
            assert text.count("{other}") == 1, f"{t.id}/{cond}"


def test_mention_counts_are_balanced_across_conditions(templates):
    audit = audit_templates(templates)
    for col in ("mentions_designated", "mentions_other"):
        assert audit[col].nunique() == 1, f"{col} differs across conditions"


def test_turn_structure_matches_the_condition_map(cfg, templates):
    """Only TURN_CONDITIONS carry an assistant turn (A2.9.3).

    Turn presence is a modelled factor, not an unmeasured residual: it is estimated
    at both wording levels so a probe reading turn-presence can be subtracted.
    """
    from src.config import TURN_CONDITIONS
    from src.stimuli.build import post_dv_messages

    t = templates[0]
    for condition in CONDITIONS:
        msgs = post_dv_messages(t, "X", "Y", "X", "Y", condition, "1", cfg)
        assert len(msgs) == (3 if condition in TURN_CONDITIONS else 1), condition

    audit = audit_templates(templates)
    for _, row in audit.iterrows():
        assert row["n_turns"] == (3 if row["condition"] in TURN_CONDITIONS else 1)


def test_the_2x2_borrows_antecedents_byte_for_byte(templates):
    """self-recounted must reuse chose's antecedent and structure-control yoked's.

    If either defined its own text the wording factor would be more than one
    substitution and the 2x2 edges would confound wording with phrasing.
    """
    from src.config import ANTECEDENT_SOURCE

    for t in templates:
        assert t.antecedent[ANTECEDENT_SOURCE["self-recounted"]] == t.antecedent["chose"]
        assert t.antecedent[ANTECEDENT_SOURCE["structure-control"]] == t.antecedent["yoked"]


def test_dv_question_is_identical_pre_and_post(cfg, templates):
    """The pre/post contrast is only a contrast if the question does not change."""
    from src.stimuli.build import post_dv_messages, pre_dv_messages, prefer_question

    t = templates[0]
    q = prefer_question(t, cfg.readout.option_labels)
    assert q in pre_dv_messages(t, "X", "Y", cfg)[0]["content"]
    for condition in CONDITIONS:
        msgs = post_dv_messages(t, "X", "Y", "X", "Y", condition, "1", cfg)
        assert q in msgs[-1]["content"], condition


def test_reversibility_is_a_one_clause_contrast(cfg, templates):
    """chose vs chose-provisional must differ only in the finality clause."""
    from src.stimuli.build import post_dv_messages

    t = templates[0]
    a = post_dv_messages(t, "X", "Y", "X", "Y", "chose", "1", cfg)
    b = post_dv_messages(t, "X", "Y", "X", "Y", "chose-provisional", "1", cfg)
    assert len(a) == len(b)
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert diffs == [len(a) - 1], "only the final turn may differ"
    assert a[-1]["content"].replace(t.finality, t.provisional) == b[-1]["content"]


def test_chose_requires_the_models_own_label(cfg, templates):
    with pytest.raises(ValueError, match="requires the model's own chosen label"):
        post_messages(templates[0], "X", "Y", "X", "X", "Y", "chose", None, cfg)


def test_pre_context_never_contains_a_rating_or_the_manipulation(cfg, templates):
    """No anchoring channel: the model must not see its own prior rating, and the
    pre context must not leak the manipulation."""
    t = templates[0]
    msgs = pre_messages(t, "a trip to X", "a trip to Y", "a trip to X", cfg)
    assert len(msgs) == 1
    body = msgs[0]["content"]
    # NEGATIVE CONTROL: an empty or stub body would satisfy every check below.
    assert len(body) > 40 and "X" in body and "Y" in body, body

    leaks = ("you chose", "assigned", "another person", "receive")
    for leak in leaks:
        assert leak.lower() not in body.lower()
    # ...and the same predicate must actually detect a leak when one is present.
    contaminated = body + " Another person chose X for you; you receive X."
    assert any(k in contaminated.lower() for k in leaks)


def test_pre_context_is_identical_across_conditions_by_construction(cfg, templates):
    """Pre is measured once per (pair, template, option order) and reused, which
    is exact rather than an approximation: it precedes the manipulation."""
    t = templates[0]
    baseline = pre_messages(t, "X", "Y", "X", cfg)
    assert baseline == pre_messages(t, "X", "Y", "X", cfg)


# ---------------------------------------------------------------------------
# counterbalancing


def test_pass_a_is_a_complete_crossing(cfg):
    trials = list(pass_a_trials(cfg))
    assert len(trials) == 400 * 5 * 2
    frame = pd.DataFrame(trials)
    cell = frame.groupby(["item_id", "template", "polarity"]).size()
    assert (cell == 1).all(), "Pass A must be a complete crossing, each cell once"


def test_pass_a_has_no_per_run_randomness(cfg):
    """Counterbalancing by construction: two enumerations must be identical."""
    a = pd.DataFrame(list(pass_a_trials(cfg)))
    b = pd.DataFrame(list(pass_a_trials(cfg)))
    pd.testing.assert_frame_equal(a, b)


def test_balanced_designation_is_exactly_balanced_within_stratum():
    pair_ids = [f"p{i}" for i in range(40)]
    strata = ["difficult/foods"] * 20 + ["easy/foods"] * 20
    out = balanced_designation(pair_ids, strata, seed=7)
    frame = pd.DataFrame({"pair": pair_ids, "s": strata})
    frame["d"] = frame["pair"].map(out)
    for _, g in frame.groupby("s"):
        assert g["d"].sum() == len(g) // 2


def test_balanced_designation_is_seed_deterministic():
    ids = [f"p{i}" for i in range(12)]
    strata = ["x"] * 12
    assert balanced_designation(ids, strata, 3) == balanced_designation(ids, strata, 3)
    assert balanced_designation(ids, strata, 3) != balanced_designation(ids, strata, 4)


# ---------------------------------------------------------------------------
# Pass B constraints


def _synthetic_scores(cfg, seed: int = 0) -> pd.DataFrame:
    """Item scores with real spread, so pair construction has something to work on."""
    rng = np.random.default_rng(seed)
    items = load_items(cfg)
    sel = rng.normal(5.0, 1.5, size=len(items))
    return pd.DataFrame(
        {
            "item_id": [i.id for i in items],
            "domain": [i.domain for i in items],
            "score_selection": sel,
            # Analysis score is an independent measurement of the same quantity.
            "score_analysis": sel + rng.normal(0.0, 0.3, size=len(items)),
        }
    )


#: The synthetic scores below are on the theta scale with unit-ish spread, so this stands
#: in for a model's fitted sigma_item. A2.9.2 makes the tolerance 0.26 x this.
SIGMA_ITEM = 1.573


@pytest.fixture(scope="module")
def pairs(cfg):
    return build_pairs(cfg, _synthetic_scores(cfg), SIGMA_ITEM)


def test_pair_counts_and_domain_balance(cfg, pairs):
    counts = pairs.groupby(["domain", "difficulty"]).size()
    expected = cfg.n_difficult // len(cfg.stimuli.domains)
    assert (counts == expected).all()
    assert (pairs["difficulty"] == "difficult").sum() == cfg.n_difficult
    assert (pairs["difficulty"] == "easy").sum() == cfg.n_easy


def test_item_reuse_is_capped(cfg, pairs):
    long = pd.concat([pairs["item1_id"], pairs["item2_id"]])
    assert long.value_counts().max() <= cfg.pass_b.max_uses_per_item


def test_no_item_appears_in_both_difficulty_levels(pairs):
    long = pd.concat(
        [
            pairs[["item1_id", "difficulty"]].rename(columns={"item1_id": "item"}),
            pairs[["item2_id", "difficulty"]].rename(columns={"item2_id": "item"}),
        ]
    )
    assert long.groupby("item")["difficulty"].nunique().max() == 1


def test_pairs_are_within_domain(pairs):
    assert (
        pairs["item1_id"].str.split("/").str[0] == pairs["item2_id"].str.split("/").str[0]
    ).all()


def test_difficult_pairs_have_smaller_diff_than_easy(pairs):
    hard = pairs[pairs.difficulty == "difficult"]["diff_selection"]
    easy = pairs[pairs.difficulty == "easy"]["diff_selection"]
    assert hard.max() < easy.min(), "the difficulty manipulation does not separate"


def test_difficult_and_easy_are_matched_on_mean_rating(cfg, pairs):
    """Design-level matching removes the extremity/ceiling confound. Without it,
    ceiling compression alone could manufacture the difficulty interaction."""
    wide = pairs.pivot_table(
        index="matched_set", columns="difficulty", values="mean_selection"
    )
    gap = (wide["difficult"] - wide["easy"]).abs()
    assert gap.max() <= cfg.pass_b.match_tolerance(SIGMA_ITEM) + 1e-9

    hard = pairs[pairs.difficulty == "difficult"]["mean_analysis"].mean()
    easy = pairs[pairs.difficulty == "easy"]["mean_analysis"].mean()
    # On the INDEPENDENT measurement the pools should still be close.
    assert abs(hard - easy) < 0.5


def test_pair_construction_is_deterministic(cfg):
    a = build_pairs(cfg, _synthetic_scores(cfg), SIGMA_ITEM)
    b = build_pairs(cfg, _synthetic_scores(cfg), SIGMA_ITEM)
    pd.testing.assert_frame_equal(a, b)


def test_pair_construction_fails_loudly_rather_than_returning_fewer_pairs(cfg):
    """Silently accepting fewer pairs would break difficulty x domain balance."""
    tight = cfg.model_copy(
        update={"pass_b": cfg.pass_b.model_copy(
            update={"match_tolerance_sigma_fraction": 1e-9})}
    )
    with pytest.raises(RuntimeError, match="matched sets could be built"):
        build_pairs(tight, _synthetic_scores(cfg), SIGMA_ITEM)


# ---------------------------------------------------------------------------
# the no-generation constraint


BANNED_GENERATION = (".generate(", "GenerationConfig", "pipeline(")


def _generation_offenders(files) -> list[str]:
    """Banned generation constructs, checked PER LINE.

    The earlier version tested `pattern in text and "prompt_pipeline" not in text`, so the
    exemption was evaluated over the WHOLE FILE: any module mentioning `prompt_pipeline`
    anywhere became exempt from all three patterns, including `.generate(`. The exemption
    now applies only to the line that earns it.
    """
    offenders = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "prompt_pipeline" in line:
                continue
            for pattern in BANNED_GENERATION:
                if pattern in line:
                    offenders.append(f"{path.name}:{lineno}: {pattern}")
    return offenders


def test_no_generate_call_anywhere_in_src():
    """Hard constraint 1: every DV is read from one forward pass at one token
    position. No open-ended generation in the measurement path, ever.

    This is the single most load-bearing constraint in the project, and it was enforced by
    a scan that passes when the scan finds nothing -- a renamed directory or a bad path
    would have retired the constraint silently. Both halves are now proved: that the walk
    reached real files, and that the detector fires on a planted violation.
    """
    root = Path(__file__).resolve().parent.parent / "src"
    files = sorted(root.rglob("*.py"))

    # NEGATIVE CONTROL 1: the walk must actually reach the source tree.
    assert len(files) >= 15, f"the scan found only {len(files)} files under {root}"
    assert any(f.name == "run.py" for f in files), "the walk missed the experiment runner"

    assert not _generation_offenders(files), (
        f"generation reached the measurement path: {_generation_offenders(files)}")


def test_the_generation_scan_actually_detects_generation(tmp_path):
    """NEGATIVE CONTROL 2, and it covers the over-broad exemption as well.

    A file containing a legitimate `prompt_pipeline` used to be exempted from the whole
    check, so a real `.generate(` alongside it passed.
    """
    planted = tmp_path / "sneaky.py"
    planted.write_text(
        "from x import prompt_pipeline\n"
        "out = model.generate(ids, max_new_tokens=8)\n", encoding="utf-8")
    found = _generation_offenders([planted])
    assert found, "the detector missed a .generate( call sitting beside prompt_pipeline"
    assert ".generate(" in found[0]

    clean = tmp_path / "fine.py"
    clean.write_text("logits = model(ids).logits  # one forward pass\n", encoding="utf-8")
    assert not _generation_offenders([clean])


# ---------------------------------------------------------------------------
# the choice prompt -- the free choice the whole paradigm rests on


def test_the_choice_prompt_actually_shows_both_options(cfg, templates):
    """The model cannot make a free choice between options it was never shown.

    Pass C built this prompt by string-surgery on the pre prompt --
    `pre.split("\\n\\n")[0]` -- to recover the pair block. But EVERY template's pair block
    contains a blank line, so the split truncated it to its first line ("Here are two
    options.") and the choice prompt asked the model to reply A or B having named neither.

    Six of the eight conditions designate that choice, so the primary contrast would have
    been built on a pick carrying no item content at all. Asserted per template, on both
    option orders, against the rendered text.
    """
    from src.stimuli.build import choice_messages

    for t in templates:
        for fa, fb in (("a trip to Kyoto", "a trip to Lisbon"),
                       ("a trip to Lisbon", "a trip to Kyoto")):
            msgs = choice_messages(t, fa, fb, cfg)
            assert len(msgs) == 1
            body = msgs[0]["content"]
            assert fa in body, f"{t.id}: first option missing from the choice prompt"
            assert fb in body, f"{t.id}: second option missing from the choice prompt"
            for label in cfg.readout.option_labels:
                assert label in body, f"{t.id}: option label {label} missing"


def test_pass_c_builds_the_choice_prompt_from_the_shared_builder(cfg, templates):
    """A second renderer is how the two drift apart; there must be exactly one.

    NEGATIVE CONTROL: the old reconstruction is rebuilt here and shown to lose the items,
    so this test fails the moment anyone reintroduces prompt string-surgery.
    """
    import inspect

    from src.experiments import pass_c
    from src.stimuli.build import pre_dv_messages

    src = inspect.getsource(pass_c.run_pass_c)
    # Comments are allowed to NAME the banned construct -- that is how the reason for the
    # ban survives. Only executable lines are checked.
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "choice_messages(" in code, "Pass C must use the shared choice builder"
    assert "split(chr(10)" not in code and 'split("\\n\\n")' not in code, (
        "Pass C is reconstructing a prompt by splitting another prompt")

    t = templates[0]
    fa, fb = "a trip to Kyoto", "a trip to Lisbon"
    broken = pre_dv_messages(t, fa, fb, cfg)[0]["content"].split("\n\n")[0]
    assert fa not in broken and fb not in broken, (
        "the old reconstruction no longer loses the options -- this control is stale "
        "and the test above needs rewriting rather than trusting")


def test_the_match_gap_exclusion_is_vacuous_on_the_selection_score(cfg, pairs):
    """A3.13. The tolerance is a hard filter at construction, so "exclude pairs that
    exceeded it" can never exclude anything on the score it was applied to.

    Reporting that zero as a clean result would be true and misleading, so the diagnostic
    labels it as enforced upstream.
    """
    from src.experiments.pass_b import match_gap_exclusions

    out = match_gap_exclusions(cfg, pairs, SIGMA_ITEM)
    assert out["n_over_on_selection"] == 0
    assert out["selection_is_bounded_by_construction"] is True
    assert out["tolerance"] == pytest.approx(cfg.pass_b.match_tolerance(SIGMA_ITEM))

    # NEGATIVE CONTROL: the zero must come from the filter, not from a broken comparison.
    # Realized gaps sit just under the bound -- the signature of a hard filter.
    wide = pairs.pivot_table(index="matched_set", columns="difficulty",
                             values="mean_selection")
    gap = (wide["difficult"] - wide["easy"]).abs()
    assert gap.max() <= out["tolerance"] + 1e-9
    assert gap.max() > 0.5 * out["tolerance"], (
        "gaps nowhere near the bound would mean the filter is not what produced the zero")


def test_the_analysis_scale_gap_is_not_bounded_by_matching(cfg, pairs):
    """A3.13's substantive half: matching binds T1-T3, the primary model uses T4-T5.

    Section 4.4's disjoint split makes this unavoidable -- the price of an uncontaminated
    regressor is that matching cannot bind it -- so it is measured rather than assumed away.
    """
    from src.experiments.pass_b import match_gap_exclusions

    out = match_gap_exclusions(cfg, pairs, SIGMA_ITEM)
    assert out["gap_analysis_max"] > out["tolerance"], (
        "if the analysis gap were also bounded, this diagnostic would be vacuous too "
        "and A3.13's substantive half would not exist")
    assert out["n_over_on_analysis"] > 0
    assert 0.0 < out["frac_over_on_analysis"] <= 1.0
    assert len(out["excluded_matched_sets"]) == out["n_over_on_analysis"]
    assert out["n_matched_sets"] == pairs["matched_set"].nunique()
