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
    for leak in ("you chose", "assigned", "another person", "receive"):
        assert leak.lower() not in body.lower()


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


def test_no_generate_call_anywhere_in_src():
    """Hard constraint 1: every DV is read from one forward pass at one token
    position. No open-ended generation in the measurement path, ever."""
    root = Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        for pattern in (".generate(", "GenerationConfig", "pipeline("):
            if pattern in text and "prompt_pipeline" not in text:
                offenders.append(f"{path.relative_to(root)}: {pattern}")
    assert not offenders, f"generation reached the measurement path: {offenders}"
