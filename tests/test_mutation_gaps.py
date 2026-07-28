"""Tests written to kill surviving mutants.

A mutation sweep broke src/ one edit at a time and asked whether any test objected. 51 of
96 mutants survived -- a 47% kill rate. Most survivors were equivalent mutants (config
defaults that base.yaml overrides) or uncovered CLI code, but the ones below changed
behaviour that a preregistered constraint depends on, and nothing noticed.

Each test names the mutation it kills, so the claim is checkable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


# --- config.py:134  range(scale_min, scale_max + 1) -> + 2 --------------------

def test_the_digit_set_is_exactly_one_through_nine():
    """3.2's hard constraint. Mutating the `+ 1` to `+ 2` yields digits 1..10 and no test
    objected -- yet a 1-10 scale is UNREADABLE at one token position on Qwen2.5 and
    Gemma-2, where "10" is two tokens whose first is byte-identical to "1"."""
    cfg = load_config(CONFIG)
    assert cfg.readout.digits == tuple(range(1, 10))
    assert len(cfg.readout.digits) == 9
    assert 10 not in cfg.readout.digits, "a 1-10 scale is unreadable at one position (3.2)"
    assert cfg.readout.reversal_constant == 10, "10 - x for a 1-9 scale, never 11 - x"


# --- readout/digits.py:98  owner[tid] != d -> == d ---------------------------

def test_a_digit_token_claimed_by_two_digits_is_fatal():
    """The collision guard. Flipping `!=` to `==` disables it and nothing objected.

    Silent partial coverage would be a degraded measurement masquerading as a clean one,
    which is why digits.py makes it fatal rather than a warning.
    """
    from src.readout.digits import DigitMapError, build_digit_map

    class Colliding:
        """Every digit encodes to the SAME id, so all nine collide."""

        def encode(self, s, add_special_tokens=False):
            return [7]

        def get_vocab(self):
            return {"x": 7}

        def convert_tokens_to_string(self, toks):
            return "1"

    with pytest.raises(DigitMapError, match="collision"):
        build_digit_map(Colliding(), tuple(range(1, 10)))


# --- readout/choice.py:55  mass.unsqueeze(-1) -> unsqueeze(0) ----------------

def test_choice_probabilities_renormalise_per_row_not_across_the_batch():
    """Mutating the broadcast axis survived because every existing test used one or two
    rows, where the two axes coincide. With several rows of DIFFERENT mass they do not:
    normalising across the batch would make each row's probabilities depend on the others.
    """
    from src.readout.choice import read_choice
    from src.readout.digits import DigitMap

    lmap = DigitMap(digits=(0, 1), ids_by_digit={0: (10,), 1: (20,)},
                    flat_ids=(10, 20), digit_index=(0, 1),
                    surfaces_by_digit={0: ("A",), 1: ("B",)})
    logits = torch.zeros(4, 64)
    for r, (a, b) in enumerate([(8.0, 4.0), (-2.0, -6.0), (1.0, 1.0), (12.0, 0.0)]):
        logits[r, 10], logits[r, 20] = a, b

    out = read_choice(logits, lmap)
    assert out.probs.shape == (4, 2)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-5)
    # Rows 0 and 1 share a label-logit DIFFERENCE of 4, so identical probs...
    np.testing.assert_allclose(out.probs[0], out.probs[1], atol=1e-5)
    # ...while their MASS differs by orders of magnitude. Per-row normalisation is the
    # only way both can hold.
    assert out.mass[0] > 10 * out.mass[1]
    assert out.probs[2, 0] == pytest.approx(0.5, abs=1e-5)


# --- readout/expected_value.py:69  mass <= 0 -> mass < 0 ---------------------

def test_zero_digit_mass_raises_in_the_expected_value_readout_too():
    """The choice readout had this test; the rating readout did not, so weakening its
    guard from `<= 0` to `< 0` survived."""
    from src.readout.digits import DigitMap
    from src.readout.expected_value import read_expected_value

    dmap = DigitMap(digits=(1, 2), ids_by_digit={1: (10,), 2: (20,)},
                    flat_ids=(10, 20), digit_index=(0, 1),
                    surfaces_by_digit={1: ("1",), 2: ("2",)})
    dead = torch.full((1, 64), -30.0)
    dead[0, 10] = dead[0, 20] = -float("inf")
    with pytest.raises(ValueError, match="zero probability mass"):
        read_expected_value(dead, dmap)


# --- bradley_terry.py:462  the sigmoid ---------------------------------------

def test_the_sigmoid_is_a_sigmoid():
    """Mutating a constant inside `1/(1+exp(-z))` survived, which means the functions
    built on it were never checked numerically -- only structurally."""
    from src.analysis.bradley_terry import _sigmoid

    z = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    got = _sigmoid(z)
    np.testing.assert_allclose(got, 1.0 / (1.0 + np.exp(-z)), rtol=1e-12)
    assert got[2] == pytest.approx(0.5), "sigmoid(0) must be exactly 0.5"
    np.testing.assert_allclose(got + _sigmoid(-z), 1.0, rtol=1e-12)
    assert np.all(np.diff(got) > 0), "must be strictly increasing"


# --- readout/pairwise.py:341  wide[0] == wide[1] -> != ------------------------

def test_order_invariance_counts_agreement_not_disagreement():
    """Flipping the comparison inverts the statistic and no test objected.

    1.0 is content-driven and 0.0 is pure position capture, so an inverted statistic would
    report a position-only responder as perfectly content-driven.
    """
    import pandas as pd

    from src.readout.pairwise import order_invariance

    # A content-driven responder: same winner under both orders.
    same = pd.DataFrame([
        {"arm": "digits", "template": "t0", "item_id": f"i{i}", "anchor_id": "a",
         "order": o, "item_wins": True, "readout_valid": True}
        for i in range(6) for o in (0, 1)
    ])
    # A position-only responder: always picks slot 1, so the winner flips every reversal.
    flips = pd.DataFrame([
        {"arm": "digits", "template": "t0", "item_id": f"i{i}", "anchor_id": "a",
         "order": o, "item_wins": (o == 0), "readout_valid": True}
        for i in range(6) for o in (0, 1)
    ])
    a = order_invariance(same).iloc[0]
    b = order_invariance(flips).iloc[0]
    assert float(a["order_invariance"]) == pytest.approx(1.0)
    assert float(b["order_invariance"]) == pytest.approx(0.0)
    assert a["regime"] == "content-driven"
    assert b["regime"] == "position-dominated", (
        "an inverted statistic would call a position-only responder content-driven")


# --- stimuli/build.py:355  condition == "structure-control" -> != ------------

def test_structure_control_asks_for_confirmation_and_chose_asks_for_a_choice():
    """Existing tests checked the turn COUNT, not its CONTENT, so flipping the condition
    test survived. The 2x2 rests on structure-control carrying a turn WITHOUT a choice:
    if it asked for a choice too, the contrast would isolate nothing.
    """
    from src.stimuli.build import load_templates, post_dv_messages

    cfg = load_config(CONFIG)
    t = load_templates(cfg)[0]
    chose = post_dv_messages(t, "X", "Y", "X", "Y", "chose", "A", cfg)
    ctrl = post_dv_messages(t, "X", "Y", "X", "Y", "structure-control", "A", cfg)

    assert len(chose) == len(ctrl) == 3, "both carry an assistant turn"
    # t.choice / t.confirm carry {label_a}/{label_b} placeholders; compare rendered text.
    labels = cfg.readout.option_labels
    choice_q = t.choice.format(label_a=labels[0], label_b=labels[1])
    confirm_q = t.confirm.format(label_a=labels[0], label_b=labels[1])
    assert choice_q in chose[0]["content"], "chose must ask for a choice"
    assert confirm_q in ctrl[0]["content"], "structure-control must ask for confirmation"
    assert choice_q not in ctrl[0]["content"], (
        "structure-control must NOT ask for a choice, or the 2x2 isolates nothing")
    assert chose[1]["content"] != ctrl[1]["content"], "the replies must differ"


# --- bradley_terry.py:359  (wide[0] == wide[1]) -> != ------------------------

def test_excess_consistency_measures_agreement_between_orders():
    """A2.1's fit-quality statistic, checked numerically rather than structurally.

    Flipping `==` to `!=` inverts observed consistency and survived. That matters more here
    than anywhere: R3 and R4 were BOTH estimator errors in this exact area -- a statistic
    computed correctly but compared against the wrong null -- so the arithmetic gets
    pinned against data whose consistency is known by construction.
    """
    import pandas as pd

    from src.analysis.bradley_terry import _sigmoid, excess_consistency_slope

    items = [f"i{i:02d}" for i in range(30)]
    theta = dict(zip(items, np.linspace(-2.0, 2.0, len(items))))
    alpha = {"a0": -0.4, "a1": 0.4}
    beta = 0.8

    class _Fit:
        pass

    fit = _Fit()
    fit.theta = pd.DataFrame({"item_id": items, "theta_mean": [theta[i] for i in items]})
    fit.anchors = pd.DataFrame({"anchor_id": list(alpha), "alpha_mean": list(alpha.values())})
    fit.beta = pd.DataFrame({"template": ["t0"], "beta_mean": [beta]})

    def frame(consistent: bool):
        rows = []
        for i in items:
            for a in alpha:
                for order in (0, 1):
                    # consistent -> same winner under both orders; else it flips.
                    wins = True if consistent else (order == 0)
                    rows.append({"arm": "digits", "readout_valid": True, "template": "t0",
                                 "item_id": i, "anchor_id": a, "order": order,
                                 "item_wins": wins})
        return pd.DataFrame(rows)

    # The model's own prediction for these cells, which `excess` subtracts.
    x = np.array([abs(theta[i] - alpha[a]) for i in items for a in alpha])
    pred = _sigmoid(x + beta) * _sigmoid(x - beta) + (1 - _sigmoid(x + beta)) * (1 - _sigmoid(x - beta))

    always = excess_consistency_slope(frame(True), fit)
    never = excess_consistency_slope(frame(False), fit)

    assert always["mean_excess"] == pytest.approx(1.0 - pred.mean(), abs=1e-6), (
        "perfectly consistent data must give excess = 1 - pred")
    assert never["mean_excess"] == pytest.approx(0.0 - pred.mean(), abs=1e-6), (
        "perfectly inconsistent data must give excess = 0 - pred")
    # NEGATIVE CONTROL: the two must differ by exactly 1, or the statistic is not
    # measuring agreement at all.
    assert always["mean_excess"] - never["mean_excess"] == pytest.approx(1.0, abs=1e-6)
    assert always["n_cells"] == len(items) * len(alpha)
