"""The choice readout and Pass C's design validation -- both previously untested.

`read_choice` produces the discrete commitment that six of the eight conditions designate,
so an error here propagates into every own-pick contrast. `pass_c._validate` asserts the
design properties the causal interpretation rests on, and nothing asserted that it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.readout.choice import read_choice
from src.readout.digits import DigitMap

VOCAB = 32


def _lmap() -> DigitMap:
    """Two labels: label 0 owns ids 10 and 11 (e.g. 'A' and ' A'), label 1 owns id 20."""
    return DigitMap(
        digits=(0, 1),
        ids_by_digit={0: (10, 11), 1: (20,)},
        flat_ids=(10, 11, 20),
        digit_index=(0, 0, 1),
        surfaces_by_digit={0: ("A", " A"), 1: ("B",)},
    )


def _logits(rows: list[dict[int, float]], background: float = -30.0) -> torch.Tensor:
    """Non-label tokens sit at `background`.

    It matters: with the background pinned very low, EVERY row puts ~all mass on the
    labels and `mass` is 1.0 regardless -- which made the first version of the mass test
    fail against correct code. A realistic background is what lets mass vary.
    """
    out = torch.full((len(rows), VOCAB), background)
    for r, spec in enumerate(rows):
        for token_id, value in spec.items():
            out[r, token_id] = value
    return out


def test_argmax_is_over_labels_and_surface_variants_are_summed():
    """Label 0's two surface forms must be pooled BEFORE the argmax.

    Compared per-id, id 20 (5.0) beats each of ids 10 and 11 (4.5) individually, so a
    readout that argmaxed over token ids would pick label 1. Pooling label 0's two
    surfaces is what makes label 0 win, and that pooling is the point of the map.
    """
    out = read_choice(_logits([{10: 4.5, 11: 4.5, 20: 5.0}]), _lmap())
    assert out.index[0] == 0, "surface variants were not summed before the argmax"
    assert out.probs[0].sum() == pytest.approx(1.0)
    assert out.probs[0, 0] > out.probs[0, 1]


def test_probabilities_are_renormalised_within_the_labels_but_mass_is_not():
    """`probs` answers "which label", `mass` answers "was this the label position at all".

    Conflating them would hide a prompt whose readout position is not the decision
    position -- A1.6's readout-mass floor is built on `mass` staying un-normalised.
    """
    # Background at 0.0, so the rest of the vocabulary competes for mass. Every label id
    # is set explicitly -- leaving one at the background silently hands it that mass, and
    # doing so is what made the first two drafts of this test wrong.
    on_position = read_choice(
        _logits([{10: 8.0, 11: -20.0, 20: 4.0}], background=0.0), _lmap())
    off_position = read_choice(
        _logits([{10: 0.0, 11: -28.0, 20: -4.0}], background=0.0), _lmap())

    for out in (on_position, off_position):
        assert out.probs[0].sum() == pytest.approx(1.0), "probs must renormalise"

    # THE POINT: the two rows carry the SAME label preference and wildly different mass.
    # A1.6's readout-mass floor lives on `mass`; renormalising it away would hide a prompt
    # whose readout position is not the decision position.
    assert on_position.probs[0, 0] == pytest.approx(off_position.probs[0, 0], abs=1e-4)
    assert on_position.mass[0] > 0.9
    assert off_position.mass[0] < 0.1


def test_margin_is_the_gap_between_the_two_labels():
    out = read_choice(_logits([{10: 6.0, 20: 6.0}, {10: 10.0, 20: -2.0}]), _lmap())
    assert out.margin[0] == pytest.approx(0.0, abs=1e-6), "a tie must have zero margin"
    assert out.margin[1] > 0.9
    assert out.margin[1] == pytest.approx(abs(out.probs[1, 0] - out.probs[1, 1]))


def test_an_exact_tie_resolves_deterministically():
    """A tie is pure position capture. It must not be resolved by anything random, or the
    same prompt would yield different choices across runs and break the seeded design."""
    logits = _logits([{10: 6.0, 20: 6.0}])
    picks = {int(read_choice(logits, _lmap()).index[0]) for _ in range(5)}
    assert len(picks) == 1, f"a tie resolved non-deterministically: {picks}"


def test_zero_label_mass_raises_rather_than_returning_a_plausible_choice():
    """NEGATIVE CONTROL for the whole readout: with no probability on any label token the
    argmax is meaningless, and a meaningless choice would still look like a choice."""
    dead = torch.full((1, VOCAB), -30.0)
    dead[0, 10] = dead[0, 11] = dead[0, 20] = -float("inf")
    with pytest.raises(ValueError, match="zero probability mass"):
        read_choice(dead, _lmap())


def test_wrong_logit_shape_is_refused():
    with pytest.raises(ValueError, match=r"expected \[n, vocab\]"):
        read_choice(torch.zeros(VOCAB), _lmap())
    with pytest.raises(ValueError, match=r"expected \[n, vocab\]"):
        read_choice(torch.zeros(2, 3, VOCAB), _lmap())


# ---------------------------------------------------------------------------
# Pass C design validation


def _frame(cfg, n_pairs=3, break_=None) -> pd.DataFrame:
    from src.analysis.spread_model import PRE_SENTINEL
    from src.experiments.pass_c import OWN_PICK_CONDITIONS

    rows = []
    for p in range(n_pairs):
        i1, i2 = f"d/a{p}", f"d/b{p}"
        own = i1 if p % 2 == 0 else i2
        rnd = i2 if p % 2 == 0 else i1
        for order in (0, 1):
            base = dict(pair_id=f"d/p{p}", template="t0", option_order=order,
                        item1_id=i1, item2_id=i2)
            rows.append({**base, "condition": PRE_SENTINEL, "timepoint": "pre",
                         "designated_item_id": i1})
            for c in cfg.pass_c.conditions:
                d = own if c in OWN_PICK_CONDITIONS else rnd
                if break_ == "own" and c == "3p-yoked" and p == 0 and order == 0:
                    d = i2 if own == i1 else i1
                if break_ == "random" and c == "3p-random" and p == 0:
                    d = i1 if rnd == i2 else i2
                if break_ == "stray" and c == "chose" and p == 0 and order == 0:
                    d = "d/SOMETHING-ELSE"
                rows.append({**base, "condition": c, "timepoint": "post",
                             "designated_item_id": d})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg():
    from src.config import load_config

    return load_config("configs/stage0_qwen2.5-0.5b.yaml")


def test_a_well_formed_pass_c_frame_validates(cfg):
    """PRECONDITION for every rejection below: the honest frame must pass."""
    from src.experiments.pass_c import _validate

    _validate(cfg, _frame(cfg))


@pytest.mark.parametrize("break_,expected", [
    ("own", "designation-matched"),
    ("random", "confound endorsement with designation"),
    ("stray", "not in the pair"),
])
def test_designation_violations_are_caught(cfg, break_, expected):
    """Each is a property a contrast depends on, not a data-hygiene nicety.

    `own`    -- the 2x2 and 3p-yoked must share the model's own pick, or the primary
                contrast is not designation-matched.
    `random` -- random and 3p-random must share one designation, or 3p-random - random
                confounds endorsement with which item was designated.
    `stray`  -- a designated item outside the pair is a designation bug that would
                otherwise flow into the DV as a valid-looking row.
    """
    from src.experiments.pass_c import _validate

    with pytest.raises(RuntimeError, match=expected):
        _validate(cfg, _frame(cfg, break_=break_))


def test_the_shared_pre_row_must_be_emitted_once_per_cell(cfg):
    """Emitting pre per condition would feed one measurement to the likelihood eight
    times and over-weight the baseline."""
    from src.analysis.spread_model import PRE_SENTINEL
    from src.experiments.pass_c import _validate

    frame = _frame(cfg)
    duplicated = pd.concat([frame, frame[frame["condition"] == PRE_SENTINEL]])
    with pytest.raises(RuntimeError, match="pre rows"):
        _validate(cfg, duplicated)


# ---------------------------------------------------------------------------
# A4.5: the graded DV


@pytest.mark.parametrize("index,margin,expected", [
    (0, 0.0, 0.50), (1, 0.0, 0.50),          # a tie is 0.5 whichever way argmax broke it
    (0, 0.5, 0.75), (1, 0.5, 0.25),
    (0, 1.0, 1.00), (1, 1.0, 0.00),
])
def test_p_slot1_recovers_the_two_label_distribution_exactly(index, margin, expected):
    """`item1_wins` keeps the SIGN of the model's preference and discards its STRENGTH.

    Nothing is approximated in the recovery: read_choice renormalises over the labels, so
    with two of them p(top) = (1+margin)/2 and the pair (index, margin) determines the
    distribution exactly. The strength was computed on every trial and thrown away before
    it reached the artifact (A4.5).
    """
    from src.experiments.pass_c import p_slot1

    assert float(p_slot1(index, margin)) == pytest.approx(expected)


def test_the_graded_readout_agrees_with_the_argmax_it_replaces():
    """The graded value must never contradict the binary one, or the two analyses of the
    same trial would disagree about which item the model preferred."""
    import numpy as np

    from src.experiments.pass_c import p_slot1

    rng = np.random.default_rng(0)
    idx = rng.integers(0, 2, 500)
    margin = rng.uniform(0.0, 1.0, 500)
    p = p_slot1(idx, margin)

    slot1_preferred = p > 0.5
    assert np.array_equal(slot1_preferred, idx == 0), (
        "the graded readout and the argmax disagree on the winner")
    # ...and it carries strictly more: the same winner spans the full confidence range.
    won = p[idx == 0]
    assert won.min() < 0.6 and won.max() > 0.95, (
        "the graded value is not actually varying; it would add nothing over the argmax")


def test_pass_c_persists_the_graded_dv():
    """It costs no forward passes -- the number is already computed. It was discarded."""
    import inspect

    from src.experiments import pass_c

    src = inspect.getsource(pass_c.run_pass_c)
    assert '"p_item1"' in src, "the graded pre readout is not persisted"
    assert 'frame.loc[idx, "p_item1"]' in src, "the graded post readout is not persisted"
    # the binary DV must SURVIVE: it is the preregistered primary's observable
    assert '"item1_wins"' in src, "item1_wins was removed; the preregistered DV needs it"
