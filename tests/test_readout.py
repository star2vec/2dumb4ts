"""Readout correctness. These guard the measurement, so they are the tests that
matter most: a silent readout bug would produce plausible numbers that mean
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import ReadoutConfig, load_config
from src.readout.digits import DigitMapError, build_digit_map, build_label_map
from src.readout.expected_value import read_expected_value, reverse_polarity

TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
]

DIGITS = tuple(range(1, 10))


def _tokenizer(name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{name} unavailable: {type(exc).__name__}")


@pytest.mark.parametrize("name", TOKENIZERS)
def test_digit_map_covers_1_to_9_without_collision(name):
    """The readout must be available and unambiguous on every model in the set."""
    dmap = build_digit_map(_tokenizer(name), DIGITS)
    assert set(dmap.ids_by_digit) == set(DIGITS)
    for d in DIGITS:
        assert dmap.ids_by_digit[d], f"digit {d} has no single-token surface form"
    flat = list(dmap.flat_ids)
    assert len(flat) == len(set(flat)), "a token id is claimed by two digits"
    assert len(dmap.digit_index) == len(flat)


@pytest.mark.parametrize("name", TOKENIZERS)
def test_ten_is_not_single_token_or_collides_with_one(name):
    """The empirical fact that forced the 1-9 scale.

    Either "10" is multi-token (so unreadable at one position), or its single
    token differs from "1". Whichever holds, a 1-10 scale cannot be read
    uniformly across the set at one position -- which is why the scale is 1-9.
    """
    tok = _tokenizer(name)
    ten = tok.encode("10", add_special_tokens=False)
    one = tok.encode("1", add_special_tokens=False)
    assert len(one) == 1
    if len(ten) > 1:
        assert ten[0] == one[0], (
            "multi-token '10' whose first token is not '1' would be a different "
            "problem than the one documented; re-derive the scale choice"
        )


@pytest.mark.parametrize("name", TOKENIZERS)
def test_option_labels_are_single_tokens(name):
    lmap = build_label_map(_tokenizer(name), ("A", "B"))
    assert len(lmap.digits) == 2
    assert lmap.ids_by_digit[0] and lmap.ids_by_digit[1]
    assert not set(lmap.ids_by_digit[0]) & set(lmap.ids_by_digit[1])


def test_digit_map_aborts_when_a_digit_is_missing():
    """Incomplete coverage is fatal, never a silently degraded measurement."""

    class Stub:
        def encode(self, s, add_special_tokens=False):
            return [1, 2] if s.strip() == "7" else [ord(s.strip())]

        def get_vocab(self):
            return {}

        def convert_tokens_to_string(self, toks):
            return ""

    with pytest.raises(DigitMapError, match="no single-token surface form"):
        build_digit_map(Stub(), DIGITS)


# ---------------------------------------------------------------------------
# expected value


def _fake_map(digits, n):
    """A minimal DigitMap over a toy vocabulary, for exercising the arithmetic."""
    from src.readout.digits import DigitMap

    return DigitMap(
        digits=tuple(digits),
        ids_by_digit={d: (i,) for i, d in enumerate(digits)},
        flat_ids=tuple(range(n)),
        digit_index=tuple(range(n)),
        surfaces_by_digit={d: (str(d),) for d in digits},
    )


def test_expected_value_is_not_argmax_on_a_bimodal_distribution():
    """The reason ratings are an expected value.

    Mass split evenly between 3 and 9 has EV 6 -- a rating the model never
    'says'. Argmax reports 3 or 9 arbitrarily and discards the graded
    information the difference-of-differences DV is built on.
    """
    dmap = _fake_map(DIGITS, 9)
    logits = torch.full((1, 9), -30.0)
    logits[0, 2] = 0.0  # digit 3
    logits[0, 8] = 0.0  # digit 9

    out = read_expected_value(logits, dmap)
    assert out.value[0] == pytest.approx(6.0, abs=1e-4)
    assert out.argmax[0] in (3.0, 9.0)
    assert out.value[0] != out.argmax[0]


def test_expected_value_matches_hand_computation():
    dmap = _fake_map(DIGITS, 9)
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    out = read_expected_value(logits, dmap)
    expected = 1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4
    assert out.value[0] == pytest.approx(expected, abs=1e-5)
    assert out.mass[0] == pytest.approx(1.0, abs=1e-5)


def test_surface_form_variants_sum_into_one_digit_bin():
    """' 1' and '1' are the same rating and must not be double counted."""
    from src.readout.digits import DigitMap

    dmap = DigitMap(
        digits=(1, 2),
        ids_by_digit={1: (0, 1), 2: (2,)},
        flat_ids=(0, 1, 2),
        digit_index=(0, 0, 1),
        surfaces_by_digit={1: ("1", " 1"), 2: ("2",)},
    )
    # 0.25 + 0.25 on the two variants of "1", 0.5 on "2" -> EV = 1*0.5 + 2*0.5
    logits = torch.log(torch.tensor([[0.25, 0.25, 0.5]]))
    out = read_expected_value(logits, dmap)
    assert out.value[0] == pytest.approx(1.5, abs=1e-5)


def test_digit_mass_reports_probability_outside_the_scale():
    """digit_mass is the diagnostic that catches a misplaced readout position."""
    dmap = _fake_map((1, 2), 2)
    # Only 20% of the full-vocabulary mass lands on digits.
    logits = torch.log(torch.tensor([[0.1, 0.1, 0.8]]))
    out = read_expected_value(logits, dmap)
    assert out.mass[0] == pytest.approx(0.2, abs=1e-5)
    # Renormalization is over the digit bins only.
    assert out.value[0] == pytest.approx(1.5, abs=1e-5)


def test_expected_value_rejects_wrong_shape():
    dmap = _fake_map(DIGITS, 9)
    with pytest.raises(ValueError, match=r"\[n, vocab\]"):
        read_expected_value(torch.zeros(2, 3, 9), dmap)


# ---------------------------------------------------------------------------
# polarity


def test_reversal_constant_is_ten_for_a_1_to_9_scale():
    """11 - x belongs to a 1-10 scale and would bias every collapsed mean by +0.5."""
    assert ReadoutConfig().reversal_constant == 10


def test_reverse_polarity_maps_endpoints_and_midpoint():
    rc = ReadoutConfig().reversal_constant
    assert reverse_polarity(np.array([1.0]), rc)[0] == pytest.approx(9.0)
    assert reverse_polarity(np.array([9.0]), rc)[0] == pytest.approx(1.0)
    assert reverse_polarity(np.array([5.0]), rc)[0] == pytest.approx(5.0)


def test_polarity_collapse_is_identity_when_the_model_is_consistent():
    """A model that follows both anchor definitions collapses to its own rating."""
    rc = ReadoutConfig().reversal_constant
    asc = np.array([7.0, 3.0, 8.5])
    desc = rc - asc  # a perfectly polarity-consistent model
    collapsed = (asc + reverse_polarity(desc, rc)) / 2
    np.testing.assert_allclose(collapsed, asc)


def test_polarity_collapse_flattens_a_polarity_blind_model():
    """The failure mode the validity gate exists to catch.

    A model that ignores the anchor definition and always answers ascending has
    asc == desc, so the collapsed score is the constant midpoint for every item
    and sigma_between goes to zero.
    """
    rc = ReadoutConfig().reversal_constant
    asc = np.array([7.0, 3.0, 8.5])
    desc = asc.copy()
    collapsed = (asc + reverse_polarity(desc, rc)) / 2
    np.testing.assert_allclose(collapsed, np.full_like(asc, rc / 2))
    assert collapsed.std() == pytest.approx(0.0)


def test_config_scale_is_locked_to_1_9():
    cfg = load_config("configs/stage0_qwen2.5-0.5b.yaml")
    assert cfg.readout.digits == DIGITS
    assert cfg.readout.reversal_constant == 10
