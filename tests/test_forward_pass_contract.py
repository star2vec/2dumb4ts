"""The two functions every measurement passes through, neither of which had a test.

`Runner.render` builds the prompt and `Runner.last_logits` reads position -1 of it. Coverage
put runner.py at 36%, and both were in the uncovered part -- they need weights, so nothing
exercised them. But their CONTRACTS do not need weights, and the contracts are where the
damage would be:

  - `last_logits` reads `[:, -1, :]`, which is the true final token only under LEFT padding.
    `load_runner` sets `padding_side = "left"`, and nothing asserted it. Under right padding
    every sequence shorter than the batch maximum would be read at a PAD position, silently,
    producing plausible numbers for the wrong token.
  - the tokenizer is called with `add_special_tokens=False` on the grounds that
    `apply_chat_template` already emits BOS. That is an assumption about five different
    model families and it was never checked.
"""

from __future__ import annotations

import pytest
import torch

TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
]
MESSAGES = [{"role": "user", "content": "Here are two options.\n\nA. x\nB. y\n\nReply A or B."}]


def _tokenizer(name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{name} unavailable: {type(exc).__name__}")


def test_load_runner_sets_left_padding_and_the_readout_depends_on_it():
    """The invariant and its consumer, pinned together so they cannot drift apart."""
    import inspect

    from src.models import runner as rn

    setup = inspect.getsource(rn.load_runner)
    assert 'padding_side = "left"' in setup, "left padding is no longer established"

    read = inspect.getsource(rn.Runner.last_logits)
    assert "[:, -1, :]" in read, "the readout no longer takes the final position"


@pytest.mark.parametrize("name", TOKENIZERS)
def test_left_padding_puts_the_real_final_token_at_index_minus_one(name):
    """The actual failure, demonstrated on real tokenizers rather than asserted.

    Two prompts of different lengths batched together: under LEFT padding both end on their
    own final token; under RIGHT padding the shorter one ends on a pad. This is the exact
    condition every batched readout in the project runs under.
    """
    tok = _tokenizer(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    short = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                    tokenize=False, add_generation_prompt=True)
    long = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)

    tok.padding_side = "left"
    left = tok([short, long], return_tensors="pt", padding=True, add_special_tokens=False)
    tok.padding_side = "right"
    right = tok([short, long], return_tensors="pt", padding=True, add_special_tokens=False)

    unpadded = tok(short, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    true_final = int(unpadded[-1])

    assert int(left["input_ids"][0, -1]) == true_final, "left padding is not doing its job"
    assert int(left["attention_mask"][0, -1]) == 1

    # NEGATIVE CONTROL: right padding must actually break it, or the invariant is moot.
    if right["input_ids"].shape[1] > unpadded.shape[0]:
        assert int(right["input_ids"][0, -1]) == tok.pad_token_id
        assert int(right["attention_mask"][0, -1]) == 0


@pytest.mark.parametrize("name", TOKENIZERS)
def test_add_special_tokens_false_loses_nothing_the_template_did_not_add(name):
    """`last_logits` passes add_special_tokens=False because the chat template emits BOS.

    If a family's template does NOT emit its BOS, that flag silently drops it and every
    prompt for that model starts wrong. Checked per family rather than assumed.
    """
    tok = _tokenizer(name)
    rendered = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    without = tok(rendered, add_special_tokens=False)["input_ids"]
    with_specials = tok(rendered, add_special_tokens=True)["input_ids"]

    if tok.bos_token_id is None:
        assert without == with_specials, "no BOS exists, so the flag must change nothing"
        return
    # Either the template already emitted BOS (so the flag is a no-op and the prompt keeps
    # it), or add_special_tokens would have DOUBLED it.
    assert without[0] == tok.bos_token_id, (
        f"{name}: the template does not emit BOS, so add_special_tokens=False drops it")
    assert with_specials[:2] == [tok.bos_token_id, tok.bos_token_id] or (
        with_specials == without), f"{name}: unexpected special-token behaviour"


@pytest.mark.parametrize("name", TOKENIZERS)
def test_render_opens_the_assistant_turn_and_leaves_the_readout_position_next(name):
    """The token after the rendered prompt IS the measurement. If the template did not open
    the assistant turn, position -1 would be the end of the user turn instead."""
    from src.models.runner import Runner

    tok = _tokenizer(name)
    rendered = Runner.render(type("R", (), {"tokenizer": tok})(), MESSAGES)
    plain = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False)

    assert rendered.startswith(plain[: len(plain) // 2])
    assert len(rendered) > len(plain), "add_generation_prompt added nothing"
    # Family-agnostic: Qwen and Llama open an "assistant" turn, Gemma-2 a "model" one.
    suffix = rendered[len(plain) - 1 :].lower()
    assert "assistant" in suffix or "model" in suffix, (
        f"{name}: the generation prompt does not open a reply turn; position -1 would be "
        f"the end of the user turn instead. suffix={suffix!r}")
