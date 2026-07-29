"""The activation replay, and the digest assertion that is its reason for existing.

`collect_activations` is specified in preregistration_stage1.md 3 and was recorded as
unbuilt in A3.5. The probe study depends on it, and it depends on one property: that the
prompts it replays are the prompts the DV was read from. Everything else is bookkeeping.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "collect_activations", ROOT / "scripts" / "collect_activations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_pre_cells_are_rebuilt_in_pass_c_order():
    """The digest only proves anything if the cells are rebuilt in the SAME order.

    Reconstructed through the builders rather than read back from the trials frame -- a
    digest over the file would only prove the file is unchanged, not that the prompts
    still render the same way.
    """
    import pandas as pd

    from src.config import load_config
    from src.experiments import pass_c as stage_c

    cfg = load_config("configs/stage0_qwen2.5-0.5b.yaml")
    items = ["destinations/a", "destinations/b", "destinations/c", "destinations/d"]
    pairs = pd.DataFrame([
        {"pair_id": "d/p0", "item1_id": items[0], "item2_id": items[1],
         "diff_analysis": 0.2, "difficulty": "difficult",
         "theta_item1": 0.5, "theta_item2": 0.3},
        {"pair_id": "d/p1", "item1_id": items[2], "item2_id": items[3],
         "diff_analysis": 1.5, "difficulty": "easy",
         "theta_item1": -0.2, "theta_item2": 1.3},
    ])
    framed = {i: f"a trip to {i[-1].upper()}" for i in items}

    mod = _script()
    import src.stimuli.build as build

    # Patch the SCRIPT's binding, not the module's: `from ... import load_items` binds the
    # name at import, so patching src.stimuli.build.load_items does not reach it.
    original = mod.load_items
    mod.load_items = lambda c: [type("I", (), {"id": k, "framed": v})()
                                for k, v in framed.items()]
    try:
        cells = mod._cells(cfg, pairs)
    finally:
        mod.load_items = original

    n_templates = len(build.load_templates(cfg))
    assert len(cells) == 2 * n_templates * cfg.pass_c.n_option_orders
    # pair-major, then template, then order -- exactly pass_c's nesting
    assert [c["pair_id"] for c in cells[:n_templates * 2]] == ["d/p0"] * n_templates * 2
    assert [c["option_order"] for c in cells[:2]] == [0, 1]
    # and slot 1 must alternate with order, or the readout position means something else
    assert cells[0]["slot1_item_id"] == items[0]
    assert cells[1]["slot1_item_id"] == items[1]
    assert all("_msgs" in c and len(c["_msgs"]) == 1 for c in cells)
    # signed theta must survive to every row, or the positive control cannot be run
    assert all({"theta_item1", "theta_item2"} <= set(c) for c in cells)
    assert cells[0]["theta_item1"] == 0.5 and cells[0]["theta_item2"] == 0.3


def test_a_drifted_prompt_changes_the_digest():
    """The whole assertion. If a template, the chat template or the tokenizer moves, the
    digest must move with it -- otherwise activations get attached to prompts that were
    never read."""
    a = ["prompt one", "prompt two", "prompt three"]
    b = ["prompt one", "prompt two", "prompt three!"]      # one character
    dig = lambda xs: hashlib.sha256("\x00".join(xs).encode()).hexdigest()[:16]  # noqa: E731

    assert dig(a) == dig(list(a)), "the digest is not deterministic"
    assert dig(a) != dig(b), "a changed prompt did not change the digest"
    # order matters: the same prompts in a different order are different data
    assert dig(a) != dig(list(reversed(a))), "the digest ignores cell order"


def test_hidden_states_are_taken_at_the_readout_position_every_layer():
    """[n, n_layers+1, hidden], from the final position under left padding -- the same
    position and padding `last_logits` uses, or the activation and the DV are not one
    forward pass at one token."""
    from src.models.runner import Runner

    n_layers, hidden, seq, batch = 4, 8, 6, 3

    class _Model:
        dtype = torch.bfloat16

        def __call__(self, **kw):
            n = kw["input_ids"].shape[0]
            # layer L is filled with L at the LAST position and 0 elsewhere, so a wrong
            # position or a wrong axis is visible in the values themselves.
            hs = []
            for layer in range(n_layers + 1):
                h = torch.zeros(n, seq, hidden)
                h[:, -1, :] = float(layer)
                hs.append(h)
            return type("O", (), {"hidden_states": tuple(hs)})()

    class _Tok:
        pad_token_id = 0

        def __call__(self, batch_prompts, **kw):
            n = len(batch_prompts)
            return type("E", (), {
                "to": lambda self, d: {"input_ids": torch.ones(n, seq, dtype=torch.long),
                                       "attention_mask": torch.ones(n, seq,
                                                                    dtype=torch.long)}})()

    r = Runner.__new__(Runner)
    r.model, r.tokenizer, r.device, r.batch_size, r.n_forward = _Model(), _Tok(), "cpu", batch, 0
    out = r.last_hidden_states(["a", "b", "c", "d", "e"])

    assert out.shape == (5, n_layers + 1, hidden), out.shape
    for layer in range(n_layers + 1):
        assert torch.allclose(out[:, layer, :], torch.full((5, hidden), float(layer))), (
            f"layer {layer} was not read at the final position")
    assert r.n_forward == 5


def test_the_script_refuses_when_the_digest_does_not_match(tmp_path, monkeypatch, capsys):
    """NEGATIVE CONTROL for the assertion: it must REFUSE, not warn and continue."""
    mod = _script()
    src = (ROOT / "scripts" / "collect_activations.py").read_text(encoding="utf-8")
    assert "REFUSING TO WRITE" in src
    assert "return 1" in src.split("REFUSING TO WRITE")[1][:400], (
        "a digest mismatch must abort before anything is written")
    # the write must come after the check, not before
    assert src.index("digest != recorded") < src.index("np.savez_compressed")
    assert hasattr(mod, "_cells")


def test_signed_theta_is_carried_through_for_the_positive_control():
    """|diff| is unsigned and cannot say WHICH item is preferred.

    Without signed theta the sign(diff) control cannot be run, and preregistration_probe.md
    §5 makes a magnitude result uninterpretable without it. The collector must carry it and
    must refuse if Pass B predates it -- found by writing the consumer, which is the only
    reason it was noticed.
    """
    src = (ROOT / "scripts" / "collect_activations.py").read_text(encoding="utf-8")
    assert '"theta_item1": p["theta_item1"]' in src, "signed theta not read from pairs"
    assert "theta_item1=np.array(" in src, "signed theta not written to the npz"
    assert "rebuild Pass B" in src, "a Pass B predating signed theta must be refused"

    runner = (ROOT / "scripts" / "run_probe.py").read_text(encoding="utf-8")
    assert "REFUSING" in runner and "POSITIVE CONTROL" in runner, (
        "the probe runner must refuse rather than report a magnitude result alone")
    # and the layer must be chosen by the CONTROL, not by the question
    assert 'max(rows, key=lambda r: r["sign"]["rho_pair"])' in runner, (
        "the layer is being selected on the outcome")
