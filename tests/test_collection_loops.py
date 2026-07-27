"""The collection loops, run for real against a stub model.

`readout/pairwise.py` sits at 37% and `pass_c.py` at 36%; the uncovered parts are the
batching, the checkpoint/resume path, and Pass C's two-phase choice-then-post structure.
They were uncovered because exercising them needs weights -- but only the *logits* need a
model, and those can be stubbed deterministically.

The resume path matters most. At full scale one model is 40,000 comparisons over hours on
a machine with a known OOM risk, so resume is what stands between a crash at 95% and
losing everything. It had never executed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"
VOCAB = 4096


def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {type(exc).__name__}")


class StubRunner:
    """Real tokenizer, synthetic logits. Deterministic in the prompt text.

    Determinism is the point: a resumed run must reproduce the answers the interrupted run
    would have given, and that can only be checked if the same prompt always reads the
    same way.
    """

    def __init__(self, tokenizer, label_ids: list[int], batch_size: int = 8):
        self.tokenizer = tokenizer
        self.label_ids = label_ids
        self.batch_size = batch_size
        self.device = "cpu"
        self.n_forward = 0
        self.calls: list[int] = []

    def render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def last_logits(self, prompts):
        self.calls.append(len(prompts))
        self.n_forward += len(prompts)
        out = torch.full((len(prompts), VOCAB), -8.0)
        for i, p in enumerate(prompts):
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            # Most of the mass on the labels, split by a prompt-determined margin.
            for k, tid in enumerate(self.label_ids):
                out[i, tid] = 6.0 + (2.0 if (h >> k) & 1 else 0.0)
        return out


@pytest.fixture(scope="module")
def cfg():
    base = load_config(CONFIG)
    # batch_size 1 -> chunk = max(1*8, 1) = 8. With 24 prompts that is three chunks, so
    # resume has a boundary to land on. At batch_size 4 the chunk is 32, one chunk covers
    # everything, start_at is always 0 and the resume path never executes -- which is how
    # the first draft of this file "passed" while testing nothing.
    return base.model_copy(update={"smoke": True, "smoke_items_per_domain": 3,
                                   "batch_size": 1})


def _stub(cfg, tok, labels=None):
    """Mass must land on the labels the CALLER will read.

    Pairwise uses DIGIT_LABELS; Pass C uses cfg.readout.option_labels ("A", "B"). Stubbing
    the wrong set puts every readout under the A1.6 mass floor and Pass C aborts -- which
    is the guard working correctly on a bad stub.
    """
    from src.readout.digits import build_label_map
    from src.readout.pairwise import DIGIT_LABELS

    lmap = build_label_map(tok, labels or DIGIT_LABELS)
    return StubRunner(tok, list(lmap.flat_ids), cfg.batch_size)


# ---------------------------------------------------------------------------
# pairwise collection and its checkpoint


def test_collection_produces_both_orders_for_every_cell(cfg, tmp_path):
    """Both orders must always be present -- that is what makes order averaging structural
    rather than a modelling assumption."""
    from src.readout.pairwise import collect_comparisons, load_anchors
    from src.stimuli.build import load_items, load_templates

    tok = _tokenizer()
    items, anchors = load_items(cfg)[:3], load_anchors(cfg)[:2]
    templates = load_templates(cfg)[:2]
    frame = collect_comparisons(cfg, _stub(cfg, tok), items, anchors, templates,
                                desc="test", checkpoint=tmp_path / "ck.parquet")

    assert len(frame) == 3 * 2 * 2 * 2, "one row per item x anchor x template x order"
    per_cell = frame.groupby(["item_id", "anchor_id", "template"])["order"].nunique()
    assert (per_cell == 2).all(), "a cell is missing one of its two orders"
    assert set(frame["item_wins"].unique()) <= {True, False}
    assert frame["readout_valid"].all()


def test_a_resumed_run_reproduces_the_uninterrupted_one(cfg, tmp_path):
    """The whole point of the checkpoint. Never executed before this test.

    A full run is compared against one that is interrupted after the first chunk and
    resumed -- the second must land on exactly the same answers.
    """
    from src.readout.pairwise import collect_comparisons, load_anchors
    from src.stimuli.build import load_items, load_templates

    tok = _tokenizer()
    items, anchors = load_items(cfg)[:3], load_anchors(cfg)[:2]
    templates = load_templates(cfg)[:2]

    full = collect_comparisons(cfg, _stub(cfg, tok), items, anchors, templates,
                               desc="full", checkpoint=tmp_path / "a.parquet")

    # Interrupt after exactly one completed chunk, as a crash on a boundary would leave.
    # Resume rounds DOWN to a chunk boundary, so a partial smaller than one chunk rounds
    # to zero and re-runs everything -- which is why the slice must be `chunk`, not less.
    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    assert len(full) > chunk, "the fixture must span more than one chunk or resume is inert"
    ck = tmp_path / "b.parquet"
    pd.read_parquet(tmp_path / "a.parquet").iloc[:chunk].to_parquet(ck, index=False)

    resumed_runner = _stub(cfg, tok)
    resumed = collect_comparisons(cfg, resumed_runner, items, anchors, templates,
                                  desc="resumed", checkpoint=ck)

    pd.testing.assert_frame_equal(
        full.drop(columns=[c for c in full.columns if c.startswith("prov_")]),
        resumed.drop(columns=[c for c in resumed.columns if c.startswith("prov_")]),
    )
    # NEGATIVE CONTROL: a resume that silently re-ran everything would also pass the
    # equality above, so assert work was actually skipped.
    assert resumed_runner.n_forward < len(full), (
        f"the resumed run did {resumed_runner.n_forward} forward passes for {len(full)} "
        "prompts -- it skipped nothing, so resume is not being exercised")


def test_a_checkpoint_longer_than_the_plan_is_refused(cfg, tmp_path):
    """It belongs to a different configuration; resuming would misalign every row."""
    from src.readout.pairwise import collect_comparisons, load_anchors
    from src.stimuli.build import load_items, load_templates

    tok = _tokenizer()
    items, anchors = load_items(cfg)[:2], load_anchors(cfg)[:2]
    templates = load_templates(cfg)[:1]
    ck = tmp_path / "toolong.parquet"
    pd.DataFrame({"pick_slot": [0] * 500, "margin": [0.1] * 500,
                  "readout_mass_raw": [0.9] * 500}).to_parquet(ck, index=False)

    with pytest.raises(RuntimeError, match="different configuration"):
        collect_comparisons(cfg, _stub(cfg, tok), items, anchors, templates,
                            desc="x", checkpoint=ck)


def test_a_shorter_checkpoint_from_a_different_plan_is_NOT_refused(cfg, tmp_path):
    """A gap, recorded rather than fixed under the freeze.

    Resume validates the checkpoint by ROW COUNT only, and rejects just the case where it
    is too LONG. A checkpoint that is too SHORT is indistinguishable from an interrupted
    run of the current plan, so it is resumed -- attributing one plan's answers to another
    plan's prompts.

    Not reachable from `run.py`, which names the checkpoint
    `_checkpoint_{cfg.hash('pass_a')}.parquet` and always passes arms=("digits",). But
    `arms` is a FUNCTION ARGUMENT and is not in that hash, so two calls differing only in
    `arms` share one checkpoint file and differ in prompt count. Same class as the Pass B
    cache key: a key that does not cover everything determining the output.
    """
    from src.readout.pairwise import collect_comparisons, load_anchors
    from src.stimuli.build import load_items, load_templates

    tok = _tokenizer()
    items, anchors = load_items(cfg)[:3], load_anchors(cfg)[:2]
    templates = load_templates(cfg)[:2]
    ck = tmp_path / "short.parquet"
    # Exactly one chunk of deliberately alien values, so they land on a resume boundary
    # and survive into the output if the gap is real.
    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    pd.DataFrame({"pick_slot": [1] * chunk, "margin": [-99.0] * chunk,
                  "readout_mass_raw": [0.5] * chunk}).to_parquet(ck, index=False)

    frame = collect_comparisons(cfg, _stub(cfg, tok), items, anchors, templates,
                                desc="x", checkpoint=ck)
    assert len(frame) == 24
    # Documents the current behaviour: the alien rows are accepted, not rejected. If a
    # future change makes this raise, that is an IMPROVEMENT and this test should be
    # rewritten to assert the raise.
    assert (frame["readout_mass"] == 0.5).sum() >= 1, (
        "alien checkpoint rows no longer survive -- the gap may have been closed")


# ---------------------------------------------------------------------------
# Pass C's two-phase collection


def test_pass_c_runs_end_to_end_against_a_stub_and_validates(cfg, tmp_path):
    """Choice elicitation, write-back, all eight conditions, and _validate -- for real."""
    from src.experiments.pass_c import OWN_PICK_CONDITIONS, run_pass_c
    from src.provenance import Provenance

    tok = _tokenizer()
    pairs = pd.DataFrame([
        {"pair_id": f"destinations/p{i}", "domain": "destinations",
         "difficulty": "difficult" if i < 2 else "easy",
         "matched_set": f"destinations/ms{i % 2}",
         "item1_id": f"destinations/a{i}", "item2_id": f"destinations/b{i}",
         "diff_selection": 0.2, "mean_selection": 0.0,
         "diff_analysis": 0.2 + 0.5 * i, "mean_analysis": 0.0}
        for i in range(4)
    ])
    framed = {f"destinations/{p}{i}": f"a trip to City{p}{i}"
              for i in range(4) for p in ("a", "b")}

    import src.experiments.pass_c as pc

    original = pc.load_items
    pc.load_items = lambda c: [type("I", (), {"id": k, "framed": v})() for k, v in framed.items()]
    try:
        prov = Provenance(
            device="cpu", device_name="stub", dtype="bfloat16",
            attn_implementation="eager", torch_version="2.8.0",
            transformers_version="4.57.0", pymc_version="5", pytensor_version="2",
            pytensor_mode="Mode", model_name=cfg.model.name, model_hf_id=cfg.model.hf_id,
            model_revision="a" * 40, model_revision_pinned=True, seed=cfg.seed,
            config_hash=cfg.hash(), git_sha="c" * 40, git_dirty=False, smoke=True,
            platform="stub", python_version="3.11.15",
            created_utc="2026-07-28T00:00:00+00:00",
        )
        cfg2 = cfg.model_copy(update={"artifacts_dir": tmp_path})
        trials = run_pass_c(cfg2, pairs,
                            _stub(cfg2, tok, cfg2.readout.option_labels), prov)
    finally:
        pc.load_items = original

    n_cells = trials.groupby(["pair_id", "template", "option_order"]).ngroups
    assert len(trials) == n_cells * (1 + len(cfg.pass_c.conditions))

    # The own-pick family must agree on the designated item within every cell -- that is
    # what makes the primary contrast designation-matched.
    own = trials[trials["condition"].isin(OWN_PICK_CONDITIONS)]
    agree = own.groupby(["pair_id", "template", "option_order"])["designated_item_id"].nunique()
    assert (agree == 1).all()

    # The chosen item must be one of the pair's two, per cell.
    post = trials[trials["timepoint"] == "post"]
    assert (post["designated_item_id"].isin(set(framed))).all()
    assert trials["item1_wins"].dtype == bool
    assert np.isfinite(trials["readout_mass"]).all()
