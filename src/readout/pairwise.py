"""Anchor-based pairwise readout. preregistration.md A1.1, A1.3, A1.6.

Replaces the absolute 1-9 Likert judgement, which compressed almost all
between-item variance out of the measurement (ranges of 2.3-8.0, sigma_between
0.43-0.58) while the underlying ordering stayed intact -- Qwen2.5-3B assigns
p = 0.958 to Vienna over a motorway service station when asked to CHOOSE, and
rates both in the 7-9 band when asked to RATE.

Design
    each pool item vs each of 10 fixed anchors, BOTH orders  -> 20 outcomes/item
    digit labels "1"/"2" primary, item-name first token secondary
    order averaged STRUCTURALLY (both always run), never corrected post hoc
    mass floor applied per trial; below floor -> invalid and logged

Order is averaged structurally rather than modelled because a position-bias term
in the likelihood would be estimated from the same comparisons it is meant to
purge, and would trade off against theta wherever an item's anchors are unevenly
distributed across slots. Running both orders makes the design balanced by
construction, which needs no estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.config import RunConfig
from src.models.runner import Runner
from src.readout.choice import read_choice
from src.readout.digits import DigitMap, build_label_map
from src.readout.validity import attach
from src.stimuli.build import render_choice

# D3: digits, not letters. Letters were the worst scheme on every model tested,
# and drove Qwen2.5-3B to pick the first-displayed option 85% of the time on
# close pairs versus 42% under digits.
DIGIT_LABELS = ("1", "2")

ARMS = ("digits", "name")

_ARTICLES = ("a ", "an ", "the ")


@dataclass(frozen=True)
class Anchor:
    id: str
    label: str
    tier: str


def load_anchors(cfg: RunConfig) -> list[Anchor]:
    """Load anchors and assert they do not overlap the pool (A1.1)."""
    from src.stimuli.build import load_items

    raw = yaml.safe_load(cfg.resolve(Path("src/stimuli/anchors.yaml")).read_text())
    anchors = [Anchor(a["id"], a["label"], a["tier"]) for a in raw["anchors"]]

    pool_labels = {i.label.strip().lower() for i in load_items(cfg)}
    clash = [a.label for a in anchors if a.label.strip().lower() in pool_labels]
    if clash:
        raise ValueError(
            f"anchors must lie OUTSIDE the item pool; these also appear in it: {clash}"
        )
    if len({a.id for a in anchors}) != len(anchors):
        raise ValueError("duplicate anchor ids")
    return anchors


def bare(label: str) -> str:
    """Strip a leading article, so an item-name readout keys on a content word."""
    low = label.lower()
    for art in _ARTICLES:
        if low.startswith(art):
            return label[len(art) :]
    return label


# ---------------------------------------------------------------------------
# prompts


def comparison_prompt(
    arm: str, first: str, second: str, template, labels: tuple[str, str]
) -> str:
    """One binary comparison. `first` is displayed in slot 1.

    Uses the template's OWN `pair_prefix`, not a fixed one. Hardcoding a single
    prefix across all five templates would collapse template variation in the
    presentation while still varying the question, which quietly demotes a
    modelled factor to a half-varied one.

    The `name` arm shows NO labels at all: position must not be an available
    answer. Keeping labels visible while reading item-name tokens is exactly the
    error that produced a readout with 0.8% of its mass on the tokens being read.
    """
    if arm == "name":
        return (
            "Here are two options.\n\n"
            f"- {first}\n- {second}\n\n"
            "Which one would you choose for yourself? Reply with the option you "
            "choose, exactly as written, and nothing else."
        )
    la, lb = labels
    return (
        f"{lead_in(template)}\n\n"
        f"{la}. {first}\n{lb}. {second}\n\n"
        f"{render_choice(template, labels)}"
    )


def lead_in(template) -> str:
    """The template's own opening prose, without its option-rendering format.

    Template paraphrase variation lives in the PROSE -- the lead-in sentence and
    the question. The LABEL RENDERING is standardised to "N. item" because it is
    part of the readout mechanism, not part of the manipulation.

    This is not cosmetic. Template t4 renders options as "[1] item", and the model
    copies the format and answers "[1]" -- so at the single readout position it
    emits "[" and the digit tokens lose their mass. Observed median readout mass
    for t4 under its native format was 0.38, with 568 of 800 comparisons below the
    floor, versus ~1.00 under a standardised format. Varying the label syntax
    across templates tests the parser, not the model's robustness to wording.
    """
    return template.pair_prefix.split("\n\n", 1)[0].strip()


def name_map(runner: Runner, first: str, second: str) -> DigitMap | None:
    """First-token map for the item-name arm; None if the labels collide."""
    def firsts(text: str) -> set[int]:
        out: set[int] = set()
        for surface in (text, f" {text}"):
            enc = runner.tokenizer.encode(surface, add_special_tokens=False)
            if enc:
                out.add(enc[0])
        return out

    a, b = firsts(first), firsts(second)
    if not a or not b or (a & b):
        return None
    return DigitMap(
        digits=(0, 1),
        ids_by_digit={0: tuple(sorted(a)), 1: tuple(sorted(b))},
        flat_ids=tuple(sorted(a) + sorted(b)),
        digit_index=tuple([0] * len(a) + [1] * len(b)),
        surfaces_by_digit={0: (first,), 1: (second,)},
    )


# ---------------------------------------------------------------------------
# collection


def collect_comparisons(
    cfg: RunConfig,
    runner: Runner,
    items: list,
    anchors: list[Anchor],
    templates: list,
    *,
    arms: tuple[str, ...] = ("digits",),
    desc: str = "pairwise",
    checkpoint: Path | None = None,
) -> pd.DataFrame:
    """Every (item x anchor x template x order x arm) comparison.

    Returns one row per comparison with `item_wins` = did the POOL item beat the
    anchor. Both orders are always present, which is what makes order averaging
    structural.
    """
    from tqdm import tqdm

    label_map = build_label_map(runner.tokenizer, DIGIT_LABELS)
    rows: list[dict] = []
    prompts: list[str] = []
    maps: list[DigitMap] = []
    dropped = 0

    for item in items:
        for anchor in anchors:
            for t in templates:
                for order in (0, 1):
                    # order 0: pool item in slot 1. order 1: anchor in slot 1.
                    if order == 0:
                        first_id, second_id = item.id, anchor.id
                        first_txt, second_txt = item.label, anchor.label
                    else:
                        first_id, second_id = anchor.id, item.id
                        first_txt, second_txt = anchor.label, item.label

                    for arm in arms:
                        if arm == "name":
                            f_txt, s_txt = bare(first_txt), bare(second_txt)
                            dmap = name_map(runner, f_txt, s_txt)
                            if dmap is None:
                                dropped += 1
                                continue
                        else:
                            f_txt, s_txt = first_txt, second_txt
                            dmap = label_map

                        prompts.append(
                            runner.render([{
                                "role": "user",
                                "content": comparison_prompt(
                                    arm, f_txt, s_txt, t, DIGIT_LABELS
                                ),
                            }])
                        )
                        maps.append(dmap)
                        rows.append({
                            "item_id": item.id,
                            "domain": item.domain,
                            "anchor_id": anchor.id,
                            "anchor_tier": anchor.tier,
                            "template": t.id,
                            "template_index": t.index,
                            "order": order,
                            "arm": arm,
                            "item_in_slot1": order == 0,
                            "first_id": first_id,
                            "second_id": second_id,
                        })

    if dropped:
        print(f"  dropped {dropped} name-arm comparisons (labels share a first token)")

    picks, margins, masses = _run_with_checkpoint(
        cfg, runner, prompts, maps, desc=desc, checkpoint=checkpoint
    )

    frame = pd.DataFrame(rows)
    frame["pick_slot"] = picks
    frame["margin"] = margins
    # The pool item wins when the chosen slot holds it.
    frame["item_wins"] = np.where(
        frame["item_in_slot1"], frame["pick_slot"] == 0, frame["pick_slot"] == 1
    )
    return attach(frame, np.asarray(masses), context=f"{desc} comparisons")


# ---------------------------------------------------------------------------
# order invariance (A1.4)


def _run_with_checkpoint(
    cfg: RunConfig, runner: Runner, prompts: list[str], maps: list[DigitMap],
    *, desc: str, checkpoint: Path | None, every: int = 20,
) -> tuple[list[int], list[float], list[float]]:
    """Read every prompt, saving partial results so a long run is resumable.

    At full scale one model is 400 items x 10 anchors x 5 templates x 2 orders =
    40,000 comparisons -- hours. Without this, a failure at 95% loses everything,
    which is a bad property for a job running unattended overnight on a machine
    with 8 GB of VRAM and a known OOM risk.

    Resume is by ROW COUNT, which is safe only because prompt order is fully
    deterministic: items, anchors and templates are all iterated in sorted order
    with no sampling anywhere in the loop.
    """
    from tqdm import tqdm

    picks: list[int] = []
    margins: list[float] = []
    masses: list[float] = []

    if checkpoint is not None and checkpoint.exists():
        done = pd.read_parquet(checkpoint)
        if len(done) > len(prompts):
            raise RuntimeError(
                f"checkpoint {checkpoint} holds {len(done)} rows but only "
                f"{len(prompts)} prompts are planned -- it belongs to a different "
                "configuration. Delete it rather than resuming from it."
            )
        picks = done["pick_slot"].tolist()
        margins = done["margin"].tolist()
        masses = done["readout_mass_raw"].tolist()
        print(f"  resuming from checkpoint: {len(picks)}/{len(prompts)} done")

    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    start_at = (len(picks) // chunk) * chunk
    # Drop any partial chunk so resumption lands on a chunk boundary.
    picks, margins, masses = picks[:start_at], margins[:start_at], masses[:start_at]

    steps = list(range(start_at, len(prompts), chunk))
    for n, start in enumerate(tqdm(steps, desc=desc, unit="chunk", leave=False)):
        logits = runner.last_logits(prompts[start : start + chunk])
        for j in range(logits.shape[0]):
            out = read_choice(logits[j : j + 1], maps[start + j])
            picks.append(int(out.index[0]))
            margins.append(float(out.margin[0]))
            masses.append(float(out.mass[0]))
        if checkpoint is not None and (n + 1) % every == 0:
            _save_checkpoint(checkpoint, picks, margins, masses)

    if checkpoint is not None:
        _save_checkpoint(checkpoint, picks, margins, masses)
    return picks, margins, masses


def _save_checkpoint(path: Path, picks, margins, masses) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame({
        "pick_slot": picks, "margin": margins, "readout_mass_raw": masses,
    }).to_parquet(tmp, index=False)
    # Atomic replace, so a crash mid-write cannot leave a truncated checkpoint
    # that would then be resumed from.
    tmp.replace(path)


def order_invariance(frame: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Proportion of comparisons whose winner is the same under both orders.

    1.0 content-driven, 0.5 random, 0.0 purely positional. The statistic
    separates the two failure modes: a position-only responder always picks the
    same slot, so the winner flips on every reversal and it scores 0, not 0.5.

    Invalid-readout trials are excluded; a pair missing either order is dropped.
    """
    valid = frame[frame["readout_valid"]]
    keys = ["item_id", "anchor_id", "template", "arm"]
    # Carry through any grouping column the caller asked for that is constant
    # within a comparison (anchor_tier, domain, ...), so invariance can be
    # decomposed without re-deriving the pivot.
    passthrough = [
        c for c in (by or []) if c not in keys and c in valid.columns
    ]
    wide = valid.pivot_table(
        index=keys + passthrough, columns="order", values="item_wins", aggfunc="first"
    ).dropna()
    if not {0, 1} <= set(wide.columns):
        return pd.DataFrame()

    wide = wide.reset_index()
    wide["invariant"] = wide[0] == wide[1]
    # An unambiguous win: the pool item beats the anchor under BOTH orders. This
    # is the content signal that survives order averaging.
    wide["unambiguous_item_win"] = wide[0] & wide[1]

    group = by or ["arm"]
    rows = []
    for key, block in wide.groupby(group, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        inv = float(block["invariant"].mean())
        rows.append({
            **dict(zip(group, key)),
            "n_pairs": len(block),
            "order_invariance": inv,
            "unambiguous_item_win": float(block["unambiguous_item_win"].mean()),
            "regime": (
                "position-dominated" if inv < 0.5
                else "random-dominated" if inv < 0.6
                else "content-driven"
            ),
        })
    return pd.DataFrame(rows)
