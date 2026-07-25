"""Stimulus loading, template invariants, and counterbalanced prompt assembly.

Nothing here is random-per-run. Pass A is a full crossing of item x template x
polarity, and Pass C is a full crossing of pair x template x option order x
condition, so counterbalancing is achieved by construction rather than by
sampling. The only seeded draw is which item a `random`/`3p-random` trial
designates, and that draw is exactly balanced within every
(difficulty, domain) stratum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd
import yaml

from src.config import CONDITIONS, POLARITIES, RunConfig

Polarity = Literal["ascending", "descending"]
Role = Literal["user", "assistant"]

# yoked/random and 3p-yoked/3p-random must differ ONLY in which item is
# designated. Their antecedent text is required to be byte-identical.
VERBATIM_PAIRS = (("yoked", "random"), ("3p-yoked", "3p-random"))

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# items


@dataclass(frozen=True)
class Item:
    id: str
    domain: str
    label: str
    framed: str


def load_items(cfg: RunConfig) -> list[Item]:
    """Load the item pool. Ids are slug-derived, so reordering the YAML is safe."""
    raw = yaml.safe_load(cfg.resolve(cfg.stimuli.items_path).read_text())
    per_domain = cfg.items_per_domain

    items: list[Item] = []
    for domain in cfg.stimuli.domains:
        if domain not in raw["domains"]:
            raise ValueError(f"domain {domain!r} missing from items file")
        block = raw["domains"][domain]
        frame = block["frame"]
        labels = block["items"]
        if len(labels) < per_domain:
            raise ValueError(
                f"domain {domain!r} has {len(labels)} items, config needs {per_domain}"
            )
        # Deterministic subset for smoke mode: the leading N, never a sample.
        for label in labels[:per_domain]:
            items.append(
                Item(
                    id=f"{domain}/{slug(label)}",
                    domain=domain,
                    label=label,
                    framed=frame.format(item=label),
                )
            )

    ids = [i.id for i in items]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise ValueError(f"duplicate item ids (slug collision): {dupes}")
    return items


# ---------------------------------------------------------------------------
# templates


@dataclass(frozen=True)
class Template:
    id: str
    index: int
    rating: dict[str, str]
    pair_prefix: str
    choice: str
    antecedent: dict[str, str]
    receipt: str


def load_templates(cfg: RunConfig) -> list[Template]:
    raw = yaml.safe_load(cfg.resolve(cfg.stimuli.templates_path).read_text())
    blocks = raw["templates"]
    if len(blocks) != cfg.stimuli.n_templates:
        raise ValueError(
            f"templates file has {len(blocks)} templates, config declares "
            f"{cfg.stimuli.n_templates}"
        )
    templates = [
        Template(
            id=b["id"],
            index=i,
            rating=b["rating"],
            pair_prefix=b["pair_prefix"],
            choice=b["choice"],
            antecedent=b["antecedent"],
            receipt=b["receipt"],
        )
        for i, b in enumerate(blocks)
    ]
    assert_template_invariants(templates)
    return templates


def assert_template_invariants(templates: list[Template]) -> None:
    """Enforce the three invariants documented at the top of templates.yaml."""
    problems: list[str] = []

    for t in templates:
        for pol in POLARITIES:
            if pol not in t.rating:
                problems.append(f"{t.id}: rating.{pol} missing")
        missing = set(CONDITIONS) - set(t.antecedent)
        if missing:
            problems.append(f"{t.id}: antecedent missing {sorted(missing)}")
            continue

        # (1) verbatim-identical wording for the designation-only contrasts
        for a, b in VERBATIM_PAIRS:
            if t.antecedent[a] != t.antecedent[b]:
                problems.append(
                    f"{t.id}: antecedent[{a}] and antecedent[{b}] must be byte-identical; "
                    "they may differ only in which item is designated"
                )

        # (3) every antecedent names each item exactly once
        for cond, text in t.antecedent.items():
            for ph in ("{designated}", "{other}"):
                n = text.count(ph)
                if n != 1:
                    problems.append(
                        f"{t.id}/{cond}: antecedent contains {ph} {n} times, must be exactly 1"
                    )

        # (2) receipt is one shared string; check it is well-formed
        for ph in ("{designated}", "{other}"):
            if t.receipt.count(ph) != 1:
                problems.append(
                    f"{t.id}: receipt must contain {ph} exactly once "
                    f"(found {t.receipt.count(ph)})"
                )

    if problems:
        raise ValueError("template invariant violations:\n  - " + "\n  - ".join(problems))


def audit_templates(templates: list[Template]) -> pd.DataFrame:
    """Per condition x template: turn count and item-mention counts.

    The `chose` condition is structurally longer by design (it carries the choice
    question and the model's own answer). This table is what makes that residual
    measured rather than hidden; it is reported alongside the results.
    """
    rows = []
    for t in templates:
        for cond in CONDITIONS:
            ante, rec = t.antecedent[cond], t.receipt
            body = f"{t.pair_prefix} {ante} {rec}"
            if cond == "chose":
                body = f"{body} {t.choice}"
            rows.append(
                {
                    "template": t.id,
                    "condition": cond,
                    "n_turns": 3 if cond == "chose" else 1,
                    "mentions_designated": body.count("{designated}")
                    + body.count("{item_a}"),
                    "mentions_other": body.count("{other}") + body.count("{item_b}"),
                    "chars": len(body),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# prompt assembly
#
# These return chat message lists. Applying the tokenizer chat template and
# locating the readout position is the model layer's job (src/models/prompt.py).


def rating_question(t: Template, framed_item: str, polarity: Polarity, cfg: RunConfig) -> str:
    return t.rating[polarity].format(
        item=framed_item, lo=cfg.readout.scale_min, hi=cfg.readout.scale_max
    )


def pair_block(t: Template, framed_a: str, framed_b: str, cfg: RunConfig) -> str:
    la, lb = cfg.readout.option_labels
    return t.pair_prefix.format(item_a=framed_a, item_b=framed_b, label_a=la, label_b=lb)


def pass_a_messages(
    t: Template, framed_item: str, polarity: Polarity, cfg: RunConfig
) -> list[dict[str, str]]:
    """Single item, no pair context, no shared context with any other item."""
    return [{"role": "user", "content": rating_question(t, framed_item, polarity, cfg)}]


def pre_messages(
    t: Template, framed_a: str, framed_b: str, framed_target: str, cfg: RunConfig
) -> list[dict[str, str]]:
    """Pre-rating. Identical across all five conditions -- it precedes the
    manipulation -- so it is measured once per (pair, template, option order) and
    reused. The model never sees its own prior rating; there is no anchoring
    channel into the difference-of-differences DV."""
    prefix = pair_block(t, framed_a, framed_b, cfg)
    q = rating_question(t, framed_target, "ascending", cfg)
    return [{"role": "user", "content": f"{prefix}\n\n{q}"}]


def choice_messages(
    t: Template, framed_a: str, framed_b: str, cfg: RunConfig
) -> list[dict[str, str]]:
    la, lb = cfg.readout.option_labels
    prefix = pair_block(t, framed_a, framed_b, cfg)
    q = t.choice.format(label_a=la, label_b=lb)
    return [{"role": "user", "content": f"{prefix}\n\n{q}"}]


def post_messages(
    t: Template,
    framed_a: str,
    framed_b: str,
    framed_target: str,
    framed_designated: str,
    framed_other: str,
    condition: str,
    chosen_label: str | None,
    cfg: RunConfig,
) -> list[dict[str, str]]:
    """Post-rating context: pair prefix, manipulation, rating question.

    `chose` writes the model's own choice back as an assistant turn -- a discrete
    commitment, so argmax is the right token -- which is what supplies authorship.
    The other four conditions state the designation in the user turn. The receipt
    sentence is identical across all five.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")

    la, lb = cfg.readout.option_labels
    prefix = pair_block(t, framed_a, framed_b, cfg)
    ante = t.antecedent[condition].format(
        designated=framed_designated, other=framed_other
    )
    rec = t.receipt.format(designated=framed_designated, other=framed_other)
    q = rating_question(t, framed_target, "ascending", cfg)

    if condition == "chose":
        if chosen_label is None:
            raise ValueError("condition 'chose' requires the model's own chosen label")
        return [
            {
                "role": "user",
                "content": f"{prefix}\n\n{t.choice.format(label_a=la, label_b=lb)}",
            },
            {"role": "assistant", "content": chosen_label},
            {"role": "user", "content": f"{ante} {rec}\n\n{q}"},
        ]
    return [{"role": "user", "content": f"{prefix}\n\n{ante} {rec}\n\n{q}"}]


# ---------------------------------------------------------------------------
# counterbalanced enumeration


def pass_a_trials(cfg: RunConfig) -> Iterator[dict]:
    """Full crossing: item x template x polarity. 400 x 5 x 2 = 4000."""
    items = load_items(cfg)
    templates = load_templates(cfg)
    for item in items:
        for t in templates:
            for polarity in POLARITIES:
                yield {
                    "item_id": item.id,
                    "domain": item.domain,
                    "item_label": item.label,
                    "item_framed": item.framed,
                    "template": t.id,
                    "template_index": t.index,
                    "polarity": polarity,
                }


def balanced_designation(
    pair_ids: list[str], strata: list[str], seed: int
) -> dict[str, int]:
    """Which pair member a `random`/`3p-random` trial designates: 0 or 1.

    Exactly balanced within every stratum (difficulty x domain), fixed by the run
    seed, so designation is counterbalanced rather than sampled per run. The same
    assignment is used for `random` and `3p-random`, which makes
    3p-random - random designation-matched and therefore a clean endorsement
    contrast.
    """
    if len(pair_ids) != len(strata):
        raise ValueError("pair_ids and strata must be the same length")
    rng = np.random.default_rng(seed)
    out: dict[str, int] = {}
    frame = pd.DataFrame({"pair_id": pair_ids, "stratum": strata})
    for _, group in frame.groupby("stratum", sort=True):
        ids = sorted(group["pair_id"])
        n = len(ids)
        assign = np.array([0] * (n // 2) + [1] * (n - n // 2))
        rng.shuffle(assign)
        out.update(dict(zip(ids, assign.tolist())))
    return {k: int(v) for k, v in out.items()}
