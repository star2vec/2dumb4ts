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

from src.config import (
    ANTECEDENT_SOURCE, CONDITIONS, POLARITIES, TURN_CONDITIONS, RunConfig,
)

Polarity = Literal["ascending", "descending"]
Role = Literal["user", "assistant"]

# yoked/random and 3p-yoked/3p-random must differ ONLY in which item is
# designated. Their antecedent text is required to be byte-identical.
VERBATIM_PAIRS = (("yoked", "random"), ("3p-yoked", "3p-random"))

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def label_noun(labels: tuple[str, ...]) -> str:
    """What to call the option labels in prompt text: "number", "letter", "option".

    Templates originally hardcoded "letter", which D3's switch to digits turned into
    "the single letter 1 or 2" -- incoherent, and models answered in prose. Banning
    the noun outright was worse: "Give only 1 or 2" reads as a QUANTITY ("give only
    one or two of them") and "Output just 1 or 2" led Gemma to emit the item name
    instead. Naming the label type is what makes the instruction unambiguous, so the
    noun is derived from the labels actually in use rather than written into the
    template or removed from it.
    """
    if all(x.isdigit() for x in labels):
        return "number"
    if all(x.isalpha() and len(x) == 1 for x in labels):
        return "letter"
    return "option"


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
    #: Amendment 2 A2.9. The DV question (identical pre and post), the neutral
    #: instruction that elicits the structure control's acknowledgement, and the
    #: finality / provisional clauses that make reversibility a one-clause contrast.
    prefer: str = ""
    confirm: str = ""
    finality: str = ""
    provisional: str = ""


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
            prefer=b.get("prefer", ""),
            confirm=b.get("confirm", ""),
            finality=b.get("finality", ""),
            provisional=b.get("provisional", ""),
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
        # Conditions BORROW antecedents via ANTECEDENT_SOURCE (A2.9.3), so only the
        # source set must be present. Requiring one per condition would demand text
        # for `self-recounted` and `structure-control` that must NOT exist separately
        # -- their whole point is being byte-identical to chose and yoked.
        required = set(ANTECEDENT_SOURCE.values())
        missing = required - set(t.antecedent)
        if missing:
            problems.append(f"{t.id}: antecedent missing {sorted(missing)}")
            continue

        # A2.9.3: the 2x2 is only crossed if the borrowing is exact.
        for borrower, source in ANTECEDENT_SOURCE.items():
            if t.antecedent.get(borrower, t.antecedent[source]) != t.antecedent[source]:
                problems.append(
                    f"{t.id}: {borrower!r} defines its own antecedent but must borrow "
                    f"{source!r} byte-for-byte, or the 2x2 is not crossed"
                )
        for field in ("prefer", "confirm", "finality", "provisional"):
            if not getattr(t, field):
                problems.append(f"{t.id}: {field} missing (required by A2.9)")

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
            ante, rec = t.antecedent[ANTECEDENT_SOURCE[cond]], t.receipt
            clause = t.provisional if cond == "chose-provisional" else t.finality
            body = f"{t.pair_prefix} {ante} {rec} {clause}"
            if cond in TURN_CONDITIONS:
                body = f"{body} {t.confirm if cond == 'structure-control' else t.choice}"
            rows.append(
                {
                    "template": t.id,
                    "condition": cond,
                    "n_turns": 3 if cond in TURN_CONDITIONS else 1,
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


def render_choice(t: Template, labels: tuple[str, str]) -> str:
    """Fill a template's choice question, including the label noun."""
    la, lb = labels
    return t.choice.format(label_a=la, label_b=lb, label_noun=label_noun(labels))


def choice_messages(
    t: Template, framed_a: str, framed_b: str, cfg: RunConfig
) -> list[dict[str, str]]:
    prefix = pair_block(t, framed_a, framed_b, cfg)
    q = render_choice(t, cfg.readout.option_labels)
    return [{"role": "user", "content": f"{prefix}\n\n{q}"}]


def prefer_question(t: Template, labels: tuple[str, str]) -> str:
    """The DV question. IDENTICAL at pre and post -- that is what makes the
    pre/post contrast a contrast rather than two different measurements."""
    return t.prefer.format(label_a=labels[0], label_b=labels[1],
                           label_noun=label_noun(labels))


def pre_dv_messages(
    t: Template, framed_a: str, framed_b: str, cfg: RunConfig
) -> list[dict[str, str]]:
    """Pre-manipulation preference comparison. Shared across all conditions."""
    prefix = pair_block(t, framed_a, framed_b, cfg)
    return [{"role": "user",
             "content": f"{prefix}\n\n{prefer_question(t, cfg.readout.option_labels)}"}]


def post_dv_messages(
    t: Template,
    framed_a: str,
    framed_b: str,
    framed_designated: str,
    framed_other: str,
    condition: str,
    chosen_label: str | None,
    cfg: RunConfig,
) -> list[dict[str, str]]:
    """Post-manipulation preference comparison, for any of the eight conditions.

    Amendment 2 A2.9.3. The first four conditions form a fully crossed 2x2:

                          antecedent "you chose"   antecedent "assigned"
      turn present        chose                    structure-control
      turn absent         self-recounted           yoked

    `self-recounted` borrows `chose`'s antecedent byte-for-byte and
    `structure-control` borrows `yoked`'s, so the wording factor is exactly one
    substitution and the structure factor is exactly the assistant turn. Any
    departure from that breaks the crossing, which is why the mapping lives in
    ANTECEDENT_SOURCE rather than being written out per condition.

    A finality clause is appended in EVERY condition so receipt matching survives;
    `chose-provisional` swaps it for the provisional wording, making reversibility a
    one-clause contrast against `chose`.
    """
    from src.config import ANTECEDENT_SOURCE, TURN_CONDITIONS

    if condition not in ANTECEDENT_SOURCE:
        raise ValueError(f"unknown condition {condition!r}")

    la, lb = cfg.readout.option_labels
    prefix = pair_block(t, framed_a, framed_b, cfg)
    ante = t.antecedent[ANTECEDENT_SOURCE[condition]].format(
        designated=framed_designated, other=framed_other
    )
    rec = t.receipt.format(designated=framed_designated, other=framed_other)
    clause = t.provisional if condition == "chose-provisional" else t.finality
    tail = f"{ante} {rec} {clause}\n\n{prefer_question(t, (la, lb))}"

    if condition not in TURN_CONDITIONS:
        return [{"role": "user", "content": f"{prefix}\n\n{tail}"}]

    # An assistant turn is present. What it contains is the structure factor's
    # content: its own choice for `chose`, a content-free acknowledgement for
    # `structure-control`. Both are single tokens, so turn LENGTH is matched too.
    if condition == "structure-control":
        lead, reply = t.confirm, "OK"
    else:
        if chosen_label is None:
            raise ValueError(f"condition {condition!r} requires the model's chosen label")
        lead, reply = render_choice(t, (la, lb)), chosen_label

    return [
        {"role": "user", "content": f"{prefix}\n\n{lead}"},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": tail},
    ]


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
