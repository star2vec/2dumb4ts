"""Pass C -- five conditions x difficulty.

    condition   designated item    authorship   endorsement
    chose       model's own pick   self         --
    yoked       model's own pick   none         none
    3p-yoked    model's own pick   other        other person
    3p-random   random             other        other person
    random      random             none         none

Receipt is matched across all five: every condition states that the model
receives the designated item. Without it, 3p-yoked - yoked would confound
endorsement with ownership.

Context structure (preregistration.md 6.3). All four ratings hang off the same
pair prefix, and the model NEVER sees its own prior rating -- writing a
pre-rating back into the context would give the post-rating a token to copy,
which is a first-order threat to a difference-of-differences DV:

    pre_X  = fwd(P + "rate X")            measured ONCE per (pair, template,
    pre_Y  = fwd(P + "rate Y")            option order), reused by all 5 conditions
    post_X = fwd(P + manipulation + "rate X")
    post_Y = fwd(P + manipulation + "rate Y")

Only `chose` writes anything back: its own choice, as an assistant turn. That is
what supplies authorship, and it is the sole place in Stage 0 where a model
output re-enters a prompt.

DV: spread = (designated_post - designated_pre) - (other_post - other_pre)
Bounded +/-16 on a 1-9 scale; interpreted as points of divergence between the two
options, not as a rating.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import RunConfig
from src.models.runner import Runner
from src.provenance import Provenance, read_parquet, write_parquet
from src.readout.validity import attach
from src.stimuli.build import (
    Template,
    balanced_designation,
    choice_messages,
    load_items,
    load_templates,
    post_messages,
    pre_messages,
)

# Conditions whose designated item is the model's own pick.
YOKED_CONDITIONS = frozenset({"chose", "yoked", "3p-yoked"})


def artifact_path(cfg: RunConfig) -> Path:
    return cfg.artifact_dir("pass_c") / f"trials_{cfg.hash('pass_c')}.parquet"


def _framed(cfg: RunConfig) -> dict[str, str]:
    return {item.id: item.framed for item in load_items(cfg)}


def _displayed(item1: str, item2: str, option_order: int) -> tuple[str, str]:
    """Which item carries label A. Option order is counterbalanced, not sampled."""
    return (item1, item2) if option_order == 0 else (item2, item1)


def run_pass_c(
    cfg: RunConfig, pairs: pd.DataFrame, runner: Runner, prov: Provenance
) -> pd.DataFrame:
    framed = _framed(cfg)
    templates: list[Template] = load_templates(cfg)
    labels = cfg.readout.option_labels

    # Random designation for `random` / `3p-random`: exactly balanced within every
    # (difficulty, domain) stratum, fixed by the run seed. The SAME assignment
    # serves both conditions, which makes 3p-random - random designation-matched
    # and therefore a clean endorsement contrast.
    designation = balanced_designation(
        pairs["pair_id"].tolist(),
        (pairs["difficulty"] + "/" + pairs["domain"]).tolist(),
        cfg.seed,
    )

    pair_rows = pairs.to_dict("records")

    # ---- phase 1: pre-ratings and choices ---------------------------------
    trials: list[dict] = []
    pre_msgs: list[list[dict[str, str]]] = []
    choice_msgs: list[list[dict[str, str]]] = []

    for p in pair_rows:
        for t in templates:
            for order in range(cfg.pass_c.n_option_orders):
                a_id, b_id = _displayed(p["item1_id"], p["item2_id"], order)
                fa, fb = framed[a_id], framed[b_id]
                trials.append(
                    {
                        "pair_id": p["pair_id"],
                        "domain": p["domain"],
                        "difficulty": p["difficulty"],
                        "matched_set": p["matched_set"],
                        "item1_id": p["item1_id"],
                        "item2_id": p["item2_id"],
                        "template": t.id,
                        "template_index": t.index,
                        "option_order": order,
                        "label_a_item": a_id,
                        "label_b_item": b_id,
                        "diff_selection": p["diff_selection"],
                        "diff_analysis": p["diff_analysis"],
                        "mean_analysis": p["mean_analysis"],
                        "_template_obj": t,
                    }
                )
                # Pre is identical across conditions, so it is measured once here.
                pre_msgs.append(pre_messages(t, fa, fb, framed[p["item1_id"]], cfg))
                pre_msgs.append(pre_messages(t, fa, fb, framed[p["item2_id"]], cfg))
                choice_msgs.append(choice_messages(t, fa, fb, cfg))

    pre_out = _chunked_rate(runner, pre_msgs, cfg, "pass C pre")
    choice_out = runner.choose(choice_msgs)

    for i, tr in enumerate(trials):
        tr["pre_item1"] = float(pre_out.value[2 * i])
        tr["pre_item2"] = float(pre_out.value[2 * i + 1])
        tr["pre_mass_min"] = float(min(pre_out.mass[2 * i], pre_out.mass[2 * i + 1]))
        idx = int(choice_out.index[i])
        tr["choice_label_index"] = idx
        tr["choice_label"] = labels[idx]
        tr["choice_margin"] = float(choice_out.margin[i])
        tr["choice_mass"] = float(choice_out.mass[i])
        tr["chosen_item_id"] = tr["label_a_item"] if idx == 0 else tr["label_b_item"]

    # ---- phase 2: post-ratings, five conditions ---------------------------
    rows: list[dict] = []
    post_msgs: list[list[dict[str, str]]] = []

    for tr in trials:
        t: Template = tr["_template_obj"]
        fa, fb = framed[tr["label_a_item"]], framed[tr["label_b_item"]]
        for condition in cfg.pass_c.conditions:
            if condition in YOKED_CONDITIONS:
                designated = tr["chosen_item_id"]
            else:
                which = designation[tr["pair_id"]]
                designated = tr["item1_id"] if which == 0 else tr["item2_id"]
            other = tr["item2_id"] if designated == tr["item1_id"] else tr["item1_id"]

            row = {k: v for k, v in tr.items() if k != "_template_obj"}
            row["condition"] = condition
            row["designated_item_id"] = designated
            row["other_item_id"] = other
            row["designated_is_chosen"] = designated == tr["chosen_item_id"]
            row["designated_pre"] = (
                tr["pre_item1"] if designated == tr["item1_id"] else tr["pre_item2"]
            )
            row["other_pre"] = (
                tr["pre_item1"] if other == tr["item1_id"] else tr["pre_item2"]
            )
            rows.append(row)

            chosen_label = tr["choice_label"] if condition == "chose" else None
            for target in (designated, other):
                post_msgs.append(
                    post_messages(
                        t,
                        fa,
                        fb,
                        framed[target],
                        framed[designated],
                        framed[other],
                        condition,
                        chosen_label,
                        cfg,
                    )
                )

    post_out = _chunked_rate(runner, post_msgs, cfg, "pass C post")

    for i, row in enumerate(rows):
        row["designated_post"] = float(post_out.value[2 * i])
        row["other_post"] = float(post_out.value[2 * i + 1])
        row["post_mass_min"] = float(min(post_out.mass[2 * i], post_out.mass[2 * i + 1]))

    frame = pd.DataFrame(rows)
    frame["spread"] = (frame["designated_post"] - frame["designated_pre"]) - (
        frame["other_post"] - frame["other_pre"]
    )
    frame["digit_mass_min"] = frame[["pre_mass_min", "post_mass_min"]].min(axis=1)
    # A1.6: a trial is only valid if EVERY one of its four ratings cleared the
    # floor -- the DV is a difference of all four, so one bad readout poisons it.
    frame = attach(frame, frame["digit_mass_min"].to_numpy(), context="pass C ratings")

    _validate(cfg, frame)
    return write_parquet(frame, artifact_path(cfg), prov)


def _chunked_rate(runner: Runner, msgs: list, cfg: RunConfig, desc: str):
    from src.readout.expected_value import Readout

    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    values, masses, probs, argmaxes = [], [], [], []
    for start in tqdm(range(0, len(msgs), chunk), desc=desc, unit="chunk", leave=False):
        out = runner.rate(msgs[start : start + chunk])
        values.append(out.value)
        masses.append(out.mass)
        probs.append(out.probs)
        argmaxes.append(out.argmax)
    return Readout(
        value=np.concatenate(values),
        mass=np.concatenate(masses),
        probs=np.concatenate(probs),
        argmax=np.concatenate(argmaxes),
    )


def _validate(cfg: RunConfig, frame: pd.DataFrame) -> None:
    """Every counterbalancing cell filled exactly once, and the DV well-formed."""
    problems: list[str] = []

    expected = (
        (cfg.n_difficult + cfg.n_easy)
        * cfg.stimuli.n_templates
        * cfg.pass_c.n_option_orders
        * len(cfg.pass_c.conditions)
    )
    if len(frame) != expected:
        problems.append(f"{len(frame)} trials, expected {expected}")

    cell = frame.groupby(
        ["pair_id", "template", "option_order", "condition"], observed=True
    ).size()
    if (cell != 1).any():
        bad = cell[cell != 1]
        problems.append(
            f"{len(bad)} counterbalancing cell(s) not filled exactly once, e.g. "
            f"{list(bad.index[:3])}"
        )

    # yoked conditions must all designate the same item within a trial: that is
    # what makes the selection artifact cancel in chose - yoked.
    grouped = frame[frame["condition"].isin(YOKED_CONDITIONS)].groupby(
        ["pair_id", "template", "option_order"], observed=True
    )["designated_item_id"].nunique()
    if (grouped != 1).any():
        problems.append(
            "chose / yoked / 3p-yoked disagree on the designated item within a trial; "
            "the primary contrast would not be designation-matched"
        )

    # random and 3p-random must share their designation, making 3p-random - random
    # a clean endorsement contrast.
    rnd = frame[frame["condition"].isin({"random", "3p-random"})]
    if len(rnd):
        g = rnd.groupby(["pair_id", "template", "option_order"], observed=True)[
            "designated_item_id"
        ].nunique()
        if (g != 1).any():
            problems.append("random and 3p-random disagree on the designated item")

    if frame["designated_item_id"].eq(frame["other_item_id"]).any():
        problems.append("designated and other item are the same in some rows")
    if not np.isfinite(frame["spread"]).all():
        problems.append("non-finite spread values")

    bound = cfg.readout.scale_max - cfg.readout.scale_min
    if frame["spread"].abs().max() > 2 * bound + 1e-6:
        problems.append(f"spread exceeds its bound of +/-{2 * bound}")

    if problems:
        raise RuntimeError("Pass C validation failures:\n  - " + "\n  - ".join(problems))


def choice_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Position-bias diagnostics: label-A rate and flip rate across option orders.

    Yoking is defined WITHIN option order, so a flip does not desynchronise chose
    from yoked. The flip rate is reported because it is the natural measure of how
    much of the choice is driven by position rather than preference.
    """
    trials = frame.drop_duplicates(["pair_id", "template", "option_order"])
    wide = trials.pivot_table(
        index=["pair_id", "template"],
        columns="option_order",
        values="chosen_item_id",
        aggfunc="first",
    )
    rows = []
    if {0, 1} <= set(wide.columns):
        flip = (wide[0] != wide[1]).mean()
    else:
        flip = float("nan")
    for template, block in trials.groupby("template"):
        sub = wide.loc[wide.index.get_level_values("template") == template]
        rows.append(
            {
                "template": template,
                "label_a_rate": float((block["choice_label_index"] == 0).mean()),
                "flip_rate_across_option_order": (
                    float((sub[0] != sub[1]).mean()) if {0, 1} <= set(wide.columns) else float("nan")
                ),
                "mean_choice_margin": float(block["choice_margin"].mean()),
                "mean_choice_mass": float(block["choice_mass"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["overall_flip_rate"] = float(flip)
    return out


def load_or_run(
    cfg: RunConfig, pairs: pd.DataFrame, runner_factory, prov: Provenance
) -> pd.DataFrame:
    path = artifact_path(cfg)
    if path.exists():
        return read_parquet(path)
    return run_pass_c(cfg, pairs, runner_factory(), prov)
