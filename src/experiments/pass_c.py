"""Pass C -- the experiment. preregistration.md sections 6 and Amendment 2 A2.9.

    python -m src.experiments.pass_c --config configs/stage0_gemma-2-2b.yaml

WHAT IT MEASURES. For each pair: the model's preference between the two items BEFORE
any manipulation, then again AFTER. Whether that preference pulls apart, and whether it
pulls apart more for near-equal pairs when the model chose than when it was assigned, is
H1.

    logit P(item1 beats item2)
        = u_pair + u_template + beta_t*s + post*d*(gamma_c + lambda_c*diff_z)

    PRIMARY: lambda_chose - lambda_yoked, predicted negative.

The DV question is byte-identical at pre and post -- a pre/post contrast is not a
contrast if the question moves.

EIGHT CONDITIONS (A2.9.3). The first four are a 2x2 of transcript structure x
attribution wording:

                          "you chose"          "assigned"
    assistant turn        chose                structure-control
    no assistant turn     self-recounted       yoked

plus 3p-yoked, 3p-random, random, and chose-provisional. Every condition carries a
finality clause so receipt matching survives; chose-provisional swaps it, making
reversibility a one-clause contrast.

PASS STRUCTURE, per model:
    pre     200 pairs x 5 templates x 2 orders          =  2,000   shared by all conditions
    choice  200 x 5 x 2                                 =  2,000   designation + write-back
    post    200 x 5 x 2 x 8                             = 16,000
                                                          ------
                                                          20,000

NO ACTIVATION CACHE HERE. Hidden states are collected in a separate pass. On the run
machine weights and activations must coexist on one 6.7 GB volume and three of five
models exceed it; forward passes are deterministic given the same weights, seed and
batch size, so a later activation pass corresponds exactly to the behaviour banked here.
Building a half-working cache now would be worse than building none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis.spread_model import PRE_SENTINEL
from src.config import RunConfig, load_config
from src.models.runner import Runner
from src.provenance import Provenance, capture, read_parquet, write_parquet
from src.readout import validity
from src.readout.choice import read_choice
from src.readout.digits import build_label_map
from src.stimuli.build import (
    Template,
    balanced_designation,
    choice_messages,
    load_items,
    load_templates,
    post_dv_messages,
    pre_dv_messages,
)

#: Conditions whose designated item is the model's own pick. The 2x2 is entirely
#: inside this set, which is what makes its four cells designation-matched.
OWN_PICK_CONDITIONS = frozenset({
    "chose", "structure-control", "self-recounted", "yoked", "3p-yoked",
    "chose-provisional",
})


def artifact_path(cfg: RunConfig) -> Path:
    h = cfg.hash("pass_c")
    return cfg.artifact_dir("pass_c") / f"trials_{h}.parquet"


def _displayed(item1: str, item2: str, option_order: int) -> tuple[str, str]:
    """Which item occupies slot 1. Counterbalanced, never sampled."""
    return (item1, item2) if option_order == 0 else (item2, item1)


def run_pass_c(
    cfg: RunConfig, pairs: pd.DataFrame, runner: Runner, prov: Provenance
) -> pd.DataFrame:
    framed = {i.id: i.framed for i in load_items(cfg)}
    templates: list[Template] = load_templates(cfg)
    labels = cfg.readout.option_labels
    label_map = build_label_map(runner.tokenizer, labels)

    # `random` / `3p-random` designation: exactly balanced within every
    # (difficulty, domain) stratum and fixed by the run seed. The SAME assignment
    # serves both, which makes 3p-random - random a clean endorsement contrast.
    designation = balanced_designation(
        pairs["pair_id"].tolist(),
        (pairs["difficulty"] + "/" + pairs["domain"]).tolist(),
        cfg.seed,
    )

    # ---- phase 1: shared pre-measurement and the choice --------------------
    cells: list[dict] = []
    pre_msgs, choice_msgs = [], []
    for p in pairs.to_dict("records"):
        for t in templates:
            for order in range(cfg.pass_c.n_option_orders):
                s1, s2 = _displayed(p["item1_id"], p["item2_id"], order)
                fa, fb = framed[s1], framed[s2]
                cells.append({
                    "pair_id": p["pair_id"], "domain": p["domain"],
                    "difficulty": p["difficulty"], "matched_set": p["matched_set"],
                    "item1_id": p["item1_id"], "item2_id": p["item2_id"],
                    "template": t.id, "template_index": t.index,
                    "option_order": order, "slot1_item_id": s1, "slot2_item_id": s2,
                    "diff_selection": p["diff_selection"],
                    "diff_analysis": p["diff_analysis"],
                    "_t": t,
                })
                pre_msgs.append(pre_dv_messages(t, fa, fb, cfg))
                # Built by the shared template builder, NOT by string-surgery on the pre
                # prompt. The previous version took `pre.split("\n\n")[0]` to recover the
                # pair block -- but every template's pair block CONTAINS a blank line, so
                # the split truncated it to its first line and the model was asked to
                # choose between A and B without being shown what A and B were. The choice
                # it returned carried no item content, and six of the eight conditions
                # designate that choice.
                choice_msgs.append(choice_messages(t, fa, fb, cfg))

    pre = _read(runner, pre_msgs, label_map, cfg, "pass C pre")
    cho = _read(runner, choice_msgs, label_map, cfg, "pass C choice")

    rows: list[dict] = []
    post_msgs: list[list[dict[str, str]]] = []
    for i, cell in enumerate(cells):
        s1 = cell["slot1_item_id"]
        # The readout indexes SLOTS; map to items via which item is in slot 1.
        pre_slot = int(pre["index"][i])
        cell["pre_item1_wins"] = bool(
            (pre_slot == 0) == (s1 == cell["item1_id"])
        )
        ch_slot = int(cho["index"][i])
        cell["chosen_item_id"] = s1 if ch_slot == 0 else cell["slot2_item_id"]
        cell["choice_label"] = labels[ch_slot]
        cell["choice_margin"] = float(cho["margin"][i])

        base = {k: v for k, v in cell.items() if k != "_t"}
        # The shared baseline, emitted ONCE -- not once per condition.
        # The GRADED readout, persisted beside the binary one. `item1_wins` is an argmax:
        # it keeps the sign of the model's preference and discards its strength, which is a
        # dichotomisation of the outcome. Section 13 item 9 rejected dichotomising the
        # REGRESSOR for costing ~36% of the variance, and then the outcome was dichotomised
        # anyway; 3.1 states "expected values ... never argmax" and 3.5 exempts only the
        # CHOICE, which is a discrete commitment. The DV is a measurement. See A4.5.
        #
        # This costs no forward passes -- the number is already computed and was thrown
        # away before it reached the artifact.
        pre_p1 = float(p_slot1(pre["index"][i], pre["margin"][i]))
        if s1 != cell["item1_id"]:
            pre_p1 = 1.0 - pre_p1
        cell["pre_p_item1"] = pre_p1
        rows.append({**base, "condition": PRE_SENTINEL, "timepoint": "pre",
                     "designated_item_id": cell["item1_id"],
                     "item1_wins": cell["pre_item1_wins"],
                     "p_item1": pre_p1,
                     "readout_mass": float(pre["mass"][i]),
                     "readout_valid": bool(pre["valid"][i])})

        t: Template = cell["_t"]
        fa, fb = framed[s1], framed[cell["slot2_item_id"]]
        for condition in cfg.pass_c.conditions:
            if condition in OWN_PICK_CONDITIONS:
                designated = cell["chosen_item_id"]
            else:
                which = designation[cell["pair_id"]]
                designated = cell["item1_id"] if which == 0 else cell["item2_id"]
            other = cell["item2_id"] if designated == cell["item1_id"] else cell["item1_id"]

            rows.append({**base, "condition": condition, "timepoint": "post",
                         "designated_item_id": designated, "other_item_id": other,
                         "designated_is_chosen": designated == cell["chosen_item_id"]})
            post_msgs.append(post_dv_messages(
                t, fa, fb, framed[designated], framed[other], condition,
                cell["choice_label"], cfg,
            ))

    # ---- phase 2: post-manipulation measurement ---------------------------
    post = _read(runner, post_msgs, label_map, cfg, "pass C post")

    frame = pd.DataFrame(rows)
    post_mask = frame["timepoint"] == "post"
    idx = frame.index[post_mask]
    slot_pick = post["index"]
    s1_is_item1 = (frame.loc[idx, "slot1_item_id"] == frame.loc[idx, "item1_id"]).to_numpy()
    frame.loc[idx, "item1_wins"] = (slot_pick == 0) == s1_is_item1
    post_p1 = p_slot1(slot_pick, post["margin"])
    frame.loc[idx, "p_item1"] = np.where(s1_is_item1, post_p1, 1.0 - post_p1)
    frame.loc[idx, "readout_mass"] = post["mass"]
    frame.loc[idx, "readout_valid"] = post["valid"]
    frame["item1_wins"] = frame["item1_wins"].astype(bool)

    _validate(cfg, frame)
    return write_parquet(frame, artifact_path(cfg), prov)


def p_slot1(index, margin):
    """Renormalised probability on SLOT 1, recovered from the argmax and its margin.

    `read_choice` renormalises over the option labels, so with two labels
    p(top) = (1 + margin)/2 and p(other) = (1 - margin)/2. Nothing is approximated here:
    the pair (index, margin) determines the two-label distribution exactly.
    """
    top = (1.0 + np.asarray(margin, dtype=float)) / 2.0
    return np.where(np.asarray(index) == 0, top, 1.0 - top)


def _read(runner: Runner, msgs: list, label_map, cfg: RunConfig, desc: str) -> dict:
    """Batched one-position choice readout with the A1.6 mass floor applied."""
    idxs, margins, masses = [], [], []
    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    prompts = [runner.render(m) for m in msgs]
    for start in tqdm(range(0, len(prompts), chunk), desc=desc, unit="chunk", leave=False):
        logits = runner.last_logits(prompts[start : start + chunk])
        out = read_choice(logits, label_map)
        idxs.extend(out.index.tolist())
        margins.extend(out.margin.tolist())
        masses.extend(out.mass.tolist())
    mass = np.asarray(masses)
    valid, _ = validity.check_mass(mass, context=desc)
    return {"index": np.asarray(idxs), "margin": np.asarray(margins),
            "mass": mass, "valid": valid}


def _validate(cfg: RunConfig, frame: pd.DataFrame) -> None:
    problems: list[str] = []
    # Cell count is taken from the data, not from config. Pass B already validates that
    # it produced n_difficult + n_easy pairs; re-asserting that contract here would make
    # Pass C untestable on any subset and would fail for the wrong reason.
    n_cells = frame.groupby(
        ["pair_id", "template", "option_order"], observed=True
    ).ngroups
    expected = n_cells * (1 + len(cfg.pass_c.conditions))
    if len(frame) != expected:
        problems.append(
            f"{len(frame)} rows for {n_cells} cells and "
            f"{len(cfg.pass_c.conditions)} conditions, expected {expected}"
        )

    pre = frame[frame["condition"] == PRE_SENTINEL]
    if len(pre) != n_cells:
        problems.append(
            f"{len(pre)} pre rows, expected {n_cells} -- the shared baseline must be "
            "emitted once per cell, not once per condition"
        )
    if not (pre["timepoint"] == "pre").all():
        problems.append("a PRE_SENTINEL row is not timepoint 'pre'")

    cell = frame[frame["timepoint"] == "post"].groupby(
        ["pair_id", "template", "option_order", "condition"], observed=True
    ).size()
    if (cell != 1).any():
        problems.append(f"{int((cell != 1).sum())} post cell(s) not filled exactly once")

    # The 2x2 and the 3p-yoked arm must all designate the model's own pick, or the
    # contrasts among them are not designation-matched.
    own = frame[frame["condition"].isin(OWN_PICK_CONDITIONS & set(cfg.pass_c.conditions))]
    if len(own):
        g = own.groupby(["pair_id", "template", "option_order"], observed=True)[
            "designated_item_id"].nunique()
        if (g != 1).any():
            problems.append(
                "own-pick conditions disagree on the designated item within a cell; "
                "the primary contrast would not be designation-matched"
            )

    # The random family must share one designation, or 3p-random - random is not an
    # endorsement contrast. balanced_designation returns a single per-pair assignment
    # serving both, so this asserts a property of the code rather than hoping for it.
    rnd = frame[frame["condition"].isin({"random", "3p-random"} & set(cfg.pass_c.conditions))]
    if rnd["condition"].nunique() > 1:
        g = rnd.groupby(["pair_id", "template", "option_order"], observed=True)[
            "designated_item_id"].nunique()
        if (g != 1).any():
            problems.append(
                "random and 3p-random disagree on the designated item within a cell; "
                "3p-random - random would confound endorsement with designation"
            )

    if frame["designated_item_id"].isna().any():
        problems.append("null designated_item_id")
    post = frame[frame["timepoint"] == "post"]
    stray = post[~(
        (post["designated_item_id"] == post["item1_id"])
        | (post["designated_item_id"] == post["item2_id"])
    )]
    if len(stray):
        problems.append(
            f"{len(stray)} post row(s) designate an item that is not in the pair")

    if problems:
        raise RuntimeError("Pass C validation failures:\n  - " + "\n  - ".join(problems))


def load_or_run(cfg: RunConfig, pairs: pd.DataFrame, runner_factory, prov: Provenance):
    path = artifact_path(cfg)
    if path.exists():
        return read_parquet(path)
    return run_pass_c(cfg, pairs, runner_factory(), prov)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"artifacts_dir": args.artifacts} if args.artifacts else None)

    from src.experiments import pass_b

    prov = capture(cfg)
    pairs_path = pass_b.artifact_path(cfg)
    if not pairs_path.exists():
        print(f"no pairs at {pairs_path}; run Pass B first")
        return 1
    pairs = read_parquet(pairs_path)

    print(f"\n{'=' * 72}\nPass C  |  {cfg.model.name}  |  {prov.device} {prov.dtype}\n{'=' * 72}")
    n_cells = len(pairs) * cfg.stimuli.n_templates * cfg.pass_c.n_option_orders
    print(f"{len(pairs)} pairs x {cfg.stimuli.n_templates} templates x "
          f"{cfg.pass_c.n_option_orders} orders = {n_cells:,} cells")
    print(f"  pre {n_cells:,} + choice {n_cells:,} + post "
          f"{n_cells * len(cfg.pass_c.conditions):,} = "
          f"{n_cells * (2 + len(cfg.pass_c.conditions)):,} forward passes")

    from src.models.runner import load_runner

    frame = run_pass_c(cfg, pairs, load_runner(cfg), prov)
    print(f"wrote: {artifact_path(cfg)}")
    print("\nreadout mass by condition:")
    print(validity.summarize(frame, by=["condition"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
