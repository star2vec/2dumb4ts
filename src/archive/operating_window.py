"""RETIRED. Not imported, not invoked, not tested. See src/archive/README.md.

This module is kept only so the numbers it produced remain traceable to the code that
produced them. Do not import it, do not run it, and do not quote its output.
"""

"""Operating-window diagnostic, stratified on theta. preregistration.md A1.7.

    python -m src.experiments.operating_window --config configs/stage0_qwen2.5-3b.yaml

WHY THIS EXISTS SEPARATELY FROM THE ANCHOR-BASED VERSION.

`pass_a_pairwise.operating_window` reuses the anchor comparisons already collected,
which costs nothing extra -- but anchors are chosen to SPAN the appeal range, so
almost every pool-vs-anchor gap is wide. That leaves the diagnostic thinnest exactly
where the paradigm lives. Spreading of alternatives requires NEAR-EQUAL pairs, so the
small-gap end is the only end that decides anything.

This module fixes that by sampling item-vs-item pairs stratified on |theta_i -
theta_j| from the fitted scores, deliberately including the smallest gaps available,
and running both orders.

THE HEADLINE NUMBER is not "is there any band". It is: **is order-reversal
consistency above chance at the gap Pass B's difficult decile would actually
select?** A window that opens only where the choice is easy does not support the
paradigm, because Pass C never presents easy pairs to the difficult cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RunConfig, load_config
from src.provenance import capture, read_parquet, write_parquet
from src.readout import validity
from src.readout.choice import read_choice
from src.readout.digits import build_label_map
from src.readout.pairwise import DIGIT_LABELS, comparison_prompt
from src.stimuli.build import load_items, load_templates

N_STRATA = 8
PAIRS_PER_STRATUM = 24


def artifact_path(cfg: RunConfig) -> Path:
    h = cfg.hash("pass_a")
    return cfg.artifacts_dir / "operating_window" / cfg.model.name / h / f"window_{h}.parquet"


def theta_path(cfg: RunConfig) -> Path:
    h = cfg.hash("pass_a")
    return cfg.artifacts_dir / "pass_a_pairwise" / cfg.model.name / h / f"theta_{h}.parquet"


def stratified_pairs(
    cfg: RunConfig, theta: pd.DataFrame, *, n_strata: int = N_STRATA,
    per_stratum: int = PAIRS_PER_STRATUM,
) -> tuple[pd.DataFrame, float]:
    """Sample within-domain pairs across gap strata. Returns (pairs, difficult_gap).

    `difficult_gap` is the bottom-decile |theta| gap over ALL within-domain pairs --
    the gap Pass B's difficult decile would select on this scale. It is the reference
    point the verdict is measured against.

    Within-domain because Pass B pairs within domain; a cross-domain window would
    measure a comparison Pass C never makes.
    """
    rng = np.random.default_rng(cfg.seed)
    items = {i.id: i for i in load_items(cfg)}
    th = dict(zip(theta["item_id"], theta["theta_mean"]))

    rows = []
    for domain in cfg.stimuli.domains:
        ids = sorted([i for i in th if items[i].domain == domain])
        for a, b in combinations(ids, 2):
            rows.append({"item_a": a, "item_b": b, "domain": domain,
                         "gap": abs(th[a] - th[b])})
    allpairs = pd.DataFrame(rows)
    if allpairs.empty:
        raise ValueError("no within-domain candidate pairs")

    difficult_gap = float(allpairs["gap"].quantile(cfg.pass_b.difficult_quantile))

    allpairs["stratum"] = pd.qcut(
        allpairs["gap"], n_strata, duplicates="drop", labels=False
    )
    picks = []
    for s, block in allpairs.groupby("stratum", observed=True):
        take = min(per_stratum, len(block))
        idx = rng.choice(len(block), size=take, replace=False)
        chosen = block.iloc[idx].copy()
        chosen["stratum"] = s
        picks.append(chosen)
    out = pd.concat(picks, ignore_index=True)
    out["pair_id"] = out["item_a"].str.split("/").str[-1] + "|" + out["item_b"].str.split("/").str[-1]
    return out, difficult_gap


def collect(cfg: RunConfig, runner, pairs: pd.DataFrame, templates) -> pd.DataFrame:
    """Item-vs-item comparisons, both orders, digits arm."""
    from tqdm import tqdm

    items = {i.id: i for i in load_items(cfg)}
    lmap = build_label_map(runner.tokenizer, DIGIT_LABELS)

    rows, prompts = [], []
    for p in pairs.itertuples():
        for t in templates:
            for order in (0, 1):
                first, second = (
                    (p.item_a, p.item_b) if order == 0 else (p.item_b, p.item_a)
                )
                prompts.append(runner.render([{
                    "role": "user",
                    "content": comparison_prompt(
                        "digits", items[first].label, items[second].label, t, DIGIT_LABELS
                    ),
                }]))
                rows.append({
                    "pair_id": p.pair_id, "domain": p.domain, "stratum": int(p.stratum),
                    "gap": float(p.gap), "item_a": p.item_a, "item_b": p.item_b,
                    "template": t.id, "order": order,
                    "a_in_slot1": order == 0,
                })

    picks, masses = [], []
    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    for start in tqdm(range(0, len(prompts), chunk), desc="window", unit="chunk", leave=False):
        logits = runner.last_logits(prompts[start : start + chunk])
        out = read_choice(logits, lmap)
        picks.extend(out.index.tolist())
        masses.extend(out.mass.tolist())

    frame = pd.DataFrame(rows)
    frame["pick_slot"] = picks
    # Did item_a win, regardless of which slot it occupied?
    frame["a_wins"] = np.where(
        frame["a_in_slot1"], frame["pick_slot"] == 0, frame["pick_slot"] == 1
    )
    return validity.attach(frame, np.asarray(masses), context="operating window")


def summarize(frame: pd.DataFrame, theta: pd.DataFrame) -> pd.DataFrame:
    """Consistency and theta-agreement per gap stratum."""
    valid = frame[frame["readout_valid"]]
    wide = valid.pivot_table(
        index=["pair_id", "template", "stratum", "gap", "item_a", "item_b"],
        columns="order", values="a_wins", aggfunc="first",
    ).dropna().reset_index()
    if not {0, 1} <= set(wide.columns):
        return pd.DataFrame()

    wide["consistent"] = wide[0] == wide[1]
    th = dict(zip(theta["item_id"], theta["theta_mean"]))
    # Under order averaging, a_wins in both orders means an unambiguous win.
    wide["a_higher_theta"] = [th[a] > th[b] for a, b in zip(wide["item_a"], wide["item_b"])]
    wide["agrees_theta"] = np.where(
        wide["consistent"], wide[0] == wide["a_higher_theta"], np.nan
    )

    rows = []
    for s, block in wide.groupby("stratum", observed=True):
        n = len(block)
        p = float(block["consistent"].mean())
        se = float(np.sqrt(max(p * (1 - p), 1e-9) / n))
        agree = block["agrees_theta"].dropna()
        rows.append({
            "stratum": int(s),
            "gap_mean": float(block["gap"].mean()),
            "gap_min": float(block["gap"].min()),
            "n": n,
            "consistency": p,
            "se": se,
            "ci_low": p - 1.96 * se,
            "above_chance": bool(p - 1.96 * se > 0.5),
            "agrees_theta": float(agree.mean()) if len(agree) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("gap_mean").reset_index(drop=True)


def verdict(summary: pd.DataFrame, difficult_gap: float) -> str:
    """Is there signal at the gap Pass B's difficult decile would select?"""
    if summary.empty:
        return "operating window: not computable (missing an order)"

    lines = [
        f"Pass B's difficult decile corresponds to a theta gap of "
        f"{difficult_gap:.3f} on this model's scale."
    ]
    at_difficult = summary[summary["gap_mean"] <= difficult_gap]
    usable = summary[summary["above_chance"]]

    if at_difficult.empty:
        lines.append(
            "  no sampled stratum sits at or below that gap; the window cannot speak "
            "to the difficult cell. Increase strata resolution at the small-gap end."
        )
    else:
        best = at_difficult.loc[at_difficult["consistency"].idxmax()]
        ok = bool(best["above_chance"])
        lines.append(
            f"  best consistency at or below it: {best['consistency']:.3f} "
            f"(95% CI low {best['ci_low']:.3f}, n={int(best['n'])}) -> "
            f"{'ABOVE chance' if ok else 'NOT above chance'}"
        )
        if ok:
            lines.append(
                "  WINDOW OPEN: near-equal pairs still elicit content-driven choice. "
                "Pass C's difficult cell is measurable."
            )
        else:
            lines.append(
                "  WINDOW CLOSED at the difficult end. Choices on near-equal pairs are "
                "indistinguishable from position or noise, so the difficult cell -- "
                "which the primary interaction depends on -- carries no signal. "
                "Pass C as designed cannot work; reconsider the paradigm before "
                "spending run-machine time."
            )

    if not usable.empty:
        lines.append(
            f"  ({len(usable)}/{len(summary)} strata above chance overall; narrowest "
            f"usable gap {usable['gap_mean'].min():.3f})"
        )
    else:
        lines.append("  no stratum at any gap is above chance -- the instrument, not "
                     "just the window, is failing here.")
    return "\n".join(lines)


def run(cfg: RunConfig) -> int:
    prov = capture(cfg)
    tpath = theta_path(cfg)
    if not tpath.exists():
        print(f"no theta artifact at {tpath}\n"
              "Run `python -m src.experiments.pass_a_pairwise` first; if it halted on "
              "the order-invariance gate, the window is undefined for this model.")
        return 1

    theta = pd.read_parquet(tpath)
    templates = load_templates(cfg)
    pairs, difficult_gap = stratified_pairs(cfg, theta)

    print(f"\n{'=' * 72}\nOperating window (A1.7)  |  {cfg.model.name}\n{'=' * 72}")
    print(f"{len(pairs)} pairs across {pairs['stratum'].nunique()} gap strata "
          f"x {len(templates)} templates x 2 orders = "
          f"{len(pairs) * len(templates) * 2:,} comparisons")

    path = artifact_path(cfg)
    if path.exists():
        print(f"cached: {path}")
        frame = read_parquet(path)
    else:
        from src.models.runner import load_runner

        frame = collect(cfg, load_runner(cfg), pairs, templates)
        frame = write_parquet(frame, path, prov)
        print(f"wrote: {path}")

    print("\nreadout mass:")
    print(validity.summarize(frame).to_string(index=False))

    summary = summarize(frame, theta)
    print("\nconsistency under order reversal by theta gap:")
    print(summary.to_string(index=False))
    print()
    print(verdict(summary, difficult_gap))

    out = path.parent
    summary.to_csv(out / f"window_summary_{cfg.hash('pass_a')}.csv", index=False)
    (out / f"window_{cfg.hash()}.json").write_text(json.dumps({
        "model": cfg.model.name,
        "difficult_decile_gap": difficult_gap,
        "strata": summary.to_dict("records"),
        "provenance": prov.model_dump(),
    }, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args(argv)
    overrides = {"artifacts_dir": args.artifacts} if args.artifacts else None
    return run(load_config(args.config, overrides))


if __name__ == "__main__":
    sys.exit(main())
