"""Pairwise Pass A: anchor comparisons -> order-invariance gate -> theta.

preregistration.md A1.1, A1.3, A1.4, A1.7.

    python -m src.experiments.pass_a_pairwise --config configs/stage0_qwen2.5-3b.yaml

Order of operations is deliberate. The order-invariance gate is evaluated BEFORE
Bradley-Terry is fitted: fitting theta to comparisons that carry no content signal
would produce a tidy posterior that means nothing, and the gate exists precisely
to stop that from reaching a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bradley_terry import anchor_ordering_check, fit_bradley_terry, test_retest
from src.config import RunConfig, load_config
from src.provenance import Provenance, capture, read_parquet, write_parquet
from src.readout import validity
from src.readout.pairwise import collect_comparisons, load_anchors, order_invariance
from src.stimuli.build import load_items, load_templates

# A1.4, preregistered before any pairwise data was collected.
ORDER_INVARIANCE_MIN = 0.60
MIN_TEMPLATES_CLEARING = 3

HALT = 2


def artifact_path(cfg: RunConfig, kind: str = "comparisons") -> Path:
    h = cfg.hash("pass_a")
    return cfg.artifacts_dir / "pass_a_pairwise" / cfg.model.name / h / f"{kind}_{h}.parquet"


def _echo(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def evaluate_order_invariance_gate(comparisons: pd.DataFrame) -> dict:
    """A1.4. Median across templates >= 0.60, and >= 3 of 5 templates clearing."""
    per_template = order_invariance(comparisons, by=["arm", "template"])
    per_arm = order_invariance(comparisons, by=["arm"])

    digits = per_template[per_template["arm"] == "digits"]
    if digits.empty:
        return {"passed": False, "reason": "no valid digit-arm comparisons",
                "per_template": per_template, "per_arm": per_arm}

    median = float(digits["order_invariance"].median())
    clearing = sorted(digits.loc[digits["order_invariance"] >= ORDER_INVARIANCE_MIN, "template"])

    reasons: list[str] = []
    passed = True
    if median < ORDER_INVARIANCE_MIN:
        passed = False
        regime = "position-dominated" if median < 0.5 else "random-dominated"
        reasons.append(
            f"EXCLUDED: median order invariance {median:.3f} < {ORDER_INVARIANCE_MIN} "
            f"({regime}). "
            + ("Responding is driven by slot rather than content."
               if median < 0.5 else
               "Comparisons carry no recoverable content signal.")
        )
    if len(clearing) < MIN_TEMPLATES_CLEARING:
        passed = False
        reasons.append(
            f"EXCLUDED: only {len(clearing)} template(s) reach "
            f"{ORDER_INVARIANCE_MIN}, need {MIN_TEMPLATES_CLEARING}."
        )
    elif len(clearing) < len(digits):
        reasons.append(
            f"template(s) {sorted(set(digits['template']) - set(clearing))} fall below "
            f"{ORDER_INVARIANCE_MIN} and are reported."
        )

    return {
        "passed": passed,
        "median_order_invariance": median,
        "clearing_templates": clearing,
        "threshold": ORDER_INVARIANCE_MIN,
        "reasons": reasons,
        "per_template": per_template,
        "per_arm": per_arm,
    }


def operating_window(
    comparisons: pd.DataFrame, theta: pd.DataFrame, anchors_alpha: pd.DataFrame,
    n_bins: int = 8,
) -> pd.DataFrame:
    """A1.7. Order-reversal consistency against |theta_i - alpha_a|, binned.

    Uses the item-vs-anchor comparisons already collected, so it costs nothing
    extra. A usable window needs a band where consistency is above chance AND the
    gap is small. If none exists, Pass C cannot work as designed.
    """
    valid = comparisons[comparisons["readout_valid"] & (comparisons["arm"] == "digits")]
    wide = valid.pivot_table(
        index=["item_id", "anchor_id", "template"], columns="order",
        values="item_wins", aggfunc="first",
    ).dropna()
    if not {0, 1} <= set(wide.columns):
        return pd.DataFrame()

    wide = wide.reset_index()
    wide["consistent"] = wide[0] == wide[1]
    wide = wide.merge(theta[["item_id", "theta_mean"]], on="item_id")
    wide = wide.merge(anchors_alpha[["anchor_id", "alpha_mean"]], on="anchor_id")
    wide["gap"] = (wide["theta_mean"] - wide["alpha_mean"]).abs()

    wide["bin"] = pd.qcut(wide["gap"], n_bins, duplicates="drop")
    rows = []
    for b, block in wide.groupby("bin", observed=True):
        n = len(block)
        p = float(block["consistent"].mean())
        # Binomial SE; chance is 0.5.
        se = float(np.sqrt(max(p * (1 - p), 1e-9) / n))
        rows.append({
            "gap_bin": str(b),
            "gap_mid": float(block["gap"].mean()),
            "n": n,
            "consistency": p,
            "se": se,
            "above_chance": bool(p - 1.96 * se > 0.5),
        })
    return pd.DataFrame(rows)


def describe_window(window: pd.DataFrame) -> str:
    if window.empty:
        return "operating window: could not be computed (missing both orders)"
    usable = window[window["above_chance"]]
    if usable.empty:
        return (
            "OPERATING WINDOW EMPTY: consistency is not above chance at ANY gap. "
            "Pass C as designed cannot work -- reconsider the paradigm before "
            "spending run-machine time."
        )
    narrowest = usable.loc[usable["gap_mid"].idxmin()]
    widest_gap = window["gap_mid"].max()
    lines = [
        f"operating window: consistency above chance in {len(usable)}/{len(window)} bins; "
        f"narrowest usable gap = {narrowest['gap_mid']:.3f} "
        f"(consistency {narrowest['consistency']:.3f})"
    ]
    if narrowest["gap_mid"] > 0.4 * widest_gap:
        lines.append(
            "  WARNING: the usable band sits at LARGE gaps only. Spreading of "
            "alternatives requires near-equal pairs, so a window that opens only "
            "where the choice is easy does not support the paradigm."
        )
    return "\n".join(lines)


def run(cfg: RunConfig, *, progressbar: bool = True) -> int:
    prov = capture(cfg)
    out_dir = cfg.artifacts_dir / "pass_a_pairwise" / cfg.model.name / cfg.hash("pass_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": cfg.model.name, "config_hash": cfg.hash(),
                     "provenance": prov.model_dump()}

    _echo(f"Pairwise Pass A  |  {cfg.model.name}  |  {prov.device} {prov.dtype}")

    anchors = load_anchors(cfg)
    items = load_items(cfg)
    templates = load_templates(cfg)
    print(f"{len(items)} items x {len(anchors)} anchors x {len(templates)} templates "
          f"x 2 orders = {len(items) * len(anchors) * len(templates) * 2:,} comparisons")
    print(f"  -> {len(anchors) * 2} binary outcomes per item per template")

    # ---- collect --------------------------------------------------------
    path = artifact_path(cfg)
    if path.exists():
        print(f"cached: {path}")
        comparisons = read_parquet(path)
    else:
        from src.models.runner import load_runner

        runner = load_runner(cfg)
        comparisons = collect_comparisons(
            cfg, runner, items, anchors, templates, arms=("digits",), desc="pairwise"
        )
        comparisons = write_parquet(comparisons, path, prov)
        print(f"wrote: {path}")

    mass = validity.summarize(comparisons, by=["arm"])
    print("\nreadout mass (A1.6):")
    print(mass.to_string(index=False))
    results["readout_mass"] = mass.to_dict("records")

    # ---- gate BEFORE fitting -------------------------------------------
    _echo("Order-invariance gate (A1.4)")
    gate = evaluate_order_invariance_gate(comparisons)
    print(f"threshold {ORDER_INVARIANCE_MIN}; "
          f"chance 0.50; pure position responding 0.00\n")
    print(gate["per_template"].to_string(index=False))
    print()
    print(f"median (digits arm) = {gate.get('median_order_invariance', float('nan')):.3f}"
          f"  ->  {'PASS' if gate['passed'] else 'HALT'}")
    for r in gate["reasons"]:
        print(f"  - {r}")

    results["order_invariance_gate"] = {
        k: v for k, v in gate.items() if k not in ("per_template", "per_arm")
    }
    gate["per_template"].to_csv(out_dir / f"order_invariance_{cfg.hash('pass_a')}.csv",
                                index=False)

    if not gate["passed"]:
        (out_dir / f"results_{cfg.hash()}.json").write_text(
            json.dumps({**results, "outcome": "halted-order-invariance"}, indent=2, default=str)
        )
        _echo("HALTED. theta is not fitted: a posterior from signal-free comparisons "
              "would look tidy and mean nothing.")
        return HALT

    # ---- Bradley-Terry --------------------------------------------------
    _echo("Hierarchical Bradley-Terry (A1.1)")
    fit = fit_bradley_terry(cfg, comparisons, arm="digits", progressbar=progressbar)
    print(fit.summary())

    fit.theta.to_parquet(out_dir / f"theta_{cfg.hash('pass_a')}.parquet", index=False)
    results["theta"] = {
        "n_items": len(fit.theta),
        "sigma_item": fit.sigma_item,
        "posterior_sd_median": float(fit.theta["theta_sd"].median()),
        "separation_ratio": float(fit.sigma_item / fit.theta["theta_sd"].median()),
    }

    print("\nanchor abilities (intended tier is a design annotation, never shown "
          "to the model):")
    print(anchor_ordering_check(fit, anchors).to_string(index=False))

    # ---- test-retest ----------------------------------------------------
    _echo("Test-retest across disjoint template splits (A1.1)")
    names = [t.id for t in templates]
    split_a = [names[i] for i in cfg.pass_a.selection_templates]
    split_b = [names[i] for i in cfg.pass_a.analysis_templates]
    tr = test_retest(cfg, comparisons, split_a, split_b, arm="digits")
    print(f"splits {tr['split_a']} vs {tr['split_b']}")
    print(f"  Pearson  r = {tr['pearson']:.3f}")
    print(f"  Spearman r = {tr['spearman']:.3f}")
    print(f"  median posterior SD: {tr['median_posterior_sd_a']:.3f} / "
          f"{tr['median_posterior_sd_b']:.3f}")
    results["test_retest"] = {k: v for k, v in tr.items() if k != "theta"}

    # ---- operating window ----------------------------------------------
    _echo("Operating-window diagnostic (A1.7) -- evaluated BEFORE Pass C")
    window = operating_window(comparisons, fit.theta, fit.anchors)
    if not window.empty:
        print(window.to_string(index=False))
    print()
    print(describe_window(window))
    window.to_csv(out_dir / f"operating_window_{cfg.hash('pass_a')}.csv", index=False)
    results["operating_window"] = window.to_dict("records")

    results["outcome"] = "completed"
    (out_dir / f"results_{cfg.hash()}.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nresults: {out_dir}")
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
