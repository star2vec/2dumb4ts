"""Pairwise Pass A: anchor comparisons -> theta -> reliability gate -> window.

preregistration.md A1.1, A1.3, A1.6 and Amendment 2 (A2.1, A2.2, A2.6).

    python -m src.experiments.pass_a_pairwise --config configs/stage0_qwen2.5-3b.yaml

Fitting now precedes the gate. Amendment 1 gated first, on the grounds that fitting
signal-free comparisons yields a meaningless posterior. The A2.2 gate is computed
FROM fits (empirical split-half reliability of theta), so it cannot precede them. The
protection is preserved by the gate itself: a model whose theta does not replicate
across disjoint template splits is excluded and no theta is reported for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bradley_terry import (
    anchor_ordering_check, convergence_summary, excess_consistency_slope,
    excess_slope_ppc_null, fit_bradley_terry, test_retest,
)
from src.config import RunConfig, load_config
from src.provenance import Provenance, capture, read_parquet, write_parquet
from src.readout import validity
from src.readout.pairwise import collect_comparisons, load_anchors, order_invariance
from src.stimuli.build import load_items, load_templates

# A2.2: the A1.4 order-invariance gate is RETIRED. Its null, 2*s*(1-s) with
# s = sigmoid(beta), moves with a signed nuisance parameter that ranges 0.464 to
# 0.026 across templates in a single model, so a flat threshold was predominantly a
# test of |beta| rather than of content signal. Replaced by empirical split-half
# reliability of theta, which is beta-free once beta is modelled and is the quantity
# Pass B actually depends on.
RELIABILITY_MIN = 0.70

# Retained for reporting only -- never an exclusion criterion.
ORDER_INVARIANCE_REPORTED_ONLY = True

HALT = 2


def artifact_path(cfg: RunConfig, kind: str = "comparisons") -> Path:
    h = cfg.hash("pass_a")
    return cfg.artifacts_dir / "pass_a_pairwise" / cfg.model.name / h / f"{kind}_{h}.parquet"


def _echo(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def report_order_invariance(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Order invariance, REPORTED ONLY (A2.2). No longer a gate.

    Kept because it is an interpretable description of how much of a model's
    responding is slot-driven, and because the retired threshold's failure is part of
    the record. It must not be used to exclude a model: its null moves with beta.
    """
    return order_invariance(comparisons, by=["arm", "template"])


def evaluate_reliability_gate(
    cfg: RunConfig, comparisons: pd.DataFrame, templates: list
) -> dict:
    """A2.2 gate: empirical split-half reliability of theta across the disjoint
    template sets of section 4.4.

    Empirical rather than model-internal, because a misspecified likelihood can
    report a confident posterior while the two halves disagree. The gap between the
    two is itself a preregistered misspecification diagnostic.
    """
    names = [t.id for t in templates]
    split_a = [names[i] for i in cfg.pass_a.selection_templates if i < len(names)]
    split_b = [names[i] for i in cfg.pass_a.analysis_templates if i < len(names)]

    tr = test_retest(cfg, comparisons, split_a, split_b, arm="digits")
    empirical = tr["spearman"]

    passed = empirical >= RELIABILITY_MIN
    reasons: list[str] = []
    if not passed:
        reasons.append(
            f"EXCLUDED: empirical split-half reliability of theta "
            f"(Spearman {empirical:.3f}) < {RELIABILITY_MIN}. Items cannot be ranked "
            "well enough to select near-equal pairs or to detect movement."
        )
    return {
        "passed": passed,
        "empirical_reliability_spearman": empirical,
        "empirical_reliability_pearson": tr["pearson"],
        "threshold": RELIABILITY_MIN,
        "split_a": split_a,
        "split_b": split_b,
        "reasons": reasons,
    }


def operating_window(
    comparisons: pd.DataFrame, fit, n_bins: int = 8,
) -> pd.DataFrame:
    """A2.6. Discriminability of pairs by gap, plus a beta-aware fit check.

    REDEFINED. A1.7 asked whether order-reversal consistency exceeded 0.5. That was
    wrong twice over: the null under position bias is 2*s*(1-s) with s = sigmoid(beta),
    not 0.5; and once beta IS modelled, consistency carries no information about
    content signal beyond what theta already encodes, so testing it becomes vacuous.

    The question that actually decides Pass C viability is a PRECISION question: at
    the gaps Pass B would select, is the sign of theta_i - alpha_a credibly
    determined? So the window now reports, per gap stratum:

      discriminable  fraction of comparisons whose theta-alpha difference has a
                     credible sign given posterior uncertainty
      consistency    observed, alongside the beta-model prediction, as a FIT CHECK
                     only -- these should agree, and a gap between them indicates
                     misspecification rather than absence of signal
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

    th = fit.theta.set_index("item_id")
    al = fit.anchors.set_index("anchor_id")
    bt = dict(zip(fit.beta["template"], fit.beta["beta_mean"]))

    wide["gap"] = (th.loc[wide["item_id"], "theta_mean"].to_numpy()
                   - al.loc[wide["anchor_id"], "alpha_mean"].to_numpy())
    # Posterior sd of the difference, treating theta and alpha as independent.
    wide["gap_sd"] = np.sqrt(
        th.loc[wide["item_id"], "theta_sd"].to_numpy() ** 2
        + al.loc[wide["anchor_id"], "alpha_sd"].to_numpy() ** 2
    )
    wide["discriminable"] = wide["gap"].abs() > 1.96 * wide["gap_sd"]
    wide["abs_gap"] = wide["gap"].abs()

    b = np.array([bt[t] for t in wide["template"]])
    p0, p1 = _sig(wide["abs_gap"].to_numpy() + b), _sig(wide["abs_gap"].to_numpy() - b)
    wide["pred_consistency"] = p0 * p1 + (1 - p0) * (1 - p1)

    wide["bin"] = pd.qcut(wide["abs_gap"], n_bins, duplicates="drop")
    rows = []
    for bn, blk in wide.groupby("bin", observed=True):
        n = len(blk)
        disc = float(blk["discriminable"].mean())
        rows.append({
            "gap_mid": float(blk["abs_gap"].mean()),
            "n": n,
            "discriminable": disc,
            "disc_se": float(np.sqrt(max(disc * (1 - disc), 1e-9) / n)),
            "consistency": float(blk["consistent"].mean()),
            "pred_consistency": float(blk["pred_consistency"].mean()),
            "excess": float(blk["consistent"].mean() - blk["pred_consistency"].mean()),
        })
    return pd.DataFrame(rows)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def describe_window(window: pd.DataFrame, difficult_gap: float | None = None) -> str:
    """Verdict on whether Pass C's difficult cell is measurable (A2.6)."""
    if window.empty:
        return "operating window: not computable (an order is missing)"
    lines = [
        f"discriminability rises from {window['discriminable'].iloc[0]:.2f} at gap "
        f"{window['gap_mid'].iloc[0]:.2f} to {window['discriminable'].iloc[-1]:.2f} at gap "
        f"{window['gap_mid'].iloc[-1]:.2f}"
    ]
    worst = float(window["excess"].abs().max())
    lines.append(
        f"fit check: |excess consistency| <= {worst:.3f} across bins "
        + ("(consistent with the beta model)" if worst < 0.08 else
           "(LARGE -- the order model is still misspecified)")
    )
    if difficult_gap is not None:
        at = window[window["gap_mid"] <= difficult_gap]
        if at.empty:
            lines.append(
                f"  no stratum at or below Pass B's difficult-decile gap "
                f"({difficult_gap:.3f}); increase small-gap resolution"
            )
        else:
            d = float(at["discriminable"].max())
            lines.append(
                f"  at Pass B's difficult-decile gap ({difficult_gap:.3f}), "
                f"discriminability = {d:.2f}"
            )
            lines.append(
                "  -> difficult pairs are individually resolvable"
                if d >= 0.5 else
                "  -> difficult pairs are NOT individually resolvable; the difficulty "
                "regressor will be heavily attenuated and Pass C needs a power "
                "simulation against that attenuation before it is run"
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
            cfg, runner, items, anchors, templates, arms=("digits",), desc="pairwise",
            # Resumable: at full scale this is 40,000 comparisons per model.
            checkpoint=out_dir / f"_checkpoint_{cfg.hash('pass_a')}.parquet",
        )
        comparisons = write_parquet(comparisons, path, prov)
        print(f"wrote: {path}")

    mass = validity.summarize(comparisons, by=["arm"])
    print("\nreadout mass (A1.6):")
    print(mass.to_string(index=False))
    # Per template as well: a prompt bug shows up here long before it reaches a
    # posterior, and it is invisible in the pooled figure.
    per_t = validity.summarize(comparisons, by=["arm", "template"])
    print("\nreadout mass by template:")
    print(per_t.to_string(index=False))
    results["readout_mass"] = mass.to_dict("records")
    results["readout_mass_by_template"] = per_t.to_dict("records")

    # ---- order invariance: REPORTED, not a gate (A2.2) -------------------
    _echo("Order invariance -- reported only, no longer a gate (A2.2)")
    inv = report_order_invariance(comparisons)
    print("retired as an exclusion criterion: its null is 2*s*(1-s) with "
          "s = sigmoid(beta),\nwhich moves with a signed, template-varying nuisance "
          "parameter.\n")
    print(inv.to_string(index=False))
    results["order_invariance_reported"] = inv.to_dict("records")
    inv.to_csv(out_dir / f"order_invariance_{cfg.hash('pass_a')}.csv", index=False)

    # ---- Bradley-Terry, then the gate -----------------------------------
    # Order reversed relative to Amendment 1. The old gate ran first on the grounds
    # that fitting signal-free comparisons yields a meaningless posterior; the new
    # gate is computed FROM fits, so it cannot precede them. The protection that
    # rationale sought is preserved by the gate itself: a model whose theta does not
    # replicate across template splits is excluded, and no theta is reported for it.
    _echo("Hierarchical Bradley-Terry with per-template order term (A2.1)")
    fit = fit_bradley_terry(cfg, comparisons, arm="digits", progressbar=progressbar)
    print(fit.summary())
    print(convergence_summary(cfg, fit.idata))

    print("\nposition bias by template (beta):")
    print(fit.beta.round(3).to_string(index=False))
    fit.beta.to_parquet(out_dir / f"beta_{cfg.hash('pass_a')}.parquet", index=False)
    results["beta"] = fit.beta.to_dict("records")

    # Preregistered fit-quality diagnostic (A2.1): should be flat.
    ex = excess_consistency_slope(comparisons, fit)
    null = excess_slope_ppc_null(cfg, comparisons, fit)
    z = ((ex["slope"] - null["null_mean"]) / null["null_sd"]
         if null["null_sd"] > 0 else float("nan"))
    print(f"\nfit quality -- excess consistency on gap (A2.1):")
    print(f"  observed slope {ex['slope']:+.4f} 95% CI "
          f"[{ex['ci_low']:+.4f}, {ex['ci_high']:+.4f}] over {ex['n_cells']} cells")
    # The null is NOT zero: plug-in prediction of a nonlinear quantity biases this
    # statistic even under correct specification.
    print(f"  posterior-predictive null {null['null_mean']:+.4f} "
          f"(sd {null['null_sd']:.4f}, {null['n_replicates']} replicates)")
    print(f"  observed sits {z:+.2f} sd from the null")
    print("  " + ("CONSISTENT with correct specification"
                  if abs(z) < 2 else
                  "BEYOND the null: residual misspecification is real"))
    results["excess_consistency_slope"] = {**ex, **null, "z_vs_null": z}

    # ---- reliability gate (A2.2) ----------------------------------------
    _echo("Reliability gate (A2.2)")
    gate = evaluate_reliability_gate(cfg, comparisons, templates)
    print(f"empirical split-half reliability of theta across {gate['split_a']} vs "
          f"{gate['split_b']}:")
    print(f"  Spearman {gate['empirical_reliability_spearman']:.3f} "
          f"(Pearson {gate['empirical_reliability_pearson']:.3f}), "
          f"threshold {gate['threshold']}")
    print(f"  model-internal reliability {fit.model_reliability:.3f} -- a gap between "
          "these two signals misspecification")
    print(f"  ->  {'PASS' if gate['passed'] else 'HALT'}")
    for r in gate["reasons"]:
        print(f"  - {r}")
    results["reliability_gate"] = {**gate, "model_reliability": fit.model_reliability}

    if not gate["passed"]:
        (out_dir / f"results_{cfg.hash()}.json").write_text(
            json.dumps({**results, "outcome": "halted-reliability"}, indent=2, default=str)
        )
        _echo("HALTED on reliability. theta is not reported for this model.")
        return HALT

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

    # ---- operating window ----------------------------------------------
    _echo("Operating-window diagnostic (A1.7) -- evaluated BEFORE Pass C")
    window = operating_window(comparisons, fit)
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
