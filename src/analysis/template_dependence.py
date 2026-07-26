"""Are template responses independent? preregistration.md A1.1 open question.

    python -m src.analysis.template_dependence --config configs/stage0_gemma-2-2b.yaml

WHY THIS MATTERS. The Bradley-Terry model treats each of an item's ~100 comparisons
as an independent Bernoulli observation. It has no template effect and no
over-dispersion term. But the same (item, anchor, order) cell is measured under five
templates, and if those five responses are correlated, the model is counting
correlated evidence as independent -- inflating the effective sample size and
UNDERSTATING the posterior SD of theta.

That is not a cosmetic concern: A1.1 designates the posterior SD of theta as the
precision measure, and the operating-window and Pass-B-selection decisions are made
against it.

There is already a hint in the real data. Gemma's split-half test-retest came in at
0.736 while its reported posterior SD implies something higher. This module turns
that hint into a number.

HOW. Cells (item x anchor x order) are targets, templates are raters, so this is the
same two-way structure the Pass A reliability code already handles -- `icc_two_way`
is reused rather than reimplemented. From the ICC:

    design effect      deff = 1 + (m - 1) * ICC
    effective N        n_nominal / deff
    SD understatement  sqrt(deff)

ICC near 0 means templates are independent draws and the model is fine. ICC near 1
means five templates carry barely more information than one, and posterior SD is
understated by up to sqrt(5) ~ 2.2x.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.reliability import icc_two_way
from src.config import RunConfig, load_config


def template_matrix(comparisons: pd.DataFrame, arm: str = "digits") -> pd.DataFrame:
    """Cells (item, anchor, order) x templates, of binary outcomes."""
    block = comparisons[(comparisons["arm"] == arm) & comparisons["readout_valid"]]
    wide = block.pivot_table(
        index=["item_id", "anchor_id", "order"],
        columns="template",
        values="item_wins",
        aggfunc="first",
    )
    # Only fully observed cells: a partially observed cell would bias the ICC
    # toward whichever templates happened to survive the mass floor.
    return wide.dropna().astype(float)


def analyse(comparisons: pd.DataFrame, *, arm: str = "digits") -> dict:
    mat = template_matrix(comparisons, arm)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError(f"need >=2 cells and >=2 templates, got {mat.shape}")

    m = mat.shape[1]
    icc = icc_two_way(mat.to_numpy())
    # Consistency form: an additive per-template offset (one template being
    # generally more yes-prone) does not make responses redundant, whereas
    # cell-to-cell co-movement does.
    rho = float(np.clip(icc["icc_c1"], 0.0, 0.999))
    deff = 1.0 + (m - 1) * rho

    # Raw pairwise agreement, against what independence would predict.
    p = float(mat.to_numpy().mean())
    agree_obs = []
    cols = list(mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            agree_obs.append(float((mat[cols[i]] == mat[cols[j]]).mean()))
    agree_indep = p ** 2 + (1 - p) ** 2

    return {
        "n_cells": int(mat.shape[0]),
        "n_templates": m,
        "icc_c1": rho,
        "icc_ck": float(icc["icc_ck"]),
        "design_effect": deff,
        "effective_n_fraction": 1.0 / deff,
        "sd_understatement_factor": float(np.sqrt(deff)),
        "mean_pairwise_agreement": float(np.mean(agree_obs)),
        "agreement_under_independence": agree_indep,
        "win_rate": p,
    }


def predicted_test_retest(sigma_item: float, posterior_sd: float) -> float:
    """Split-half correlation implied by a fit's own stated precision.

        r = var_true / (var_true + var_err)

    Comparing this against the OBSERVED split-half correlation is an independent
    check on whether posterior SD is honest.
    """
    vt, ve = sigma_item ** 2, posterior_sd ** 2
    return float(vt / (vt + ve)) if vt + ve > 0 else float("nan")


def report(res: dict, *, sigma_item: float | None = None,
           posterior_sd: float | None = None,
           observed_test_retest: float | None = None) -> str:
    lines = [
        f"cells = {res['n_cells']}, templates = {res['n_templates']}, "
        f"pool-item win rate = {res['win_rate']:.3f}",
        f"ICC(C,1) across templates = {res['icc_c1']:.3f}",
        f"mean pairwise template agreement = {res['mean_pairwise_agreement']:.3f} "
        f"(independence would give {res['agreement_under_independence']:.3f})",
        f"design effect = {res['design_effect']:.2f}  ->  effective N is "
        f"{res['effective_n_fraction']:.0%} of nominal",
        f"posterior SD understated by ~{res['sd_understatement_factor']:.2f}x "
        "if templates are modelled as independent",
    ]
    if None not in (sigma_item, posterior_sd, observed_test_retest):
        pred = predicted_test_retest(sigma_item, posterior_sd)
        lines += [
            "",
            f"independent check: split-half r predicted from posterior SD = {pred:.3f}, "
            f"observed = {observed_test_retest:.3f}",
        ]
        if observed_test_retest < pred - 0.05:
            lines.append(
                "  observed BELOW predicted -- consistent with an understated "
                "posterior SD, i.e. the same direction as the ICC above."
            )
        else:
            lines.append("  observed is consistent with predicted.")

    rho = res["icc_c1"]
    lines.append("")
    if rho < 0.05:
        lines.append(
            "VERDICT: templates behave as near-independent draws. The current "
            "Bradley-Terry specification is adequate and needs no template term."
        )
    elif rho < 0.20:
        lines.append(
            "VERDICT: mild template dependence. Posterior SD is somewhat optimistic; "
            "report the design effect alongside it rather than respecifying."
        )
    else:
        lines.append(
            "VERDICT: substantial template dependence. Posterior SD cannot be "
            "reported as precision without a template random effect or an "
            "over-dispersion term in the Bradley-Terry model. That is a change to "
            "A1.1 and requires an amendment, not a code edit."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args(argv)

    overrides = {"artifacts_dir": args.artifacts} if args.artifacts else None
    cfg: RunConfig = load_config(args.config, overrides)
    h = cfg.hash("pass_a")
    base = cfg.artifacts_dir / "pass_a_pairwise" / cfg.model.name / h

    comp_path = base / f"comparisons_{h}.parquet"
    if not comp_path.exists():
        print(f"no comparisons at {comp_path}")
        return 1

    from src.provenance import read_parquet

    comparisons = read_parquet(comp_path)
    res = analyse(comparisons)

    print(f"\n{'=' * 72}\nTemplate dependence  |  {cfg.model.name}\n{'=' * 72}")

    sigma_item = posterior_sd = None
    theta_path = base / f"theta_{h}.parquet"
    if theta_path.exists():
        theta = pd.read_parquet(theta_path)
        posterior_sd = float(theta["theta_sd"].median())

    results_json = base / f"results_{cfg.hash()}.json"
    observed = None
    if results_json.exists():
        import json

        r = json.loads(results_json.read_text())
        sigma_item = (r.get("theta") or {}).get("sigma_item")
        observed = (r.get("test_retest") or {}).get("pearson")

    print(report(res, sigma_item=sigma_item, posterior_sd=posterior_sd,
                 observed_test_retest=observed))
    pd.DataFrame([res]).to_csv(base / f"template_dependence_{h}.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
