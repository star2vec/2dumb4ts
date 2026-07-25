"""Hierarchical Bradley-Terry appeal scores. preregistration.md A1.1.

    P(item i beats anchor a) = sigmoid(theta_i - alpha_a)

Partial pooling across items: theta_i ~ Normal(0, sigma_item), with sigma_item
estimated. Items are shrunk toward the pool mean by an amount the data chooses,
which matters because each item carries only 20 binary outcomes -- an unpooled
per-item MLE is badly behaved for any item that beats or loses to every anchor
(the estimate runs to +/-inf).

Anchor abilities alpha_a are free parameters with a zero-sum constraint. Without
it, theta and alpha are jointly unidentified: adding a constant to every theta and
every alpha leaves every comparison probability unchanged. This is the same
identification problem that showed up in the spread model, and it gets the same
fix.

PRECISION IS REPORTED AS THE POSTERIOR SD OF theta, not as an ICC. Reliability of
a single comparison is not the quantity of interest; the quantity of interest is
how well theta is pinned down after 20 of them.

No order term appears in the likelihood. Order is averaged structurally by always
running both -- a position-bias parameter would be estimated from the very
comparisons it is meant to purge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.config import RunConfig


@dataclass
class BTFit:
    theta: pd.DataFrame           # item_id, theta_mean, theta_sd, hdi_low, hdi_high
    anchors: pd.DataFrame         # anchor_id, alpha_mean, alpha_sd
    sigma_item: float
    idata: az.InferenceData = field(repr=False, default=None)
    n_comparisons: int = 0
    n_dropped_invalid: int = 0

    def summary(self) -> str:
        sd = self.theta["theta_sd"]
        return (
            f"Bradley-Terry: {len(self.theta)} items from {self.n_comparisons} valid "
            f"comparisons ({self.n_dropped_invalid} dropped below the mass floor)\n"
            f"  theta spread: sd across items = {self.theta['theta_mean'].std(ddof=1):.3f}, "
            f"range {self.theta['theta_mean'].min():.2f} to {self.theta['theta_mean'].max():.2f}\n"
            f"  precision: posterior SD of theta, median {sd.median():.3f} "
            f"(min {sd.min():.3f}, max {sd.max():.3f})\n"
            f"  sigma_item (between-item scale) = {self.sigma_item:.3f}\n"
            f"  separation ratio sigma_item / median posterior SD = "
            f"{self.sigma_item / sd.median():.2f}"
        )


def fit_bradley_terry(
    cfg: RunConfig,
    comparisons: pd.DataFrame,
    *,
    arm: str = "digits",
    templates: list[str] | None = None,
    progressbar: bool = False,
) -> BTFit:
    """Fit theta from item-vs-anchor comparisons.

    Args:
        comparisons: output of pairwise.collect_comparisons.
        arm: which elicitation arm to score. Digits is primary (A1.3).
        templates: restrict to these templates, for the disjoint-split test-retest.
    """
    block = comparisons[comparisons["arm"] == arm]
    if templates is not None:
        block = block[block["template"].isin(templates)]

    n_before = len(block)
    # A1.6: invalid readouts are excluded from scoring, and the count is reported.
    block = block[block["readout_valid"]]
    n_dropped = n_before - len(block)
    if block.empty:
        raise ValueError(
            f"no valid comparisons for arm={arm!r}"
            + (f", templates={templates}" if templates else "")
        )

    items = sorted(block["item_id"].unique())
    anchors = sorted(block["anchor_id"].unique())
    item_idx = block["item_id"].map({v: i for i, v in enumerate(items)}).to_numpy()
    anchor_idx = block["anchor_id"].map({v: i for i, v in enumerate(anchors)}).to_numpy()
    wins = block["item_wins"].to_numpy().astype(int)

    coords = {"item": items, "anchor": anchors}
    with pm.Model(coords=coords) as model:
        # Partial pooling: the shrinkage strength is estimated, not assumed.
        sigma_item = pm.HalfNormal("sigma_item", sigma=2.0)
        theta = pm.Normal("theta", mu=0.0, sigma=sigma_item, dims="item")
        # Zero-sum on anchors fixes the location that theta and alpha otherwise
        # share.
        alpha = pm.ZeroSumNormal("alpha", sigma=2.0, dims="anchor")

        pm.Bernoulli(
            "wins",
            logit_p=theta[item_idx] - alpha[anchor_idx],
            observed=wins,
        )

        idata = pm.sample(
            draws=cfg.analysis.draws,
            tune=cfg.analysis.tune,
            chains=cfg.analysis.chains,
            random_seed=cfg.seed,
            target_accept=0.9,
            progressbar=progressbar,
        )

    post = idata.posterior
    th = post["theta"].stack(sample=("chain", "draw"))
    hdi = az.hdi(idata, var_names=["theta"], hdi_prob=cfg.analysis.hdi_prob)["theta"]

    theta_frame = pd.DataFrame({
        "item_id": items,
        "theta_mean": th.mean("sample").to_numpy(),
        "theta_sd": th.std("sample").to_numpy(),
        "hdi_low": hdi.sel(hdi="lower").to_numpy(),
        "hdi_high": hdi.sel(hdi="higher").to_numpy(),
    })

    al = post["alpha"].stack(sample=("chain", "draw"))
    anchor_frame = pd.DataFrame({
        "anchor_id": anchors,
        "alpha_mean": al.mean("sample").to_numpy(),
        "alpha_sd": al.std("sample").to_numpy(),
    })

    return BTFit(
        theta=theta_frame,
        anchors=anchor_frame,
        sigma_item=float(post["sigma_item"].mean()),
        idata=idata,
        n_comparisons=len(block),
        n_dropped_invalid=n_dropped,
    )


def test_retest(
    cfg: RunConfig,
    comparisons: pd.DataFrame,
    split_a: list[str],
    split_b: list[str],
    *,
    arm: str = "digits",
) -> dict:
    """Correlate theta across disjoint template sets (A1.1).

    This replaces the retired polarity check as the stability measure. Because
    each split is fitted independently, the correlation is between two genuinely
    separate estimates rather than two views of one fit.
    """
    if set(split_a) & set(split_b):
        raise ValueError("test-retest template splits must be disjoint")

    from scipy import stats

    fit_a = fit_bradley_terry(cfg, comparisons, arm=arm, templates=split_a)
    fit_b = fit_bradley_terry(cfg, comparisons, arm=arm, templates=split_b)

    merged = fit_a.theta.merge(fit_b.theta, on="item_id", suffixes=("_a", "_b"))
    pearson = float(np.corrcoef(merged["theta_mean_a"], merged["theta_mean_b"])[0, 1])
    spearman = float(stats.spearmanr(merged["theta_mean_a"], merged["theta_mean_b"]).statistic)

    return {
        "split_a": split_a,
        "split_b": split_b,
        "n_items": len(merged),
        "pearson": pearson,
        "spearman": spearman,
        "sigma_item_a": fit_a.sigma_item,
        "sigma_item_b": fit_b.sigma_item,
        "median_posterior_sd_a": float(fit_a.theta["theta_sd"].median()),
        "median_posterior_sd_b": float(fit_b.theta["theta_sd"].median()),
        "theta": merged,
    }


def anchor_ordering_check(fit: BTFit, anchors: list) -> pd.DataFrame:
    """Did the estimated anchor abilities recover the intended spread?

    `tier` is a design annotation never shown to the model. If the estimated
    ordering collapses -- all anchors at the same alpha -- the anchor set does not
    span the range and theta is being estimated against a degenerate reference,
    which would reproduce the compression the pairwise switch exists to escape.
    """
    tiers = {a.id: a.tier for a in anchors}
    order = {"high": 0, "upper-mid": 1, "mid": 2, "lower-mid": 3, "low": 4}
    out = fit.anchors.copy()
    out["tier"] = out["anchor_id"].map(tiers)
    out["tier_rank"] = out["tier"].map(order)
    out = out.sort_values("alpha_mean", ascending=False).reset_index(drop=True)
    out["alpha_rank"] = np.arange(len(out))
    return out
