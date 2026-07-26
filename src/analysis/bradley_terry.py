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

AN ORDER TERM IS INCLUDED, per template (Amendment 2, A2.1):

    P(item beats anchor) = sigmoid(theta_i - alpha_a + s * beta_t)
    s = +1 when the item is in slot 1, -1 when in slot 2

This file previously argued the opposite -- that a position term "would be
estimated from the very comparisons it is meant to purge." That was backwards.
Because both orders are run for EVERY cell, per cell we see logit p0 = x + beta and
logit p1 = x - beta, so the symmetric contrast identifies x = theta - alpha and the
antisymmetric contrast identifies beta. They are orthogonal contrasts of the same
two observations; balance is what makes beta identifiable, not what makes it
unnecessary. Simulation: true beta 1.0, recovered 1.016.

Omitting it is not harmless. Fitting one Bernoulli to two cells with probabilities
sigmoid(x+beta) and sigmoid(x-beta) lands the MLE on their average, which by Jensen
sits closer to 0.5 than sigmoid(x). The induced bias is zero at x=0, saturates at
-ln cosh(beta), and its local slope dx_hat/dx is MINIMAL at x=0 -- so the latent
scale is compressed most at small gaps, the only regime the paradigm operates in.
Measured: slope on true theta 0.746 without the term, 0.907 with it; sigma_item
0.928 vs 1.118 against a true 1.086. Rankings survive, the metric does not.

beta is per template because it is strongly heterogeneous: +1.645, +1.360, +1.211,
+4.302, +0.551 across t0..t4 on Gemma-2-2b, and a single global beta leaves residual
compression (excess-on-gap slope +0.0240, 95% CI [+0.0140, +0.0338]). Its sign is
also model-specific -- Gemma favours slot 1, Qwen2.5-3B slot 2.

NO cell-level random effect is included. An earlier analysis reported template
non-independence (ICC 0.529, design effect 3.12), but that ICC treated templates as
raters over cells and so conflated genuine between-cell variation in p -- which
theta - alpha already captures -- with excess dependence. The correct over-dispersion
test against the binomial expectation gives 1.070, 95% CI [0.907, 1.258],
i.e. consistent with independence. See A2.3 W2.
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
    beta: pd.DataFrame            # template, beta_mean, beta_sd, hdi
    sigma_item: float
    model_reliability: float
    idata: az.InferenceData = field(repr=False, default=None)
    n_comparisons: int = 0
    n_dropped_invalid: int = 0

    def summary(self) -> str:
        sd = self.theta["theta_sd"]
        b = self.beta
        signed = "slot 1" if b["beta_mean"].mean() > 0 else "slot 2"
        return (
            f"Bradley-Terry (+ per-template order term): {len(self.theta)} items from "
            f"{self.n_comparisons} valid comparisons "
            f"({self.n_dropped_invalid} dropped below the mass floor)\n"
            f"  theta spread: sd across items = {self.theta['theta_mean'].std(ddof=1):.3f}, "
            f"range {self.theta['theta_mean'].min():.2f} to {self.theta['theta_mean'].max():.2f}\n"
            f"  precision: posterior SD of theta, median {sd.median():.3f} "
            f"(min {sd.min():.3f}, max {sd.max():.3f})\n"
            f"  sigma_item (between-item scale) = {self.sigma_item:.3f}\n"
            f"  separation ratio = {self.sigma_item / sd.median():.2f}\n"
            f"  model reliability = {self.model_reliability:.3f} "
            "(compare against the EMPIRICAL split-half figure)\n"
            f"  beta: mean {b['beta_mean'].mean():+.3f} (favours {signed}), "
            f"range {b['beta_mean'].min():+.3f} to {b['beta_mean'].max():+.3f} "
            f"across {len(b)} templates"
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
    tmpl = sorted(block["template"].unique())
    item_idx = block["item_id"].map({v: i for i, v in enumerate(items)}).to_numpy()
    anchor_idx = block["anchor_id"].map({v: i for i, v in enumerate(anchors)}).to_numpy()
    tmpl_idx = block["template"].map({v: i for i, v in enumerate(tmpl)}).to_numpy()
    wins = block["item_wins"].to_numpy().astype(int)
    # +1 when the pool item occupies slot 1, -1 when it occupies slot 2. This is the
    # antisymmetric contrast that identifies beta.
    slot = np.where(block["order"].to_numpy() == 0, 1.0, -1.0)

    coords = {"item": items, "anchor": anchors, "template": tmpl}
    with pm.Model(coords=coords) as model:
        # Partial pooling: the shrinkage strength is estimated, not assumed.
        sigma_item = pm.HalfNormal("sigma_item", sigma=2.0)
        theta = pm.Normal("theta", mu=0.0, sigma=sigma_item, dims="item")
        # Zero-sum on anchors fixes the location that theta and alpha otherwise
        # share.
        alpha = pm.ZeroSumNormal("alpha", sigma=2.0, dims="anchor")
        # Position bias, per template (A2.1). Orthogonal to theta and alpha, which
        # enter symmetrically across order.
        beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims="template")

        pm.Bernoulli(
            "wins",
            logit_p=theta[item_idx] - alpha[anchor_idx] + beta[tmpl_idx] * slot,
            observed=wins,
        )

        kwargs = dict(
            draws=cfg.analysis.draws,
            tune=cfg.analysis.tune,
            chains=cfg.analysis.chains,
            random_seed=cfg.seed,
            target_accept=0.9,
            progressbar=progressbar,
        )
        if cfg.analysis.sampler_cores is not None:
            kwargs["cores"] = cfg.analysis.sampler_cores
        idata = pm.sample(**kwargs)

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

    bt = post["beta"].stack(sample=("chain", "draw"))
    beta_frame = pd.DataFrame({
        "template": tmpl,
        "beta_mean": bt.mean("sample").to_numpy(),
        "beta_sd": bt.std("sample").to_numpy(),
        "beta_hdi_low": az.hdi(idata, var_names=["beta"], hdi_prob=cfg.analysis.hdi_prob)
            ["beta"].sel(hdi="lower").to_numpy(),
        "beta_hdi_high": az.hdi(idata, var_names=["beta"], hdi_prob=cfg.analysis.hdi_prob)
            ["beta"].sel(hdi="higher").to_numpy(),
    })

    sigma = float(post["sigma_item"].mean())
    return BTFit(
        theta=theta_frame,
        anchors=anchor_frame,
        beta=beta_frame,
        sigma_item=sigma,
        # sigma^2 / (sigma^2 + E[posterior var]). Reported alongside the EMPIRICAL
        # split-half figure; a gap between them signals misspecification (A2.2).
        model_reliability=float(
            sigma ** 2 / (sigma ** 2 + np.mean(theta_frame["theta_sd"].to_numpy() ** 2))
        ),
        idata=idata,
        n_comparisons=len(block),
        n_dropped_invalid=n_dropped,
    )


def convergence(cfg: RunConfig, idata: az.InferenceData) -> pd.DataFrame:
    """R-hat and ESS for the BT parameters.

    Divergences and posterior SD alone are not enough: a fit can report zero
    divergences and a tidy posterior while chains disagree. Reported for every fit,
    and failures are reported rather than silently re-tuned.
    """
    summ = az.summary(idata, var_names=["theta", "alpha", "sigma_item"])
    summ["rhat_ok"] = summ["r_hat"] <= cfg.analysis.rhat_max
    summ["ess_ok"] = summ["ess_bulk"] >= cfg.analysis.ess_min
    return summ


def convergence_summary(cfg: RunConfig, idata: az.InferenceData) -> str:
    conv = convergence(cfg, idata)
    bad = conv[~(conv["rhat_ok"] & conv["ess_ok"])]
    n_div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    line = (
        f"convergence: max R-hat {conv['r_hat'].max():.4f} (limit {cfg.analysis.rhat_max}), "
        f"min ESS {conv['ess_bulk'].min():.0f} (limit {cfg.analysis.ess_min}), "
        f"divergences {n_div}, {len(bad)} parameter(s) failing"
    )
    if len(bad):
        worst = bad.sort_values("r_hat", ascending=False).head(5)
        line += "\n" + worst[["r_hat", "ess_bulk"]].to_string()
    return line


def theta_item_scores(
    cfg: RunConfig, comparisons: pd.DataFrame, *, arm: str = "digits",
) -> pd.DataFrame:
    """Per-item selection and analysis scores on the THETA scale (audit item T2.2).

    Pass B was fed polarity-collapsed absolute ratings, an instrument Amendment 1
    retired. This replaces them with theta fitted independently on the two disjoint
    template sets of section 4.4, which preserves exactly the property that split
    exists for: the score used to SELECT difficulty is measured on different templates
    from the score used to ANALYSE it, so selecting on noise cannot contaminate the
    regressor.
    """
    from src.stimuli.build import load_items

    templates = sorted(comparisons["template"].unique())
    sel = [templates[i] for i in cfg.pass_a.selection_templates if i < len(templates)]
    ana = [templates[i] for i in cfg.pass_a.analysis_templates if i < len(templates)]
    if set(sel) & set(ana):
        raise ValueError("selection and analysis template sets overlap")

    fit_sel = fit_bradley_terry(cfg, comparisons, arm=arm, templates=sel)
    fit_ana = fit_bradley_terry(cfg, comparisons, arm=arm, templates=ana)

    domain = {i.id: i.domain for i in load_items(cfg)}
    out = (
        fit_sel.theta[["item_id", "theta_mean"]]
        .rename(columns={"theta_mean": "score_selection"})
        .merge(
            fit_ana.theta[["item_id", "theta_mean"]]
            .rename(columns={"theta_mean": "score_analysis"}),
            on="item_id",
        )
    )
    out["domain"] = out["item_id"].map(domain)
    missing = out["domain"].isna().sum()
    if missing:
        raise ValueError(f"{missing} scored item(s) are not in the item pool")
    return out[["item_id", "domain", "score_selection", "score_analysis"]]


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


def excess_consistency_slope(
    comparisons: pd.DataFrame, fit: BTFit, *, arm: str = "digits", n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Preregistered fit-quality diagnostic (A2.1).

    Under a correctly specified order model, observed order-reversal consistency
    should match what content-plus-position predicts, with no residual trend in gap.
    A slope credibly different from zero means remaining compression.

    THE NULL IS NOT ZERO. Because the prediction plugs in posterior-mean theta and
    alpha and consistency is nonlinear in the gap, this statistic is biased away from
    zero even under correct specification. On Gemma's design the posterior-predictive
    null is +0.0044 (sd 0.0025), not 0. Compare against `excess_slope_ppc_null`,
    never against zero -- an earlier version of this docstring said "should be flat",
    which would have reported a correctly specified model as broken.
    """
    block = comparisons[(comparisons["arm"] == arm) & comparisons["readout_valid"]]
    wide = block.pivot_table(
        index=["item_id", "anchor_id", "template"], columns="order",
        values="item_wins", aggfunc="first",
    ).dropna().reset_index()
    if not {0, 1} <= set(wide.columns) or len(wide) < 20:
        return {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_cells": len(wide), "flat": None}

    th = dict(zip(fit.theta["item_id"], fit.theta["theta_mean"]))
    al = dict(zip(fit.anchors["anchor_id"], fit.anchors["alpha_mean"]))
    bt = dict(zip(fit.beta["template"], fit.beta["beta_mean"]))

    x = np.array([abs(th[i] - al[a]) for i, a in zip(wide["item_id"], wide["anchor_id"])])
    b = np.array([bt[t] for t in wide["template"]])
    p0, p1 = _sigmoid(x + b), _sigmoid(x - b)
    pred = p0 * p1 + (1 - p0) * (1 - p1)
    excess = (wide[0] == wide[1]).to_numpy().astype(float) - pred

    design = np.column_stack([np.ones(len(x)), x])
    slope = float(np.linalg.lstsq(design, excess, rcond=None)[0][1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(x), len(x))
        boots.append(np.linalg.lstsq(design[k], excess[k], rcond=None)[0][1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"slope": slope, "ci_low": float(lo), "ci_high": float(hi),
            "n_cells": int(len(x)), "mean_excess": float(excess.mean()),
            "flat": bool(lo < 0 < hi)}


def predicted_split_half(model_reliability: float, frac_a: float, frac_b: float) -> float:
    """What split-half correlation a full-data reliability implies.

    A2.2 compares model-internal reliability against an EMPIRICAL split-half figure,
    but those are not the same length. Model reliability comes from a fit using all
    templates; the split-half correlation is between two SHORTER fits. Each half has
    less data, hence a noisier theta, so its correlation is lower even under perfect
    specification. Comparing them directly guarantees an apparent gap and would report
    every model as misspecified.

    With r_full = sigma^2/(sigma^2+v) and k = 1/r_full - 1, a half holding fraction f
    of the data has v/f, so

        r_split = 1 / sqrt((1 + k/frac_a) * (1 + k/frac_b))

    Compare the observed split-half against THIS, not against r_full.
    """
    if not 0 < model_reliability < 1:
        return float("nan")
    k = 1.0 / model_reliability - 1.0
    return float(1.0 / np.sqrt((1.0 + k / frac_a) * (1.0 + k / frac_b)))


def excess_slope_ppc_null(
    cfg: RunConfig, comparisons: pd.DataFrame, fit: BTFit, *, n_rep: int = 24,
    arm: str = "digits", seed: int = 100,
) -> dict:
    """Posterior-predictive null for the excess-consistency slope.

    Regenerates outcomes from the fitted model on the SAME design -- same cells,
    templates, and mass-floor missingness -- refits, and recomputes the statistic.
    Whatever bias the plug-in prediction induces is then present in the null too, so
    the comparison isolates genuine misspecification.

    WHY THE NULL MUST BE SIMULATED RATHER THAN DERIVED. The observed statistic and the
    null both route through the same path -- posterior means plugged into a nonlinear
    consistency function -- so the Jensen bias that inflates the statistic is
    COMMON-MODE and cancels in the comparison. That is the property that makes this
    null valid, and it is why the plug-in bias must NOT be "fixed" in
    excess_consistency_slope: correcting the observed statistic while comparing it
    against a null that still carries the bias would break the cancellation and
    manufacture a discrepancy. The bias is deliberate here. If it is ever removed,
    every stored null must be regenerated in the same change.

    n_rep defaults to 24: at 8 replicates the null sd carries ~25% relative
    uncertainty and "0 of 8" is only a one-sided bound of ~0.11.
    """
    block = comparisons[(comparisons["arm"] == arm) & comparisons["readout_valid"]].copy()
    th = dict(zip(fit.theta["item_id"], fit.theta["theta_mean"]))
    al = dict(zip(fit.anchors["anchor_id"], fit.anchors["alpha_mean"]))
    bt = dict(zip(fit.beta["template"], fit.beta["beta_mean"]))
    slot = np.where(block["order"].to_numpy() == 0, 1.0, -1.0)
    p = _sigmoid(
        np.array([th[i] for i in block["item_id"]])
        - np.array([al[a] for a in block["anchor_id"]])
        + np.array([bt[t] for t in block["template"]]) * slot
    )

    slopes = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed + r)
        rep = block.copy()
        rep["item_wins"] = rng.random(len(p)) < p
        f2 = fit_bradley_terry(cfg, rep, arm=arm)
        slopes.append(excess_consistency_slope(rep, f2, arm=arm)["slope"])
    s = np.array(slopes, dtype=float)
    return {"null_mean": float(s.mean()), "null_sd": float(s.std(ddof=1)),
            "null_min": float(s.min()), "null_max": float(s.max()),
            "n_replicates": int(n_rep), "slopes": s.tolist()}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


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
