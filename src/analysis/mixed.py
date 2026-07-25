"""Bayesian mixed-effects model, planned contrasts, and the gate decision.

Parameterization is CELL MEANS, not treatment coding:

    mu = b_cond[c] + b_slope[c] * diff_z + b_order * order
         + u_pair[p] + u_template[t]        (+ u_item[i1] + u_item[i2] in the
                                             robustness model)

Every condition gets its own intercept and its own diff_z slope, so each planned
contrast is a plain difference of posterior draws and depends on no
contrast-coding scheme. The model is written directly in PyMC rather than through
a formula interface precisely so those names and that parameterization are exact.

PRIMARY TEST: b_slope[chose] - b_slope[yoked], predicted NEGATIVE. The agency
effect on spread should be larger for difficult (small-|diff|) pairs. A main
effect of agency is not the primary test -- it is fully consistent with context
sensitivity, which is how the prior version of this claim was eliminated.

diff_z is z-scored within model, so the interaction coefficient is in spread
points per SD of |diff| and is directly comparable to the SESOI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.config import RunConfig

# name -> (condition_a, condition_b, predicted direction on the SLOPE contrast)
PLANNED_CONTRASTS: dict[str, tuple[str, str, int]] = {
    # primary
    "chose - yoked": ("chose", "yoked", -1),
    # authorship + self-relevance, information held constant
    "chose - 3p-yoked": ("chose", "3p-yoked", -1),
    # information effect, designation held constant
    "3p-yoked - yoked": ("3p-yoked", "yoked", 0),
    # selection artifact
    "yoked - random": ("yoked", "random", 0),
    # pure context effect: the rebuttal's mechanism
    "3p-random - random": ("3p-random", "random", 0),
}

PRIMARY = "chose - yoked"

# The two routes to the same selection-artifact quantity. They should agree; the
# discrepancy is reported, and a large one means Stage 0 is uninterpretable rather
# than a pass or a fail.
ARTIFACT_ROUTES = ("yoked - random", "3p-yoked - 3p-random")


@dataclass
class Design:
    frame: pd.DataFrame
    conditions: list[str]
    pairs: list[str]
    templates: list[str]
    items: list[str]
    diff_mean: float
    diff_sd: float


def prepare_design(cfg: RunConfig, pass_c: pd.DataFrame) -> Design:
    """Index vectors and the z-scored |diff| regressor."""
    frame = pass_c.copy()

    diff = frame["diff_analysis"].to_numpy(dtype=float)
    mu, sd = float(diff.mean()), float(diff.std(ddof=1))
    if sd <= 0:
        raise ValueError("|diff| has zero variance; the difficulty regressor is undefined")
    frame["diff_z"] = (diff - mu) / sd

    conditions = [c for c in cfg.pass_c.conditions if c in set(frame["condition"])]
    pairs = sorted(frame["pair_id"].unique())
    templates = sorted(frame["template"].unique())
    items = sorted(set(frame["item1_id"]) | set(frame["item2_id"]))

    frame["cond_idx"] = frame["condition"].map({c: i for i, c in enumerate(conditions)})
    frame["pair_idx"] = frame["pair_id"].map({p: i for i, p in enumerate(pairs)})
    frame["template_idx"] = frame["template"].map({t: i for i, t in enumerate(templates)})
    item_index = {it: i for i, it in enumerate(items)}
    frame["item1_idx"] = frame["item1_id"].map(item_index)
    frame["item2_idx"] = frame["item2_id"].map(item_index)
    frame["order_c"] = frame["option_order"].astype(float)

    return Design(
        frame=frame,
        conditions=conditions,
        pairs=pairs,
        templates=templates,
        items=items,
        diff_mean=mu,
        diff_sd=sd,
    )


def _build_model(
    cfg: RunConfig, design: Design, *, with_item: bool
) -> pm.Model:
    f = design.frame
    coords = {
        "condition": design.conditions,
        "pair": design.pairs,
        "template": design.templates,
    }
    if with_item:
        coords["item"] = design.items

    with pm.Model(coords=coords) as model:
        # Weakly informative on the spread scale (bounded +/-16, expected ~+/-2).
        b_cond = pm.Normal("b_cond", mu=0.0, sigma=1.0, dims="condition")
        b_slope = pm.Normal("b_slope", mu=0.0, sigma=1.0, dims="condition")
        b_order = pm.Normal("b_order", mu=0.0, sigma=1.0)

        # ZeroSumNormal, not Normal. With cell-means coding there is no global
        # intercept, so unconstrained random intercepts leave the overall level
        # unidentified: adding a constant to every b_cond and subtracting it from
        # every u_pair leaves the likelihood unchanged. The contrasts are still
        # identified -- they are differences -- but the ridge wrecks sampling
        # efficiency (b_cond had the worst ESS of any parameter) and makes b_cond
        # uninterpretable as a cell mean. Constraining the random effects to sum to
        # zero removes the ridge and costs nothing the design needs.
        sd_pair = pm.HalfNormal("sd_pair", sigma=1.0)
        u_pair = pm.ZeroSumNormal("u_pair", sigma=sd_pair, dims="pair")
        sd_template = pm.HalfNormal("sd_template", sigma=1.0)
        u_template = pm.ZeroSumNormal("u_template", sigma=sd_template, dims="template")

        cond_idx = f["cond_idx"].to_numpy()
        mu = (
            b_cond[cond_idx]
            + b_slope[cond_idx] * f["diff_z"].to_numpy()
            + b_order * f["order_c"].to_numpy()
            + u_pair[f["pair_idx"].to_numpy()]
            + u_template[f["template_idx"].to_numpy()]
        )

        if with_item:
            # Multi-membership: the DV is per-pair and each pair contains two
            # items, so the item effect enters as the sum of its two members'
            # intercepts. This is not expressible as a formula grouping term,
            # which is why the robustness model is written out by hand.
            sd_item = pm.HalfNormal("sd_item", sigma=1.0)
            u_item = pm.ZeroSumNormal("u_item", sigma=sd_item, dims="item")
            mu = mu + u_item[f["item1_idx"].to_numpy()] + u_item[f["item2_idx"].to_numpy()]

        sigma = pm.HalfNormal("sigma", sigma=2.0)
        pm.Normal("spread", mu=mu, sigma=sigma, observed=f["spread"].to_numpy())

    return model


def fit(
    cfg: RunConfig,
    design: Design,
    *,
    with_item: bool = False,
    progressbar: bool = False,
) -> az.InferenceData:
    model = _build_model(cfg, design, with_item=with_item)
    with model:
        idata = pm.sample(
            draws=cfg.analysis.draws,
            tune=cfg.analysis.tune,
            chains=cfg.analysis.chains,
            random_seed=cfg.seed,
            target_accept=0.9,
            progressbar=progressbar,
        )
    return idata


# ---------------------------------------------------------------------------
# contrasts


@dataclass
class ContrastSummary:
    name: str
    term: Literal["slope", "intercept"]
    median: float
    mean: float
    hdi_low: float
    hdi_high: float
    p_negative: float
    p_positive: float
    excludes_zero: bool
    inside_rope: bool
    exceeds_sesoi: bool
    decision: str

    def as_row(self) -> dict:
        return self.__dict__.copy()


def _draws(idata: az.InferenceData, var: str, condition: str) -> np.ndarray:
    return (
        idata.posterior[var]
        .sel(condition=condition)
        .stack(sample=("chain", "draw"))
        .to_numpy()
    )


def summarize_contrast(
    name: str,
    draws: np.ndarray,
    *,
    term: Literal["slope", "intercept"],
    sesoi: float,
    hdi_prob: float,
    direction: int,
) -> ContrastSummary:
    """Posterior summary plus the preregistered three-way gate decision.

    pass          HDI excludes 0 in the predicted direction AND |median| > SESOI
    equivalent    HDI lies entirely inside the ROPE [-SESOI, +SESOI]
    inconclusive  anything else -- resolved by scaling items, not by reanalysis
    """
    low, high = az.hdi(draws, hdi_prob=hdi_prob)
    median = float(np.median(draws))
    excludes_zero = bool(low > 0 or high < 0)
    inside_rope = bool(low >= -sesoi and high <= sesoi)
    exceeds = bool(abs(median) > sesoi)

    if direction < 0:
        directional = bool(high < 0)
    elif direction > 0:
        directional = bool(low > 0)
    else:
        directional = excludes_zero

    if directional and exceeds:
        decision = "pass"
    elif inside_rope:
        decision = "equivalent-to-null"
    else:
        decision = "inconclusive"

    return ContrastSummary(
        name=name,
        term=term,
        median=median,
        mean=float(draws.mean()),
        hdi_low=float(low),
        hdi_high=float(high),
        p_negative=float((draws < 0).mean()),
        p_positive=float((draws > 0).mean()),
        excludes_zero=excludes_zero,
        inside_rope=inside_rope,
        exceeds_sesoi=exceeds,
        decision=decision,
    )


def contrast_table(
    cfg: RunConfig, idata: az.InferenceData, sesoi: float
) -> pd.DataFrame:
    """All planned contrasts, on both the slope (interaction) and the intercept."""
    available = set(idata.posterior["condition"].to_numpy().tolist())
    rows: list[dict] = []

    for name, (a, b, direction) in PLANNED_CONTRASTS.items():
        if {a, b} - available:
            continue
        for term, var, dirn in (
            ("slope", "b_slope", direction),
            # The intercept contrast is the main effect at mean |diff|. Reported,
            # never primary: on its own it is consistent with context sensitivity.
            ("intercept", "b_cond", 0),
        ):
            draws = _draws(idata, var, a) - _draws(idata, var, b)
            rows.append(
                summarize_contrast(
                    name,
                    draws,
                    term=term,
                    sesoi=sesoi,
                    hdi_prob=cfg.analysis.hdi_prob,
                    direction=dirn,
                ).as_row()
            )

    # Artifact cross-check: the second route to the same quantity.
    if {"3p-yoked", "3p-random"} <= available:
        draws = _draws(idata, "b_cond", "3p-yoked") - _draws(idata, "b_cond", "3p-random")
        rows.append(
            summarize_contrast(
                "3p-yoked - 3p-random",
                draws,
                term="intercept",
                sesoi=sesoi,
                hdi_prob=cfg.analysis.hdi_prob,
                direction=0,
            ).as_row()
        )

    return pd.DataFrame(rows)


def artifact_agreement(contrasts: pd.DataFrame) -> dict:
    """Discrepancy between the two selection-artifact estimates."""
    inter = contrasts[contrasts["term"] == "intercept"].set_index("name")
    if not set(ARTIFACT_ROUTES) <= set(inter.index):
        return {}
    a, b = (inter.loc[r, "median"] for r in ARTIFACT_ROUTES)
    return {
        "route_a": ARTIFACT_ROUTES[0],
        "route_b": ARTIFACT_ROUTES[1],
        "estimate_a": float(a),
        "estimate_b": float(b),
        "discrepancy": float(a - b),
    }


def convergence(cfg: RunConfig, idata: az.InferenceData) -> pd.DataFrame:
    """R-hat and ESS for every reported parameter. Failures are reported, not retuned."""
    summ = az.summary(
        idata, var_names=["b_cond", "b_slope", "b_order", "sd_pair", "sd_template", "sigma"]
    )
    summ["rhat_ok"] = summ["r_hat"] <= cfg.analysis.rhat_max
    summ["ess_ok"] = summ["ess_bulk"] >= cfg.analysis.ess_min
    return summ


def equivalence_ratio(
    numerator: np.ndarray, denominator: np.ndarray, f: float, hdi_prob: float
) -> dict:
    """Stage 2 equivalence rule, preregistered here and applied later.

    Equivalence is expressed as the posterior of the RATIO
    (truth-vector effect / consistency-vector effect), with the claim being that
    the ratio's HDI falls within +/-f. Taking the ratio's posterior propagates
    uncertainty in BOTH effects, rather than dividing a point estimate by a point
    estimate and understating the denominator's uncertainty.
    """
    ratio = numerator / denominator
    low, high = az.hdi(ratio, hdi_prob=hdi_prob)
    return {
        "f": f,
        "ratio_median": float(np.median(ratio)),
        "hdi_low": float(low),
        "hdi_high": float(high),
        "equivalent": bool(low >= -f and high <= f),
    }
