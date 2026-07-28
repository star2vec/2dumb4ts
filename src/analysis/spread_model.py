"""The Pass C spread model. preregistration.md A2.9.1.

The DV is the designated-versus-other comparison, measured pre and post. Outcomes
are modelled DIRECTLY; no per-pair spread is ever formed.

    logit P(item1 beats item2)
        = u_pair + u_template + beta_t * s + post * d * (gamma_c + lambda_c * diff_z)

    s     +1 if item1 occupies slot 1, -1 otherwise      (position bias)
    d     +1 if item1 is the designated item, -1 otherwise
    post  0 at pre, 1 at post
    c     condition, t template

PRIMARY TEST: lambda_chose - lambda_yoked, predicted NEGATIVE.

TWO CORRECTIONS TO A2.9.1's WRITTEN FORM, both identification issues found on
implementation and both stated rather than silently applied:

1. A2.9.1 lists `delta_pair` AND `u_pair`. Those are the same quantity -- a per-pair
   baseline log-odds -- and are not separately identifiable. Implemented as a single
   partially pooled `u_pair`, which is the right choice anyway since each pair carries
   few observations.

2. A2.9.1 orients the outcome as "designated beats other". That makes the BASELINE
   condition-dependent, because which item is designated differs by condition within a
   pair: in the yoked family it is the model's own pick and so correlates with the
   pre-existing preference, while in the random family it does not. The per-pair
   baseline would then absorb part of the selection artifact. Orienting the outcome on
   a FIXED pair axis (item1 vs item2, canonical by sorted id) and carrying designation
   in the sign `d` removes that entirely: the baseline is a property of the pair, and
   the spread effect pushes toward whichever item was designated.

WHY THE TWO-STAGE ALTERNATIVE IS PROHIBITED, structurally and not by convention.
Estimating a per-pair spread and regressing it on gap would give a quantity whose
sampling variance depends on where the pair sits on the logit curve -- maximal at
p = 0.5, which is exactly where difficult pairs sit by construction. That manufactures
the predicted interaction out of noise. This module therefore never exposes a per-pair
spread: there is no function that returns one, and the interaction exists only as
`lambda_c` inside the likelihood.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path as _Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.config import RunConfig

#: (name, condition_a, condition_b, predicted sign on the lambda contrast)
PLANNED: dict[str, tuple[str, str, int]] = {
    # PRIMARY
    "chose - yoked": ("chose", "yoked", -1),
    # A2.9.3 2x2 edges
    "chose - self-recounted": ("chose", "self-recounted", 0),
    "structure-control - yoked": ("structure-control", "yoked", 0),
    "self-recounted - yoked": ("self-recounted", "yoked", 0),
    # reversibility: dissonance predicts an effect, self-perception does not (A2.8)
    "chose - chose-provisional": ("chose", "chose-provisional", -1),
    # rebuttal mechanism
    "chose - 3p-yoked": ("chose", "3p-yoked", -1),
    "3p-yoked - yoked": ("3p-yoked", "yoked", 0),
    "yoked - random": ("yoked", "random", 0),
    "3p-random - random": ("3p-random", "random", 0),
}

PRIMARY = "chose - yoked"

#: Label for the shared pre-manipulation baseline. Not a condition -- one physical
#: measurement per (pair, template, option order), emitted once.
PRE_SENTINEL = "pre"


#: Which form of the template random effect this module implements. Recorded in every
#: artifact so a fit can say which parameterization produced it, rather than leaving it to
#: be inferred from a filename. Artifacts written before A4.1 carry no field at all and are
#: read as "centered", which is what they are.
PARAMETERIZATION = "non-centered"


def source_digest() -> str:
    """Digest of this module, for the saved posterior's filename.

    The posterior was stored as `spread_posterior_{cfg.hash()}.nc` -- a CONFIG hash, which
    cannot see the model code. A4.1 non-centered `u_template` without changing any config
    field, so the notebook's `if posterior_path.exists()` would have loaded the CENTERED
    posterior and reported it as the new fit. Same class as the Pass B cache key.

    It also does the thing A4.1 requires: the centered and non-centered fits land under
    different names, so both are retained and neither overwrites the other.
    """
    return hashlib.sha256(_Path(__file__).read_bytes()).hexdigest()[:8]


def posterior_path(cfg, out_dir):
    return out_dir / f"spread_posterior_{cfg.hash()}-{source_digest()}.nc"


@dataclass
class SpreadDesign:
    frame: pd.DataFrame
    conditions: list[str]
    pairs: list[str]
    templates: list[str]
    diff_mean: float
    diff_sd: float


def prepare(cfg: RunConfig, trials: pd.DataFrame) -> SpreadDesign:
    """Build the long-format design: one row per observed comparison.

    `trials` must carry, per row: pair_id, template, option_order, condition,
    timepoint ('pre'|'post'), item1_id, item2_id, designated_item_id,
    slot1_item_id, diff_analysis, and the binary outcome item1_wins.
    """
    f = trials.copy()

    need = {"pair_id", "template", "condition", "timepoint", "item1_id", "item2_id",
            "designated_item_id", "slot1_item_id", "diff_analysis", "item1_wins"}
    missing = need - set(f.columns)
    if missing:
        raise ValueError(f"trials missing columns: {sorted(missing)}")

    diff = f["diff_analysis"].to_numpy(dtype=float)
    mu, sd = float(diff.mean()), float(diff.std(ddof=1))
    # Scale-RELATIVE, not `sd <= 0`. A constant column of 1e-6 has sd of order 1e-22, which
    # sails past an absolute test while making diff_z garbage of order 1e6. `power.py` was
    # corrected for this and its comment named this function as carrying the same hole;
    # the weaker guard was left on the LIVE model and the stricter one on the diagnostic.
    if sd <= 1e-9 * max(1.0, abs(mu)):
        raise ValueError(
            f"|diff| has no usable variance (sd {sd:.3g} against mean {mu:.3g}); "
            "the difficulty regressor is undefined."
        )
    f["diff_z"] = (diff - mu) / sd

    # Fixed pair axis. s and d are both +/-1 relative to item1, so the baseline is a
    # property of the pair rather than of the condition.
    f["s"] = np.where(f["slot1_item_id"] == f["item1_id"], 1.0, -1.0)
    f["d"] = np.where(f["designated_item_id"] == f["item1_id"], 1.0, -1.0)
    f["post"] = (f["timepoint"] == "post").astype(float)

    # Pre observations are ONE physical measurement per (pair, template, order),
    # shared by every condition. They are emitted once, labelled with the PRE_SENTINEL,
    # and carry post = 0 -- so they inform u_pair, u_template and beta_t but contribute
    # nothing to gamma or lambda. Emitting them once per condition instead would feed
    # the same observation to the likelihood eight times and over-weight the baseline.
    pre_rows = f["condition"] == PRE_SENTINEL
    if pre_rows.any() and not (f.loc[pre_rows, "timepoint"] == "pre").all():
        raise ValueError(f"rows labelled {PRE_SENTINEL!r} must all have timepoint 'pre'")

    conditions = [c for c in cfg.pass_c.conditions
                  if c in set(f.loc[~pre_rows, "condition"])]
    pairs = sorted(f["pair_id"].unique())
    templates = sorted(f["template"].unique())

    # NEVER fillna here. An unmapped condition silently became condition 0 -- i.e.
    # `chose` -- which mixed conditions together and attenuated every lambda toward
    # their average while leaving gamma looking correct. That is exactly the failure
    # the recovery test caught, and it was invisible because the fallback was silent.
    if not conditions:
        raise ValueError(
            "no configured Pass C condition is present in the trials. Every row is either "
            f"the {PRE_SENTINEL!r} baseline or an unrecognised condition, so there is "
            "nothing to estimate lambda from."
        )
    unknown = sorted(set(f.loc[~pre_rows, "condition"]) - set(conditions))
    if unknown:
        raise ValueError(
            f"conditions present in the data but not in cfg.pass_c.conditions: {unknown}. "
            "Refusing to guess -- an unmapped condition would be silently scored as "
            f"{conditions[0]!r}."
        )
    # Pre rows get index 0 purely so the array is well-formed; post = 0 makes it inert.
    f["cond_idx"] = (
        f["condition"].map({c: i for i, c in enumerate(conditions)}).fillna(0).astype(int)
    )
    f.loc[pre_rows, "cond_idx"] = 0
    f["pair_idx"] = f["pair_id"].map({p: i for i, p in enumerate(pairs)})
    f["tmpl_idx"] = f["template"].map({t: i for i, t in enumerate(templates)})

    return SpreadDesign(f, conditions, pairs, templates, mu, sd)


def fit(cfg: RunConfig, design: SpreadDesign, *, progressbar: bool = False) -> az.InferenceData:
    f = design.frame
    coords = {"condition": design.conditions, "pair": design.pairs,
              "template": design.templates}

    with pm.Model(coords=coords):
        # Priors on the logit scale (A2.9.1). A shift of 1 logit moves a coin flip to
        # p = 0.73, so these are weakly informative rather than restrictive.
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="condition")   # post shift
        lam = pm.Normal("lambda", 0.0, 1.0, dims="condition")    # difficulty slope
        beta = pm.Normal("beta", 0.0, 1.5, dims="template")      # position bias

        sd_pair = pm.HalfNormal("sd_pair", 2.0)
        u_pair = pm.ZeroSumNormal("u_pair", sigma=sd_pair, dims="pair")
        sd_tmpl = pm.HalfNormal("sd_template", 1.0)
        # NON-CENTERED (A4.1). The centered form, ZeroSumNormal(sigma=sd_template), funnels
        # when the group SD approaches zero: llama's templates produce nearly identical
        # spread (sd_template ~ 0.043), its posterior piles against zero, and 4.5% of draws
        # diverged there. Divergence count tracked sd_template exactly across the three
        # models -- 0.303/7, 0.132/70, 0.043/362 -- which is what identifies the mechanism.
        #
        # Scaling a zero-sum vector by a positive scalar preserves the sum-to-zero
        # constraint, so this is a pure reparameterization: same model, same posterior in
        # expectation, different geometry for the sampler to traverse.
        #
        # `u_pair` is deliberately left CENTERED. Its diagnostics show no funnel (divergent
        # vs non-divergent sd_pair: 1.017 vs 1.014), and altering an unimplicated part of
        # the model after seeing the results would have no justification.
        z_tmpl = pm.ZeroSumNormal("z_template", sigma=1.0, dims="template")
        u_tmpl = pm.Deterministic("u_template", z_tmpl * sd_tmpl, dims="template")

        ci = f["cond_idx"].to_numpy()
        logit_p = (
            u_pair[f["pair_idx"].to_numpy()]
            + u_tmpl[f["tmpl_idx"].to_numpy()]
            + beta[f["tmpl_idx"].to_numpy()] * f["s"].to_numpy()
            + f["post"].to_numpy() * f["d"].to_numpy()
            * (gamma[ci] + lam[ci] * f["diff_z"].to_numpy())
        )
        pm.Bernoulli("item1_wins", logit_p=logit_p,
                     observed=f["item1_wins"].to_numpy().astype(int))

        kwargs = dict(draws=cfg.analysis.draws, tune=cfg.analysis.tune,
                      chains=cfg.analysis.chains, random_seed=cfg.seed,
                      target_accept=0.9, progressbar=progressbar)
        if cfg.analysis.sampler_cores is not None:
            kwargs["cores"] = cfg.analysis.sampler_cores
        return pm.sample(**kwargs)


def fit_graded(cfg: RunConfig, design: SpreadDesign, *, family: str = "normal",
               progressbar: bool = False) -> az.InferenceData:
    """A4.5 PILOT MODEL. Same linear predictor, graded outcome instead of a coin flip.

    NOT A PREREGISTERED PRIMARY. The continuous DV changes the likelihood, so its priors,
    the SESOI's units and 9.2's thresholds all have to be restated on the new scale in a
    preregistration written blind. This exists to measure ONE thing: how much precision the
    binarisation costs. Stage 0's reported result is unaffected by anything computed here.

        y = logit(p_item1)
        mu = u_pair + u_template + beta_t*s + post*d*(gamma_c + lambda_c*diff_z)
        y ~ Normal(mu, sigma_obs)   or   StudentT(nu, mu, sigma_obs)

    The linear predictor is IDENTICAL to the Bernoulli model's, which is the point: `gamma`
    and `lambda` keep their units, so the two fits' standard errors are directly comparable
    and the ratio means what it appears to mean.

    WHY BOTH FAMILIES. `p_item1` reaches 1.5e-06 and 0.99999, i.e. logits near +/-13. A
    Gaussian with constant sigma is misspecified in those tails and will report intervals
    that are too NARROW -- which would look exactly like a precision win. Fitting a
    Student-t as well is the check: if the two agree the gain is real, and if the Gaussian
    is much tighter than the t then part of the "gain" is the tails being modelled badly.
    Reporting only the Gaussian would be the most flattering and least honest choice.
    """
    if "p_item1" not in design.frame.columns:
        raise ValueError(
            "trials carry no `p_item1`; they predate A4.5 and the graded readout was "
            "discarded when they were collected. Re-run Pass C."
        )
    if family not in ("normal", "studentt"):
        raise ValueError(f"family must be 'normal' or 'studentt', got {family!r}")

    f = design.frame
    p_obs = f["p_item1"].to_numpy(dtype=float)
    if not np.all((p_obs > 0.0) & (p_obs < 1.0)):
        n_bad = int((~((p_obs > 0.0) & (p_obs < 1.0))).sum())
        raise ValueError(
            f"{n_bad} row(s) have p_item1 at exactly 0 or 1, whose logit is infinite. "
            "Clipping would invent a value for the most extreme observations, which are "
            "the ones with the most leverage; decide explicitly rather than here."
        )
    y = np.log(p_obs / (1.0 - p_obs))

    coords = {"condition": design.conditions, "pair": design.pairs,
              "template": design.templates}
    with pm.Model(coords=coords):
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="condition")
        lam = pm.Normal("lambda", 0.0, 1.0, dims="condition")
        beta = pm.Normal("beta", 0.0, 1.5, dims="template")

        sd_pair = pm.HalfNormal("sd_pair", 2.0)
        u_pair = pm.ZeroSumNormal("u_pair", sigma=sd_pair, dims="pair")
        sd_tmpl = pm.HalfNormal("sd_template", 1.0)
        z_tmpl = pm.ZeroSumNormal("z_template", sigma=1.0, dims="template")
        u_tmpl = pm.Deterministic("u_template", z_tmpl * sd_tmpl, dims="template")

        ci = f["cond_idx"].to_numpy()
        mu = (
            u_pair[f["pair_idx"].to_numpy()]
            + u_tmpl[f["tmpl_idx"].to_numpy()]
            + beta[f["tmpl_idx"].to_numpy()] * f["s"].to_numpy()
            + f["post"].to_numpy() * f["d"].to_numpy()
            * (gamma[ci] + lam[ci] * f["diff_z"].to_numpy())
        )
        # The logit of a readout spans a wide range, so the observation scale is given room.
        sigma_obs = pm.HalfNormal("sigma_obs", 3.0)
        if family == "normal":
            pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y)
        else:
            nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
            pm.StudentT("y", nu=nu, mu=mu, sigma=sigma_obs, observed=y)

        kwargs = dict(draws=cfg.analysis.draws, tune=cfg.analysis.tune,
                      chains=cfg.analysis.chains, random_seed=cfg.seed,
                      target_accept=0.9, progressbar=progressbar)
        if cfg.analysis.sampler_cores is not None:
            kwargs["cores"] = cfg.analysis.sampler_cores
        return pm.sample(**kwargs)


def _draws(idata: az.InferenceData, var: str, condition: str) -> np.ndarray:
    return (idata.posterior[var].sel(condition=condition)
            .stack(sample=("chain", "draw")).to_numpy())


def contrasts(cfg: RunConfig, idata: az.InferenceData, sesoi: float) -> pd.DataFrame:
    """Every planned contrast, on the difficulty slope and on the post shift.

    `lambda` is the interaction -- the primary term. `gamma` is the main effect at mean
    |diff|, reported but never primary: on its own it is consistent with context
    sensitivity, which is the whole reason H1 is an interaction (section 1.1).
    """
    from src.analysis.mixed import summarize_contrast

    available = set(idata.posterior["condition"].to_numpy().tolist())
    rows = []
    for name, (a, b, sign) in PLANNED.items():
        if {a, b} - available:
            continue
        for term, var, dirn in (("lambda", "lambda", sign), ("gamma", "gamma", 0)):
            rows.append(
                summarize_contrast(
                    name, _draws(idata, var, a) - _draws(idata, var, b),
                    term=term, sesoi=sesoi,
                    hdi_prob=cfg.analysis.hdi_prob, direction=dirn,
                ).as_row()
            )
    return pd.DataFrame(rows)


def structure_factor_agreement(contrasts_frame: pd.DataFrame) -> dict:
    """Do the two turn-presence estimates agree? (A2.9.3 correction)

    Turn presence is estimated at both wording levels -- `chose - self-recounted` and
    `structure-control - yoked`. If they agree, a probe or effect reading turn-presence
    is identified and can be subtracted. If they disagree, turn presence interacts with
    wording and neither edge is interpretable alone.
    """
    lam = contrasts_frame[contrasts_frame["term"] == "lambda"].set_index("name")
    a, b = "chose - self-recounted", "structure-control - yoked"
    if not {a, b} <= set(lam.index):
        return {}
    return {"edge_a": a, "edge_b": b,
            "estimate_a": float(lam.loc[a, "median"]),
            "estimate_b": float(lam.loc[b, "median"]),
            "discrepancy": float(lam.loc[a, "median"] - lam.loc[b, "median"])}


def convergence(cfg: RunConfig, idata: az.InferenceData) -> pd.DataFrame:
    summ = az.summary(idata, var_names=["gamma", "lambda", "beta", "sd_pair", "sd_template"])
    summ["rhat_ok"] = summ["r_hat"] <= cfg.analysis.rhat_max
    summ["ess_ok"] = summ["ess_bulk"] >= cfg.analysis.ess_min
    return summ
