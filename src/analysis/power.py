"""Power for the primary interaction, on the logit scale it is actually estimated on.

WHY THIS MODULE WAS REWRITTEN RATHER THAN ADJUSTED. The previous version computed power
for `spread = (chosen_post - chosen_pre) - (rejected_post - rejected_pre)`, a difference
of four ratings on a 1-9 scale, with measurement noise derived from ICC(C,1). None of
those quantities exists any more. Amendment 1 replaced the absolute instrument with
pairwise comparisons and A2.9.1 made the DV a Bernoulli outcome modelled directly; the
interaction is now `lambda_c`, a log-odds shift per SD of |diff|. Rescaling the old
answer would have produced a confident number about a quantity the experiment no longer
measures, so the old module was NOT run. This one replaces it.

WHAT IT COMPUTES. The Fisher information of the exact Pass C design under the exact
likelihood in `spread_model`, evaluated at a stated parameter vector:

    I = X' W X,   W = diag(p (1 - p)),   SE(c) = sqrt(c' I^-1 c)

with `c` selecting `lambda_chose - lambda_yoked`. This is a design calculation, not a
simulation: given the design matrix there is nothing left to sample. It is checked
against actual fits in `test_power.py` rather than trusted on its algebra alone.

THREE THINGS IT MAKES VISIBLE THAT A SIMULATION WOULD HAVE BURIED.

1. `W = p(1-p)` is the information weight, and it is maximal at p = 0.5. Difficult pairs
   sit there by construction, so the design concentrates its information exactly where
   the interaction is identified. This is the favourable face of the same fact that makes
   the two-stage estimator biased (see `spread_model`'s docstring): there, near-0.5 pairs
   have the largest sampling variance in a per-pair spread; here, they carry the most
   information in the joint likelihood. Same geometry, opposite sign, because the joint
   model weights by information instead of averaging noisy per-pair estimates.

2. The position term costs real power. beta runs +0.67 to +1.73 across gemma's templates,
   and beta = 1.73 puts a coin-flip pair at p = 0.85, where W = 0.127 against 0.25 at
   p = 0.5 -- a 49% loss of information on that template. That cost is a property of the
   model's position bias rather than of anything we chose, but it belongs in the power
   statement instead of being discovered afterwards.

3. Whether the equivalence branch is even REACHABLE. Section 9.2 declares `fail` when the
   95% HDI lies entirely inside the ROPE, which requires 1.96 * SE < SESOI. If the design
   cannot meet that, `fail` is unreachable and the only possible outcomes are `pass` and
   `inconclusive` -- so a null could never be reported as equivalence no matter how the
   data came out. That has to be known BEFORE the data, which is why this is computed
   while blind to Pass C.

THE PAIR GAP COMES FROM `diff_analysis`, NEVER `diff_selection`. Difficult pairs have a
selection gap near zero by construction, but that is selection on noise: their gap
measured on the disjoint templates is 0.478 for gemma, not 0.002. Feeding the selection
gap in would place every difficult pair at exactly p = 0.5 and overstate the available
information. `diff_analysis` still carries its own measurement error, which inflates the
spread of gaps slightly, so the resulting figure is mildly CONSERVATIVE for the difficult
stratum -- the direction to err in.

CONSERVATISM -- FALSIFIED BY THE COMPLETED RUN (A4.2). This paragraph claimed the design
SE errs WIDE. Measured against the realized fits it errs NARROW, by 1.05x for llama, 3.44x
for qwen-1.5b and 5.38x for gemma, and the error tracks the fitted between-template SD.
Realized MDE is 2.1-6.6x the SESOI, not the 1.37-1.99x this module reported. The claim is
left in place, struck, because it is what the preregistered figures were computed under.

Pair and template effects are absorbed as fixed effects, whereas the fitted
model partially pools them. Fixed-effect absorption spends more degrees of freedom and so
gives a wider SE than partial pooling. The figures here therefore understate power rather
than overstate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.spread_model import PLANNED, PRIMARY
from src.config import RunConfig

#: Prior on lambda in the fitted model (spread_model.fit). Included because with a wide
#: SE the prior is not negligible: it shrinks the posterior mean toward zero, which costs
#: power, and narrows the posterior sd, which buys some back. Ignoring it would overstate
#: power at exactly the sample sizes where the answer matters.
LAMBDA_PRIOR_SD = 1.0


@dataclass
class PowerResult:
    se: float
    sesoi: float
    target: float
    grid: pd.DataFrame
    power_at_sesoi: float
    min_detectable: float
    equivalence_reachable: bool
    assumptions: dict = field(default_factory=dict)
    information: pd.DataFrame | None = None

    def summary(self) -> str:
        lines = [
            f"SE(lambda_chose - lambda_yoked) = {self.se:.4f} logits per SD of |diff|",
            f"SESOI = {self.sesoi:.4f}  (0.15 * sigma_item, Amendment 2)",
            f"MINIMUM DETECTABLE EFFECT at {self.target:.0%} power = {self.min_detectable:.4f}"
            f"  = {self.min_detectable / self.sesoi:.2f}x SESOI",
            f"power at the SESOI itself = {self.power_at_sesoi:.3f} -- NOT compared against "
            "a target, see below",
        ]
        lines.append(
            "SECTION 8's CRITERION IS UNSATISFIABLE AND IS NOT USED HERE. It asks for >= 80% "
            "power AT the SESOI, but section 9.2's `pass` cell requires the posterior median "
            "to EXCEED the SESOI. When the true effect IS the SESOI the median is centred on "
            "that threshold, so power at the SESOI converges to exactly 0.500 as SE -> 0 and "
            "can never reach 0.80 at any sample size. This is a property of the two rules "
            "together, not of this design. The minimum detectable effect is reported instead; "
            "its floor is the SESOI itself. Raised as Amendment 3, not silently patched."
        )
        if self.equivalence_reachable:
            lines.append(
                "equivalence branch REACHABLE: the HDI can fit inside the ROPE, so a null "
                "may be reported as equivalence"
            )
        else:
            lines.append(
                "equivalence branch UNREACHABLE: the 95% HDI is wider than the ROPE, so "
                "the `fail` cell of the section 9.2 rule cannot be entered at ANY observed "
                "value. A null result can only be `inconclusive`. Report it as such and do "
                "not describe it as equivalence."
            )
        if self.information is not None:
            lines.append("information for lambda by stratum:")
            lines.append(self.information.to_string(index=False))
        lines.append("assumptions: " + ", ".join(f"{k}={v}" for k, v in self.assumptions.items()))
        return "\n".join(lines)


def _design(
    pairs: pd.DataFrame,
    beta: np.ndarray,
    conditions: list[str],
    n_orders: int,
    gamma: float,
    lam: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build the Pass C design matrix, its linear predictor, and the contrast vector.

    Rows are exactly what Pass C emits: one shared pre measurement per
    (pair, template, order), plus one post measurement per condition on top of it.
    """
    rng = np.random.default_rng(seed)
    n_pair, n_tmpl, n_cond = len(pairs), len(beta), len(conditions)

    gap = pairs["diff_analysis"].to_numpy(dtype=float)
    # Scale-RELATIVE, not `<= 0`. A column of constant 1e-6 has sd 2e-22 rather than
    # exactly zero, so an absolute test passes it and diff_z comes out as garbage of
    # order 1e6 instead of raising. spread_model.prepare has the same absolute test and
    # the same hole; this is the stricter of the two.
    if gap.std(ddof=1) <= 1e-9 * max(1.0, abs(float(gap.mean()))):
        raise ValueError(
            f"|diff| has no usable variance (sd {gap.std(ddof=1):.3g} against mean "
            f"{gap.mean():.3g}); the difficulty regressor is undefined."
        )
    # |diff| is unsigned; the item1/item2 axis is canonical by sorted id and so unrelated
    # to preference. Signs are therefore symmetric and assigned by the seeded rng.
    u_pair = gap * rng.choice([-1.0, 1.0], size=n_pair)
    diff_z = (gap - gap.mean()) / gap.std(ddof=1)

    # DESIGNATION IS NOT A FREE DESIGN CHOICE -- it is inherited from the model's own
    # behaviour, and getting it wrong changes the answer. Six of the eight conditions
    # designate the model's own pick (pass_c.OWN_PICK_CONDITIONS); only `random` and
    # `3p-random` use an independent per-pair designation. And the own pick is CORRELATED
    # WITH THE PAIR GAP: on an easy pair the model picks the higher-theta item almost
    # deterministically, so d aligns with the sign of u_pair and the post linear predictor
    # moves further from 0.5, where W is small. Easy pairs therefore lose information
    # twice over -- once for their gap, once for their predictability. Modelling d as
    # alternating (the obvious synthetic choice) misses that entirely.
    from src.experiments.pass_c import OWN_PICK_CONDITIONS

    own = np.array([c in OWN_PICK_CONDITIONS for c in conditions])
    rand_designation = rng.choice([-1.0, 1.0], size=n_pair)

    rows = []
    # pre: post = 0, so these rows inform u_pair, u_template and beta only. They carry no
    # information about lambda, but they are part of the design and change the fit's
    # conditioning, so they are included rather than assumed away.
    for p in range(n_pair):
        for t in range(n_tmpl):
            for o in range(n_orders):
                rows.append((p, t, 1.0 if o == 0 else -1.0, 0, 0, 0.0))
    for p in range(n_pair):
        for t in range(n_tmpl):
            for o in range(n_orders):
                s = 1.0 if o == 0 else -1.0
                # The choice is elicited per (pair, template, order) and is subject to the
                # same position bias as the DV, which is what the flip-rate diagnostic
                # measures. Draw it from the same linear predictor.
                p_item1 = 1.0 / (1.0 + np.exp(-(u_pair[p] + beta[t] * s)))
                d_own = 1.0 if rng.random() < p_item1 else -1.0
                for c in range(n_cond):
                    rows.append((p, t, s, 1, c,
                                 d_own if own[c] else rand_designation[p]))

    arr = np.array(rows, dtype=float)
    pi = arr[:, 0].astype(int)
    ti = arr[:, 1].astype(int)
    s, post, ci, d = arr[:, 2], arr[:, 3], arr[:, 4].astype(int), arr[:, 5]
    dz, up = diff_z[pi], u_pair[pi]
    n = len(arr)

    n_col = n_pair + 2 * n_tmpl + 2 * n_cond
    g0 = n_pair + 2 * n_tmpl
    l0 = g0 + n_cond
    idx = np.arange(n)
    X = np.zeros((n, n_col))
    X[idx, pi] = 1.0                              # pair intercepts
    X[idx, n_pair + ti] = 1.0                     # template intercepts
    X[idx, n_pair + n_tmpl + ti] = s              # beta_t * s
    live = np.where(post == 1)[0]
    X[live, g0 + ci[live]] = d[live]              # gamma_c
    X[live, l0 + ci[live]] = d[live] * dz[live]   # lambda_c

    eta = up + beta[ti] * s + post * d * (gamma + lam[ci] * dz)

    a, b = PLANNED[PRIMARY][0], PLANNED[PRIMARY][1]
    c_vec = np.zeros(n_col)
    c_vec[l0 + conditions.index(a)] = 1.0
    c_vec[l0 + conditions.index(b)] = -1.0

    meta = {"n_obs": n, "n_col": n_col, "diff_z": dz, "post": post,
            "difficulty": pairs["difficulty"].to_numpy()[pi]}
    return X, eta, c_vec, meta


def _se(X: np.ndarray, eta: np.ndarray, c_vec: np.ndarray) -> tuple[float, np.ndarray]:
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1.0 - p)
    info = X.T @ (X * w[:, None])
    # The pair and template intercept blocks are jointly rank-deficient (a constant can
    # shift between them), which is exactly what ZeroSumNormal fixes in the fitted model.
    # pinv takes the minimum-norm solution; the contrast is on lambda alone and lies in
    # the row space, so it is unaffected by the deficiency.
    return float(np.sqrt(c_vec @ np.linalg.pinv(info) @ c_vec)), w


def _posterior(se: float) -> tuple[float, float]:
    """Posterior sd and mean-shrinkage factor for the contrast under its prior.

    The contrast is a difference of two lambdas, so its prior sd is sqrt(2) times the
    per-condition prior sd.
    """
    prior_sd = LAMBDA_PRIOR_SD * np.sqrt(2.0)
    prec = 1.0 / se**2 + 1.0 / prior_sd**2
    return float(np.sqrt(1.0 / prec)), float((1.0 / se**2) / prec)


def _decide_pass(true_lambda: float, se: float, sesoi: float, z: float) -> float:
    """P(the section 9.2 `pass` cell) at a given true effect.

    `pass` requires BOTH that the 95% HDI excludes 0 in the predicted (negative)
    direction AND that the posterior median magnitude exceeds the SESOI. Powering on the
    HDI alone would overstate power wherever the SESOI is the binding constraint.

    Removing the second conjunct was considered and **declined** (A3.6). It would have
    raised power at the SESOI from 0.495 to 0.641, but it also sets the effective
    threshold to `1.96 * SE`, which floats with sample size and shrinks toward zero --
    the SESOI exists precisely so the bar does not do that. The `max` below is what keeps
    the bar fixed.
    """
    from scipy import stats

    post_sd, shrink = _posterior(se)
    threshold = -max(z * post_sd, sesoi) / shrink
    return float(stats.norm.cdf((threshold - true_lambda) / se))


def analyze(
    cfg: RunConfig,
    pairs: pd.DataFrame,
    beta: pd.DataFrame,
    sigma_item: float,
    *,
    gamma: float = 0.4,
    grid: np.ndarray | None = None,
) -> PowerResult:
    """Power for `lambda_chose - lambda_yoked` on this model's own realized design.

    `beta` is the per-template position term from the Bradley-Terry fit and `sigma_item`
    its between-item scale -- both measured, neither assumed. `gamma`, the post shift at
    mean |diff|, is the one quantity unknown before Pass C; it enters only through
    `p(1-p)` and is reported as an assumption with a sensitivity band
    (`gamma_sensitivity`).
    """
    from scipy import stats

    conditions = list(cfg.pass_c.conditions)
    b = beta.sort_values("template")["beta_mean"].to_numpy(dtype=float)
    sesoi = cfg.analysis.sesoi_sigma_fraction * float(sigma_item)
    z = stats.norm.ppf(0.5 + cfg.analysis.hdi_prob / 2.0)

    # Evaluated under the null on lambda: at the small effects this powers for, the null
    # and the alternative give near-identical W, and the null is the conventional choice.
    lam = np.zeros(len(conditions))
    X, eta, c_vec, meta = _design(pairs, b, conditions, cfg.pass_c.n_option_orders,
                                  gamma, lam, cfg.seed)
    se, w = _se(X, eta, c_vec)
    post_sd, _ = _posterior(se)

    scale = max(sesoi, z * post_sd)
    if grid is None:
        grid = np.linspace(0.0, -3.0 * scale, 13)
    grid_df = pd.DataFrame(
        [{"true_lambda": float(g), "power_pass": _decide_pass(float(g), se, sesoi, z)}
         for g in grid]
    )

    fine = np.linspace(0.0, -8.0 * scale, 8001)
    pw = np.array([_decide_pass(float(g), se, sesoi, z) for g in fine])
    hit = np.where(pw >= cfg.analysis.power_target)[0]
    min_det = float(abs(fine[hit[0]])) if len(hit) else float("nan")

    # Where does the estimate actually come from? d(eta)/d(lambda) = d * diff_z and
    # |d| = 1, so each post observation contributes w * diff_z^2 to lambda's information.
    live = meta["post"] == 1
    contrib = w * meta["diff_z"] ** 2 * live
    strata = []
    for lvl in ("difficult", "easy"):
        m = (meta["difficulty"] == lvl) & live
        if not m.any():
            continue
        strata.append({"stratum": lvl, "n_post_obs": int(m.sum()),
                       "mean_w": float(w[m].mean()),
                       "share_of_lambda_information": float(contrib[m].sum() / contrib.sum())})
    info_df = pd.DataFrame(strata)

    return PowerResult(
        se=se,
        sesoi=sesoi,
        target=cfg.analysis.power_target,
        grid=grid_df,
        power_at_sesoi=_decide_pass(-sesoi, se, sesoi, z),
        min_detectable=min_det,
        equivalence_reachable=bool(z * post_sd < sesoi),
        information=info_df,
        assumptions={
            "n_pairs": len(pairs),
            "n_templates": len(b),
            "n_orders": cfg.pass_c.n_option_orders,
            "n_conditions": len(conditions),
            "n_observations": meta["n_obs"],
            "gamma_assumed": gamma,
            "beta_measured": f"{b.min():+.3f}..{b.max():+.3f}",
            "sigma_item_measured": round(float(sigma_item), 4),
            "gap_source": "diff_analysis (disjoint templates), never diff_selection",
            "random_effects": "absorbed as fixed -> conservative vs partial pooling",
        },
    )


def gamma_sensitivity(
    cfg: RunConfig,
    pairs: pd.DataFrame,
    beta: pd.DataFrame,
    sigma_item: float,
    gammas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.8, 1.5),
) -> pd.DataFrame:
    """How much does the one unmeasured quantity move the answer?

    gamma is the post shift, unknown until Pass C runs. It enters only through p(1-p), so
    a large gamma pushes post observations away from 0.5 and costs information. If power
    at the SESOI is stable across this range then the assumption is not load-bearing and
    can be reported as such instead of defended.
    """
    rows = []
    for g in gammas:
        r = analyze(cfg, pairs, beta, sigma_item, gamma=g)
        rows.append({"gamma": g, "se": r.se, "power_at_sesoi": r.power_at_sesoi,
                     "min_detectable": r.min_detectable,
                     "equivalence_reachable": r.equivalence_reachable})
    return pd.DataFrame(rows)
