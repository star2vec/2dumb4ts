"""Power simulation for the primary interaction.

Run AFTER Pass A and BEFORE Pass C, from that model's own measured quantities:
sigma_between, the measurement noise implied by ICC(C,1), and the realized |diff|
distribution from Pass B. Nothing here is borrowed from another model or from the
literature.

Noise derivation. For the consistency-form ICC on a fixed template battery,
    ICC(C,1) = var_item / (var_item + var_err)
so with var_item = sigma_between^2,
    sd_err = sigma_between * sqrt((1 - ICC) / ICC)
The DV is a difference of four independent ratings, so its measurement-noise SD is
    sd_spread = 2 * sd_err.

The interaction is estimated by least squares with pair and template absorbed as
FIXED effects. This is a deliberate approximation to the Bayesian fit, and it is
conservative in the right direction: fixed-effect absorption gives wider standard
errors than partial pooling, so it will not overstate power. Fitting the full
Bayesian model hundreds of times would take longer than the experiment.

The assumptions are returned alongside the numbers rather than buried, because a
power claim is only as good as its stated generative model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RunConfig

# Random-effect scales are unknown before Pass C. They are expressed as multiples
# of the measurement-noise SD and reported, so the assumption is explicit.
PAIR_RE_MULTIPLE = 1.0
TEMPLATE_RE_MULTIPLE = 1.0


@dataclass
class PowerResult:
    target: float
    sesoi: float
    grid: pd.DataFrame
    power_at_sesoi: float
    min_detectable: float
    assumptions: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"power at SESOI ({self.sesoi:.3f} spread points per SD of |diff|) = "
            f"{self.power_at_sesoi:.2f}  [target {self.target:.2f}]",
            f"minimum detectable interaction at {self.target:.0%} power = "
            f"{self.min_detectable:.3f}",
            "assumptions: "
            + ", ".join(f"{k}={v}" for k, v in self.assumptions.items()),
        ]
        if self.power_at_sesoi < self.target:
            lines.append(
                "  UNDERPOWERED at the SESOI. Scale ITEMS first, pairs second: adding "
                "pairs beyond the item pool's capacity forces reuse and pushes the "
                "selection thresholds toward the middle of the |diff| distribution, "
                "diluting the difficulty manipulation itself."
            )
        return "\n".join(lines)


def _simulate_once(
    rng: np.random.Generator,
    diff_z: np.ndarray,
    pair_idx: np.ndarray,
    template_idx: np.ndarray,
    cond: np.ndarray,
    true_interaction: float,
    sd_spread: float,
    sd_pair: float,
    sd_template: float,
) -> tuple[float, float]:
    """Generate one dataset and return (estimate, standard error) of the contrast."""
    n_pairs = pair_idx.max() + 1
    n_templates = template_idx.max() + 1

    u_pair = rng.normal(0.0, sd_pair, size=n_pairs)
    u_template = rng.normal(0.0, sd_template, size=n_templates)
    # cond is 1 for `chose`, 0 for `yoked`; the interaction is the differential
    # slope on diff_z between them.
    y = (
        true_interaction * cond * diff_z
        + u_pair[pair_idx]
        + u_template[template_idx]
        + rng.normal(0.0, sd_spread, size=len(diff_z))
    )

    # Design: cond, diff_z, cond:diff_z, plus pair and template fixed effects.
    n = len(y)
    blocks = [
        np.ones((n, 1)),
        cond.reshape(-1, 1),
        diff_z.reshape(-1, 1),
        (cond * diff_z).reshape(-1, 1),
        np.eye(n_pairs)[pair_idx][:, 1:],
        np.eye(n_templates)[template_idx][:, 1:],
    ]
    x = np.hstack(blocks)

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = n - np.linalg.matrix_rank(x)
    if dof <= 0:
        return float("nan"), float("nan")
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(s2 * xtx_inv[3, 3]))
    return float(beta[3]), se


def simulate_power(
    cfg: RunConfig,
    *,
    sigma_between: float,
    icc_c1: float,
    diff_analysis: np.ndarray,
    sesoi: float,
    n_templates: int | None = None,
    grid: np.ndarray | None = None,
) -> PowerResult:
    if not 0 < icc_c1 < 1:
        raise ValueError(
            f"ICC(C,1) = {icc_c1} is outside (0, 1); the noise floor is undefined and "
            "power cannot be simulated. Investigate this as an implementation fault."
        )

    sd_err = sigma_between * np.sqrt((1.0 - icc_c1) / icc_c1)
    sd_spread = 2.0 * sd_err
    sd_pair = PAIR_RE_MULTIPLE * sd_err
    sd_template = TEMPLATE_RE_MULTIPLE * sd_err

    n_templates = n_templates or cfg.stimuli.n_templates
    n_orders = cfg.pass_c.n_option_orders

    diff = np.asarray(diff_analysis, dtype=float)
    diff_z_pair = (diff - diff.mean()) / diff.std(ddof=1)
    n_pairs = len(diff_z_pair)

    # The primary contrast involves two conditions: chose and yoked.
    rows_pair, rows_template, rows_cond, rows_diff = [], [], [], []
    for p in range(n_pairs):
        for t in range(n_templates):
            for _ in range(n_orders):
                for c in (0, 1):
                    rows_pair.append(p)
                    rows_template.append(t)
                    rows_cond.append(c)
                    rows_diff.append(diff_z_pair[p])

    pair_idx = np.array(rows_pair)
    template_idx = np.array(rows_template)
    cond = np.array(rows_cond, dtype=float)
    diff_z = np.array(rows_diff, dtype=float)

    if grid is None:
        grid = np.round(np.linspace(0.0, max(4.0 * sesoi, 0.4), 9), 4)

    rng = np.random.default_rng(cfg.seed)
    crit = stats.norm.ppf(1.0 - (1.0 - cfg.analysis.hdi_prob) / 2.0)

    records = []
    for true_effect in grid:
        hits = 0
        n_ok = 0
        for _ in range(cfg.analysis.power_n_sims):
            est, se = _simulate_once(
                rng,
                diff_z,
                pair_idx,
                template_idx,
                cond,
                -float(true_effect),  # predicted direction is negative
                sd_spread,
                sd_pair,
                sd_template,
            )
            if not np.isfinite(est) or not np.isfinite(se) or se <= 0:
                continue
            n_ok += 1
            # Interval excludes zero in the predicted (negative) direction.
            if est + crit * se < 0:
                hits += 1
        records.append(
            {
                "true_interaction": float(true_effect),
                "power": hits / n_ok if n_ok else float("nan"),
                "n_sims": n_ok,
            }
        )

    table = pd.DataFrame(records)
    power_at_sesoi = float(np.interp(sesoi, table["true_interaction"], table["power"]))

    above = table[table["power"] >= cfg.analysis.power_target]
    min_detectable = (
        float(above["true_interaction"].iloc[0]) if len(above) else float("nan")
    )

    return PowerResult(
        target=cfg.analysis.power_target,
        sesoi=sesoi,
        grid=table,
        power_at_sesoi=power_at_sesoi,
        min_detectable=min_detectable,
        assumptions={
            "sigma_between": round(sigma_between, 4),
            "icc_c1": round(icc_c1, 4),
            "sd_rating_error": round(sd_err, 4),
            "sd_spread_noise": round(sd_spread, 4),
            "sd_pair_re": round(sd_pair, 4),
            "sd_template_re": round(sd_template, 4),
            "n_pairs": n_pairs,
            "n_templates": n_templates,
            "n_option_orders": n_orders,
            "estimator": "OLS with pair+template absorbed as fixed effects (conservative)",
        },
    )
