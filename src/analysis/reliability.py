"""Pass A psychometrics: polarity collapse, validity gate, ICC, sigma_between.

Two distinct properties are measured, and conflating them is the mistake this
module exists to avoid:

  VALIDITY  -- does the model read the response scale at all? Measured as the
               Spearman correlation between ascending scores and reversed
               descending scores, per template. This is the sole categorical
               exclusion (rho >= 0.6).

  RELIABILITY -- how noisy is it? Measured as ICC across the five templates. This
               is NOT a scientific halt; it is a tripwire for implementation
               faults below 0.4, and otherwise feeds the power simulation.

A model can be reliable and invalid (consistently reading the scale backwards) or
valid and noisy. Only the first is disqualifying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RunConfig
from src.readout.expected_value import reverse_polarity


# ---------------------------------------------------------------------------
# polarity


def ascending_scores(pass_a: pd.DataFrame) -> pd.DataFrame:
    """Item x template matrix of ASCENDING-polarity scores only.

    Reliability is measured here rather than on polarity-collapsed scores. The two
    are not interchangeable: if a model ignores the descending anchor definition,
    collapsing gives (asc + C - asc)/2 = C/2 for every item, which drives the
    between-item variance and the ICC toward zero as an artifact of the collapse
    rather than a property of the instrument. The ascending-only ICC is the actual
    noise floor of a single rating.
    """
    wide = pass_a[pass_a["polarity"] == "ascending"].pivot_table(
        index=["item_id", "domain"], columns="template", values="rating", aggfunc="first"
    )
    return wide.reset_index()


def collapse_polarity(pass_a: pd.DataFrame, reversal_constant: int) -> pd.DataFrame:
    """Item x template matrix of polarity-collapsed scores.

    s[item, t] = mean( ascending, reversal_constant - descending )
    """
    wide = pass_a.pivot_table(
        index=["item_id", "domain"],
        columns=["template", "polarity"],
        values="rating",
        aggfunc="first",
    )
    templates = sorted({c[0] for c in wide.columns})
    out = {}
    for t in templates:
        asc = wide[(t, "ascending")].to_numpy(dtype=float)
        desc = reverse_polarity(wide[(t, "descending")].to_numpy(dtype=float), reversal_constant)
        out[t] = (asc + desc) / 2.0
    frame = pd.DataFrame(out, index=wide.index).reset_index()
    return frame


def validity_table(pass_a: pd.DataFrame, reversal_constant: int) -> pd.DataFrame:
    """Per template: Spearman rho between ascending and reversed descending."""
    wide = pass_a.pivot_table(
        index="item_id", columns=["template", "polarity"], values="rating", aggfunc="first"
    )
    rows = []
    for t in sorted({c[0] for c in wide.columns}):
        asc = wide[(t, "ascending")].to_numpy(dtype=float)
        desc = reverse_polarity(wide[(t, "descending")].to_numpy(dtype=float), reversal_constant)
        ok = np.isfinite(asc) & np.isfinite(desc)
        # A constant vector makes Spearman undefined; that is a degenerate
        # instrument, reported as rho = 0 rather than NaN so the gate can act.
        # Scale-relative, matching power.py and spread_model.prepare. A near-constant
        # vector has sd of order 1e-22 rather than exactly 0, so `== 0` lets it through
        # and Spearman returns NaN, which then propagates into median_rho. This path is
        # inert (A1.5 gates nothing), so this is consistency rather than a fix.
        a_sd, d_sd = float(np.std(asc[ok])), float(np.std(desc[ok]))
        tiny = 1e-9 * max(1.0, abs(float(np.mean(asc[ok]))))
        if ok.sum() < 3 or a_sd <= tiny or d_sd <= tiny:
            rho, p = 0.0, 1.0
        else:
            rho, p = stats.spearmanr(asc[ok], desc[ok])
        rows.append(
            {
                "template": t,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n_items": int(ok.sum()),
                "sd_ascending": float(np.std(asc[ok], ddof=1)) if ok.sum() > 1 else 0.0,
                "sd_descending_reversed": float(np.std(desc[ok], ddof=1)) if ok.sum() > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ICC


def icc_two_way(x: np.ndarray) -> dict[str, float]:
    """Two-way random-effects ICCs from an items x raters matrix.

    Rows are items (targets), columns are templates (raters). Returns all four
    Shrout-Fleiss forms:

      C,1 / C,k  consistency  -- additive per-template offsets are tolerated
      A,1 / A,k  absolute agreement -- offsets count against reliability

    The CONSISTENCY form is the one this design needs: templates are a fixed
    battery, and an additive per-template offset cancels in the within-pair
    differences that constitute the DV.

    The k-measurement forms are computed on the actual k columns supplied, not
    projected from ICC(.,1) via Spearman-Brown. Spearman-Brown assumes
    independent errors across measurements; template errors are correlated, so
    the projection would overstate the reliability of an averaged score.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("expected a 2-d items x raters matrix")
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError(f"need at least 2 items and 2 raters, got {n}x{k}")

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    def safe(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    return {
        "n_items": float(n),
        "k_raters": float(k),
        "icc_c1": safe(ms_rows - ms_err, ms_rows + (k - 1) * ms_err),
        "icc_ck": safe(ms_rows - ms_err, ms_rows),
        "icc_a1": safe(
            ms_rows - ms_err,
            ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n,
        ),
        "icc_ak": safe(ms_rows - ms_err, ms_rows + (ms_cols - ms_err) / n),
        "ms_rows": float(ms_rows),
        "ms_cols": float(ms_cols),
        "ms_error": float(ms_err),
    }


# ---------------------------------------------------------------------------
# gates


@dataclass
class GateResult:
    model_name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    validity: pd.DataFrame = field(default_factory=pd.DataFrame)
    surviving_templates: list[str] = field(default_factory=list)
    median_rho: float = float("nan")
    #: reliability on ascending-only scores -- the noise floor of a single rating
    icc_all: dict[str, float] = field(default_factory=dict)
    icc_selection: dict[str, float] = field(default_factory=dict)
    #: the same on polarity-collapsed scores, reported for comparison. A large gap
    #: between the two means the collapse is destroying variance, i.e. the model is
    #: not tracking the descending anchors.
    icc_all_collapsed: dict[str, float] = field(default_factory=dict)
    sigma_between: float = float("nan")
    sigma_between_ascending: float = float("nan")
    sesoi_primary: float = float("nan")
    sesoi_secondary: float = float("nan")
    icc_tripwire_hit: bool = False

    def summary(self) -> str:
        head = "PASS" if self.passed else "HALT"
        lines = [
            f"[{head}] {self.model_name}",
            f"  validity: median rho = {self.median_rho:.3f} "
            f"(threshold {'>=' } gate); surviving templates = {self.surviving_templates}",
            f"  reliability (ascending only): ICC(C,1) over 5 templates = "
            f"{self.icc_all.get('icc_c1', float('nan')):.3f}",
            f"  selection score reliability: ICC(C,3) = {self.icc_selection.get('icc_ck', float('nan')):.3f}, "
            f"ICC(A,3) = {self.icc_selection.get('icc_ak', float('nan')):.3f}",
            f"  same on polarity-collapsed scores: ICC(C,1) = "
            f"{self.icc_all_collapsed.get('icc_c1', float('nan')):.3f}  "
            "(a large gap means the collapse is destroying variance)",
            f"  sigma_between = {self.sigma_between:.3f} rating points (collapsed, "
            f"drives the SESOI)  |  ascending only = {self.sigma_between_ascending:.3f}",
            f"  SESOI (primary, 0.15*sigma) = {self.sesoi_primary:.3f}; "
            f"secondary fixed anchor = {self.sesoi_secondary:.3f}",
        ]
        if self.icc_tripwire_hit:
            lines.append(
                "  TRIPWIRE: ICC(C,1) below threshold -- investigate for an "
                "implementation fault before trusting Pass C"
            )
        for r in self.reasons:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def evaluate_gates(cfg: RunConfig, pass_a: pd.DataFrame) -> GateResult:
    """Apply the preregistered exclusion criteria. See preregistration.md 5."""
    rc = cfg.readout.reversal_constant
    validity = validity_table(pass_a, rc)
    collapsed = collapse_polarity(pass_a, rc)

    ascending = ascending_scores(pass_a)

    all_templates = [c for c in collapsed.columns if c not in ("item_id", "domain")]
    surviving = sorted(
        validity.loc[validity["spearman_rho"] >= cfg.gates.validity_rho_min, "template"]
    )
    median_rho = float(np.median(validity["spearman_rho"])) if len(validity) else float("nan")

    sel_names = [all_templates[i] for i in cfg.pass_a.selection_templates]
    ana_names = [all_templates[i] for i in cfg.pass_a.analysis_templates]

    # Reliability on ascending-only scores: the noise floor of a single rating,
    # uncontaminated by whether the model tracks the descending anchors.
    icc_all = icc_two_way(ascending[all_templates].to_numpy())
    icc_sel = icc_two_way(ascending[sel_names].to_numpy())
    # The same on collapsed scores, for comparison only.
    icc_all_collapsed = icc_two_way(collapsed[all_templates].to_numpy())

    # sigma_between drives the SESOI and is measured on the same collapsed score
    # that supplies the analysis |diff| (preregistration.md 4.4).
    analysis_score = collapsed[ana_names].mean(axis=1).to_numpy(dtype=float)
    sigma = float(np.std(analysis_score, ddof=1))
    sigma_ascending = float(
        np.std(ascending[ana_names].mean(axis=1).to_numpy(dtype=float), ddof=1)
    )

    reasons: list[str] = []
    passed = True

    # 5.1 polarity validity -- the sole categorical exclusion
    if not (median_rho >= cfg.gates.validity_rho_min):
        passed = False
        reasons.append(
            f"EXCLUDED: median polarity validity rho = {median_rho:.3f} < "
            f"{cfg.gates.validity_rho_min}. The model has not demonstrated that it "
            "reads the response scale; this is not evidence about the hypotheses in "
            "either direction."
        )
    if len(surviving) < cfg.gates.validity_min_surviving_templates:
        passed = False
        reasons.append(
            f"EXCLUDED: only {len(surviving)} template(s) reach rho >= "
            f"{cfg.gates.validity_rho_min}, need "
            f"{cfg.gates.validity_min_surviving_templates}."
        )
    elif len(surviving) < len(all_templates):
        reasons.append(
            f"template(s) {sorted(set(all_templates) - set(surviving))} dropped from "
            "this model's battery for failing polarity validity (reported)."
        )

    # 5.2 dynamic range
    if not (sigma >= cfg.gates.sigma_between_min):
        passed = False
        reasons.append(
            f"EXCLUDED: sigma_between = {sigma:.3f} < {cfg.gates.sigma_between_min}. "
            "With no between-item variance there is no |diff| range, so the "
            "difficulty manipulation is undefined rather than null."
        )

    # 5.3 reliability tripwire -- explicitly NOT an exclusion
    tripwire = bool(icc_all["icc_c1"] < cfg.gates.icc_tripwire)
    if tripwire:
        reasons.append(
            f"TRIPWIRE (not an exclusion): ICC(C,1) = {icc_all['icc_c1']:.3f} < "
            f"{cfg.gates.icc_tripwire}. Check for an implementation fault; if none, "
            "the binding criterion is power on the primary contrast at the SESOI."
        )

    return GateResult(
        model_name=cfg.model.name,
        passed=passed,
        reasons=reasons,
        validity=validity,
        surviving_templates=surviving,
        median_rho=median_rho,
        icc_all=icc_all,
        icc_selection=icc_sel,
        icc_all_collapsed=icc_all_collapsed,
        sigma_between=sigma,
        sigma_between_ascending=sigma_ascending,
        sesoi_primary=cfg.analysis.sesoi_sigma_fraction * sigma,
        sesoi_secondary=cfg.analysis.sesoi_raw_secondary,
        icc_tripwire_hit=tripwire,
    )


def item_scores(cfg: RunConfig, pass_a: pd.DataFrame) -> pd.DataFrame:
    """Per-item selection and analysis scores from the disjoint template split."""
    collapsed = collapse_polarity(pass_a, cfg.readout.reversal_constant)
    templates = [c for c in collapsed.columns if c not in ("item_id", "domain")]
    sel = [templates[i] for i in cfg.pass_a.selection_templates]
    ana = [templates[i] for i in cfg.pass_a.analysis_templates]
    return pd.DataFrame(
        {
            "item_id": collapsed["item_id"],
            "domain": collapsed["domain"],
            "score_selection": collapsed[sel].mean(axis=1),
            "score_analysis": collapsed[ana].mean(axis=1),
        }
    )
