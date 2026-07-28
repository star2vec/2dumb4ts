"""Is the model's confidence PREFERENCE, or POSITION, or a renormalisation artifact?

    python scripts/decompose_confidence.py --config configs/stage0_gemma-2-2b.yaml

A4.7 found that stated confidence barely tracks measured preference (rho = 0.093) and
concluded the model answers hard comparisons at 97.7% confidence regardless. That is a claim
about the MODEL. Before it can be believed it has to survive three ways it could instead be
a claim about the MEASUREMENT.

THE DECOMPOSITION. Every pair is asked in both option orders, and the readout is

    logit p(item1) = d + beta * s        s = +1 when item1 is in slot 1, -1 otherwise

so averaging and differencing the two orders separates them EXACTLY:

    d    = (logit_order0 + logit_order1) / 2      preference, position cancelled
    beta = (logit_order0 - logit_order1) / 2      position, preference cancelled

A4.7 correlated |diff| against the RAW readout, which contains both. If position dominates,
the raw readout is large and flat regardless of preference -- exactly the reported pattern --
and the finding would be an artifact of not separating them. `d` is the quantity A4.7 should
have used.

THREE CHECKS, in the order that would kill the finding fastest:

  1. POSITION. Does |d| track |diff| once position is removed? If yes, A4.7 measured
     position bias and the headline is wrong.
  2. RENORMALISATION. `read_choice` renormalises over two label tokens only. If the model
     puts most of its mass elsewhere, two tiny numbers can renormalise to an extreme ratio.
     If confidence rises as readout mass FALLS, the saturation is manufactured by that
     division rather than stated by the model.
  3. FLIPPING. If the model's preferred item changes with option order, its "confidence"
     is not about the items at all.

Uses PRE rows only -- before any manipulation exists, so nothing here bears on H1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.spread_model import PRE_SENTINEL  # noqa: E402
from src.config import load_config  # noqa: E402
from src.experiments import pass_c as stage_c  # noqa: E402
from src.provenance import read_parquet  # noqa: E402


def _bins(x, y, n=6, label="x"):
    order = np.argsort(x)
    out = []
    for b in np.array_split(order, n):
        out.append((x[b].min(), x[b].max(), len(b), float(np.median(y[b]))))
    return out


def _self_check() -> None:
    """NEGATIVE CONTROL. Recover a known (d, beta) before trusting any real output.

    Fifth instance of the vacuous-match class if skipped: a decomposition that silently
    returns the wrong component would look exactly like a finding.
    """
    rng = np.random.default_rng(0)
    d_true = rng.normal(0, 5.8, 4000)          # gemma's implied sd(d)
    beta_true = 1.33
    l0, l1 = d_true + beta_true, d_true - beta_true
    d_hat, b_hat = (l0 + l1) / 2, (l0 - l1) / 2
    assert np.allclose(d_hat, d_true), "the decomposition does not recover d"
    assert np.allclose(b_hat, beta_true), "the decomposition does not recover beta"

    # ...and it must SEPARATE them: a correlation planted in d must survive, and beta
    # must not leak into it.
    from scipy import stats

    gap = np.abs(d_true) + rng.normal(0, 2, len(d_true))
    r_true = stats.spearmanr(gap, np.abs(d_true))[0]
    r_hat = stats.spearmanr(gap, np.abs(d_hat))[0]
    assert abs(r_true - r_hat) < 1e-9, "a planted correlation did not survive the split"
    print(f"  self-check: d and beta recovered exactly; planted rho {r_hat:+.3f} preserved\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args(argv)

    from scipy import stats

    _self_check()
    cfg = load_config(args.config)
    if args.artifacts:
        cfg = cfg.model_copy(update={"artifacts_dir": Path(args.artifacts)})
    trials = read_parquet(stage_c.artifact_path(cfg))
    if "p_item1" not in trials:
        print("trials carry no p_item1; re-run Pass C (A4.5).")
        return 1

    pre = trials[(trials["condition"] == PRE_SENTINEL) | (trials["timepoint"] == "pre")]
    pre = pre.drop_duplicates(["pair_id", "template", "option_order"]).copy()
    pre["logit"] = np.log(pre["p_item1"] / (1.0 - pre["p_item1"]))

    wide = pre.pivot_table(index=["pair_id", "template"], columns="option_order",
                           values=["logit", "readout_mass", "diff_analysis"])
    if not {0, 1} <= set(wide["logit"].columns):
        print("both option orders are needed to separate position from preference.")
        return 1
    # Both orders must exist for every cell or `d` is not computable and the script would
    # silently run on whatever subset happens to be complete.
    complete = wide["logit"].notna().all(axis=1)
    if not complete.all():
        print(f"  WARNING: {int((~complete).sum())} of {len(wide)} cells lack both orders; "
              "dropping them. d is undefined without the pair.")
        wide = wide[complete]

    l0 = wide["logit"][0].to_numpy()
    l1 = wide["logit"][1].to_numpy()
    d = (l0 + l1) / 2.0                       # preference, position cancelled
    b = (l0 - l1) / 2.0                       # position, preference cancelled
    gap = wide["diff_analysis"][0].to_numpy()
    mass = wide["readout_mass"].mean(axis=1).to_numpy()
    raw = np.abs(pre["logit"].to_numpy())

    print(f"{cfg.model.name}   {len(d)} (pair, template) cells, both orders\n")
    print("  component            median |.|      share of |raw readout|")
    print(f"  preference  d        {np.median(np.abs(d)):>9.2f}"
          f"{np.median(np.abs(d)) / np.median(raw):>22.0%}")
    print(f"  position    beta     {np.median(np.abs(b)):>9.2f}"
          f"{np.median(np.abs(b)) / np.median(raw):>22.0%}")
    print(f"  raw readout          {np.median(raw):>9.2f}")

    print("\n--- CHECK 1: does preference track the measured gap once position is out? ---")
    raw_cell = (np.abs(l0) + np.abs(l1)) / 2.0   # what A4.7 correlated, per cell
    r_raw, _ = stats.spearmanr(gap, raw_cell)
    r_d, p_d = stats.spearmanr(gap, np.abs(d))
    print(f"  Spearman(|diff|, RAW confidence)          {r_raw:+.3f}   (A4.7 measured +0.093)")
    print(f"  Spearman(|diff|, |d|) position removed    {r_d:+.3f}   (p = {p_d:.2g})")
    print(f"\n  {'|diff| bin':<24}{'n':>6}{'median |d|':>13}")
    for lo, hi, n, med in _bins(gap, np.abs(d)):
        print(f"  {lo:.2f} - {hi:<16.2f}{n:>6}{med:>13.2f}")

    print("\n--- CHECK 2: is the saturation made by the two-token renormalisation? ---")
    r_m, _ = stats.spearmanr(mass, np.abs(d))
    print(f"  Spearman(readout mass, |d|) = {r_m:+.3f}")
    print("  (a NEGATIVE correlation means low-mass rows are the confident ones, i.e. the "
          "division is manufacturing it)")
    print(f"  readout mass: median {np.median(mass):.3f}, "
          f"10th pct {np.percentile(mass, 10):.3f}")

    print("\n--- CHECK 3: does the preferred item survive reversing the options? ---")
    flip = float((np.sign(l0) != np.sign(l1)).mean())
    print(f"  preference flips with option order: {flip:.1%}")
    print("  (0% = position irrelevant; 50% = the answer is entirely position)")

    print("\n  READING:")
    print("    |d| tracks |diff| (rho >= ~0.3)  -> A4.7 measured POSITION, headline is wrong")
    print("    |d| still flat                   -> the finding survives; it is the model")
    print("    mass strongly negative           -> renormalisation artifact")
    print("    flip rate near 50%               -> the readout is not about the items")
    print("\n  PRE rows only; nothing here bears on H1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
