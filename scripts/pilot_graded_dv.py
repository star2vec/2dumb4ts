"""A4.5 pilot: how much precision does binarising the DV cost?

    python scripts/pilot_graded_dv.py --config configs/stage0_gemma-2-2b.yaml --sampler-cores 1

Fits the primary contrast FOUR ways on THE SAME trials and reports the spread of standard
errors. That spread decides whether Stage 0-bis is worth running.

    binary            Bernoulli on item1_wins       -- the preregistered Stage 0 DV
    graded (Normal)   Normal on logit(p_item1)      -- the candidate
    graded (StudentT) robust to the tails
    graded (Beta)     on p_item1 directly, no logit at all

WHY FOUR. p_item1 reaches ~1e-06 and ~0.999998, i.e. logits near +/-13. Both logit-scale
fits threw "overflow encountered in dot" on the first run, and both diverged. A Beta
likelihood has the right support for a probability, so nothing has to be pushed through a
logit and nothing overflows -- it is the principled answer to that warning rather than a
tuning knob. Three likelihoods with different tail behaviour agreeing is a much stronger
claim than two.

THE SCRIPT DOES NOT RENDER A VERDICT. The first version printed "consistent, the gain is
real" from a threshold applied to a percentage computed with the flattering denominator --
in the one comparison written to catch flattery. It now prints the inputs to the
pre-committed criteria and leaves the judgement to a person.

THIS IS NOT AN H1 RE-TEST. The point estimates are printed only so a reader can see the two
DVs agree on the sign; Stage 0's reported result stands, and the continuous DV is a
different primary with no preregistration yet (A4.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import spread_model  # noqa: E402
from src.config import load_config  # noqa: E402
from src.provenance import find_artifact, read_parquet  # noqa: E402


def _contrast(idata) -> np.ndarray:
    return (spread_model._draws(idata, "lambda", "chose")
            - spread_model._draws(idata, "lambda", "yoked"))


def _summarise(idata, label: str) -> dict:
    """Everything the pre-committed distrust criteria need, in the output and the JSON.

    The first version reported neither R-hat nor divergences, so two of the three criteria
    could not be checked from the output at all. The run machine had to read them out of
    the sampler's console noise.
    """
    import arviz as az

    d = _contrast(idata)
    lo, hi = np.percentile(d, [2.5, 97.5])
    summ = az.summary(idata, var_names=["gamma", "lambda", "beta"])
    diverging = int(idata.sample_stats["diverging"].sum()) \
        if "diverging" in getattr(idata, "sample_stats", {}) else -1
    n_draws = int(np.prod(idata.posterior["lambda"].shape[:2]))
    return {"label": label, "median": float(np.median(d)), "se": float(d.std(ddof=1)),
            "hdi_low": float(lo), "hdi_high": float(hi),
            "p_negative": float((d < 0).mean()),
            "max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min()),
            "divergences": diverging, "n_draws": n_draws,
            "divergence_rate": diverging / n_draws if diverging >= 0 else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--trials", default=None,
                    help="explicit trials parquet; default is the newest "
                         "that carries the columns this script needs")
    ap.add_argument("--sampler-cores", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.artifacts:
        cfg = cfg.model_copy(update={"artifacts_dir": Path(args.artifacts)})
    if args.sampler_cores is not None:
        cfg = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
            update={"sampler_cores": args.sampler_cores})})

    try:
        path = find_artifact(cfg.artifact_dir("pass_c").parent, "trials_*.parquet",
                             requires=("p_item1", "item1_wins"), explicit=args.trials)
    except FileNotFoundError as exc:
        print(f"{exc}\nRe-run Pass C: the graded readout was discarded when the existing "
              "trials were collected (A4.5).")
        return 1
    trials = read_parquet(path)
    print(f"  trials: {path.name}")

    design = spread_model.prepare(cfg, trials)
    p = trials["p_item1"].to_numpy(dtype=float)
    print(f"{cfg.model.name}   {len(trials)} rows   p_item1 in "
          f"[{p.min():.2e}, {1 - (1 - p.max()):.6f}]   distinct {len(np.unique(p))}")
    agree = ((p > 0.5) == trials["item1_wins"].to_numpy()).all()
    print(f"graded and binary agree on the winner: {bool(agree)}")
    if not agree:
        print("REFUSING: the two DVs disagree on some trial; the comparison is void.")
        return 1

    fits = [_summarise(spread_model.fit(cfg, design), "binary (Bernoulli)")]
    for fam, label in (("normal", "graded (Normal)"), ("studentt", "graded (StudentT)"),
                       ("beta", "graded (Beta)")):
        fits.append(_summarise(
            spread_model.fit_graded(cfg, design, family=fam), label))

    base = fits[0]["se"]
    print(f"\n{'fit':<24}{'median':>10}{'SE':>9}{'ratio':>8}{'max Rhat':>10}"
          f"{'min ESS':>9}{'diverg':>8}")
    for r in fits:
        print(f"{r['label']:<24}{r['median']:>+10.4f}{r['se']:>9.4f}"
              f"{base / r['se']:>7.2f}x{r['max_rhat']:>10.4f}{r['min_ess_bulk']:>9.0f}"
              f"{r['divergences']:>6} ({r['divergence_rate']:.2%})")

    graded = fits[1:]
    ses = [r["se"] for r in graded]
    # Spread as a RATIO of largest to smallest. The first version divided by the LARGER SE,
    # which always yields the smaller-looking percentage -- the flattering denominator, in
    # the one comparison written to catch flattery. Caught by the run machine.
    spread = max(ses) / min(ses)
    print(f"\n  precision gain spans {base / max(ses):.2f}x to {base / min(ses):.2f}x "
          f"across {len(graded)} likelihoods")
    print(f"  widest family disagreement: {spread:.3f}x "
          f"({(spread - 1) * 100:.1f}% on the tighter SE)")
    worst_rhat = max(r["max_rhat"] for r in graded)
    worst_div = max(r["divergence_rate"] for r in graded)
    print(f"  worst graded Rhat {worst_rhat:.4f} (need <= 1.01); "
          f"worst divergence rate {worst_div:.2%}")
    print("\n  The verdict is NOT computed here. These are the inputs to the "
          "pre-committed criteria; applying them is a judgement, not an assertion.")
    print("\n  Not an H1 re-test. Stage 0's reported result stands (A4.5).")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": cfg.model.name, "n_rows": int(len(trials)), "fits": fits},
            indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
