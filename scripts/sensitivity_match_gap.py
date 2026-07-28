"""A3.13's pre-specified sensitivity analysis: the primary, both ways.

    python scripts/sensitivity_match_gap.py --config configs/stage0_gemma-2-2b.yaml

A3.13 requires the primary contrast to be reported on the full sample AND restricted to
matched sets within the preregistered tolerance on the ANALYSIS scale. On the completed run
41-60% of matched sets exceed it, so the restriction is not cosmetic.

BOTH FITS USE THE SAME PARAMETERIZATION. Comparing a restricted non-centered fit against
the recorded centered primary would confound A3.13's restriction with A4.1's
reparameterization, and neither number would mean anything. The full-sample fit is redone
here rather than read from the artifact for exactly that reason.

NEITHER FIT IS PRIMARY. A3.13: "the full-sample estimate remains the preregistered primary;
the restricted one is a sensitivity analysis. Promoting whichever is larger after seeing
both is the move A3.6 declined."

Reported numbers must originate on the run machine (12). Run there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import spread_model  # noqa: E402
from src.config import load_config  # noqa: E402
from src.experiments import pass_b as stage_b  # noqa: E402
from src.experiments import pass_c as stage_c  # noqa: E402
from src.provenance import read_parquet  # noqa: E402


def _primary(cfg, trials: pd.DataFrame, sesoi: float, label: str) -> dict:
    design = spread_model.prepare(cfg, trials)
    idata = spread_model.fit(cfg, design, progressbar=False)
    contrasts = spread_model.contrasts(cfg, idata, sesoi)
    row = contrasts[(contrasts["name"] == spread_model.PRIMARY)
                    & (contrasts["term"] == "lambda")].iloc[0]
    conv = spread_model.convergence(cfg, idata)
    bad = conv[~(conv["rhat_ok"] & conv["ess_ok"])]
    return {
        "label": label,
        "n_pairs": int(trials["pair_id"].nunique()),
        "n_rows": int(len(trials)),
        "median": float(row["median"]),
        "hdi_low": float(row["hdi_low"]),
        "hdi_high": float(row["hdi_high"]),
        "p_negative": float(row["p_negative"]),
        "decision": str(row["decision"]),
        "convergence_failures": int(len(bad)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--sampler-cores", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the comparison as JSON")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.artifacts:
        cfg = cfg.model_copy(update={"artifacts_dir": Path(args.artifacts)})
    if args.sampler_cores is not None:
        cfg = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
            update={"sampler_cores": args.sampler_cores})})

    pairs = read_parquet(stage_b.artifact_path(cfg))
    trials = read_parquet(stage_c.artifact_path(cfg))
    sigma_item = float(pairs["sigma_item"].iloc[0])
    sesoi = cfg.analysis.sesoi_sigma_fraction * sigma_item

    ex = stage_b.match_gap_exclusions(cfg, pairs, sigma_item)
    dropped = set(ex["excluded_matched_sets"])
    keep_pairs = set(pairs[~pairs["matched_set"].isin(dropped)]["pair_id"])
    restricted = trials[trials["pair_id"].isin(keep_pairs)]

    print(f"{cfg.model.name}  sigma_item {sigma_item:.4f}  SESOI {sesoi:.4f}")
    print(f"  tolerance {ex['tolerance']:.4f} on the analysis scale")
    print(f"  matched sets over tolerance: {ex['n_over_on_analysis']}/"
          f"{ex['n_matched_sets']} ({ex['frac_over_on_analysis']:.1%})")
    print(f"  pairs kept: {len(keep_pairs)}/{pairs['pair_id'].nunique()}\n")

    if not len(restricted):
        print("REFUSING: the restriction removes every pair.")
        return 1
    if restricted["diff_analysis"].nunique() < 2:
        print("REFUSING: the restricted subset has no |diff| variance left.")
        return 1

    out = [_primary(cfg, trials, sesoi, "full sample (PRIMARY)"),
           _primary(cfg, restricted, sesoi, "within tolerance (sensitivity)")]

    print(f"{'':<34}{'n pairs':>8}{'median':>9}{'hdi_low':>9}{'hdi_high':>10}"
          f"{'P(<0)':>8}  decision")
    for r in out:
        print(f"{r['label']:<34}{r['n_pairs']:>8}{r['median']:>+9.4f}"
              f"{r['hdi_low']:>+9.4f}{r['hdi_high']:>+10.4f}{r['p_negative']:>8.3f}"
              f"  {r['decision']}")
    shift = out[1]["median"] - out[0]["median"]
    print(f"\n  shift under restriction: {shift:+.4f} "
          f"({shift / sesoi:+.2f} x SESOI)")
    print("  Neither is promoted. The full sample remains the preregistered primary "
          "(A3.13).")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": cfg.model.name, "sesoi": sesoi, "match_gap": ex, "fits": out},
            indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
