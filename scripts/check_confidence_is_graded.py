"""A4.6: is `p_item1` preference STRENGTH, or a decoding artifact?

    python scripts/check_confidence_is_graded.py --config configs/stage0_gemma-2-2b.yaml

Every graded fit in the pilot treats the model's stated confidence as a reading of how much
it prefers one item. The three likelihoods differ only in tail handling; NONE tests that
premise. If confidence saturates near 0 and 1 for reasons unrelated to preference, the
graded DV is precise about the wrong quantity -- which is worse than being noisy about the
right one, because precision lends it credibility.

THE TEST. At `pre`, before any manipulation exists, |logit(p_item1)| should rise with
`|diff|`: the theta gap the Bradley-Terry instrument measured INDEPENDENTLY, on different
prompts, in a different pass. Two measurements of the same latent preference should track.

  tracks       -> confidence is graded, and the pilot's gain is a real gain
  flat         -> confidence carries no preference information; the gain is illusory
  saturated    -> confidence is near-binary anyway and there was little to recover

Uses PRE rows only, so it carries no information about H1: at pre there is no condition and
it cannot distinguish chose from yoked.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--bins", type=int, default=6)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.artifacts:
        cfg = cfg.model_copy(update={"artifacts_dir": Path(args.artifacts)})

    trials = read_parquet(stage_c.artifact_path(cfg))
    if "p_item1" not in trials:
        print("trials carry no p_item1; re-run Pass C (A4.5).")
        return 1

    pre = trials[(trials["condition"] == PRE_SENTINEL) | (trials["timepoint"] == "pre")]
    pre = pre.drop_duplicates(["pair_id", "template", "option_order"])
    p = pre["p_item1"].to_numpy(dtype=float)
    gap = pre["diff_analysis"].to_numpy(dtype=float)
    strength = np.abs(np.log(p / (1.0 - p)))          # confidence, direction removed

    print(f"{cfg.model.name}   {len(pre)} pre measurements")
    print(f"  |logit p| : median {np.median(strength):.2f}  "
          f"IQR [{np.percentile(strength, 25):.2f}, {np.percentile(strength, 75):.2f}]  "
          f"max {strength.max():.2f}")
    sat = float((np.minimum(p, 1 - p) < 0.01).mean())
    print(f"  saturated (p<0.01 or >0.99): {sat:.1%}")

    order = np.argsort(gap)
    edges = np.array_split(order, args.bins)
    print(f"\n  {'|diff| bin':<22}{'n':>6}{'median |logit p|':>19}")
    med = []
    for b in edges:
        med.append(float(np.median(strength[b])))
        print(f"  {gap[b].min():.2f} - {gap[b].max():<14.2f}{len(b):>6}{med[-1]:>19.2f}")

    from scipy import stats

    rho, pval = stats.spearmanr(gap, strength)
    print(f"\n  Spearman(|diff|, |logit p|) = {rho:+.3f}  (p = {pval:.2g})")
    rise = med[-1] / med[0] if med[0] > 0 else float("inf")
    print(f"  median confidence, widest bin / narrowest bin = {rise:.2f}x")

    print("\n  READING (applying this is a judgement; the numbers are the inputs):")
    print("    rho >= ~0.3 and a clear rise  -> graded; the pilot's gain is real")
    print("    rho ~ 0                       -> confidence is unrelated to preference")
    print("    saturation > ~50%             -> near-binary already; little was recovered")
    print("\n  PRE rows only, so this says nothing about H1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
