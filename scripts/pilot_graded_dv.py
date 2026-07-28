"""A4.5 pilot: how much precision does binarising the DV cost?

    python scripts/pilot_graded_dv.py --config configs/stage0_gemma-2-2b.yaml --sampler-cores 1

Fits the primary contrast three ways on THE SAME trials and reports the ratio of standard
errors. That ratio is the only output that matters; it decides whether Stage 0-bis is worth
running.

    binary            Bernoulli on item1_wins       -- the preregistered Stage 0 DV
    graded (normal)   Normal on logit(p_item1)      -- the candidate
    graded (t)        StudentT on logit(p_item1)    -- the honesty check

WHY THREE AND NOT TWO. p_item1 reaches ~1e-06 and ~0.99999, i.e. logits near +/-13. A
Gaussian with constant sigma is misspecified there and will report intervals that are too
NARROW, which looks identical to a precision win. If the Student-t agrees with the Gaussian
the gain is real; if the Gaussian is much tighter, part of it is the tails being modelled
badly. Reporting only the Gaussian would be the most flattering and least honest choice.

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
from src.experiments import pass_c as stage_c  # noqa: E402
from src.provenance import read_parquet  # noqa: E402


def _contrast(idata) -> np.ndarray:
    return (spread_model._draws(idata, "lambda", "chose")
            - spread_model._draws(idata, "lambda", "yoked"))


def _summarise(idata, label: str) -> dict:
    d = _contrast(idata)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"label": label, "median": float(np.median(d)), "se": float(d.std(ddof=1)),
            "hdi_low": float(lo), "hdi_high": float(hi),
            "p_negative": float((d < 0).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--sampler-cores", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.artifacts:
        cfg = cfg.model_copy(update={"artifacts_dir": Path(args.artifacts)})
    if args.sampler_cores is not None:
        cfg = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
            update={"sampler_cores": args.sampler_cores})})

    path = stage_c.artifact_path(cfg)
    if not path.exists():
        print(f"no trials at {path}\nRe-run Pass C: the graded readout was discarded when "
              "the existing trials were collected (A4.5).")
        return 1
    trials = read_parquet(path)
    if "p_item1" not in trials.columns:
        print(f"{path.name} carries no p_item1; it predates A4.5. Re-run Pass C.")
        return 1

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
    for fam, label in (("normal", "graded (Normal)"), ("studentt", "graded (StudentT)")):
        fits.append(_summarise(
            spread_model.fit_graded(cfg, design, family=fam), label))

    base = fits[0]["se"]
    print(f"\n{'fit':<24}{'median':>10}{'SE':>9}{'95% interval':>22}{'SE ratio':>10}")
    for r in fits:
        span = f"[{r['hdi_low']:+.3f}, {r['hdi_high']:+.3f}]"
        print(f"{r['label']:<24}{r['median']:>+10.4f}{r['se']:>9.4f}{span:>22}"
              f"{base / r['se']:>9.2f}x")

    normal, student = fits[1]["se"], fits[2]["se"]
    print(f"\n  precision gain, Normal:   {base / normal:.2f}x")
    print(f"  precision gain, StudentT: {base / student:.2f}x")
    disagree = abs(normal - student) / max(normal, student)
    print(f"  families differ by {disagree:.1%} on SE -- "
          + ("consistent, the gain is real" if disagree < 0.25
             else "LARGE: part of the Gaussian's gain is tail misspecification"))
    print("\n  Not an H1 re-test. Stage 0's reported result stands (A4.5).")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": cfg.model.name, "n_rows": int(len(trials)), "fits": fits},
            indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
