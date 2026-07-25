"""Pass A ladder summary -- one table across every model that has run.

Answers the question the Ada run exists to answer: does the rating instrument
discriminate at full sample, and is the polarity failure universal?

The two columns that decide it:
  rating_range / rating_iqr  -- is everything squashed into "pretty good"?
  median_rho                 -- does the model track a reversed scale at all?

    python scripts/summarize_pass_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis.reliability import ascending_scores, evaluate_gates  # noqa: E402
from src.config import load_config  # noqa: E402
from src.experiments import pass_a as stage_a  # noqa: E402
from src.provenance import assert_poolable, read_parquet  # noqa: E402

LADDER = [
    "configs/stage0_qwen2.5-0.5b.yaml",
    "configs/stage0_qwen2.5-1.5b.yaml",
    "configs/stage0_gemma-2-2b.yaml",
    "configs/stage0_qwen2.5-3b.yaml",
    "configs/stage0_llama-3.2-3b.yaml",
]


def main() -> int:
    rows, frames = [], []
    for path in LADDER:
        cfg = load_config(REPO / path).model_copy(
            update={"artifacts_dir": REPO / "artifacts"}
        )
        art = stage_a.artifact_path(cfg)
        if not art.exists():
            rows.append({"model": cfg.model.name, "status": "not run"})
            continue

        frame = read_parquet(art)
        frames.append(frame)
        gate = evaluate_gates(cfg, frame)

        asc = ascending_scores(frame)
        tcols = [c for c in asc.columns if c not in ("item_id", "domain")]
        vals = asc[tcols].to_numpy(dtype=float)
        q1, q3 = np.percentile(vals, [25, 75])

        rows.append(
            {
                "model": cfg.model.name,
                "status": "gate pass" if gate.passed else "EXCLUDED",
                "device": frame["prov_device"].iloc[0],
                "n_ratings": len(frame),
                "rating_min": round(float(vals.min()), 2),
                "rating_max": round(float(vals.max()), 2),
                "rating_iqr": round(float(q3 - q1), 2),
                "median_rho": round(gate.median_rho, 3),
                "icc_c1_asc": round(gate.icc_all.get("icc_c1", np.nan), 3),
                "icc_c3_sel": round(gate.icc_selection.get("icc_ck", np.nan), 3),
                "sigma_asc": round(gate.sigma_between_ascending, 3),
                "sigma_coll": round(gate.sigma_between, 3),
                "digit_mass_p05": round(float(frame["digit_mass"].quantile(0.05)), 4),
            }
        )

    table = pd.DataFrame(rows)
    print("\n== Pass A ladder ==\n")
    print(table.to_string(index=False))

    if len(frames) > 1:
        # Hard guard: these must not be mixed across devices or dtypes.
        assert_poolable(frames, context="Pass A ladder summary")
        print("\npooling guard: all artifacts share device and dtype")

    done = table[table["status"] != "not run"]
    if len(done):
        print("\n-- reading --")
        squashed = done[done["rating_iqr"] < 1.0]
        if len(squashed):
            print(f"  range compression in {len(squashed)}/{len(done)} model(s) "
                  f"(ascending IQR < 1.0 point): {list(squashed['model'])}")
            print("    -> the latent preference ordering exists but the absolute rating "
                  "question flattens it; |diff| has no room to vary.")
        blind = done[done["median_rho"] < 0.6]
        if len(blind):
            print(f"  polarity failure in {len(blind)}/{len(done)} model(s): "
                  f"{list(blind['model'])}")
        if len(blind) == len(done):
            print("    -> universal, so it is a property of the elicitation rather than "
                  "of any one model or family.")
        excluded = done[done["status"] == "EXCLUDED"]
        print(f"\n  {len(done) - len(excluded)}/{len(done)} model(s) clear the "
              "preregistered gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
