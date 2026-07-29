"""Replay the Pass C PRE prompts and save hidden states. See preregistration_probe.md.

    python scripts/collect_activations.py --config configs/stage0_gemma-2-2b.yaml \
        --batch-size 4

Specified in `preregistration_stage1.md` §3 and recorded as unbuilt in A3.5. This is it.

WHAT IT DOES NOT DO. It does not re-run Pass C, touch the trials, or read any post row. It
replays the 2,000 PRE prompts -- which precede every manipulation -- and captures the hidden
state at the single readout position, every layer.

THE DIGEST ASSERTION IS THE POINT. "Identical prompts" is the entire basis for attaching
these activations to those trials. `pass_c` now records a digest of its rendered pre prompts
in cell order; this script re-renders through the SAME builder functions and refuses to
write anything unless the digests match. Without that, a silent drift in a template, a
tokenizer revision or the chat template would produce activations for a different set of
prompts than the DV was read from, and every probe result would be about the wrong thing.

MEMORY. `output_hidden_states` materialises [batch, seq, hidden] for every layer during the
forward, not just the position kept. On an 8 GB card holding ~5 GB of weights that is the
binding constraint, so --batch-size defaults LOW here rather than inheriting the config's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.experiments import pass_b as stage_b  # noqa: E402
from src.experiments import pass_c as stage_c  # noqa: E402
from src.provenance import capture, read_parquet  # noqa: E402
from src.stimuli.build import load_items, load_templates, pre_dv_messages  # noqa: E402


def _cells(cfg, pairs):
    """The PRE cells, in exactly the order `pass_c` builds them.

    Rebuilt through the same nested loop rather than read back from the trials frame: the
    digest must be reproduced from the BUILDERS, or it only proves the file is unchanged.
    """
    framed = {i.id: i.framed for i in load_items(cfg)}
    templates = load_templates(cfg)
    out = []
    for p in pairs.to_dict("records"):
        for t in templates:
            for order in range(cfg.pass_c.n_option_orders):
                s1, s2 = stage_c._displayed(p["item1_id"], p["item2_id"], order)
                out.append({
                    "pair_id": p["pair_id"], "template": t.id, "option_order": order,
                    "item1_id": p["item1_id"], "item2_id": p["item2_id"],
                    "slot1_item_id": s1, "diff_analysis": p["diff_analysis"],
                    "difficulty": p["difficulty"],
                    "_msgs": pre_dv_messages(t, framed[s1], framed[s2], cfg),
                })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="LOW by default; hidden states are the memory constraint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.artifacts:
        cfg = cfg.model_copy(update={"artifacts_dir": Path(args.artifacts)})

    trials_path = stage_c.artifact_path(cfg)
    if not trials_path.exists():
        print(f"no trials at {trials_path}\nRun Pass C first.")
        return 1
    trials = read_parquet(trials_path)
    if "pre_prompt_digest" not in trials.columns:
        print(f"{trials_path.name} carries no pre_prompt_digest. It predates the digest "
              "and the replay cannot be verified against it -- re-run Pass C.")
        return 1
    recorded = str(trials["pre_prompt_digest"].iloc[0])

    pairs = read_parquet(stage_b.artifact_path(cfg))
    cells = _cells(cfg, pairs)
    print(f"{cfg.model.name}   {len(cells)} pre cells rebuilt from the builders")

    from src.models.runner import load_runner

    runner = load_runner(cfg)
    runner.batch_size = args.batch_size
    prompts = [runner.render(c["_msgs"]) for c in cells]

    digest = hashlib.sha256("\x00".join(prompts).encode("utf-8")).hexdigest()[:16]
    print(f"  recorded digest {recorded}\n  replayed digest {digest}")
    if digest != recorded:
        print("\nREFUSING TO WRITE. The replayed prompts are not the prompts Pass C read.\n"
              "Something upstream moved -- a template, the chat template, or the tokenizer "
              "revision. Activations collected now would belong to different prompts than "
              "the DV, and every probe result would be about the wrong thing.")
        return 1
    print("  digests match; these are the prompts the DV was read from")

    hs = runner.last_hidden_states(prompts)          # [n, n_layers+1, hidden]
    # float16 on disk: the activations are bf16 in the model and probing is done in
    # float32, so 16 bits is what was measured and 32 would store invented precision.
    arr = hs.to(torch.float16).numpy()
    print(f"  hidden states {tuple(arr.shape)}  "
          f"{arr.nbytes / 1024**2:.0f} MB  dtype {arr.dtype}")

    out = Path(args.out) if args.out else (
        cfg.artifact_dir("activations") / f"pre_{cfg.hash('pass_c')}-{digest}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = [{k: v for k, v in c.items() if k != "_msgs"} for c in cells]
    prov = capture(cfg)
    np.savez_compressed(
        out, activations=arr,
        pair_id=np.array([m["pair_id"] for m in meta]),
        template=np.array([m["template"] for m in meta]),
        option_order=np.array([m["option_order"] for m in meta]),
        item1_id=np.array([m["item1_id"] for m in meta]),
        item2_id=np.array([m["item2_id"] for m in meta]),
        slot1_item_id=np.array([m["slot1_item_id"] for m in meta]),
        diff_analysis=np.array([m["diff_analysis"] for m in meta], dtype=float),
        difficulty=np.array([m["difficulty"] for m in meta]),
        prompt_digest=np.array([digest]),
        provenance=np.array([json.dumps(prov.model_dump(), default=str)]),
    )
    print(f"\nwrote {out}  ({out.stat().st_size / 1024**2:.0f} MB on disk)")
    print("  PRE rows only -- nothing here bears on H1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
