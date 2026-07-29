"""Run the preregistered probes over every layer. See `preregistration_probe.md`.

    python scripts/run_probe.py --activations artifacts/activations/<model>/pre_*.npz

Two probes on IDENTICAL activations, probe class and item-held-out folds:

    control    sign(theta_i - theta_j)   which item is preferred   -- expected to decode
    question   |theta_i - theta_j|       by how much               -- the study

THE LAYER IS CHOSEN BY THE CONTROL, NOT BY THE QUESTION (prereg §6). Picking the layer where
magnitude happens to look best would be selecting on the outcome; the control says where the
representation is legible at all, and the question is read there.

NO VERDICT IS COMPUTED. The script prints the inputs to §6's outcome table. Applying it is a
judgement, and a script that renders one is how a threshold gets quietly chosen to fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import probe  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--layers", default=None,
                    help="comma-separated subset; default ALL, which is what the prereg says")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    d = np.load(args.activations, allow_pickle=False)
    acts = d["activations"]                       # [n, n_layers+1, hidden]
    pair_id, i1, i2 = d["pair_id"], d["item1_id"], d["item2_id"]
    gap = d["diff_analysis"].astype(float)

    n, n_layers, hidden = acts.shape
    print(f"{Path(args.activations).name}")
    print(f"  {n} rows, {len(set(pair_id))} pairs, {n_layers} layers, {hidden} dims")
    print(f"  prompt digest {d['prompt_digest'][0]}")

    # |diff| is unsigned in the artifact; the control needs the SIGN, which is recoverable
    # because item1/item2 are canonical by sorted id and theta is not stored per item here.
    # It is reconstructed from the pair's own ordering below if present, else the control
    # cannot be run and the study is not interpretable (prereg §5).
    if "theta_item1" in d and "theta_item2" in d:
        sign = np.sign(d["theta_item1"].astype(float) - d["theta_item2"].astype(float))
    else:
        print("\nREFUSING: the activations carry no signed theta, so the POSITIVE CONTROL "
              "cannot be run. A magnitude result without it is uninterpretable (prereg §5).\n"
              "Re-collect with theta_item1/theta_item2 recorded.")
        return 1

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layers)))

    rows = []
    print(f"\n  {'layer':>6}{'sign rho':>11}{'|diff| rho':>13}{'|diff| 95% CI':>22}")
    for layer in layers:
        X = acts[:, layer, :]
        c = probe.run_probe(X, sign, pair_id, i1, i2, target="sign", layer=layer,
                            n_boot=args.n_boot)
        m = probe.run_probe(X, gap, pair_id, i1, i2, target="|diff|", layer=layer,
                            n_boot=args.n_boot)
        rows.append({"layer": layer, "sign": c.as_row(), "magnitude": m.as_row()})
        ci = f"[{m.ci_low:+.3f}, {m.ci_high:+.3f}]"
        print(f"  {layer:>6}{c.rho_pair:>+11.3f}{m.rho_pair:>+13.3f}{ci:>22}")

    # Layer selected by the CONTROL, per §6.
    best = max(rows, key=lambda r: r["sign"]["rho_pair"])
    L = best["layer"]
    X = acts[:, L, :]
    within = probe.within_item_control(X, gap, pair_id, i1, i2, layer=L,
                                       n_boot=args.n_boot)

    print(f"\n  === read at layer {L}, chosen by the CONTROL probe ===")
    print(f"  control  sign(diff)   rho {best['sign']['rho_pair']:+.3f}  "
          f"[{best['sign']['ci_low']:+.3f}, {best['sign']['ci_high']:+.3f}]")
    print(f"  question |diff|       rho {best['magnitude']['rho_pair']:+.3f}  "
          f"[{best['magnitude']['ci_low']:+.3f}, {best['magnitude']['ci_high']:+.3f}]")
    print(f"  |diff|, WITHIN-item   rho {within.rho_pair:+.3f}  "
          f"[{within.ci_low:+.3f}, {within.ci_high:+.3f}]   <- identity check")
    print(f"  effective n = {best['magnitude']['n_pairs']} pairs "
          f"({best['magnitude']['n_rows']} rows)")

    print("\n  §6 inputs (the reading is a judgement, not computed here):")
    print("    sign decodes AND magnitude comparable   -> (E) elicitation; change the readout")
    print("    sign decodes, magnitude does not        -> (R) representation; order not magnitude")
    print("    neither decodes                         -> UNINTERPRETABLE, failed measurement")
    print("    magnitude only WITHIN item              -> item identity; negative for both")
    print("\n  PRE rows only; nothing here bears on H1.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"activations": Path(args.activations).name,
             "prompt_digest": str(d["prompt_digest"][0]),
             "best_layer_by_control": L, "within_item": within.as_row(),
             "layers": rows}, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
