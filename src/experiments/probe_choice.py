"""DIAGNOSTIC PROBE -- does the model's choice track content, or only position?

Not part of the preregistered pipeline. It writes to its own artifact tree, reads
no Pass A/B/C output, and reuses none of their code paths beyond prompt assembly.

WHY THIS EXISTS. In the Pass C wiring check, three of five templates picked option
A on 100% of trials and the flip rate across counterbalanced option order was
0.825. If that is real, `chose` encodes position rather than preference, and the
agency manipulation the whole design rests on is not a manipulation at all.

THE HYPOTHESIS. All three Stage 0 failures may be one failure. Ascending ratings
for the 3B span 7.4-8.95: everything is "pretty good". A model that does not
discriminate among these items would show exactly what we saw -- no need to track
scale polarity (just answer high), no preference to drive a choice (fall back on
position), and no genuine difficulty variation (every pair is a near-tie).

THE DISCRIMINATING TEST. Give it a choice that is obviously not close.

    picks correctly on obvious pairs, position-bound on close ones
        -> the choice mechanism works; the ITEM POOL lacks range. Good outcome:
           a stimulus problem, not a paradigm problem.
    picks position even on obvious pairs
        -> position bias is fundamental to the elicitation, and free choice needs
           a different mechanism or the paradigm is dead.

Four elicitations are crossed with that, including one CONTENT-ADDRESSED variant
where the model names the option and the readout is the first token of the item
label. That removes position as an available answer, so it is the most plausible
fix. It stays within the one-forward-pass / one-token-position constraint, and
pairs whose labels share a first token are dropped rather than fudged.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.config import RunConfig, load_config
from src.models.runner import Runner, load_runner
from src.provenance import Provenance, capture, write_parquet
from src.readout.choice import read_choice
from src.readout.digits import DigitMap
from src.stimuli.build import Item, Template, load_items, load_templates, pair_block, slug

# Label schemes. `name` is content-addressed and has no fixed label pair.
SCHEMES: dict[str, tuple[str, str] | None] = {
    "letters": ("A", "B"),
    "digits": ("1", "2"),
    "xy": ("X", "Y"),
    "name": None,
}

N_PAIRS_PER_DOMAIN = 6  # per pair type


def artifact_path(cfg: RunConfig) -> Path:
    h = cfg.hash("pass_a")
    return cfg.artifacts_dir / "probe_choice" / cfg.model.name / h / f"choices_{h}.parquet"


# ---------------------------------------------------------------------------
# stimuli


def load_probe_items(cfg: RunConfig) -> list[Item]:
    """Low-appeal diagnostic items. Never enter the preregistered passes."""
    path = cfg.resolve(Path("src/stimuli/probe_items.yaml"))
    raw = yaml.safe_load(path.read_text())
    out: list[Item] = []
    for domain in cfg.stimuli.domains:
        block = raw["domains"][domain]
        for label in block["items"]:
            out.append(
                Item(
                    id=f"probe/{domain}/{slug(label)}",
                    domain=domain,
                    label=label,
                    framed=block["frame"].format(item=label),
                )
            )
    return out


@dataclass(frozen=True)
class ProbePair:
    pair_id: str
    domain: str
    pair_type: str  # "obvious" | "close"
    item1: Item
    item2: Item
    #: for obvious pairs, the id of the high-appeal member; None for close pairs
    expected_id: str | None


def build_probe_pairs(cfg: RunConfig, seed: int = 0) -> list[ProbePair]:
    """`obvious` = high-appeal vs deliberately low-appeal. `close` = two main-pool items."""
    rng = np.random.default_rng(seed)
    main = load_items(cfg)
    probe = load_probe_items(cfg)
    pairs: list[ProbePair] = []

    for domain in cfg.stimuli.domains:
        pool = [i for i in main if i.domain == domain]
        lows = [i for i in probe if i.domain == domain]
        n = min(N_PAIRS_PER_DOMAIN, len(lows))

        highs = rng.choice(len(pool), size=n, replace=False)
        for k in range(n):
            hi, lo = pool[int(highs[k])], lows[k]
            pairs.append(
                ProbePair(
                    pair_id=f"obvious/{domain}/{k:02d}",
                    domain=domain,
                    pair_type="obvious",
                    item1=hi,
                    item2=lo,
                    expected_id=hi.id,
                )
            )

        candidates = list(combinations(range(len(pool)), 2))
        picks = rng.choice(len(candidates), size=N_PAIRS_PER_DOMAIN, replace=False)
        for k, idx in enumerate(picks):
            a, b = candidates[int(idx)]
            pairs.append(
                ProbePair(
                    pair_id=f"close/{domain}/{k:02d}",
                    domain=domain,
                    pair_type="close",
                    item1=pool[a],
                    item2=pool[b],
                    expected_id=None,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# readout maps


def _first_token_ids(runner: Runner, text: str) -> set[int]:
    """First token of a string, bare and space-prefixed. The readout is position 0,
    so only the first token can ever be observed."""
    ids: set[int] = set()
    for surface in (text, f" {text}"):
        enc = runner.tokenizer.encode(surface, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return ids


def name_map(runner: Runner, label_a: str, label_b: str) -> DigitMap | None:
    """Content-addressed choice map. None if the two labels share a first token."""
    a, b = _first_token_ids(runner, label_a), _first_token_ids(runner, label_b)
    if not a or not b or (a & b):
        return None  # ambiguous at one position -- dropped, never fudged
    flat = sorted(a) + sorted(b)
    return DigitMap(
        digits=(0, 1),
        ids_by_digit={0: tuple(sorted(a)), 1: tuple(sorted(b))},
        flat_ids=tuple(flat),
        digit_index=tuple([0] * len(a) + [1] * len(b)),
        surfaces_by_digit={0: (label_a,), 1: (label_b,)},
    )


def scheme_map(runner: Runner, labels: tuple[str, str]) -> DigitMap:
    from src.readout.digits import build_label_map

    return build_label_map(runner.tokenizer, labels)


# ---------------------------------------------------------------------------
# run


_ARTICLES = ("a ", "an ", "the ")


def bare(label: str) -> str:
    """Drop a leading article.

    Many labels begin with "a ", so without this the first token of a reply is "a"
    for both options and the content-addressed readout is ambiguous for a large,
    non-random subset of pairs -- which would bias which pairs the name scheme can
    speak to. Stripping the article makes the first token a content word.
    """
    low = label.lower()
    for art in _ARTICLES:
        if low.startswith(art):
            return label[len(art) :]
    return label


def name_prompt(label_a: str, label_b: str) -> str:
    """Content-addressed choice: NO letter labels anywhere.

    The first version of this kept the lettered pair block, so the model dutifully
    answered "A" -- 98.7% of the probability mass sat on the letters and 0.8% on
    the item name. It was measuring the letter scheme twice. Position has to be
    removed as an available answer, not merely discouraged.
    """
    return (
        "Here are two options.\n\n"
        f"- {label_a}\n- {label_b}\n\n"
        "Which one would you choose for yourself? Reply with the option you "
        "choose, exactly as written, and nothing else."
    )


def run_probe(cfg: RunConfig, runner: Runner, prov: Provenance) -> pd.DataFrame:
    templates: list[Template] = load_templates(cfg)
    pairs = build_probe_pairs(cfg, seed=cfg.seed)

    fixed_maps = {
        name: scheme_map(runner, labels)
        for name, labels in SCHEMES.items()
        if labels is not None
    }

    rows: list[dict] = []
    prompts: list[str] = []
    maps: list[DigitMap] = []
    dropped = 0

    for pair in pairs:
        for t in templates:
            for order in (0, 1):
                first, second = (
                    (pair.item1, pair.item2) if order == 0 else (pair.item2, pair.item1)
                )
                for scheme, labels in SCHEMES.items():
                    if scheme == "name":
                        a_txt, b_txt = bare(first.label), bare(second.label)
                        dmap = name_map(runner, a_txt, b_txt)
                        if dmap is None:
                            dropped += 1
                            continue
                        # No template, no letters: position must not be answerable.
                        messages = [
                            {"role": "user", "content": name_prompt(a_txt, b_txt)}
                        ]
                        prompts.append(runner.render(messages))
                        maps.append(dmap)
                        rows.append(
                            {
                                "pair_id": pair.pair_id,
                                "domain": pair.domain,
                                "pair_type": pair.pair_type,
                                "template": t.id,
                                "option_order": order,
                                "scheme": scheme,
                                "first_item_id": first.id,
                                "second_item_id": second.id,
                                "expected_id": pair.expected_id,
                            }
                        )
                        continue
                    else:
                        dmap = fixed_maps[scheme]
                        la, lb = labels
                        prefix = t.pair_prefix.format(
                            item_a=first.framed, item_b=second.framed,
                            label_a=la, label_b=lb,
                        )
                        question = t.choice.format(label_a=la, label_b=lb)

                    messages = [{"role": "user", "content": f"{prefix}\n\n{question}"}]
                    prompts.append(runner.render(messages))
                    maps.append(dmap)
                    rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "domain": pair.domain,
                            "pair_type": pair.pair_type,
                            "template": t.id,
                            "option_order": order,
                            "scheme": scheme,
                            "first_item_id": first.id,
                            "second_item_id": second.id,
                            "expected_id": pair.expected_id,
                        }
                    )

    if dropped:
        print(f"  dropped {dropped} name-scheme trials (labels share a first token)")

    # One forward pass per trial, read at one position. Maps vary per trial in the
    # name scheme, so decode row by row against the batch's logits.
    picks, margins, masses = [], [], []
    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    for start in tqdm(range(0, len(prompts), chunk), desc="probe", unit="chunk", leave=False):
        logits = runner.last_logits(prompts[start : start + chunk])
        for j in range(logits.shape[0]):
            out = read_choice(logits[j : j + 1], maps[start + j])
            picks.append(int(out.index[0]))
            margins.append(float(out.margin[0]))
            masses.append(float(out.mass[0]))

    frame = pd.DataFrame(rows)
    frame["pick_index"] = picks  # 0 = the option displayed FIRST
    frame["choice_margin"] = margins
    frame["choice_mass"] = masses
    frame["chose_first"] = frame["pick_index"] == 0
    frame["chosen_item_id"] = np.where(
        frame["chose_first"], frame["first_item_id"], frame["second_item_id"]
    )
    frame["chose_expected"] = np.where(
        frame["expected_id"].notna(),
        frame["chosen_item_id"] == frame["expected_id"],
        None,
    )
    return write_parquet(frame, artifact_path(cfg), prov)


# ---------------------------------------------------------------------------
# summary


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """position_rate near 1.0 means position-bound. content_accuracy is only
    defined for obvious pairs, where there is a right answer."""
    rows = []
    for (scheme, pair_type), g in frame.groupby(["scheme", "pair_type"], observed=True):
        wide = g.pivot_table(
            index=["pair_id", "template"], columns="option_order",
            values="chosen_item_id", aggfunc="first",
        )
        flip = (
            float((wide[0] != wide[1]).mean()) if {0, 1} <= set(wide.columns) else float("nan")
        )
        acc = (
            float(g["chose_expected"].astype("boolean").mean())
            if pair_type == "obvious"
            else float("nan")
        )
        rows.append(
            {
                "scheme": scheme,
                "pair_type": pair_type,
                "n": len(g),
                "position_rate": float(g["chose_first"].mean()),
                "flip_rate": flip,
                "content_accuracy": acc,
                "mean_margin": float(g["choice_margin"].mean()),
                "mean_mass": float(g["choice_mass"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["pair_type", "scheme"]).reset_index(drop=True)


# A scheme whose probability mass does not land on the candidate tokens is not a
# valid readout, however clean its accuracy number looks. The first version of the
# name scheme had mass 0.003 -- the model was answering with a letter instead.
MASS_FLOOR = 0.5


def verdict(summary: pd.DataFrame) -> str:
    obv = summary[summary["pair_type"] == "obvious"].copy()
    if obv.empty:
        return "no obvious-pair trials; cannot adjudicate"

    invalid = obv[obv["mean_mass"] < MASS_FLOOR]
    lines = []
    for _, row in invalid.iterrows():
        lines.append(
            f"scheme={row['scheme']!r}: READOUT INVALID -- only "
            f"{row['mean_mass']:.3f} of the probability mass lands on the candidate "
            "tokens, so its accuracy figure is meaningless. The model is answering "
            "with something else entirely."
        )
    obv = obv[obv["mean_mass"] >= MASS_FLOOR]
    if obv.empty:
        return "\n".join(lines + ["no scheme produced a valid readout; cannot adjudicate"])

    best = obv.loc[obv["content_accuracy"].idxmax()]
    lines.append(
        f"best content accuracy on obvious pairs: {best['content_accuracy']:.2f} "
        f"via scheme={best['scheme']!r} (position_rate {best['position_rate']:.2f}, "
        f"mass {best['mean_mass']:.2f})"
    )
    if best["content_accuracy"] >= 0.85:
        lines.append(
            "  -> the choice DOES track content when the gap is wide. Position bias on "
            "close pairs is then an item-range problem, not a broken paradigm."
        )
    elif best["content_accuracy"] <= 0.6:
        lines.append(
            "  -> the choice does NOT track content even when the gap is wide. Position "
            "bias is fundamental to this elicitation; free choice needs a different "
            "mechanism."
        )
    else:
        lines.append("  -> partial content sensitivity; inconclusive, needs more pairs.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore cached output (the probe's artifact key does not track probe code)",
    )
    args = ap.parse_args(argv)

    overrides = {"artifacts_dir": args.artifacts} if args.artifacts else None
    cfg = load_config(args.config, overrides)
    prov = capture(cfg)

    print(f"\n{'=' * 72}\nCHOICE PROBE (diagnostic)  |  {cfg.model.name}\n{'=' * 72}")
    path = artifact_path(cfg)
    if path.exists() and not args.force:
        from src.provenance import read_parquet

        print(f"cached: {path}")
        frame = read_parquet(path)
    else:
        runner = load_runner(cfg)
        print(f"device={runner.device} dtype={runner.model.dtype}")
        frame = run_probe(cfg, runner, prov)
        print(f"wrote: {path}")

    summary = summarize(frame)
    print()
    print(summary.to_string(index=False))
    print()
    print(verdict(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
