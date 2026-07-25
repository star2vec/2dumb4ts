"""Pass A -- item rating.

Full crossing of 400 items x 5 templates x 2 polarities = 4,000 forward passes
per model. One item per forward pass, no shared context between items, so no
item's rating can be influenced by any other item's presence or position.

Counterbalancing is by construction (a complete crossing), not by sampling, so
there is no per-run randomness here at all.

Note on why polarity rather than a repeated run: with one item per forward pass
and an expected-value readout, two runs differing only in item order return
bit-identical numbers and test-retest correlation is trivially 1.0. Reliability
needs a real source of variation, which is what template and polarity supply.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import RunConfig
from src.models.runner import Runner
from src.provenance import Provenance, write_parquet
from src.readout.validity import attach
from src.stimuli.build import load_items, load_templates, pass_a_messages


def artifact_path(cfg: RunConfig) -> Path:
    return cfg.artifact_dir("pass_a") / f"ratings_{cfg.hash('pass_a')}.parquet"


def run_pass_a(cfg: RunConfig, runner: Runner, prov: Provenance) -> pd.DataFrame:
    items = load_items(cfg)
    templates = load_templates(cfg)

    rows: list[dict] = []
    message_lists: list[list[dict[str, str]]] = []
    for item in items:
        for t in templates:
            for polarity in ("ascending", "descending"):
                rows.append(
                    {
                        "item_id": item.id,
                        "domain": item.domain,
                        "item_label": item.label,
                        "template": t.id,
                        "template_index": t.index,
                        "polarity": polarity,
                    }
                )
                message_lists.append(pass_a_messages(t, item.framed, polarity, cfg))

    ratings, masses, argmaxes = [], [], []
    chunk = max(cfg.batch_size * 8, cfg.batch_size)
    for start in tqdm(
        range(0, len(message_lists), chunk), desc="pass A", unit="chunk", leave=False
    ):
        out = runner.rate(message_lists[start : start + chunk])
        ratings.extend(out.value.tolist())
        masses.extend(out.mass.tolist())
        argmaxes.extend(out.argmax.tolist())

    frame = pd.DataFrame(rows)
    frame["rating"] = ratings
    frame["digit_mass"] = masses
    frame["rating_argmax"] = argmaxes
    # A1.6: the mass floor is a global invariant. Trials whose probability does
    # not land on the digit tokens are marked invalid and logged, never scored
    # silently.
    frame = attach(frame, np.asarray(masses), context="pass A rating")

    return write_parquet(frame, artifact_path(cfg), prov)


def load_or_run(cfg: RunConfig, runner_factory, prov: Provenance) -> pd.DataFrame:
    """Re-runnable from cached upstream output: the config hash gates the cache."""
    from src.provenance import read_parquet

    path = artifact_path(cfg)
    if path.exists():
        return read_parquet(path)
    return run_pass_a(cfg, runner_factory(), prov)
