"""Execute `run()` from end to end, with every artifact pre-seeded so no model loads.

WHY THIS EXISTS. Coverage put `run.py` at 14% -- 203 of its 235 statements had never been
executed by any test, and lines 134-321 are the whole body of `run()`. That is exactly
where the `fit.sigma_item` NameError lived: a crash in the H1 stage that no test could
reach, because reaching it needs a complete Pass C. Static analysis caught that one. It
cannot catch the next one, which will be a wrong key, a shape mismatch, or a contract
between two stages that only breaks when both run.

The trick is that `Lazy` only loads a model when a stage needs a forward pass. Seed every
artifact at the current stage hashes and the orchestrator runs with no weights at all --
the full gate logic, the power block, Pass C's validation, the spread model, and the
results.json assembly, in seconds.

The DATA is synthetic and the H1 result is meaningless by construction. Nothing here is
evidence about anything; it is a wiring test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.spread_model import PRE_SENTINEL
from src.config import load_config
from src.experiments import pass_a as stage_a
from src.experiments import pass_a_pairwise as stage_pw
from src.experiments import pass_b as stage_b
from src.experiments import pass_c as stage_c
from src.experiments.pass_c import OWN_PICK_CONDITIONS
from src.provenance import Provenance, write_parquet

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


def _prov(cfg) -> Provenance:
    return Provenance(
        device="cuda", device_name="synthetic", dtype="bfloat16",
        attn_implementation="eager", torch_version="2.8.0",
        transformers_version="4.57.0", pymc_version="5.28.5",
        pytensor_version="2.38.3", pytensor_mode="Mode",
        model_name=cfg.model.name, model_hf_id=cfg.model.hf_id,
        model_revision="a" * 40, model_revision_pinned=True, seed=cfg.seed,
        config_hash=cfg.hash(), git_sha="c" * 40, git_dirty=False, smoke=False,
        platform="synthetic", python_version="3.11.15",
        created_utc="2026-07-28T00:00:00+00:00",
    )


def _seed_artifacts(cfg, rng):
    """Write every upstream artifact so no stage needs a forward pass."""
    from src.readout.pairwise import load_anchors
    from src.stimuli.build import load_items, load_templates

    prov = _prov(cfg)
    # Per domain, not the first N overall -- items are ordered by domain, so a flat slice
    # leaves later domains empty and Pass B dies with "no candidate pairs".
    per_domain = 8
    by_dom: dict[str, list] = {}
    for i in load_items(cfg):
        by_dom.setdefault(i.domain, [])
        if len(by_dom[i.domain]) < per_domain:
            by_dom[i.domain].append(i)
    items = [i for d in cfg.stimuli.domains for i in by_dom[d]]
    templates = load_templates(cfg)
    anchors = load_anchors(cfg)
    truth = {i.id: v for i, v in zip(items, np.linspace(-1.6, 1.6, len(items)))}
    astr = {a.id: v for a, v in zip(anchors, np.linspace(-0.8, 0.8, len(anchors)))}
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

    # --- absolute Pass A: the A1.5 record. Gates nothing, but run() reads it. -----
    rows = []
    for i in items:
        for t in templates:
            for pol in ("ascending", "descending"):
                base = 5.0 + 2.0 * truth[i.id]
                r = base if pol == "ascending" else cfg.readout.reversal_constant - base
                rows.append({"item_id": i.id, "domain": i.domain, "item_label": i.label,
                             "template": t.id, "template_index": t.index,
                             "polarity": pol, "rating": float(r),
                             "digit_mass": 0.97, "rating_argmax": float(round(r))})
    write_parquet(pd.DataFrame(rows), stage_a.artifact_path(cfg), prov)

    # --- pairwise Pass A: the live instrument ------------------------------------
    rows = []
    for i in items:
        for a in anchors:
            for t in templates:
                for order in (0, 1):
                    s = 1.0 if order == 0 else -1.0
                    p = sig(truth[i.id] - astr[a.id] + 0.35 * s)
                    rows.append({
                        "item_id": i.id, "domain": i.domain, "anchor_id": a.id,
                        "anchor_tier": a.tier, "template": t.id,
                        "template_index": t.index, "order": order, "arm": "digits",
                        "item_in_slot1": order == 0,
                        "first_id": i.id if order == 0 else a.id,
                        "second_id": a.id if order == 0 else i.id,
                        "item_wins": bool(rng.random() < p),
                        "readout_valid": True, "readout_mass": 0.98,
                    })
    write_parquet(pd.DataFrame(rows), stage_pw.artifact_path(cfg), prov)
    return items, templates, truth, prov


def _seed_pass_b_and_c(cfg, items, templates, truth, prov, rng, sigma_item, inst_key):
    """Pairs and trials, matching the schemas prepare()/_validate() require."""
    n_per = cfg.n_difficult // len(cfg.stimuli.domains)
    by_domain: dict[str, list] = {}
    for i in items:
        by_domain.setdefault(i.domain, []).append(i)

    pair_rows, cells = [], []
    for domain, pool in by_domain.items():
        # Disjoint item blocks per difficulty level: no item may appear in both, which is
        # a design constraint Pass B enforces and this fixture must not quietly violate.
        blocks = {"difficult": pool[: len(pool) // 2], "easy": pool[len(pool) // 2 :]}
        for k in range(min(n_per, len(pool) // 4)):
            for level in ("difficult", "easy"):
                a, b = blocks[level][2 * k], blocks[level][2 * k + 1]
                gap = 0.15 if level == "difficult" else 1.4
                t1 = truth[a.id]
                t2 = t1 - gap
                pid = f"{domain}/{level}{k}"
                pair_rows.append({
                    "pair_id": pid, "domain": domain, "difficulty": level,
                    "matched_set": f"{domain}/ms{k:02d}",
                    "item1_id": a.id, "item2_id": b.id,
                    "diff_selection": abs(t1 - t2), "mean_selection": (t1 + t2) / 2,
                    "diff_analysis": abs(t1 - t2), "mean_analysis": (t1 + t2) / 2,
                    "theta_item1": t1, "theta_item2": t2,
                    "domain_q_difficult": 0.1, "domain_q_easy": 0.75,
                    "match_tolerance_realized": cfg.pass_b.match_tolerance(sigma_item),
                    "sigma_item": sigma_item, "instrument_cache_key": inst_key,
                })
                cells.append((pid, domain, level, a.id, b.id, abs(t1 - t2)))
    write_parquet(pd.DataFrame(pair_rows), stage_b.artifact_path(cfg), prov)

    trials = []
    for pid, domain, level, i1, i2, gap in cells:
        own = i1 if rng.random() < 0.5 else i2
        for t in templates:
            for order in (0, 1):
                s1 = i1 if order == 0 else i2
                base = dict(pair_id=pid, domain=domain, difficulty=level,
                            matched_set=pid.replace(level, "ms"), item1_id=i1,
                            item2_id=i2, template=t.id, template_index=t.index,
                            option_order=order, slot1_item_id=s1,
                            slot2_item_id=i2 if order == 0 else i1,
                            diff_selection=gap, diff_analysis=gap,
                            chosen_item_id=own, choice_label="A", choice_margin=0.2)
                trials.append({**base, "condition": PRE_SENTINEL, "timepoint": "pre",
                               "designated_item_id": i1,
                               "item1_wins": bool(rng.random() < 0.5),
                               "readout_mass": 0.98, "readout_valid": True})
                for c in cfg.pass_c.conditions:
                    des = own if c in OWN_PICK_CONDITIONS else i1
                    trials.append({
                        **base, "condition": c, "timepoint": "post",
                        "designated_item_id": des,
                        "other_item_id": i2 if des == i1 else i1,
                        "designated_is_chosen": des == own,
                        "item1_wins": bool(rng.random() < 0.5),
                        "readout_mass": 0.98, "readout_valid": True,
                    })
    write_parquet(pd.DataFrame(trials), stage_c.artifact_path(cfg), prov)


@pytest.mark.slow
def test_the_whole_pipeline_executes_with_no_model_loaded(tmp_path, monkeypatch):
    """The orchestrator, start to finish. Would have caught `fit.sigma_item` instantly."""
    from src.experiments import run as run_mod

    base = load_config(CONFIG)
    cfg = base.model_copy(update={
        "artifacts_dir": tmp_path,
        "pass_b": base.pass_b.model_copy(update={
            "n_difficult": 8, "n_easy": 8, "match_tolerance_sigma_fraction": 4.0}),
        "analysis": base.analysis.model_copy(update={
            "chains": 2, "tune": 200, "draws": 200, "sampler_cores": 1,
            "ppc_null_replicates": 2}),
    })
    monkeypatch.setattr(run_mod, "capture", lambda c: _prov(c))

    rng = np.random.default_rng(4)
    items, templates, truth, prov = _seed_artifacts(cfg, rng)

    # Run once to Pass B: this fits the instrument and writes its cache, which gives us
    # the real sigma_item and cache key to build consistent Pass C against.
    assert run_mod.run(cfg, stop_after="pass_b", progressbar=False) == 0

    fit_dir = cfg.artifacts_dir / "instrument_fit" / cfg.model.name
    rec = json.loads(sorted(fit_dir.glob("*/fit_*.json"))[-1].read_text())
    _seed_pass_b_and_c(cfg, items, templates, truth, prov, rng,
                       rec["sigma_item"], rec["cache_key"])

    # NEGATIVE CONTROL: if the model were ever loaded this would fail rather than quietly
    # download weights, which is what makes "no forward pass happened" a real assertion.
    monkeypatch.setattr(run_mod.Lazy, "__call__",
                        lambda self: pytest.fail("a stage tried to load the model"))

    assert run_mod.run(cfg, progressbar=False) == 0

    results = json.loads(
        (cfg.artifact_dir("analysis") / f"results_{cfg.hash()}.json").read_text())
    assert results["outcome"].startswith("primary-")
    assert results["outcome"] != "primary-unavailable"
    for key in ("reliability_gate", "power", "contrasts", "primary", "sesoi",
                "instrument_validation", "readout_mass_by_template"):
        assert key in results, f"results.json is missing {key}"

    primary = results["primary"]
    assert {"median", "hdi_low", "hdi_high", "decision"} <= set(primary)
    assert results["sesoi"] == pytest.approx(
        cfg.analysis.sesoi_sigma_fraction * rec["sigma_item"])
    assert results["power"]["min_detectable_effect"] > 0
