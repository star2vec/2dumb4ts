"""Same seed, same numbers -- the constraint that makes every artifact re-derivable.

"All randomness seeded. Item order, option order, template assignment counterbalanced --
not random-per-run." Nothing checked that end to end. Each source of randomness in `src/`
is exercised here: seeded RNGs must repeat, and must ALSO differ when the seed differs,
because a generator that ignores its seed passes the first check trivially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def test_random_designation_is_reproducible_and_seed_sensitive():
    """`random`/`3p-random` designation is counterbalanced by construction, not sampled."""
    from src.stimuli.build import balanced_designation

    ids = [f"d/p{i:03d}" for i in range(40)]
    strata = ["difficult/d"] * 20 + ["easy/d"] * 20

    a = balanced_designation(ids, strata, seed=7)
    assert a == balanced_designation(ids, strata, seed=7)
    # NEGATIVE CONTROL: a generator ignoring its seed would pass the line above.
    assert a != balanced_designation(ids, strata, seed=8)

    # ...and it must be exactly balanced within every stratum, not merely repeatable.
    frame = pd.DataFrame({"pair_id": ids, "stratum": strata})
    frame["assigned"] = frame["pair_id"].map(a)
    for _, g in frame.groupby("stratum"):
        assert g["assigned"].sum() * 2 == len(g), "designation is not balanced in-stratum"


def test_designation_does_not_depend_on_the_order_pairs_arrive_in():
    """Pairs reach Pass C in whatever order Pass B emitted; designation must not move."""
    from src.stimuli.build import balanced_designation

    ids = [f"d/p{i:03d}" for i in range(24)]
    strata = ["difficult/d"] * 12 + ["easy/d"] * 12
    forward = balanced_designation(ids, strata, seed=3)
    rev = balanced_designation(ids[::-1], strata[::-1], seed=3)
    assert forward == rev, "designation depends on input order, so it is not counterbalanced"


def test_pair_construction_repeats_and_responds_to_the_seed(cfg):
    from src.experiments.pass_b import build_pairs

    import sys

    sys.path.insert(0, ".")
    from tests.test_design import SIGMA_ITEM, _synthetic_scores

    scores = _synthetic_scores(cfg)
    a = build_pairs(cfg, scores, SIGMA_ITEM)
    pd.testing.assert_frame_equal(a, build_pairs(cfg, scores, SIGMA_ITEM))

    # Pair construction is deterministic GIVEN the scores -- it is a greedy selection with
    # no RNG at all -- so the seed must not enter it. Pinned so nobody adds sampling.
    reseeded = cfg.model_copy(update={"seed": cfg.seed + 1})
    pd.testing.assert_frame_equal(a, build_pairs(reseeded, scores, SIGMA_ITEM))


def test_power_analysis_repeats_exactly(cfg):
    """power.analyze draws random designations internally; it is seeded from cfg."""
    from src.analysis import power

    pairs = pd.DataFrame({
        "pair_id": [f"d/p{i:03d}" for i in range(60)],
        "item1_id": [f"d/a{i}" for i in range(60)],
        "item2_id": [f"d/b{i}" for i in range(60)],
        "difficulty": ["difficult"] * 30 + ["easy"] * 30,
        "diff_analysis": np.linspace(0.05, 3.0, 60),
    })
    beta = pd.DataFrame({"template": [f"t{i}" for i in range(5)],
                         "beta_mean": np.linspace(0.7, 1.7, 5)})

    a = power.analyze(cfg, pairs, beta, 1.573)
    b = power.analyze(cfg, pairs, beta, 1.573)
    assert a.se == b.se and a.min_detectable == b.min_detectable

    # NEGATIVE CONTROL: the internal draw must actually depend on the seed.
    other = power.analyze(cfg.model_copy(update={"seed": cfg.seed + 1}), pairs, beta, 1.573)
    assert other.se != a.se, "power.analyze ignores cfg.seed"


def test_the_bootstrap_ci_on_the_excess_slope_repeats():
    from src.analysis.bradley_terry import excess_consistency_slope

    class _Fit:
        theta = pd.DataFrame({"item_id": [f"i{i}" for i in range(20)],
                              "theta_mean": np.linspace(-2, 2, 20)})
        anchors = pd.DataFrame({"anchor_id": ["a0", "a1"], "alpha_mean": [-0.5, 0.5]})
        beta = pd.DataFrame({"template": ["t0"], "beta_mean": [1.2]})

    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        for a in ("a0", "a1"):
            for order in (0, 1):
                rows.append({"arm": "digits", "readout_valid": True, "template": "t0",
                             "item_id": f"i{i}", "anchor_id": a, "order": order,
                             "item_wins": bool(rng.random() < 0.5)})
    frame = pd.DataFrame(rows)

    a = excess_consistency_slope(frame, _Fit())
    b = excess_consistency_slope(frame, _Fit())
    assert (a["slope"], a["ci_low"], a["ci_high"]) == (b["slope"], b["ci_low"], b["ci_high"])
    c = excess_consistency_slope(frame, _Fit(), seed=999)
    assert c["slope"] == a["slope"], "the point estimate must not depend on the bootstrap"
    assert (c["ci_low"], c["ci_high"]) != (a["ci_low"], a["ci_high"])


@pytest.mark.slow
def test_the_spread_model_posterior_repeats_under_one_seed(cfg):
    """PyMC is seeded from cfg.seed. Two fits of the same design must be bit-identical."""
    import sys

    from src.analysis import spread_model as sm

    sys.path.insert(0, ".")
    from tests.test_spread_model import _synth

    fast = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
        update={"chains": 2, "tune": 300, "draws": 300, "sampler_cores": 1})})
    trials, _ = _synth(-0.5, n_pairs=20, n_tmpl=2, seed=1)
    design = sm.prepare(fast, trials)

    a = sm._draws(sm.fit(fast, design), "lambda", "chose")
    b = sm._draws(sm.fit(fast, design), "lambda", "chose")
    assert np.array_equal(a, b), "same seed produced different draws"

    reseeded = fast.model_copy(update={"seed": fast.seed + 1})
    c = sm._draws(sm.fit(reseeded, design), "lambda", "chose")
    assert not np.array_equal(a, c), "the sampler ignores cfg.seed"
