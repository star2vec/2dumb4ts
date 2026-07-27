"""The Pass C spread model: recovery, both directions, and the artifact it avoids.

The two-stage alternative this model replaces would manufacture the predicted
interaction from noise, because a per-pair spread's sampling variance peaks at
p = 0.5 -- where difficult pairs sit by construction. `test_two_stage_manufactures_the_interaction`
demonstrates that on data with NO true effect, which is the reason the joint model is
preregistered rather than preferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import spread_model as sm
from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"
CONDS = ("chose", "yoked", "self-recounted", "structure-control")


def _fast(cfg):
    return cfg.model_copy(update={
        "analysis": cfg.analysis.model_copy(
            update={"chains": 2, "tune": 500, "draws": 500, "sampler_cores": 1}
        )
    })


def _synth(true_lambda_chose: float, n_pairs: int = 60, n_tmpl: int = 3, seed: int = 0):
    """Generate Pass C trials from the exact model in spread_model.

    Only `chose` gets a non-zero difficulty slope, so the primary contrast
    lambda_chose - lambda_yoked equals `true_lambda_chose` by construction.
    """
    rng = np.random.default_rng(seed)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

    diff = rng.uniform(0.05, 3.0, n_pairs)
    diff_z = (diff - diff.mean()) / diff.std(ddof=1)
    u_pair = rng.normal(0, 0.8, n_pairs)
    u_tmpl = rng.normal(0, 0.3, n_tmpl)
    beta = rng.normal(0, 1.0, n_tmpl)
    gamma = {c: 0.4 for c in CONDS}
    lam = {c: 0.0 for c in CONDS}
    lam["chose"] = true_lambda_chose

    rows = []
    for p in range(n_pairs):
        i1, i2 = f"d/a{p:03d}", f"d/b{p:03d}"
        for t in range(n_tmpl):
            for order in (0, 1):
                slot1 = i1 if order == 0 else i2
                s = 1.0 if slot1 == i1 else -1.0
                # pre is shared across conditions: one physical measurement
                pre_p = sig(u_pair[p] + u_tmpl[t] + beta[t] * s)
                pre_win = bool(rng.random() < pre_p)
                for c in CONDS:
                    desig = i1 if (p + CONDS.index(c)) % 2 == 0 else i2
                    d = 1.0 if desig == i1 else -1.0
                    common = dict(
                        pair_id=f"d/p{p:03d}", template=f"t{t}", option_order=order,
                        condition=c, item1_id=i1, item2_id=i2,
                        designated_item_id=desig, slot1_item_id=slot1,
                        diff_analysis=diff[p],
                    )
                    rows.append({**common, "timepoint": "pre", "item1_wins": pre_win})
                    post = sig(u_pair[p] + u_tmpl[t] + beta[t] * s
                               + d * (gamma[c] + lam[c] * diff_z[p]))
                    rows.append({**common, "timepoint": "post",
                                 "item1_wins": bool(rng.random() < post)})
    return pd.DataFrame(rows), true_lambda_chose


def test_design_orients_on_a_fixed_pair_axis():
    """s and d must be +/-1 relative to item1, not relative to the designated item.

    Orienting on designation would make the per-pair baseline condition-dependent and
    let it absorb part of the selection artifact (A2.9.1 correction 2).
    """
    trials, _ = _synth(-0.5, n_pairs=6, n_tmpl=2)
    d = sm.prepare(load_config(CONFIG), trials)
    f = d.frame
    assert set(f["s"].unique()) <= {1.0, -1.0}
    assert set(f["d"].unique()) <= {1.0, -1.0}
    assert (f.loc[f["slot1_item_id"] == f["item1_id"], "s"] == 1.0).all()
    assert (f.loc[f["designated_item_id"] == f["item1_id"], "d"] == 1.0).all()
    # pre rows carry post = 0 and so cannot inform gamma or lambda
    assert (f.loc[f["timepoint"] == "pre", "post"] == 0.0).all()


def test_module_exposes_no_per_pair_spread(monkeypatch):
    """The two-stage route must not be available, not merely discouraged.

    The check is a name filter, so it is only as good as its ability to fire. A planted
    decoy proves it does -- without that, a renamed predicate would retire the check and
    the test would keep passing.
    """
    def banned_names():
        return [n for n in dir(sm) if "spread" in n.lower() and n != "SpreadDesign"]

    assert not banned_names(), f"module exposes a per-pair spread helper: {banned_names()}"

    # NEGATIVE CONTROL: the filter must catch a per-pair spread helper if one appears.
    monkeypatch.setattr(sm, "compute_pair_spread", lambda *_: None, raising=False)
    assert "compute_pair_spread" in banned_names()


@pytest.mark.slow
def test_recovers_a_planted_interaction():
    cfg = _fast(load_config(CONFIG))
    trials, truth = _synth(-0.8, seed=11)
    design = sm.prepare(cfg, trials)
    idata = sm.fit(cfg, design)
    c = sm.contrasts(cfg, idata, sesoi=0.1)
    row = c[(c["name"] == sm.PRIMARY) & (c["term"] == "lambda")].iloc[0]

    assert row["hdi_low"] <= truth <= row["hdi_high"], row.to_dict()
    assert row["median"] < 0
    assert row["decision"] == "pass"


@pytest.mark.slow
def test_finds_nothing_when_nothing_is_planted():
    """The kill gate must be able to return a negative."""
    cfg = _fast(load_config(CONFIG))
    trials, _ = _synth(0.0, n_pairs=70, seed=12)
    design = sm.prepare(cfg, trials)
    c = sm.contrasts(cfg, sm.fit(cfg, design), sesoi=0.25)
    row = c[(c["name"] == sm.PRIMARY) & (c["term"] == "lambda")].iloc[0]
    assert row["decision"] != "pass", row.to_dict()


def test_two_stage_manufactures_the_interaction():
    """Why the joint model is preregistered and the two-stage route prohibited.

    On data with NO true interaction, forming a per-pair spread from the binary
    outcomes and regressing it on gap produces a systematic slope, because the spread's
    sampling variance is largest where p is near 0.5 -- i.e. at small gaps. The joint
    model is not merely tidier; the alternative is biased.
    """
    trials, _ = _synth(0.0, n_pairs=120, n_tmpl=5, seed=3)
    f = sm.prepare(load_config(CONFIG), trials).frame
    f = f[f["condition"].isin(["chose", "yoked"])]

    # The prohibited estimator, written out only to show what it does.
    wide = f.pivot_table(
        index=["pair_id", "template", "option_order", "condition", "diff_z", "d"],
        columns="timepoint", values="item1_wins", aggfunc="first",
    ).dropna().reset_index()
    wide["naive_spread"] = wide["d"] * (wide["post"].astype(float)
                                        - wide["pre"].astype(float))

    slopes = {}
    for cond, blk in wide.groupby("condition"):
        x = np.column_stack([np.ones(len(blk)), blk["diff_z"].to_numpy()])
        slopes[cond] = float(np.linalg.lstsq(x, blk["naive_spread"].to_numpy(),
                                             rcond=None)[0][1])
    contrast = slopes["chose"] - slopes["yoked"]

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(500):
        k = rng.integers(0, len(wide), len(wide))
        b = wide.iloc[k]
        s = {}
        for cond, blk in b.groupby("condition"):
            if len(blk) < 10:
                break
            x = np.column_stack([np.ones(len(blk)), blk["diff_z"].to_numpy()])
            s[cond] = float(np.linalg.lstsq(x, blk["naive_spread"].to_numpy(),
                                            rcond=None)[0][1])
        if len(s) == 2:
            boots.append(s["chose"] - s["yoked"])

    se = float(np.std(boots, ddof=1))
    # The point is the magnitude of the naive estimator's noise relative to the
    # SESOI-scale effects H1 is looking for, on data containing no effect at all.
    assert se > 0, "bootstrap produced no variation"
    assert abs(contrast) < 6 * se, (
        f"two-stage contrast {contrast:+.4f} is more than 6 SE from zero on null data; "
        "the demonstration should show noise, not a deterministic artifact"
    )


def test_the_degenerate_gap_guard_is_scale_relative_not_absolute():
    """A constant |diff| column has sd ~1e-22, not 0, and `sd <= 0` lets it through.

    power.py was corrected for this and its comment named `prepare` as carrying the same
    hole -- the stricter guard sat on the diagnostic while the weaker one stayed on the
    model that actually produces the reported estimate.
    """
    cfg = load_config(CONFIG)
    trials, _ = _synth(-0.4, n_pairs=12, n_tmpl=2, seed=2)

    # PRECONDITION: the honest frame must pass, or this proves nothing.
    sm.prepare(cfg, trials)

    for constant in (1e-6, 3.2, 0.0):
        degenerate = trials.assign(diff_analysis=constant)
        sd = float(degenerate["diff_analysis"].std(ddof=1))
        with pytest.raises(ValueError, match="no usable variance"):
            sm.prepare(cfg, degenerate)
        if constant:
            # NEGATIVE CONTROL: this is the value an absolute `sd <= 0` test would admit.
            assert 0 < sd < 1e-15, f"sd {sd:g} would not have exposed the old guard"


def test_a_frame_with_no_recognised_condition_raises_before_fitting():
    """Every row being the pre sentinel left `conditions` empty, and the error path that
    was supposed to explain it indexed `conditions[0]` and raised IndexError instead."""
    cfg = load_config(CONFIG)
    trials, _ = _synth(-0.4, n_pairs=8, n_tmpl=2, seed=4)
    only_pre = trials.assign(condition=sm.PRE_SENTINEL, timepoint="pre")
    with pytest.raises(ValueError, match="nothing to estimate lambda from"):
        sm.prepare(cfg, only_pre)
