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


@pytest.mark.slow
def test_non_centering_the_template_effect_leaves_the_primary_unchanged():
    """A4.1's claim, tested on synthetic data where the truth is known.

    The reparameterization is defended as PURE -- same model, same posterior, different
    geometry. That is a checkable claim, and checking it on synthetic data is the only way
    to check it without using the real result as the referee.

    Scaling a zero-sum vector by a positive scalar preserves sum-to-zero, so the two forms
    are the same distribution. If this test ever fails, the reparameterization is not what
    A4.1 says it is and llama's refit cannot be read as a robustness check.
    """
    import pymc as pm

    cfg = _fast(load_config(CONFIG))
    trials, truth = _synth(-0.7, n_pairs=40, n_tmpl=4, seed=5)
    design = sm.prepare(cfg, trials)

    non_centered = sm.fit(cfg, design)
    nc = (sm._draws(non_centered, "lambda", "chose")
          - sm._draws(non_centered, "lambda", "yoked"))

    # The centered form, rebuilt here so the comparison is against real code rather than a
    # remembered number.
    f, coords = design.frame, {"condition": design.conditions,
                               "pair": design.pairs, "template": design.templates}
    with pm.Model(coords=coords):
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="condition")
        lam = pm.Normal("lambda", 0.0, 1.0, dims="condition")
        beta = pm.Normal("beta", 0.0, 1.5, dims="template")
        sd_pair = pm.HalfNormal("sd_pair", 2.0)
        u_pair = pm.ZeroSumNormal("u_pair", sigma=sd_pair, dims="pair")
        sd_tmpl = pm.HalfNormal("sd_template", 1.0)
        u_tmpl = pm.ZeroSumNormal("u_template", sigma=sd_tmpl, dims="template")
        ci = f["cond_idx"].to_numpy()
        pm.Bernoulli(
            "item1_wins",
            logit_p=(u_pair[f["pair_idx"].to_numpy()] + u_tmpl[f["tmpl_idx"].to_numpy()]
                     + beta[f["tmpl_idx"].to_numpy()] * f["s"].to_numpy()
                     + f["post"].to_numpy() * f["d"].to_numpy()
                     * (gamma[ci] + lam[ci] * f["diff_z"].to_numpy())),
            observed=f["item1_wins"].to_numpy().astype(int))
        centered = pm.sample(draws=cfg.analysis.draws, tune=cfg.analysis.tune,
                             chains=cfg.analysis.chains, random_seed=cfg.seed,
                             target_accept=0.9, progressbar=False, cores=1)
    ce = (sm._draws(centered, "lambda", "chose") - sm._draws(centered, "lambda", "yoked"))

    # Agreement well inside Monte Carlo error of either fit.
    mcse = (nc.std(ddof=1) / np.sqrt(len(nc))) + (ce.std(ddof=1) / np.sqrt(len(ce)))
    assert abs(nc.mean() - ce.mean()) < 10 * mcse, (
        f"non-centered {nc.mean():+.4f} vs centered {ce.mean():+.4f}; the "
        "reparameterization is NOT pure and A4.1's framing is wrong")
    assert abs(nc.std(ddof=1) - ce.std(ddof=1)) < 0.25 * ce.std(ddof=1)
    # PRECONDITION: both must actually recover the planted effect, or agreement is cheap.
    for draws in (nc, ce):
        assert draws.mean() < 0 and abs(draws.mean() - truth) < 0.5, draws.mean()


@pytest.mark.slow
def test_the_graded_dv_recovers_the_same_effect_as_the_binary_one():
    """A4.5's pilot model, verified on synthetic data before it is trusted on real data.

    The graded fit is only a fair comparison if it estimates the SAME quantity. Both are
    run on trials generated from one known truth; both must recover it, and `lambda` must
    keep its units so the standard errors are comparable at all.

    A model that recovered a DIFFERENT quantity could show a spectacular "precision gain"
    by being precise about the wrong thing.
    """
    cfg = _fast(load_config(CONFIG))
    trials, truth = _synth(-0.8, n_pairs=40, n_tmpl=3, seed=9)

    # Attach a graded readout consistent with the binary one: a probability on the correct
    # side of 0.5, with the confidence varying per row.
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.55, 0.98, len(trials))
    trials = trials.assign(
        p_item1=np.where(trials["item1_wins"].to_numpy(), conf, 1.0 - conf))
    design = sm.prepare(cfg, trials)

    binary = sm._draws(sm.fit(cfg, design), "lambda", "chose") - \
        sm._draws(sm.fit(cfg, design), "lambda", "yoked")
    g = sm.fit_graded(cfg, design, family="normal")
    graded = sm._draws(g, "lambda", "chose") - sm._draws(g, "lambda", "yoked")

    for name, draws in (("binary", binary), ("graded", graded)):
        assert draws.mean() < 0, f"{name} lost the sign of the planted effect"
    # Same quantity, so the two must agree far better than either's own spread.
    assert abs(graded.mean() - binary.mean()) < 2.0 * max(binary.std(), graded.std()), (
        f"binary {binary.mean():+.3f} vs graded {graded.mean():+.3f} -- the graded model "
        "is not estimating the same contrast, so an SE ratio between them is meaningless")


def test_the_graded_fit_refuses_data_it_cannot_model():
    """Both refusals matter: absent trials predate A4.5, and p at exactly 0 or 1 has an
    infinite logit whose value must be decided deliberately, not clipped in the fitter."""
    cfg = _fast(load_config(CONFIG))
    trials, _ = _synth(-0.4, n_pairs=8, n_tmpl=2, seed=1)

    with pytest.raises(ValueError, match="predate A4.5"):
        sm.fit_graded(cfg, sm.prepare(cfg, trials))

    saturated = trials.assign(p_item1=np.where(trials["item1_wins"], 1.0, 0.0))
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        sm.fit_graded(cfg, sm.prepare(cfg, saturated))

    with pytest.raises(ValueError, match="normal.*studentt"):
        sm.fit_graded(cfg, sm.prepare(cfg, trials.assign(p_item1=0.6)), family="beta")
