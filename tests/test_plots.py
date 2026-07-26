"""The figure layer, and the three defects the rewrite fixed.

These are cheap structural tests plus one render. The render matters: the previous
version imported from a retired module and would have raised at figure time, which no
test caught because no test drew a figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.analysis import plots
from src.analysis.spread_model import PLANNED, PRE_SENTINEL, PRIMARY
from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


def _trials(n_pairs=40, n_tmpl=2, seed=3):
    """Eight conditions, pre emitted once with the sentinel, own-pick designation."""
    cfg = load_config(CONFIG)
    conds = list(cfg.pass_c.conditions)
    own_pick = [c for c in conds if c not in ("random", "3p-random")]
    rng = np.random.default_rng(seed)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

    diff = rng.uniform(0.05, 3.0, n_pairs)
    diff_z = (diff - diff.mean()) / diff.std(ddof=1)
    u_pair = rng.normal(0, 0.8, n_pairs)
    lam = dict.fromkeys(conds, 0.0)
    lam["chose"] = -0.6

    rows = []
    for p in range(n_pairs):
        i1, i2 = f"d/a{p:03d}", f"d/b{p:03d}"
        own = i1 if rng.random() < 0.5 else i2
        for t in range(n_tmpl):
            for slot1 in (i1, i2):
                s = 1.0 if slot1 == i1 else -1.0
                base = u_pair[p] + 0.3 * s
                common = dict(pair_id=f"d/p{p:03d}", template=f"t{t}", item1_id=i1,
                              item2_id=i2, slot1_item_id=slot1, diff_analysis=diff[p])
                rows.append({**common, "condition": PRE_SENTINEL, "timepoint": "pre",
                             "designated_item_id": i1,
                             "item1_wins": bool(rng.random() < sig(base))})
                for c in conds:
                    desig = own if c in own_pick else (i1 if rng.random() < 0.5 else i2)
                    d = 1.0 if desig == i1 else -1.0
                    p_post = sig(base + d * (0.4 + lam[c] * diff_z[p]))
                    rows.append({**common, "condition": c, "timepoint": "post",
                                 "designated_item_id": desig,
                                 "item1_wins": bool(rng.random() < p_post)})
    return cfg, pd.DataFrame(rows)


def test_every_configured_condition_has_a_hue_and_a_panel():
    """Bug 2: five names were listed for an eight-condition design.

    The panel builder filtered the data THROUGH the name list, so `structure-control`,
    `self-recounted` and `chose-provisional` -- the three conditions A2.9.3 added -- were
    dropped from the figure with no error. Both directions are pinned: every configured
    condition must have a style, and must be drawn by some panel.
    """
    cfg = load_config(CONFIG)
    configured = set(cfg.pass_c.conditions)
    assert configured == set(plots.CONDITION_STYLE), (
        configured ^ set(plots.CONDITION_STYLE))
    drawn = {c for _, cs in plots.PANELS for c in cs}
    assert configured <= drawn, f"never drawn: {sorted(configured - drawn)}"


def test_hue_and_style_together_identify_a_condition_uniquely():
    """Eight conditions on three hues only works if the second channel disambiguates."""
    seen = {}
    for condition, (attribution, style, _) in plots.CONDITION_STYLE.items():
        assert attribution in plots.ATTRIBUTION_ORDER
        key = (attribution, style)
        assert key not in seen, f"{condition} collides with {seen[key]} on {key}"
        seen[key] = condition
    for theme in (plots.LIGHT, plots.DARK):
        assert set(theme.hues) == set(plots.ATTRIBUTION_ORDER)
        assert len(set(theme.hues.values())) == len(plots.ATTRIBUTION_ORDER)


def test_no_panel_exceeds_the_validated_hue_count():
    """The palette is validated for three slots. A fourth hue would be a generated one."""
    for title, conditions in plots.PANELS:
        hues = {plots.CONDITION_STYLE[c][0] for c in conditions}
        assert len(hues) <= len(plots.ATTRIBUTION_ORDER), (title, hues)
        assert len(conditions) == len(set(conditions)), title


def test_the_data_overlay_is_pooled_within_a_bin_never_per_pair():
    """Bug 1, the expensive one.

    The old overlay averaged a per-pair spread inside each |diff| bin. `spread_model`
    refuses to expose that quantity at all, because its sampling variance is maximal at
    p = 0.5 -- where difficult pairs sit by construction -- so regressing it on the gap
    manufactures the predicted interaction out of noise. Drawn beneath the correct model
    line it would have looked like corroboration.

    The replacement pools raw counts per bin, so the row count is bounded by the bin
    count and never by the pair count. That bound is the observable difference between
    the two estimators, so it is what gets asserted.
    """
    from src.analysis import spread_model

    cfg, trials = _trials()
    design = spread_model.prepare(cfg, trials)
    n_bins = 4
    pts = plots._empirical_shift(design, "chose", n_bins)

    assert not pts.empty
    assert len(pts) <= n_bins, "one row per bin, not per pair"
    assert len(pts) < design.frame["pair_id"].nunique()
    assert (pts["n"] >= 8).all(), "bins below the drawing floor must be dropped"
    assert np.isfinite(pts[["shift", "lo", "hi"]].to_numpy()).all(), (
        "the continuity correction must keep a 0/1 bin finite")
    assert (pts["lo"] < pts["hi"]).all()
    assert "spread" not in design.frame.columns


def test_a_fitted_condition_that_no_panel_draws_is_an_error():
    """Silence was the defect, so the omission path must raise rather than omit."""
    from src.analysis import spread_model

    cfg, trials = _trials(n_pairs=12, n_tmpl=1)
    design = spread_model.prepare(cfg, trials)
    with pytest.raises(ValueError, match="no panel draws them"):
        plots.interaction_plot(cfg, design, None,
                               panels=(("only chose", ("chose",)),))


def test_planned_contrasts_all_reference_conditions_that_can_be_drawn():
    """The forest plot orders rows by PLANNED, so the two must not drift apart."""
    assert PRIMARY in PLANNED
    named = {c for a, b, _ in PLANNED.values() for c in (a, b)}
    assert named <= set(plots.CONDITION_STYLE), sorted(named - set(plots.CONDITION_STYLE))


@pytest.mark.slow
def test_both_figures_render_in_both_themes():
    """Bug 3: the module imported Design/PLANNED_CONTRASTS from the retired `mixed`.

    No test drew a figure, so nothing noticed that the plotting layer described a model
    `run.py` no longer fits. This renders.
    """
    from src.analysis import spread_model

    cfg, trials = _trials()
    fast = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
        update={"chains": 2, "tune": 400, "draws": 400, "sampler_cores": 1})})
    design = spread_model.prepare(fast, trials)
    idata = spread_model.fit(fast, design)
    sesoi = 0.15 * 1.573
    contrasts = spread_model.contrasts(fast, idata, sesoi)

    plots.use_paper_defaults()
    for theme in (plots.LIGHT, plots.DARK):
        fig = plots.interaction_plot(fast, design, idata, theme=theme)
        assert len(fig.axes) == len(plots.PANELS)
        fig = plots.forest_plot(fast, contrasts, sesoi, theme=theme)
        # the ROPE band and the zero line must both be present
        assert fig.axes[0].get_xlim()[1] > sesoi
