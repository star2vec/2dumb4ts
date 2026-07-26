"""Figures: the interaction plot and a forest plot of planned contrasts.

Rewritten against `spread_model` (A2.9.1). The previous version was wired to
`mixed`, the retired rating-scale Gaussian model, and was wrong in three ways that
all pointed the same direction -- see WHAT CHANGED below.

THE DV IS NOT A SPREAD. `spread_model` models the pre/post designated-versus-other
comparison directly, and its docstring states that it "never exposes a per-pair
spread: there is no function that returns one". The interaction lives only as
`lambda_c` inside the likelihood. So the y axis here is the quantity the model
actually estimates -- the post-manipulation shift in log-odds toward the designated
item -- and the data overlay is pooled within a gap bin, never per pair. See
`_empirical_shift` for why that distinction is load-bearing rather than pedantic.

PALETTE. Three hues, blue/orange/green, validated with the dataviz validator at
`--pairs all` in both modes with no warnings:

    light  #0072B2 #D55E00 #009E73   CVD dE 11.0 (deutan) - normal-vision 18.7
    dark   #3B8FD1 #E2703A #12A57B   CVD dE  9.6 (protan) - normal-vision 16.5

Eight conditions cannot be eight hues -- an 8-slot set fails the chroma floor and
collapses under deutan, and every 4-hue set that clears light mode leaves the
lightness band in dark. The prescribed remedy is to cut series or facet, so hue
carries ATTRIBUTION (three levels) and line style carries the remaining factors.
That is composite encoding, and it has the advantage of matching A2.9.3's factorial
structure rather than treating the conditions as eight arbitrary categories.
No panel shows more than three hues, every series is direct-labeled, and hue is
never reassigned by rank.

WHAT CHANGED, and why each mattered:

1. `_binned` averaged a per-pair `spread` column inside |diff| bins. That is exactly
   the two-stage estimator `spread_model` forbids: its sampling variance is maximal
   at p = 0.5, which is where difficult pairs sit BY CONSTRUCTION, so it manufactures
   the predicted interaction out of noise. Drawn as the data layer beneath the correct
   model line, it would have shown points appearing to confirm H1 for reasons that are
   pure noise geometry -- the most expensive kind of figure bug, because it looks like
   corroboration.

2. `CONDITION_ORDER` listed five conditions; the design has eight. The panel builder
   filtered the frame THROUGH that tuple, so `structure-control`, `self-recounted` and
   `chose-provisional` were silently dropped -- the three conditions A2.9.3 was written
   to add. No error, just missing series.

3. The module imported `PLANNED_CONTRASTS`, `PRIMARY` and `Design` from `mixed`, which
   `run.py` no longer fits. The figures described a model nobody reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.spread_model import PLANNED, PRE_SENTINEL, PRIMARY, SpreadDesign
from src.config import RunConfig

#: Hue carries attribution; line style carries transcript structure and designation.
#: Fixed order, never cycled, never reassigned by rank.
ATTRIBUTION_ORDER = ("self", "none", "third-party")

HUES_LIGHT = {"self": "#0072B2", "none": "#D55E00", "third-party": "#009E73"}
HUES_DARK = {"self": "#3B8FD1", "none": "#E2703A", "third-party": "#12A57B"}

#: condition -> (attribution, line style, human-readable style gloss)
CONDITION_STYLE: dict[str, tuple[str, str, str]] = {
    "chose":             ("self",        "solid",  "choice turn present"),
    "self-recounted":    ("self",        "dashed", "no choice turn"),
    "chose-provisional": ("self",        "dotted", "provisional (reversible)"),
    "structure-control": ("none",        "solid",  "choice turn present"),
    "yoked":             ("none",        "dashed", "no choice turn"),
    "random":            ("none",        "dotted", "random designation"),
    "3p-yoked":          ("third-party", "solid",  "own pick designated"),
    "3p-random":         ("third-party", "dotted", "random designation"),
}

DASHES = {"solid": (None, None), "dashed": (5, 2), "dotted": (1.4, 1.8)}

#: Facets. Each is (title, conditions) and each maps onto a contrast family in
#: `spread_model.PLANNED`, so the figure is organised by the argument rather than by
#: what happens to fit on one axis.
PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Primary: chose vs yoked", ("chose", "yoked")),
    ("A2.9.3 2x2: structure x wording",
     ("chose", "structure-control", "self-recounted", "yoked")),
    ("Attribution and the rebuttal mechanism",
     ("chose", "3p-yoked", "3p-random", "random")),
    ("Reversibility (A2.8)", ("chose", "chose-provisional")),
)


@dataclass(frozen=True)
class Theme:
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    hues: dict[str, str]

    def color(self, condition: str) -> str:
        attribution, _, _ = CONDITION_STYLE[condition]
        return self.hues[attribution]

    def dashes(self, condition: str) -> tuple:
        return DASHES[CONDITION_STYLE[condition][1]]


LIGHT = Theme("#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3de", HUES_LIGHT)
DARK = Theme("#1a1a19", "#ffffff", "#c3c2b7", "#8a8880", "#333330", HUES_DARK)


def _style(ax, theme: Theme, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive axes and grid; text wears ink tokens, never a series color."""
    ax.set_facecolor(theme.surface)
    ax.grid(True, color=theme.grid, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=theme.text_secondary, labelsize=9, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme.text_secondary, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme.text_secondary, fontsize=10)
    if title:
        ax.set_title(title, color=theme.text_primary, fontsize=10.5, loc="left", pad=8)


# ---------------------------------------------------------------------------
# the two layers of the interaction plot


def _model_line(
    idata: az.InferenceData, condition: str, grid_z: np.ndarray, hdi_prob: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and HDI for `gamma_c + lambda_c * z`, the modelled post shift.

    This is the estimand itself, not a summary of one. `lambda_c` is the slope, and the
    primary test is whether `chose`'s slope is more negative than `yoked`'s.
    """
    sel = dict(condition=condition)
    gamma = idata.posterior["gamma"].sel(**sel).stack(s=("chain", "draw")).to_numpy()
    lam = idata.posterior["lambda"].sel(**sel).stack(s=("chain", "draw")).to_numpy()
    lines = gamma[:, None] + lam[:, None] * grid_z[None, :]   # (draws, grid)
    band = az.hdi(lines[None, ...], hdi_prob=hdi_prob)        # (grid, 2)
    band = np.asarray(band)
    return lines.mean(0), band[:, 0], band[:, 1]


def _empirical_shift(
    design: SpreadDesign, condition: str, n_bins: int
) -> pd.DataFrame:
    """Observed post-minus-pre shift toward the designated item, POOLED within a gap bin.

    Why this and not a per-pair spread. `spread_model` refuses to expose a per-pair
    spread because that quantity's sampling variance depends on where the pair sits on
    the logit curve -- maximal at p = 0.5, which is exactly where difficult pairs sit by
    construction. Regressing it on the gap manufactures the predicted interaction out of
    noise. Averaging per-pair spreads inside a bin inherits the same defect, because the
    bin mean is still a mean of per-pair ratios.

    Pooling the raw counts first avoids it: one proportion at pre and one at post per
    bin, each estimated from many observations, differenced on the logit scale. The
    binomial variance still varies with p, but it is no longer amplified by a per-pair
    division, and the interval below reports it honestly.

    This is a DESCRIPTIVE overlay and nothing is inferred from it. Pooling ignores
    `u_pair`, and logits are not collapsible, so the pooled shift is not the average of
    the per-pair shifts the model estimates. It is here to show that the model line is
    not tracking something absent from the data.
    """
    f = design.frame
    post = f[(f["condition"] == condition) & (f["timepoint"] == "post")]
    if post.empty:
        return pd.DataFrame(columns=["gap", "shift", "lo", "hi", "n"])

    # The pre measurement is shared across conditions -- one per (pair, template, order),
    # keyed here by the slot-1 item since that is what fixes the order. It carries no
    # designation of its own, so it is oriented by THIS condition's `d`.
    pre = f[f["condition"] == PRE_SENTINEL]
    if pre.empty:
        pre = f[(f["condition"] == condition) & (f["timepoint"] == "pre")]
    key = ["pair_id", "template", "slot1_item_id"]
    pre_map = pre.set_index(key)["item1_wins"]

    joined = post.set_index(key)
    matched = joined.index.isin(pre_map.index)
    joined = joined[matched]
    if joined.empty:
        return pd.DataFrame(columns=["gap", "shift", "lo", "hi", "n"])
    pre_out = pre_map.reindex(joined.index).to_numpy(dtype=float)

    d = joined["d"].to_numpy(dtype=float)
    post_designated = np.where(d > 0, joined["item1_wins"], 1 - joined["item1_wins"])
    pre_designated = np.where(d > 0, pre_out, 1 - pre_out)

    gap = joined["diff_analysis"].to_numpy(dtype=float)
    edges = np.quantile(gap, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(gap, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n < 8:                       # too few to draw honestly
            continue
        # Empirical logit with a continuity correction, so a bin at 0 or 1 is finite.
        def elogit(x):
            k = float(np.nansum(x[m]))
            return (np.log((k + 0.5) / (n - k + 0.5)),
                    1.0 / (k + 0.5) + 1.0 / (n - k + 0.5))
        lp, vp = elogit(post_designated)
        lq, vq = elogit(pre_designated)
        se = float(np.sqrt(vp + vq))
        rows.append({"gap": float(gap[m].mean()), "shift": lp - lq,
                     "lo": lp - lq - 1.96 * se, "hi": lp - lq + 1.96 * se, "n": n})
    return pd.DataFrame(rows)


def _stagger(ys: list[float], min_gap: float) -> list[float]:
    """Nudge direct-label positions apart so overlapping series stay readable."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = list(ys)
    for prev, cur in zip(order, order[1:]):
        if placed[cur] - placed[prev] < min_gap:
            placed[cur] = placed[prev] + min_gap
    return placed


def interaction_plot(
    cfg: RunConfig,
    design: SpreadDesign,
    idata: az.InferenceData,
    *,
    theme: Theme = LIGHT,
    n_bins: int = 5,
    show_data: bool = True,
    panels: tuple[tuple[str, tuple[str, ...]], ...] = PANELS,
):
    """The primary figure: modelled post shift against |diff|, faceted by contrast family.

    A negative slope means the manipulation moves the model toward the designated item
    MORE on difficult pairs (small |diff|). H1 predicts `chose` slopes more steeply than
    `yoked`, which reads as the blue line diverging from the orange one to the left.
    """
    available = set(design.conditions)

    # The previous version filtered the frame through a five-condition tuple and dropped
    # the other three without a word. Silence is the defect, so a fitted condition that
    # no panel would draw is an error rather than an omission.
    covered = {c for _, cs in panels for c in cs}
    unplotted = sorted(available - covered)
    if unplotted:
        raise ValueError(
            f"conditions were fitted but no panel draws them: {unplotted}. Add them to a "
            "panel or drop them from cfg.pass_c.conditions -- refusing to omit silently."
        )
    missing_style = sorted(available - set(CONDITION_STYLE))
    if missing_style:
        raise ValueError(f"no hue/style assigned for: {missing_style}")

    live = [(t, [c for c in cs if c in available]) for t, cs in panels]
    live = [(t, cs) for t, cs in live if cs]
    if not live:
        raise ValueError(f"no panel condition present in the fit: {sorted(available)}")

    gap = design.frame["diff_analysis"].to_numpy(dtype=float)
    grid_raw = np.linspace(float(gap.min()), float(gap.max()), 120)
    grid_z = (grid_raw - design.diff_mean) / design.diff_sd

    fig, axes = plt.subplots(1, len(live), figsize=(4.1 * len(live), 3.9),
                             sharey=True, facecolor=theme.surface)
    axes = np.atleast_1d(axes)

    for ax, (title, conditions) in zip(axes, live):
        ends = []
        # Bin centres are shared across conditions, so without an offset the overlay
        # points land on top of each other and the panel reads as one smear.
        span = float(grid_raw[-1] - grid_raw[0])
        offsets = np.linspace(-0.012, 0.012, len(conditions)) * span * len(conditions)
        for condition, dx in zip(conditions, offsets):
            color, dash = theme.color(condition), theme.dashes(condition)
            mean, lo, hi = _model_line(idata, condition, grid_z, cfg.analysis.hdi_prob)
            ax.fill_between(grid_raw, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
            line, = ax.plot(grid_raw, mean, color=color, linewidth=2.0, zorder=3)
            if dash != (None, None):
                line.set_dashes(dash)
            ends.append((condition, float(mean[-1])))

            if show_data:
                pts = _empirical_shift(design, condition, n_bins)
                if not pts.empty:
                    ax.errorbar(pts["gap"] + dx, pts["shift"],
                                yerr=[pts["shift"] - pts["lo"], pts["hi"] - pts["shift"]],
                                fmt="o", markersize=4.5, color=color, ecolor=color,
                                elinewidth=1.0, capsize=0, alpha=0.75, zorder=4,
                                markeredgecolor=theme.surface, markeredgewidth=1.2)

        ax.axhline(0.0, color=theme.text_muted, linewidth=1.0, zorder=1)
        ylim = ax.get_ylim()
        placed = _stagger([y for _, y in ends], min_gap=0.075 * (ylim[1] - ylim[0]))
        for (condition, _), y in zip(ends, placed):
            ax.annotate(condition, xy=(grid_raw[-1], y),
                        xytext=(4, 0), textcoords="offset points",
                        color=theme.text_primary, fontsize=8.5,
                        va="center", ha="left", clip_on=False)
        _style(ax, theme, xlabel="|diff|  (analysis gap, logits)", title=title)

    axes[0].set_ylabel("post shift toward designated item  (log-odds)",
                       color=theme.text_secondary, fontsize=10)
    fig.subplots_adjust(right=0.86)

    # Identity is never color-alone: hue is glossed here, style is glossed per condition
    # by the direct labels plus this key.
    handles = [mpl.lines.Line2D([], [], color=theme.hues[a], linewidth=2.0,
                                label=f"attribution: {a}") for a in ATTRIBUTION_ORDER]
    handles += [mpl.lines.Line2D([], [], color=theme.text_muted, linewidth=1.6,
                                 dashes=DASHES[s] if DASHES[s] != (None, None) else (),
                                 label=g)
                for s, g in (("solid", "turn present / own pick"),
                             ("dashed", "no choice turn"),
                             ("dotted", "provisional or random"))]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    for text in leg.get_texts():
        text.set_color(theme.text_secondary)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# forest plot


def forest_plot(
    cfg: RunConfig,
    contrasts: pd.DataFrame,
    sesoi: float,
    *,
    theme: Theme = LIGHT,
    term: str = "lambda",
):
    """Planned contrasts on the difficulty slope, with HDIs and the ROPE.

    One mark per contrast, in ink -- the decision is written out as text rather than
    encoded in color, so no reader has to resolve a legend to know what passed. The
    ROPE band is the equivalence region; a contrast whose whole interval sits inside it
    is evidence FOR the null, which is a different statement from failing to exclude 0.
    """
    rows = contrasts[contrasts["term"] == term].copy()
    if rows.empty:
        raise ValueError(f"no contrasts with term={term!r}")
    order = [n for n in PLANNED if n in set(rows["name"])]
    rows = rows.set_index("name").loc[order].reset_index()
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.5), facecolor=theme.surface)
    ax.axvspan(-sesoi, sesoi, color=theme.text_muted, alpha=0.14, linewidth=0, zorder=1)
    ax.axvline(0.0, color=theme.text_secondary, linewidth=1.0, zorder=2)

    for yi, (_, r) in zip(y, rows.iterrows()):
        primary = r["name"] == PRIMARY
        color = theme.color("chose") if primary else theme.text_primary
        ax.plot([r["hdi_low"], r["hdi_high"]], [yi, yi], color=color,
                linewidth=2.4 if primary else 1.6, solid_capstyle="round", zorder=3)
        ax.plot([r["median"]], [yi], "o", markersize=7 if primary else 5.5,
                color=color, markeredgecolor=theme.surface, markeredgewidth=1.4, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["name"], color=theme.text_primary, fontsize=9)
    for tick, name in zip(ax.get_yticklabels(), rows["name"]):
        if name == PRIMARY:
            tick.set_fontweight("bold")

    # Decision as text INSIDE the axes. Annotating past `xlim` with clip_on=False put the
    # labels outside the tight bbox and they were silently cropped from the saved file, so
    # the room is made explicitly instead.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.34 * (hi - lo))
    x_text = hi + 0.03 * (hi - lo)
    for yi, (_, r) in zip(y, rows.iterrows()):
        label = str(r["decision"])
        if r["name"] == PRIMARY:
            label += "   (primary)"
        ax.annotate(label, xy=(x_text, yi), fontsize=8,
                    color=theme.text_secondary, va="center", ha="left")

    _style(ax, theme,
           xlabel=f"lambda contrast  (log-odds per SD of |diff|)      ROPE +/-{sesoi:.3f}",
           title=f"Planned contrasts on the difficulty slope  ({cfg.model.name})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------


def save(fig, path, *, dpi: int = 300) -> None:
    """300 dpi and a tight box; PDF stays vector for the camera-ready."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")


def use_paper_defaults() -> None:
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,     # embed TrueType, not Type 3 -- required by most venues
        "ps.fonttype": 42,
        "figure.facecolor": LIGHT.surface,
    })
