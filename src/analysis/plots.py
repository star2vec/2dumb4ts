"""Figures: the interaction plot and a forest plot of effect sizes with HDIs.

Palette is the validated categorical set, first five slots, in fixed order --
never cycled and never reassigned by rank. Both light and dark steps are the same
five hues re-stepped for their surface, validated as sets (light: worst adjacent
CVD dE 9.1, normal-vision 19.6; dark: 8.4 / 19.3).

Light mode raises a contrast warning on three slots, so the relief rule applies:
every series carries a visible direct label, and the notebook prints the contrast
table alongside. Identity is therefore never conveyed by color alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.mixed import PLANNED_CONTRASTS, PRIMARY, Design
from src.config import RunConfig

# Fixed categorical order. A 6th condition would fold into "other" or facet; it
# would never get a generated hue.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")

CONDITION_ORDER = ("chose", "yoked", "3p-yoked", "3p-random", "random")


@dataclass(frozen=True)
class Theme:
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    series: tuple[str, ...]

    def color(self, condition: str) -> str:
        return self.series[CONDITION_ORDER.index(condition) % len(self.series)]


LIGHT = Theme("#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e4e3de", SERIES_LIGHT)
DARK = Theme("#1a1a19", "#ffffff", "#c3c2b7", "#8a8880", "#333330", SERIES_DARK)


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
        ax.set_title(title, color=theme.text_primary, fontsize=11, loc="left", pad=10)


def _binned(frame: pd.DataFrame, condition: str, n_bins: int) -> pd.DataFrame:
    """Mean spread with a 95% t interval, per |diff| bin."""
    from scipy import stats

    block = frame[frame["condition"] == condition].copy()
    block["bin"] = pd.qcut(block["diff_analysis"], n_bins, duplicates="drop")
    rows = []
    for _, g in block.groupby("bin", observed=True):
        n = len(g)
        m = g["spread"].mean()
        se = g["spread"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
        rows.append(
            {
                "x": g["diff_analysis"].mean(),
                "mean": m,
                "lo": m - half,
                "hi": m + half,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def _fitted(
    idata: az.InferenceData, design: Design, condition: str, grid: np.ndarray, hdi_prob: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and HDI band for the condition's regression line.

    This is the MARGINAL prediction at the mean option order, not the fixed-effect
    line at order = 0. Omitting the `b_order` term offsets the whole line away from
    the binned means it is drawn against, which looks like a model misfit and is
    purely a plotting artefact. Random intercepts are zero-sum by construction, so
    they contribute nothing to the marginal.
    """
    draws = lambda var: (  # noqa: E731
        idata.posterior[var].sel(condition=condition).stack(sample=("chain", "draw")).to_numpy()
    )
    b, s = draws("b_cond"), draws("b_slope")
    order = (
        idata.posterior["b_order"].stack(sample=("chain", "draw")).to_numpy()
        * float(design.frame["order_c"].mean())
    )

    z = (grid - design.diff_mean) / design.diff_sd
    lines = (b + order)[:, None] + s[:, None] * z[None, :]  # [draws, grid]
    band = az.hdi(lines[None, ...], hdi_prob=hdi_prob)
    return lines.mean(axis=0), band[:, 0], band[:, 1]


def _stagger(ys: list[float], min_gap: float) -> list[float]:
    """Nudge direct-label positions apart so overlapping series stay readable."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for prev, cur in zip(order, order[1:]):
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def interaction_plot(
    cfg: RunConfig,
    design: Design,
    idata: az.InferenceData,
    *,
    dark: bool = False,
    n_bins: int = 5,
    figsize: tuple[float, float] = (11.0, 4.4),
):
    """Two panels: the primary contrast, then all five conditions.

    Left is the preregistered primary test -- `chose` against `yoked`, the only
    two conditions it involves. Right shows the full condition set so the
    rebuttal's mechanism (3p-*) is visible next to it. Both use ONE y axis.
    """
    theme = DARK if dark else LIGHT
    frame = design.frame
    conditions = [c for c in CONDITION_ORDER if c in set(frame["condition"])]
    grid = np.linspace(frame["diff_analysis"].min(), frame["diff_analysis"].max(), 60)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=theme.surface)

    for ax, subset, title in (
        (axes[0], [c for c in ("chose", "yoked") if c in conditions],
         "Primary test: agency x |diff| interaction"),
        (axes[1], conditions, "All conditions"),
    ):
        ends: list[tuple[str, float]] = []
        for condition in subset:
            color = theme.color(condition)
            mean, lo, hi = _fitted(idata, design, condition, grid, cfg.analysis.hdi_prob)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=1)
            ax.plot(grid, mean, color=color, linewidth=2.0, zorder=3, label=condition)
            ends.append((condition, float(mean[-1])))

            if ax is axes[0]:
                pts = _binned(frame, condition, n_bins)
                ax.errorbar(
                    pts["x"], pts["mean"],
                    yerr=[pts["mean"] - pts["lo"], pts["hi"] - pts["mean"]],
                    fmt="o", ms=7, color=color, ecolor=color, elinewidth=1.4,
                    capsize=0,
                    # 2px surface ring so overlapping markers stay separable
                    markeredgecolor=theme.surface, markeredgewidth=2.0, zorder=4,
                )

        ax.axhline(0, color=theme.text_muted, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)

        # Direct labels at the line ends, staggered so they never collide. Required
        # as relief for the light-mode contrast warning, and it means identity is
        # never conveyed by colour alone.
        span = np.ptp(ax.get_ylim())
        placed = _stagger([y for _, y in ends], min_gap=0.055 * span)
        for (condition, _), y_lab in zip(ends, placed):
            ax.annotate(
                condition,
                xy=(grid[-1], y_lab), xytext=(6, 0), textcoords="offset points",
                color=theme.text_secondary, fontsize=8.5, va="center", ha="left",
                annotation_clip=False,
            )
        _style(
            ax, theme,
            xlabel="|diff|  (pre-choice rating gap, rating points)",
            ylabel="spread  (points of divergence)" if ax is axes[0] else "",
            title=title,
        )
        ax.margins(x=0.14)

    # Legend is always present for >= 2 series.
    handles, labels = axes[1].get_legend_handles_labels()
    leg = axes[1].legend(
        handles, labels, frameon=False, fontsize=8.5, loc="best",
        labelcolor=theme.text_secondary,
    )
    for text in leg.get_texts():
        text.set_color(theme.text_secondary)

    fig.suptitle(
        f"Spreading of alternatives by choice difficulty  ({cfg.model.name})",
        color=theme.text_primary, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def forest_plot(
    cfg: RunConfig,
    contrasts: pd.DataFrame,
    sesoi: float,
    *,
    term: str = "slope",
    dark: bool = False,
    figsize: tuple[float, float] = (8.0, 3.6),
):
    """Effect sizes with credible intervals, and the ROPE band.

    One entity type per row, so no legend: the title names what is plotted. The
    preregistered primary contrast is distinguished by color AND by a bold label
    AND by an explicit annotation -- an entity distinction (which hypothesis),
    not a rank ordering.
    """
    theme = DARK if dark else LIGHT
    block = contrasts[contrasts["term"] == term].copy()
    order = [n for n in PLANNED_CONTRASTS if n in set(block["name"])]
    order += [n for n in block["name"] if n not in order]
    block = block.set_index("name").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=figsize, facecolor=theme.surface)
    y = np.arange(len(block))[::-1]

    # ROPE is a neutral band, never a status color.
    ax.axvspan(-sesoi, sesoi, color=theme.text_muted, alpha=0.13, linewidth=0, zorder=0)
    ax.axvline(0, color=theme.text_muted, linewidth=1.0, zorder=1)

    for yi, (_, row) in zip(y, block.iterrows()):
        is_primary = row["name"] == PRIMARY
        color = theme.color("chose") if is_primary else theme.text_secondary
        ax.plot(
            [row["hdi_low"], row["hdi_high"]], [yi, yi],
            color=color, linewidth=2.0, solid_capstyle="round", zorder=3,
        )
        ax.plot(
            row["median"], yi, "o", ms=7 if is_primary else 6, color=color,
            markeredgecolor=theme.surface, markeredgewidth=2.0, zorder=4,
        )
        ax.annotate(
            f"  {row['median']:+.3f}  [{row['hdi_low']:+.3f}, {row['hdi_high']:+.3f}]  {row['decision']}",
            xy=(ax.get_xlim()[1], yi), xytext=(6, 0), textcoords="offset points",
            color=theme.text_secondary, fontsize=8, va="center",
            annotation_clip=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n}  (PRIMARY)" if n == PRIMARY else n for n in block["name"]],
        fontsize=9, color=theme.text_primary,
    )
    for tick, name in zip(ax.get_yticklabels(), block["name"]):
        if name == PRIMARY:
            tick.set_fontweight("bold")

    label = (
        "interaction with |diff|  (spread points per SD of |diff|)"
        if term == "slope"
        else "main effect at mean |diff|  (spread points)"
    )
    _style(ax, theme, xlabel=label, title=f"Planned contrasts, {term} term ({cfg.model.name})")
    ax.grid(axis="y", visible=False)
    ax.annotate(
        f"shaded band = ROPE, +/-SESOI = {sesoi:.3f}",
        xy=(0.0, -0.28), xycoords="axes fraction",
        color=theme.text_muted, fontsize=8,
    )
    fig.tight_layout()
    return fig


def save(fig, path, *, dpi: int = 200) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


def use_paper_defaults() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )
