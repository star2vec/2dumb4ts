"""Pass B -- pair construction, per model, from that model's own Pass A ratings.

No shared, cross-model, or human-intuited pair list is used anywhere: difficulty
is defined by each model's own rating distribution.

Two things make this more than a quantile filter:

1. DIFFICULTY IS SELECTED on mean(T1-T3) but ANALYSED on mean(T4-T5). Selecting
   on a noisy |diff| and regressing on that same noisy |diff| would make the
   regressor regression-contaminated. The disjoint template split gives an
   independent measurement of the same underlying quantity.

2. DIFFICULT AND EASY PAIRS ARE MATCHED on mean pair rating. Difficulty is
   otherwise confounded with extremity, and ceiling/floor compression can
   manufacture the difficulty interaction with no mechanism behind it. Matching
   removes the confound; a covariate would only model it.

Selection is deterministic given the config: candidates are ordered by
(|diff|, pair_id), so there is no per-run randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RunConfig
from src.provenance import Provenance, read_parquet, write_parquet


def artifact_path(cfg: RunConfig) -> Path:
    return cfg.artifact_dir("pass_b") / f"pairs_{cfg.hash('pass_b')}.parquet"


@dataclass(frozen=True)
class Candidate:
    pair_id: str
    domain: str
    item1: str
    item2: str
    diff_selection: float
    mean_selection: float
    diff_analysis: float
    mean_analysis: float


def _short(item_id: str) -> str:
    return item_id.split("/", 1)[1] if "/" in item_id else item_id


def _candidates(domain: str, block: pd.DataFrame) -> list[Candidate]:
    """All within-domain pairs. Canonical member order is lexical by item id."""
    rows = block.sort_values("item_id").to_dict("records")
    out = []
    for a, b in combinations(rows, 2):
        out.append(
            Candidate(
                pair_id=f"{domain}/{_short(a['item_id'])}|{_short(b['item_id'])}",
                domain=domain,
                item1=a["item_id"],
                item2=b["item_id"],
                diff_selection=abs(a["score_selection"] - b["score_selection"]),
                mean_selection=(a["score_selection"] + b["score_selection"]) / 2.0,
                diff_analysis=abs(a["score_analysis"] - b["score_analysis"]),
                mean_analysis=(a["score_analysis"] + b["score_analysis"]) / 2.0,
            )
        )
    return out


def build_pairs(cfg: RunConfig, scores: pd.DataFrame) -> pd.DataFrame:
    n_diff, n_easy = cfg.n_difficult, cfg.n_easy
    domains = list(cfg.stimuli.domains)

    if cfg.pass_b.match_on_mean_rating and n_diff != n_easy:
        raise ValueError(
            "mean-rating matching pairs each difficult pair with one easy pair, so "
            f"n_difficult ({n_diff}) must equal n_easy ({n_easy})"
        )
    if n_diff % len(domains) or n_easy % len(domains):
        raise ValueError(
            f"difficulty x domain balance requires n_difficult ({n_diff}) and n_easy "
            f"({n_easy}) to divide evenly by {len(domains)} domains"
        )
    per_domain = n_diff // len(domains)

    selected: list[dict] = []
    for domain in domains:
        block = scores[scores["domain"] == domain]
        cands = _candidates(domain, block)
        if not cands:
            raise ValueError(f"no candidate pairs in domain {domain!r}")

        diffs = np.array([c.diff_selection for c in cands])
        # Quantiles are per domain: this guarantees the difficulty pools are large
        # enough in every domain for the required domain balance. Both the
        # per-domain thresholds and the realized distributions are reported.
        q_hard = float(np.quantile(diffs, cfg.pass_b.difficult_quantile))
        q_easy = float(np.quantile(diffs, cfg.pass_b.easy_quantile))

        hard_pool = sorted(
            (c for c in cands if c.diff_selection <= q_hard),
            key=lambda c: (c.diff_selection, c.pair_id),
        )
        easy_pool = sorted(
            (c for c in cands if c.diff_selection >= q_easy),
            key=lambda c: (-c.diff_selection, c.pair_id),
        )
        selected.extend(
            _match_domain(cfg, domain, hard_pool, easy_pool, per_domain, q_hard, q_easy)
        )

    frame = pd.DataFrame(selected)
    _validate(cfg, frame)
    return frame


def _match_domain(
    cfg: RunConfig,
    domain: str,
    hard_pool: list[Candidate],
    easy_pool: list[Candidate],
    target: int,
    q_hard: float,
    q_easy: float,
) -> list[dict]:
    """Greedy matched selection under the item-reuse and level-exclusivity rules."""
    uses: dict[str, int] = {}
    level: dict[str, str] = {}
    max_uses = cfg.pass_b.max_uses_per_item
    tol = cfg.pass_b.match_tolerance

    def feasible(c: Candidate, lvl: str) -> bool:
        for item in (c.item1, c.item2):
            if uses.get(item, 0) >= max_uses:
                return False
            # No item may appear in both difficulty levels.
            if level.get(item, lvl) != lvl:
                return False
        return True

    def commit(c: Candidate, lvl: str) -> None:
        for item in (c.item1, c.item2):
            uses[item] = uses.get(item, 0) + 1
            level[item] = lvl

    out: list[dict] = []
    easy_taken: set[str] = set()
    n_sets = 0

    for hard in hard_pool:
        if n_sets >= target:
            break
        if not feasible(hard, "difficult"):
            continue

        best: Candidate | None = None
        best_key: tuple | None = None
        for easy in easy_pool:
            if easy.pair_id in easy_taken or not feasible(easy, "easy"):
                continue
            if {easy.item1, easy.item2} & {hard.item1, hard.item2}:
                continue
            gap = abs(hard.mean_selection - easy.mean_selection)
            if cfg.pass_b.match_on_mean_rating and gap > tol:
                continue
            # Prefer the most extreme easy pair, then the closest mean match.
            key = (-easy.diff_selection, gap, easy.pair_id)
            if best_key is None or key < best_key:
                best, best_key = easy, key
        if best is None:
            continue

        commit(hard, "difficult")
        commit(best, "easy")
        easy_taken.add(best.pair_id)
        matched_set = f"{domain}/ms{n_sets:02d}"
        n_sets += 1
        for cand, lvl in ((hard, "difficult"), (best, "easy")):
            out.append(
                {
                    "pair_id": cand.pair_id,
                    "domain": domain,
                    "difficulty": lvl,
                    "matched_set": matched_set,
                    "item1_id": cand.item1,
                    "item2_id": cand.item2,
                    "diff_selection": cand.diff_selection,
                    "mean_selection": cand.mean_selection,
                    "diff_analysis": cand.diff_analysis,
                    "mean_analysis": cand.mean_analysis,
                    "domain_q_difficult": q_hard,
                    "domain_q_easy": q_easy,
                }
            )

    if n_sets < target:
        raise RuntimeError(
            f"domain {domain!r}: only {n_sets} of {target} matched sets could be built "
            f"under max_uses_per_item={max_uses}, match_tolerance={tol}. Widen the "
            "tolerance or enlarge the item pool -- do NOT silently accept fewer pairs, "
            "which would break difficulty x domain balance."
        )
    return out


def _validate(cfg: RunConfig, pairs: pd.DataFrame) -> None:
    """Assert every design constraint that Pass C and the analysis rely on."""
    problems: list[str] = []

    counts = pairs.groupby(["domain", "difficulty"]).size()
    expected = cfg.n_difficult // len(cfg.stimuli.domains)
    for (domain, difficulty), n in counts.items():
        if n != expected:
            problems.append(f"{domain}/{difficulty}: {n} pairs, expected {expected}")

    long = pd.concat(
        [
            pairs[["item1_id", "difficulty"]].rename(columns={"item1_id": "item_id"}),
            pairs[["item2_id", "difficulty"]].rename(columns={"item2_id": "item_id"}),
        ]
    )
    uses = long.groupby("item_id").size()
    over = uses[uses > cfg.pass_b.max_uses_per_item]
    if len(over):
        problems.append(
            f"{len(over)} item(s) exceed max_uses_per_item="
            f"{cfg.pass_b.max_uses_per_item}: {list(over.index[:5])}"
        )

    levels = long.groupby("item_id")["difficulty"].nunique()
    both = levels[levels > 1]
    if len(both):
        problems.append(
            f"{len(both)} item(s) appear in both difficulty levels: {list(both.index[:5])}"
        )

    if pairs["pair_id"].duplicated().any():
        problems.append("duplicate pair_id in the selected set")

    same = pairs[pairs["item1_id"] == pairs["item2_id"]]
    if len(same):
        problems.append(f"{len(same)} pair(s) contain the same item twice")

    if problems:
        raise RuntimeError("Pass B design violations:\n  - " + "\n  - ".join(problems))


def pair_diagnostics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Realized |diff| and matching quality, reported per preregistration.md 6.1."""
    rows = []
    for difficulty, block in pairs.groupby("difficulty"):
        rows.append(
            {
                "difficulty": difficulty,
                "n_pairs": len(block),
                "diff_sel_mean": block["diff_selection"].mean(),
                "diff_sel_sd": block["diff_selection"].std(ddof=1),
                "diff_sel_min": block["diff_selection"].min(),
                "diff_sel_max": block["diff_selection"].max(),
                "diff_ana_mean": block["diff_analysis"].mean(),
                "diff_ana_sd": block["diff_analysis"].std(ddof=1),
                "mean_rating_mean": block["mean_analysis"].mean(),
                "mean_rating_sd": block["mean_analysis"].std(ddof=1),
            }
        )
    diag = pd.DataFrame(rows)

    # Matching quality: within-set gap in mean pair rating, on the selection score
    # it was matched on and on the independent analysis score.
    wide = pairs.pivot_table(
        index="matched_set", columns="difficulty", values=["mean_selection", "mean_analysis"]
    )
    if {"difficult", "easy"} <= set(wide["mean_selection"].columns):
        gap_sel = (wide["mean_selection"]["difficult"] - wide["mean_selection"]["easy"]).abs()
        gap_ana = (wide["mean_analysis"]["difficult"] - wide["mean_analysis"]["easy"]).abs()
        diag.attrs["match_gap_selection_mean"] = float(gap_sel.mean())
        diag.attrs["match_gap_selection_max"] = float(gap_sel.max())
        diag.attrs["match_gap_analysis_mean"] = float(gap_ana.mean())
        diag.attrs["match_gap_analysis_max"] = float(gap_ana.max())
    return diag


def run_pass_b(cfg: RunConfig, scores: pd.DataFrame, prov: Provenance) -> pd.DataFrame:
    pairs = build_pairs(cfg, scores)
    return write_parquet(pairs, artifact_path(cfg), prov)


def load_or_run(cfg: RunConfig, scores: pd.DataFrame, prov: Provenance) -> pd.DataFrame:
    path = artifact_path(cfg)
    if path.exists():
        return read_parquet(path)
    return run_pass_b(cfg, scores, prov)
