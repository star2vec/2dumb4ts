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
    #: SIGNED analysis-split theta per member. `diff_analysis` is |theta1 - theta2| and
    #: so cannot say WHICH item the instrument prefers -- which A3.12's framing-transfer
    #: check needs, because it asks whether the choice-framed scale predicts the
    #: prefer-framed DV baseline.
    theta_item1: float
    theta_item2: float


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
                theta_item1=float(a["score_analysis"]),
                theta_item2=float(b["score_analysis"]),
            )
        )
    return out


def build_pairs(cfg: RunConfig, scores: pd.DataFrame, sigma_item: float) -> pd.DataFrame:
    """Matched pair construction. `sigma_item` sets the matching tolerance (A2.9.2).

    `sigma_item` is required rather than defaulted: a default would silently reinstate the
    fixed-constant bug it exists to fix, and would do so invisibly, since a too-strict
    tolerance changes WHICH easy pairs are eligible without changing how many are built.
    """
    n_diff, n_easy = cfg.n_difficult, cfg.n_easy
    tol = cfg.pass_b.match_tolerance(sigma_item)
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
            _match_domain(cfg, domain, hard_pool, easy_pool, per_domain, q_hard, q_easy,
                          tol)
        )

    frame = pd.DataFrame(selected)
    # The REALIZED tolerance, not the rule. The config hash keys on the fraction, which is
    # constant across models, while the value actually applied depends on sigma_item -- and
    # sigma_item comes from the instrument fit, whose estimator source is NOT in the pass_b
    # hash. Two different pair sets could otherwise share one hash. Recorded per row so it
    # survives the parquet round-trip and can be checked on cache reuse.
    frame["match_tolerance_realized"] = tol
    frame["sigma_item"] = float(sigma_item)
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
    tol: float,
) -> list[dict]:
    """Greedy matched selection under the item-reuse and level-exclusivity rules."""
    uses: dict[str, int] = {}
    level: dict[str, str] = {}
    max_uses = cfg.pass_b.max_uses_per_item

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
                    "theta_item1": cand.theta_item1,
                    "theta_item2": cand.theta_item2,
                    "mean_analysis": cand.mean_analysis,
                    "domain_q_difficult": q_hard,
                    "domain_q_easy": q_easy,
                }
            )

    if n_sets < target:
        raise RuntimeError(
            f"domain {domain!r}: only {n_sets} of {target} matched sets could be built "
            f"under max_uses_per_item={max_uses}, match_tolerance={tol:.4f} "
            f"(= {cfg.pass_b.match_tolerance_sigma_fraction} x sigma_item). Widen the "
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


def run_pass_b(cfg: RunConfig, scores: pd.DataFrame, prov: Provenance,
               sigma_item: float) -> pd.DataFrame:
    pairs = build_pairs(cfg, scores, sigma_item)
    return write_parquet(pairs, artifact_path(cfg), prov)


def assert_tolerance_matches(cfg: RunConfig, pairs: pd.DataFrame, sigma_item: float) -> None:
    """Refuse a cached Pass B built under a different realized tolerance.

    The pass_b config hash covers `match_tolerance_sigma_fraction`, which is the same for
    every model, but the applied tolerance is that fraction times `sigma_item` -- and
    `sigma_item` comes from the instrument fit, keyed on the SOURCE digest of the estimator
    rather than on anything in the config. Change the estimator and the tolerance moves
    while the pass_b hash stands still. That is the stimulus-digest failure again: one hash,
    two artifacts. This makes the collision loud instead of silent.
    """
    if "match_tolerance_realized" not in pairs.columns:
        raise ValueError(
            f"cached Pass B at {artifact_path(cfg)} predates A3.9 and carries no realized "
            "match tolerance, so it was built with the fixed 0.15-logit constant. Delete it "
            "and rebuild -- it cannot be verified and must not be reused."
        )
    if pairs.empty:
        raise ValueError(f"cached Pass B at {artifact_path(cfg)} is empty")
    want = cfg.pass_b.match_tolerance(sigma_item)
    got = float(pairs["match_tolerance_realized"].iloc[0])
    if abs(got - want) > 1e-9:
        raise ValueError(
            f"cached Pass B was built with match_tolerance={got:.6f} but this run computes "
            f"{want:.6f} ({cfg.pass_b.match_tolerance_sigma_fraction} x sigma_item="
            f"{sigma_item:.6f}). The pass_b hash cannot see this difference. Delete "
            f"{artifact_path(cfg)} and rebuild."
        )


def load_or_run(cfg: RunConfig, scores: pd.DataFrame, prov: Provenance,
                sigma_item: float) -> pd.DataFrame:
    path = artifact_path(cfg)
    if path.exists():
        pairs = read_parquet(path)
        assert_tolerance_matches(cfg, pairs, sigma_item)
        return pairs
    return run_pass_b(cfg, scores, prov, sigma_item)


def match_gap_exclusions(cfg: RunConfig, pairs: pd.DataFrame, sigma_item: float) -> dict:
    """Matched sets whose residual imbalance exceeds the preregistered tolerance (A3.13).

    WHY THE OBVIOUS VERSION IS VACUOUS. "Exclude pairs that exceeded their own tolerance"
    can never exclude anything on the SELECTION score: `_match_domain` applies
    `if gap > tol: continue` as a hard filter at construction and raises rather than accept
    fewer sets, so every committed set satisfies `gap <= tol` by construction. The realized
    maxima press right against the bound -- 0.406 against a tolerance of 0.409 for gemma --
    which is the signature of a hard filter, not of comfortable matching.

    WHAT IS NOT BOUNDED. Matching is enforced on `mean_selection`, from templates T1-T3.
    The analysis measurement `mean_analysis`, from the disjoint T4-T5, is a SEPARATE
    estimate of the same quantity and nothing constrains it. So residual imbalance on the
    analysis scale can exceed the tolerance, and that is the leakage the concern actually
    points at: difficulty confounded with extremity through the measurement the primary
    model uses, which is what design-level matching existed to remove.

    The threshold is the preregistered `0.26 * sigma_item` (A2.9.2), applied to the
    analysis gap. No new number is introduced.
    """
    tol = cfg.pass_b.match_tolerance(sigma_item)
    wide = pairs.pivot_table(
        index="matched_set", columns="difficulty",
        values=["mean_selection", "mean_analysis"],
    )
    if not {"difficult", "easy"} <= set(wide["mean_analysis"].columns):
        raise ValueError("pairs do not contain both difficulty levels")

    gap_sel = (wide["mean_selection"]["difficult"] - wide["mean_selection"]["easy"]).abs()
    gap_ana = (wide["mean_analysis"]["difficult"] - wide["mean_analysis"]["easy"]).abs()
    over = gap_ana[gap_ana > tol]

    return {
        "tolerance": float(tol),
        "n_matched_sets": int(len(gap_ana)),
        # Selection-scale exclusions are structurally impossible; reported so the zero is
        # read as "enforced upstream" rather than "checked and found clean".
        "n_over_on_selection": int((gap_sel > tol + 1e-9).sum()),
        "selection_is_bounded_by_construction": True,
        "n_over_on_analysis": int(len(over)),
        "frac_over_on_analysis": float(len(over) / len(gap_ana)) if len(gap_ana) else 0.0,
        "gap_analysis_max": float(gap_ana.max()),
        "gap_analysis_mean": float(gap_ana.mean()),
        "excluded_matched_sets": sorted(over.index.tolist()),
    }
