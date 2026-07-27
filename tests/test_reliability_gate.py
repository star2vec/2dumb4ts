"""A2.2's reliability gate -- the sole exclusion criterion, previously untested.

Nothing named `evaluate_reliability_gate`. It decides which models enter the study, and
that decision reaches the ladder, the project gate (A3.7), and the paper's model set. It
was found untested by a static reference map, not by reading.

The gate fits Bradley-Terry twice on disjoint template splits, so these are slow. They are
kept honest rather than fast: a mocked test_retest would assert only that a comparison
operator works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.experiments.pass_a_pairwise import evaluate_reliability_gate

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


def _comparisons(signal: float, n_items: int = 24, n_anchor: int = 4, seed: int = 0):
    """Anchor comparisons whose theta recoverability is controlled by `signal`.

    `signal = 0` gives coin flips -- no item ordering exists to recover, so the two
    template splits must disagree. Large `signal` gives a near-deterministic ordering
    both splits recover.
    """
    rng = np.random.default_rng(seed)
    items = [f"destinations/i{i:02d}" for i in range(n_items)]
    anchors = [f"anchor{a}" for a in range(n_anchor)]
    truth = dict(zip(items, np.linspace(-1.5, 1.5, n_items)))
    a_str = dict(zip(anchors, np.linspace(-1.0, 1.0, n_anchor)))

    rows = []
    for t in range(5):
        for item in items:
            for anchor in anchors:
                for order in (0, 1):
                    p = 1 / (1 + np.exp(-signal * (truth[item] - a_str[anchor])))
                    rows.append({
                        "arm": "digits", "template": f"t{t}", "item_id": item,
                        "anchor_id": anchor, "order": order,
                        "item_wins": bool(rng.random() < p),
                        "readout_valid": True, "readout_mass": 0.99,
                    })
    return pd.DataFrame(rows)


class _T:
    def __init__(self, i):
        self.id = f"t{i}"


TEMPLATES = [_T(i) for i in range(5)]


@pytest.mark.slow
def test_a_recoverable_item_ordering_passes_the_gate():
    cfg = load_config(CONFIG)
    gate = evaluate_reliability_gate(cfg, _comparisons(signal=3.0, seed=1), TEMPLATES)

    assert gate["threshold"] == cfg.gates.reliability_min
    assert gate["split_a"] == ["t0", "t1", "t2"] and gate["split_b"] == ["t3", "t4"]
    assert set(gate["split_a"]).isdisjoint(gate["split_b"]), "4.4's split must be disjoint"
    assert gate["empirical_reliability_spearman"] >= cfg.gates.reliability_min
    assert gate["passed"] is True
    assert gate["reasons"] == []


@pytest.mark.slow
def test_coin_flip_comparisons_are_excluded():
    """NEGATIVE CONTROL for the test above: with no item ordering to recover, the two
    independent fits must disagree and the model must be excluded. Without this, a gate
    that returned `passed=True` unconditionally would look correct."""
    cfg = load_config(CONFIG)
    gate = evaluate_reliability_gate(cfg, _comparisons(signal=0.0, seed=2), TEMPLATES)

    assert gate["passed"] is False
    assert gate["empirical_reliability_spearman"] < cfg.gates.reliability_min
    assert any("EXCLUDED" in r for r in gate["reasons"])
    assert any(str(cfg.gates.reliability_min) in r for r in gate["reasons"])


@pytest.mark.slow
def test_the_threshold_is_read_from_config_not_hardcoded():
    """Raising the bar above a passing model's score must flip it to excluded."""
    cfg = load_config(CONFIG)
    comparisons = _comparisons(signal=3.0, seed=1)
    passing = evaluate_reliability_gate(cfg, comparisons, TEMPLATES)
    assert passing["passed"]

    strict = cfg.model_copy(update={"gates": cfg.gates.model_copy(
        update={"reliability_min": 0.999999})})
    tightened = evaluate_reliability_gate(strict, comparisons, TEMPLATES)
    assert tightened["passed"] is False
    assert tightened["threshold"] == 0.999999
    # the measurement itself must not move -- only the verdict
    assert tightened["empirical_reliability_spearman"] == pytest.approx(
        passing["empirical_reliability_spearman"])


def test_overlapping_template_splits_are_refused():
    """4.4's whole point is independence; an overlapping split would inflate the gate."""
    from src.analysis.bradley_terry import test_retest

    cfg = load_config(CONFIG)
    with pytest.raises(ValueError, match="disjoint"):
        test_retest(cfg, _comparisons(1.0), ["t0", "t1"], ["t1", "t2"])
