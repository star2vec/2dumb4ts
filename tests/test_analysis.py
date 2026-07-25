"""Psychometrics, gate logic, and parameter recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import mixed, power
from src.analysis.reliability import (
    collapse_polarity,
    evaluate_gates,
    icc_two_way,
    validity_table,
)
from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


# ---------------------------------------------------------------------------
# ICC


def test_icc_hand_computed_fixture():
    """Perfect consistency with an additive per-column offset.

    Column 2 = column 1 + 1. Hand computation: MSR=5, MSC=2.5, MSE=0, so
    ICC(C,1) = 1.0 exactly and ICC(A,1) = 5/(5 + 2*2.5/5) = 0.8333.

    This is the distinction that matters here: templates are a fixed battery and
    an additive template offset cancels in the within-pair differences that make
    up the DV, so the CONSISTENCY form is the right one. Absolute agreement would
    penalise a harmless offset.
    """
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
    out = icc_two_way(x)
    assert out["ms_rows"] == pytest.approx(5.0)
    assert out["ms_cols"] == pytest.approx(2.5)
    assert out["ms_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["icc_c1"] == pytest.approx(1.0)
    assert out["icc_a1"] == pytest.approx(5.0 / 6.0)
    assert out["icc_ck"] == pytest.approx(1.0)


def test_icc_is_near_zero_for_pure_noise():
    rng = np.random.default_rng(0)
    out = icc_two_way(rng.normal(size=(300, 5)))
    assert abs(out["icc_c1"]) < 0.1


def test_icc_k_form_exceeds_single_measurement_form():
    """Averaging k templates is more reliable than one -- computed on the actual
    columns, never projected via Spearman-Brown (template errors correlate)."""
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(200, 1))
    x = signal + rng.normal(scale=1.0, size=(200, 3))
    out = icc_two_way(x)
    assert out["icc_ck"] > out["icc_c1"]


def test_icc_requires_two_items_and_two_raters():
    with pytest.raises(ValueError, match="at least 2 items and 2 raters"):
        icc_two_way(np.zeros((1, 5)))


# ---------------------------------------------------------------------------
# validity


def _pass_a_frame(mode: str, n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic Pass A. `mode` selects the model's polarity behaviour."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1.5, 8.5, size=n)
    rows = []
    for t in range(5):
        asc = np.clip(truth + rng.normal(0, 0.2, n), 1, 9)
        if mode == "consistent":
            desc = 10 - asc + rng.normal(0, 0.2, n)
        elif mode == "polarity-blind":
            # Ignores the anchor definition, always answers ascending.
            desc = asc + rng.normal(0, 0.2, n)
        else:
            desc = rng.uniform(1, 9, n)
        for i in range(n):
            for polarity, value in (("ascending", asc[i]), ("descending", desc[i])):
                rows.append(
                    {
                        "item_id": f"foods/item{i:03d}",
                        "domain": "foods",
                        "item_label": f"item{i:03d}",
                        "template": f"t{t}",
                        "template_index": t,
                        "polarity": polarity,
                        "rating": float(np.clip(value, 1, 9)),
                        "digit_mass": 0.99,
                        "rating_argmax": float(round(value)),
                    }
                )
    return pd.DataFrame(rows)


def test_validity_is_high_for_a_polarity_consistent_model():
    v = validity_table(_pass_a_frame("consistent"), 10)
    assert (v["spearman_rho"] > 0.9).all()


def test_validity_is_strongly_negative_for_a_polarity_blind_model():
    """The empirically observed failure mode: rho near -1 means the model answers
    on a fixed higher-is-better mapping regardless of the anchor definition."""
    v = validity_table(_pass_a_frame("polarity-blind"), 10)
    assert (v["spearman_rho"] < -0.9).all()


def test_polarity_collapse_destroys_variance_for_a_polarity_blind_model():
    collapsed = collapse_polarity(_pass_a_frame("polarity-blind"), 10)
    templates = [c for c in collapsed.columns if c.startswith("t")]
    assert collapsed[templates].to_numpy().std() < 0.3


def test_gate_passes_a_consistent_model(cfg):
    gate = evaluate_gates(cfg, _pass_a_frame("consistent"))
    assert gate.passed, gate.summary()
    assert gate.sigma_between > cfg.gates.sigma_between_min
    assert gate.sesoi_primary == pytest.approx(
        cfg.analysis.sesoi_sigma_fraction * gate.sigma_between
    )


def test_gate_excludes_a_polarity_blind_model(cfg):
    gate = evaluate_gates(cfg, _pass_a_frame("polarity-blind"))
    assert not gate.passed
    assert any("polarity validity" in r for r in gate.reasons)
    assert any("sigma_between" in r for r in gate.reasons)


def test_reliability_tripwire_does_not_exclude(cfg):
    """ICC is a tripwire for implementation faults, NOT a scientific halt.

    A model can be valid and noisy; the binding criterion there is power at the
    SESOI, not the ICC.
    """
    rng = np.random.default_rng(3)
    frame = _pass_a_frame("consistent", n=80, seed=5)
    # Inject template-specific noise: still valid, much less reliable.
    noisy = frame.copy()
    noisy["rating"] = np.clip(
        noisy["rating"] + rng.normal(0, 1.2, len(noisy)), 1, 9
    )
    gate = evaluate_gates(cfg, noisy)
    if gate.icc_tripwire_hit:
        assert gate.passed or any("polarity validity" in r for r in gate.reasons)
        assert not any(
            "EXCLUDED" in r and "ICC" in r for r in gate.reasons
        ), "ICC must never appear as an exclusion reason"


# ---------------------------------------------------------------------------
# gate decision rule


def _draws(center: float, sd: float, n: int = 8000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(center, sd, n)


def test_decision_pass_requires_direction_and_magnitude():
    out = mixed.summarize_contrast(
        "x", _draws(-0.5, 0.08), term="slope", sesoi=0.1, hdi_prob=0.95, direction=-1
    )
    assert out.decision == "pass"
    assert out.excludes_zero and out.exceeds_sesoi


def test_decision_rejects_an_effect_in_the_wrong_direction():
    """A positive interaction is not a pass, however precise: the prediction is
    that spreading is LARGER for difficult (small-|diff|) pairs."""
    out = mixed.summarize_contrast(
        "x", _draws(+0.5, 0.08), term="slope", sesoi=0.1, hdi_prob=0.95, direction=-1
    )
    assert out.excludes_zero
    assert out.decision != "pass"


def test_decision_equivalent_to_null_when_hdi_sits_inside_the_rope():
    out = mixed.summarize_contrast(
        "x", _draws(0.0, 0.01), term="slope", sesoi=0.2, hdi_prob=0.95, direction=-1
    )
    assert out.inside_rope
    assert out.decision == "equivalent-to-null"


def test_decision_inconclusive_when_hdi_straddles_zero_and_the_sesoi():
    out = mixed.summarize_contrast(
        "x", _draws(-0.15, 0.4), term="slope", sesoi=0.1, hdi_prob=0.95, direction=-1
    )
    assert not out.inside_rope and not out.excludes_zero
    assert out.decision == "inconclusive"


def test_equivalence_ratio_propagates_denominator_uncertainty():
    """Stage 2 rule: the ratio's posterior, not a ratio of point estimates."""
    rng = np.random.default_rng(0)
    num = rng.normal(0.02, 0.01, 20000)
    den = rng.normal(1.0, 0.05, 20000)
    out = mixed.equivalence_ratio(num, den, f=0.25, hdi_prob=0.95)
    assert out["equivalent"]
    assert abs(out["ratio_median"] - 0.02) < 0.01

    big = rng.normal(0.8, 0.05, 20000)
    assert not mixed.equivalence_ratio(big, den, f=0.25, hdi_prob=0.95)["equivalent"]


# ---------------------------------------------------------------------------
# parameter recovery


def _synthetic_pass_c(
    true_interaction: float, n_pairs: int = 24, n_templates: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Generate Pass C data from the exact model the analysis fits."""
    rng = np.random.default_rng(seed)
    conditions = list(mixed.PLANNED_CONTRASTS and ("chose", "yoked", "3p-yoked", "3p-random", "random"))
    # chose and yoked slopes differ by exactly `true_interaction`.
    slope = {c: 0.0 for c in conditions}
    slope["chose"] = true_interaction / 2
    slope["yoked"] = -true_interaction / 2
    intercept = {c: 0.0 for c in conditions}

    diff = rng.uniform(0.05, 3.0, n_pairs)
    u_pair = rng.normal(0, 0.3, n_pairs)
    u_tpl = rng.normal(0, 0.2, n_templates)
    diff_z = (diff - diff.mean()) / diff.std(ddof=1)

    rows = []
    for p in range(n_pairs):
        for t in range(n_templates):
            for order in (0, 1):
                for c in conditions:
                    rows.append(
                        {
                            "pair_id": f"foods/p{p:03d}",
                            "domain": "foods",
                            "difficulty": "difficult" if diff[p] < 1.5 else "easy",
                            "matched_set": f"foods/ms{p:02d}",
                            "item1_id": f"foods/a{p:03d}",
                            "item2_id": f"foods/b{p:03d}",
                            "template": f"t{t}",
                            "option_order": order,
                            "condition": c,
                            "diff_analysis": diff[p],
                            "mean_analysis": 5.0,
                            "spread": (
                                intercept[c]
                                + slope[c] * diff_z[p]
                                + u_pair[p]
                                + u_tpl[t]
                                + rng.normal(0, 0.4)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_recovers_a_known_interaction(cfg):
    """The analysis must find a planted interaction, with the truth in the HDI."""
    true_effect = -0.8
    frame = _synthetic_pass_c(true_effect, seed=11)
    fast = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"chains": 2, "tune": 500, "draws": 500})}
    )
    design = mixed.prepare_design(fast, frame)
    idata = mixed.fit(fast, design, with_item=False)

    contrasts = mixed.contrast_table(fast, idata, sesoi=0.1)
    primary = contrasts[
        (contrasts["name"] == mixed.PRIMARY) & (contrasts["term"] == "slope")
    ].iloc[0]

    assert primary["hdi_low"] <= true_effect <= primary["hdi_high"], primary.to_dict()
    assert primary["decision"] == "pass"
    assert primary["median"] < 0


@pytest.mark.slow
def test_finds_no_interaction_when_none_was_planted(cfg):
    """And must NOT find one that is not there -- the kill gate has to be able to
    return a negative."""
    frame = _synthetic_pass_c(0.0, n_pairs=30, seed=12)
    fast = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"chains": 2, "tune": 500, "draws": 500})}
    )
    design = mixed.prepare_design(fast, frame)
    idata = mixed.fit(fast, design, with_item=False)
    contrasts = mixed.contrast_table(fast, idata, sesoi=0.25)
    primary = contrasts[
        (contrasts["name"] == mixed.PRIMARY) & (contrasts["term"] == "slope")
    ].iloc[0]
    assert primary["decision"] != "pass", primary.to_dict()


def test_prepare_design_z_scores_diff(cfg):
    frame = _synthetic_pass_c(-0.5, seed=2)
    design = mixed.prepare_design(cfg, frame)
    assert design.frame["diff_z"].mean() == pytest.approx(0.0, abs=1e-9)
    assert design.frame["diff_z"].std(ddof=1) == pytest.approx(1.0, abs=1e-6)
    assert design.diff_sd > 0


def test_prepare_design_rejects_a_constant_diff(cfg):
    frame = _synthetic_pass_c(-0.5, seed=3)
    frame["diff_analysis"] = 1.0
    with pytest.raises(ValueError, match="zero variance"):
        mixed.prepare_design(cfg, frame)


# ---------------------------------------------------------------------------
# power


def test_power_increases_with_effect_size(cfg):
    fast = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"power_n_sims": 30})}
    )
    rng = np.random.default_rng(0)
    result = power.simulate_power(
        fast,
        sigma_between=1.2,
        icc_c1=0.75,
        diff_analysis=rng.uniform(0.05, 3.0, 40),
        sesoi=0.18,
        n_templates=3,
        grid=np.array([0.0, 0.3, 1.0]),
    )
    grid = result.grid.set_index("true_interaction")["power"]
    assert grid.loc[0.0] < 0.2, "false positive rate should be near alpha"
    assert grid.loc[1.0] > grid.loc[0.3] > grid.loc[0.0]
    assert "estimator" in result.assumptions


def test_power_refuses_an_impossible_icc(cfg):
    with pytest.raises(ValueError, match="noise floor is undefined"):
        power.simulate_power(
            cfg,
            sigma_between=1.0,
            icc_c1=0.0,
            diff_analysis=np.linspace(0.1, 2.0, 20),
            sesoi=0.15,
        )
