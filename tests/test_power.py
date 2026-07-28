"""The logit-scale power calculation, and the spec defect it exposes.

The old rating-scale module was deleted rather than adapted, so these tests exist to pin
the replacement's two load-bearing claims: that the design calculation agrees with actual
fits, and that section 8's power criterion is unsatisfiable as written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import power
from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


def _pairs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    gap = np.concatenate([rng.uniform(0.02, 0.9, half), rng.uniform(1.0, 3.5, n - half)])
    return pd.DataFrame({
        "pair_id": [f"d/p{i:03d}" for i in range(n)],
        "item1_id": [f"d/a{i:03d}" for i in range(n)],
        "item2_id": [f"d/b{i:03d}" for i in range(n)],
        "difficulty": ["difficult"] * half + ["easy"] * (n - half),
        "diff_analysis": gap,
    })


def _beta(n=5):
    return pd.DataFrame({"template": [f"t{i}" for i in range(n)],
                         "beta_mean": np.linspace(0.7, 1.7, n)})


def test_power_at_the_sesoi_cannot_reach_the_target_at_any_precision():
    """Section 8 asks for 80% power AT the SESOI; section 9.2 makes that impossible.

    `pass` requires the posterior median to EXCEED the SESOI. With the true effect equal
    to the SESOI, the median is centred on the threshold, so the probability of exceeding
    it converges to 0.5 from below however much data is collected. This is the arithmetic
    behind Amendment 3 and it is asserted rather than described, so nobody can quietly
    reinstate the old criterion.
    """
    sesoi, z = 0.2359, 1.959963985
    powers = [power._decide_pass(-sesoi, se, sesoi, z)
              for se in (0.2, 0.1, 0.05, 0.02, 0.005, 0.001)]
    assert powers == sorted(powers), "power at the SESOI should rise monotonically with n"
    assert all(p < 0.5 for p in powers), powers
    assert powers[-1] == pytest.approx(0.5, abs=1e-3)
    # and it is bounded away from any usable target no matter how small the SE gets
    assert power._decide_pass(-sesoi, 1e-6, sesoi, z) < 0.8


def test_minimum_detectable_effect_floors_at_the_sesoi():
    """The MDE is the quantity that CAN move, and its floor is the SESOI.

    This is what makes the A2.9.5 scaling branch nearly worthless for the primary
    contrast: once 1.96*SE is below the SESOI, extra data buys almost nothing.
    """
    cfg = load_config(CONFIG)
    beta, sigma = _beta(), 1.573
    base = _pairs(200)   # gemma's realized size, where 1.96*SE is already below the SESOI
    scaled = pd.concat([base.assign(pair_id=base["pair_id"] + f"#{i}") for i in range(4)])
    small = power.analyze(cfg, base, beta, sigma)
    big = power.analyze(cfg, scaled, beta, sigma)

    assert 1.96 * small.se < small.sesoi, "precondition: the SESOI must already bind"
    assert big.se < small.se, "more pairs must reduce the SE"
    assert big.min_detectable < small.min_detectable
    assert big.min_detectable > small.sesoi, "MDE cannot fall below the SESOI at any n"
    # THE POINT: once the SESOI binds, 4x the data barely moves the detectable effect.
    # That is why A2.9.5's scaling branch is close to worthless for the primary contrast.
    assert (small.min_detectable - big.min_detectable) < 0.25 * small.sesoi


def test_designation_follows_the_real_pass_c_rule():
    """Six of eight conditions share the model's own pick; only the random arms differ.

    Modelling designation as alternating across conditions -- the obvious synthetic
    choice -- inflates the SE by ~60%, because it invents within-pair contrast that the
    real design does not have. The rule is imported from pass_c rather than restated.
    """
    from src.experiments.pass_c import OWN_PICK_CONDITIONS

    cfg = load_config(CONFIG)
    conditions = list(cfg.pass_c.conditions)
    shared = [c for c in conditions if c in OWN_PICK_CONDITIONS]
    assert len(shared) == 6 and set(conditions) - set(shared) == {"random", "3p-random"}

    X, eta, c_vec, meta = power._design(
        _pairs(20), _beta(2)["beta_mean"].to_numpy(), conditions, 2, 0.4,
        np.zeros(len(conditions)), seed=0)
    # the contrast selects exactly lambda_chose and lambda_yoked, nothing else
    assert c_vec.sum() == pytest.approx(0.0)
    assert np.count_nonzero(c_vec) == 2


def test_degenerate_gap_fails_loudly_and_the_source_column_is_recorded():
    """Two separate protections, because only one of them can be automatic.

    A degenerate |diff| column must raise rather than emit nan -- and the check has to be
    scale-RELATIVE, since a constant column of 1e-6 has sd 2e-22, not 0, and sails past
    an absolute `<= 0` test while making diff_z garbage of order 1e6.

    But the realistic error -- passing diff_selection, whose difficult stratum is ~0.002
    with real variance around it -- CANNOT be caught by any variance check, because that
    column is not degenerate. It is merely wrong: near-zero by construction, which would
    put every difficult pair at p = 0.5 and overstate the information. Nothing detects
    that automatically, so the column actually used is recorded in `assumptions` where a
    reader can check it.
    """
    cfg = load_config(CONFIG)
    pairs = _pairs()
    honest = power.analyze(cfg, pairs, _beta(), 1.573)
    assert honest.se > 0
    assert "diff_analysis" in honest.assumptions["gap_source"]
    assert "never diff_selection" in honest.assumptions["gap_source"]

    for degenerate in (1e-6, 0.0, 3.2):
        with pytest.raises(ValueError, match="no usable variance"):
            power.analyze(cfg, pairs.assign(diff_analysis=degenerate), _beta(), 1.573)


def test_easy_pairs_carry_less_information_per_observation():
    """W = p(1-p) is maximal at p = 0.5, where difficult pairs sit by construction.

    Easy pairs lose twice: their gap pushes p away from 0.5, and the model's own pick --
    which six conditions designate -- is nearly deterministic on them, pushing it further.
    """
    r = power.analyze(load_config(CONFIG), _pairs(120), _beta(), 1.573)
    info = r.information.set_index("stratum")
    assert info.loc["easy", "mean_w"] < info.loc["difficult", "mean_w"]
    assert r.information["share_of_lambda_information"].sum() == pytest.approx(1.0)


@pytest.mark.slow
def test_design_se_agrees_with_an_actual_fit():
    """The closed form is checked against the real sampler, not trusted on its algebra.

    Fixed-effect absorption here vs partial pooling in the fitted model means the design
    SE should be slightly WIDE, never narrow -- conservative in the stated direction.
    """
    from src.analysis import spread_model as sm
    from tests.test_spread_model import _synth

    cfg = load_config(CONFIG)
    fast = cfg.model_copy(update={"analysis": cfg.analysis.model_copy(
        update={"chains": 2, "tune": 500, "draws": 500, "sampler_cores": 1})})
    trials, _ = _synth(0.0, n_pairs=60, n_tmpl=3, seed=7)
    idata = sm.fit(fast, sm.prepare(fast, trials))
    draws = (sm._draws(idata, "lambda", "chose") - sm._draws(idata, "lambda", "yoked"))
    fitted_sd = float(draws.std(ddof=1))

    pairs = (trials[["pair_id", "diff_analysis"]].drop_duplicates()
             .assign(difficulty="difficult"))
    design = power.analyze(
        fast.model_copy(update={"pass_c": fast.pass_c.model_copy(
            update={"conditions": ["chose", "yoked", "self-recounted", "structure-control"]})}),
        pairs, _beta(3), 1.573)
    assert design.se == pytest.approx(fitted_sd, rel=0.5), (design.se, fitted_sd)


def test_the_conservatism_claim_is_recorded_as_falsified():
    """A4.2. This module told the reader its error direction, and the direction was wrong.

    `power.py` claimed absorbing pair and template as fixed effects makes the design SE err
    WIDE. On the completed run it erred NARROW by up to 5.4x, so the preregistered MDE of
    1.37x SESOI was really 6.57x for gemma. The claim is struck in place rather than
    deleted, because every preregistered power figure was computed under it.

    Pinned because a future reader will otherwise take the paragraph at face value, which
    is exactly what happened to us.
    """
    import inspect

    from src.analysis import power

    doc = inspect.getsource(power)[:4000]
    assert "FALSIFIED BY THE COMPLETED RUN (A4.2)" in doc, (
        "the conservatism claim is unmarked again; a reader would trust it")
    assert "errs NARROW" in doc

    # NEGATIVE CONTROL: the original claim must still be present, not quietly deleted --
    # it is the assumption the preregistered figures were computed under.
    assert "absorbed as fixed effects" in doc
