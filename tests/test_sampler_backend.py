"""Does the sampler configuration change the answer?

The run machine cannot use PyMC's default path: without a C++ toolchain PyTensor
falls back to pure Python (~2.5 s/draw, a day per fit), and the multiprocess path
that would rescue it deadlocks on Windows spawn re-importing scipy. The fix is
`PYTENSOR_FLAGS=mode=NUMBA` (LLVM compilation, no toolchain) plus `cores=1`.

Both are compute-environment changes on the path that produces PAPER NUMBERS, so
neither gets accepted on assurance. These tests check that the posterior is
invariant to both, on a Bradley-Terry model of the same shape as the real one.

Note what is and is not claimed: draws are NOT bit-identical across backends --
different op implementations perturb floating point, and NUTS trajectories diverge
from there. The claim is that the POSTERIOR is the same, which is the only thing
any conclusion rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config

CONFIG = "configs/stage0_qwen2.5-0.5b.yaml"


def _synthetic_comparisons(
    n_items: int = 30, n_anchors: int = 10, n_templates: int = 5, seed: int = 0
):
    """Item-vs-anchor wins from a known Bradley-Terry generative model.

    Every (item, anchor, template, order) cell is an INDEPENDENT draw, matching the
    real design. An earlier version of this helper assigned one template per
    (item, anchor) and a variant replicated the same draws across template labels;
    the latter fed the model five copies of the same evidence, so posterior SD fell
    by sqrt(5) while the actual error did not move and 95% coverage collapsed to
    0.67. That was a bug in the fixture, but see `test_posterior_is_calibrated` --
    the underlying risk is real for the live design.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1.2, n_items)
    alpha = np.linspace(-2.5, 2.5, n_anchors)

    rows = []
    for i in range(n_items):
        for a in range(n_anchors):
            p = 1 / (1 + np.exp(-(theta[i] - alpha[a])))
            for t in range(n_templates):
                for order in (0, 1):
                    rows.append({
                        "item_id": f"d/item{i:03d}",
                        "anchor_id": f"anchor/a{a}",
                        "template": f"t{t}",
                        "arm": "digits",
                        "order": order,
                        "item_wins": bool(rng.random() < p),
                        "readout_valid": True,
                        "readout_mass": 0.99,
                    })
    return pd.DataFrame(rows), theta


def _fast(cfg, cores=None):
    return cfg.model_copy(update={
        "analysis": cfg.analysis.model_copy(update={
            "chains": 2, "tune": 400, "draws": 400, "sampler_cores": cores,
        })
    })


@pytest.fixture(scope="module")
def data():
    return _synthetic_comparisons()


@pytest.mark.slow
def test_sequential_sampling_gives_the_same_posterior_as_parallel(data):
    """cores=1 is a process-layout change, not a statistical one.

    Chains are seeded independently of how they are distributed across processes,
    so the posterior must not move. If this fails, the Windows workaround is not
    free and every fit from that machine would need re-running elsewhere.
    """
    from src.analysis.bradley_terry import fit_bradley_terry

    comparisons, _ = data
    cfg = load_config(CONFIG)

    par = fit_bradley_terry(_fast(cfg, cores=None), comparisons)
    seq = fit_bradley_terry(_fast(cfg, cores=1), comparisons)

    merged = par.theta.merge(seq.theta, on="item_id", suffixes=("_p", "_s"))
    # Posterior means agree well within their own uncertainty.
    diff = (merged["theta_mean_p"] - merged["theta_mean_s"]).abs()
    tol = 3 * merged[["theta_sd_p", "theta_sd_s"]].max(axis=1)
    assert (diff <= tol).all(), (
        f"cores changed the posterior: max |diff| {diff.max():.3f} vs tol {tol.min():.3f}"
    )
    assert abs(par.sigma_item - seq.sigma_item) < 0.5 * par.sigma_item


@pytest.mark.slow
def test_recovers_known_theta(data):
    """The fit must be right before backend equivalence means anything.

    Bradley-Terry fixes location only up to a constant, so recovery is judged on
    correlation and on the centred values, never on raw agreement.
    """
    from scipy import stats

    from src.analysis.bradley_terry import fit_bradley_terry

    comparisons, true_theta = data
    fit = fit_bradley_terry(_fast(load_config(CONFIG), cores=1), comparisons)

    t = fit.theta.sort_values("item_id")
    est, sd = t["theta_mean"].to_numpy(), t["theta_sd"].to_numpy()
    r = stats.pearsonr(est, true_theta).statistic

    # Judged against what the fit's OWN uncertainty implies, not a magic constant:
    # r = sqrt(var_true / (var_true + var_err)). A fit that recovers much worse than
    # its stated precision is overconfident, which matters because A1.1 designates
    # the posterior SD of theta as the precision measure.
    expected = np.sqrt(true_theta.var() / (true_theta.var() + (sd ** 2).mean()))
    assert r > expected - 0.10, (
        f"recovery r={r:.3f} far below the {expected:.3f} implied by its own "
        f"posterior SD (median {np.median(sd):.3f}) -- the fit is overconfident"
    )


@pytest.mark.slow
def test_posterior_is_calibrated(data):
    """Do 95% intervals actually cover the truth 95% of the time?

    This is the assumption behind reporting posterior SD as precision. Bradley-Terry
    fixes location only up to a constant, so both sides are centred before comparing.

    The live model treats each of an item's ~100 comparisons as independent
    Bernoulli evidence, with no template effect and no over-dispersion term. If real
    templates elicit correlated responses, that assumption inflates the effective
    sample size and understates posterior SD. There is a hint of exactly this in the
    real Gemma fit, where observed test-retest (0.835) came in below what its
    reported posterior SD implies.
    """
    from src.analysis.bradley_terry import fit_bradley_terry

    comparisons, true_theta = data
    fit = fit_bradley_terry(_fast(load_config(CONFIG), cores=1), comparisons)

    t = fit.theta.sort_values("item_id")
    est = t["theta_mean"].to_numpy()
    offset = est.mean() - true_theta.mean()
    covered = (
        (true_theta + offset >= t["hdi_low"].to_numpy())
        & (true_theta + offset <= t["hdi_high"].to_numpy())
    ).mean()
    assert covered >= 0.80, (
        f"95% HDI covers only {covered:.2f} of true values -- posterior SD is "
        "understated and cannot be reported as precision"
    )


def test_sampler_cores_is_recorded_in_the_config_hash():
    """A compute-environment change on the paper-number path must be traceable."""
    cfg = load_config(CONFIG)
    assert cfg.analysis.sampler_cores is None
    assert cfg.model_copy(update={
        "analysis": cfg.analysis.model_copy(update={"sampler_cores": 1})
    }).hash() != cfg.hash()


def test_sampler_cores_does_not_invalidate_forward_pass_caches():
    """It touches sampling only, so it must not force 40,000 comparisons to re-run."""
    cfg = load_config(CONFIG)
    changed = cfg.model_copy(update={
        "analysis": cfg.analysis.model_copy(update={"sampler_cores": 1})
    })
    for stage in ("pass_a", "pass_b", "pass_c"):
        assert changed.hash(stage) == cfg.hash(stage)


def test_provenance_records_the_pytensor_backend():
    """PYTENSOR_FLAGS is set in the environment, so nothing else would capture it."""
    from src.provenance import capture

    prov = capture(load_config(CONFIG))
    assert prov.pytensor_mode, "pytensor mode not recorded"
    assert prov.pymc_version != "absent"
    assert prov.pytensor_version != "absent"
