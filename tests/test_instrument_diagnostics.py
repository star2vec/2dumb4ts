"""A2.1's fit-quality diagnostics, previously untested.

`excess_consistency_slope` and `excess_slope_ppc_null` produce the misspecification
numbers quoted in the preregistration, and `predicted_split_half` decides what A2.4's
open discrepancy is measured against. None of the three was named by a test.

That matters more here than elsewhere: R3 and R4 were BOTH estimator errors in this
area -- a statistic computed correctly but compared against the wrong null. These tests
are mostly about the nulls.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.bradley_terry import predicted_split_half


def test_split_half_prediction_is_below_full_data_reliability():
    """The two figures are not comparable, and that is the whole point of the function.

    Model reliability comes from a fit on all five templates. The empirical split-half is
    between two SHORTER fits, each noisier, so it is lower even under perfect
    specification. Comparing them directly guarantees an apparent gap.
    """
    r_full = 0.957
    pred = predicted_split_half(r_full, 3 / 5, 2 / 5)
    assert pred < r_full
    assert pred == pytest.approx(0.9145, abs=5e-4)

    # NEGATIVE CONTROL: with the halves at full length the correction vanishes, so the
    # formula is doing length-matching rather than a constant shrink.
    assert predicted_split_half(r_full, 1.0, 1.0) == pytest.approx(r_full, abs=1e-9)


def test_the_correction_shrinks_a2_4s_recorded_gap_without_closing_it():
    """A2.4 records 0.770 against 0.957. The comparator should be the length-matched
    figure, so the recorded shortfall is overstated -- but the discrepancy survives, and
    A2.4 stays open. Both halves are asserted so neither can drift."""
    observed, pred = 0.770, predicted_split_half(0.957, 3 / 5, 2 / 5)
    assert observed - 0.957 == pytest.approx(-0.187, abs=5e-4)   # as recorded
    assert observed - pred == pytest.approx(-0.145, abs=5e-4)    # as it should be
    assert observed < pred, "the correction must not close the gap -- A2.4 stays open"


def test_split_half_prediction_is_symmetric_and_monotone():
    r = 0.9
    assert predicted_split_half(r, 0.6, 0.4) == pytest.approx(predicted_split_half(r, 0.4, 0.6))
    # a more lopsided split predicts a lower correlation
    assert predicted_split_half(r, 0.5, 0.5) > predicted_split_half(r, 0.8, 0.2)
    # a less reliable instrument predicts a lower split-half
    assert predicted_split_half(0.8, 0.6, 0.4) < predicted_split_half(0.95, 0.6, 0.4)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_split_half_prediction_refuses_impossible_reliabilities(bad):
    assert np.isnan(predicted_split_half(bad, 0.6, 0.4))


def test_run_py_reports_the_length_matched_comparator():
    """pass_a_pairwise.main() printed it; run.py -- the path the numbers come from -- did
    not, so the run showed the misleading comparison. Both paths must show it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "experiments"
    for name in ("run.py", "pass_a_pairwise.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "predicted_split_half" in src, f"{name} omits the length-matched comparator"
        assert "length-matched" in src, f"{name} does not label it for the reader"


def test_the_ppc_null_refuses_to_return_a_nan_that_reads_as_no_signal():
    """One failed replicate turned mean and sd into NaN, and run.py's `null_sd > 0` guard
    is False for NaN -- so the misspecification diagnostic reported nothing, silently."""
    import inspect

    from src.analysis import bradley_terry as bt

    src = inspect.getsource(bt.excess_slope_ppc_null)
    assert "np.isfinite" in src and "n_failed" in src
    assert "raise ValueError" in src, "too many failed replicates must be an error"

    # NEGATIVE CONTROL: the arithmetic that made this silent.
    s = np.array([0.004, 0.005, np.nan, 0.003])
    assert np.isnan(s.std(ddof=1)) and not (s.std(ddof=1) > 0), (
        "if NaN sd ever compares > 0, run.py's guard changes meaning")
    assert np.isfinite(s[np.isfinite(s)].std(ddof=1))
