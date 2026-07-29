"""The probe, and the three properties it exists to have. See preregistration_probe.md.

Every test here plants a known truth and checks the probe recovers it -- or, more
importantly, checks it does NOT recover something it should be blind to.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import probe


def _design(n_items=40, n_templates=5, seed=0):
    """Pairs of items, ten rows each, exactly the Pass C pre structure."""
    rng = np.random.default_rng(seed)
    items = [f"d/i{i:02d}" for i in range(n_items)]
    theta = dict(zip(items, rng.normal(0, 1.5, n_items)))
    pair_id, i1, i2, y_mag, y_sign = [], [], [], [], []
    for k in range(0, n_items, 2):
        a, b = items[k], items[k + 1]
        for _t in range(n_templates):
            for _o in (0, 1):
                pair_id.append(f"d/p{k:02d}")
                i1.append(a)
                i2.append(b)
                y_mag.append(abs(theta[a] - theta[b]))
                y_sign.append(np.sign(theta[a] - theta[b]))
    return (np.array(pair_id), np.array(i1), np.array(i2),
            np.array(y_mag), np.array(y_sign), theta)


def test_folds_hold_out_items_not_rows():
    """The study's hinge. A pair is test if EITHER item is held out, so training never sees
    an item that appears at test -- in any pair, template or order."""
    pair_id, i1, i2, *_ = _design()
    masks = probe.item_folds(i1, i2, n_folds=5, seed=1)

    assert len(masks) == 5
    for mask in masks:
        train_items = set(i1[~mask]) | set(i2[~mask])
        test_items = set(i1[mask]) | set(i2[mask])
        assert not (train_items & test_items), (
            "an item appears in both train and test; the identity confound is open")
        # a pair is never split across the boundary
        for p in set(pair_id):
            rows = pair_id == p
            assert len(set(mask[rows])) == 1, f"pair {p} straddles the split"

    # NEGATIVE CONTROL: random row folds would NOT have this property.
    rng = np.random.default_rng(0)
    random_mask = rng.random(len(i1)) < 0.2
    assert set(i1[~random_mask]) & set(i1[random_mask]), (
        "the fixture is degenerate; a random split should share items")


def test_it_recovers_a_magnitude_that_is_genuinely_encoded():
    """PRECONDITION for reading any null: if the signal is there, the probe finds it."""
    pair_id, i1, i2, y_mag, _, _ = _design(seed=2)
    rng = np.random.default_rng(2)
    # magnitude written into one direction, plus noise in 60 others
    X = rng.normal(0, 1, (len(y_mag), 60))
    X[:, 0] += 3.0 * y_mag

    r = probe.run_probe(X, y_mag, pair_id, i1, i2, target="|diff|", layer=0, n_boot=300)
    assert r.rho_pair > 0.7, r
    assert r.ci_low > 0.3, r
    assert r.n_pairs == 20 and r.n_rows == 200


def test_it_does_not_recover_magnitude_from_item_identity_alone():
    """THE test. Activations carry item identity and NOTHING about magnitude.

    A row-split probe can memorise which pair is which and appear to succeed. The
    item-held-out probe must not, because at test it has never seen these items.
    """
    pair_id, i1, i2, y_mag, _, _ = _design(seed=3)
    rng = np.random.default_rng(3)
    # one-hot item identity: perfectly predicts y within known items, useless across
    items = sorted(set(i1) | set(i2))
    idx = {it: k for k, it in enumerate(items)}
    X = np.zeros((len(y_mag), len(items)), dtype=np.float32)
    for r_, (a, b) in enumerate(zip(i1, i2)):
        X[r_, idx[a]] = 1.0
        X[r_, idx[b]] = 1.0
    X += rng.normal(0, 0.01, X.shape)

    held_out = probe.run_probe(X, y_mag, pair_id, i1, i2, target="|diff|", layer=0,
                               n_boot=300)
    within = probe.within_item_control(X, y_mag, pair_id, i1, i2, layer=0, n_boot=300)

    assert within.rho_pair > 0.6, (
        f"the fixture is wrong -- identity should decode within-item, got {within.rho_pair}")
    assert held_out.rho_pair < 0.35, (
        f"identity leaked across the item split: rho {held_out.rho_pair:.3f}. The confound "
        "the whole design controls for is open.")


def test_the_pair_is_the_unit_and_the_interval_reflects_it():
    """Ten rows share one target. Scoring rows counts each pair ten times."""
    pair_id, i1, i2, y_mag, _, _ = _design(seed=4)
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (len(y_mag), 40))
    X[:, 0] += 1.2 * y_mag

    r = probe.run_probe(X, y_mag, pair_id, i1, i2, target="|diff|", layer=0, n_boot=500)
    assert r.n_pairs == 20, "the unit is the pair"
    assert r.n_rows == 200
    assert r.ci_low < r.rho_pair < r.ci_high
    # 20 pairs is a small sample and the interval must SAY so
    assert (r.ci_high - r.ci_low) > 0.15, (
        "the interval is implausibly tight for 20 pairs; it is probably over rows")


def test_alpha_is_selected_on_item_respecting_inner_folds():
    """Selecting it on folds sharing items leaks the confound into the hyperparameter --
    the least visible way to get a good number."""
    import inspect

    src = inspect.getsource(probe._select_alpha)
    assert "item_folds" in src, "inner folds do not respect the item split"
    assert "seed + 1" in src, "inner folds must differ from the outer split"

    pair_id, i1, i2, y_mag, _, _ = _design(seed=5)
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (len(y_mag), 30))
    X[:, 0] += 2.0 * y_mag
    a = probe._select_alpha(X, y_mag, i1, i2, (1e2, 1e3, 1e4, 1e5), seed=5)
    assert a in (1e2, 1e3, 1e4, 1e5)


def test_sign_and_magnitude_run_through_the_identical_path():
    """The control is only a control if nothing but the target differs."""
    pair_id, i1, i2, y_mag, y_sign, _ = _design(seed=6)
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (len(y_mag), 50))
    X[:, 0] += 2.0 * y_sign          # ORDER encoded, magnitude not

    sign = probe.run_probe(X, y_sign, pair_id, i1, i2, target="sign", layer=0, n_boot=300)
    mag = probe.run_probe(X, y_mag, pair_id, i1, i2, target="|diff|", layer=0, n_boot=300)

    # This is the (R) outcome the prereg names: order decodes, magnitude does not.
    assert sign.rho_pair > 0.7, sign
    assert mag.rho_pair < 0.4, mag
    assert sign.n_pairs == mag.n_pairs and sign.n_rows == mag.n_rows
