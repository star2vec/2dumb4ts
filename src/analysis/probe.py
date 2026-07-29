"""Linear probes on Pass C pre activations. See `preregistration_probe.md`.

Asks whether `|diff|` — preference MAGNITUDE — is decodable from the hidden state at the
readout position, when the output does not express it. The preregistration fixes every
choice here; this module implements it and adds nothing.

THREE THINGS DO ALL THE WORK, and each exists to stop a specific way of being wrong:

1. **Item-held-out folds.** `|diff|` is a property of the item pair, and the activations
   have seen both items. A probe can recover it by recognising WHICH PAIR THIS IS rather
   than by reading any represented preference, and that probe decodes beautifully and means
   nothing. Folds partition the ITEM POOL; a pair is test if EITHER of its items is held
   out, so training never sees an item that appears at test.

2. **The pair is the unit.** Ten rows share one pair, the same two items, and an IDENTICAL
   target. Row-level intervals would be overstated by construction. Predictions are averaged
   within pair before scoring, and the bootstrap resamples pairs.

3. **A positive control on the same activations.** A null probe alone cannot distinguish
   "magnitude is not represented" from "the probe was inadequate". The output expresses
   ORDER but not MAGNITUDE, so `sign(diff)` is the reference: it is expected to decode, and
   the reading is the COMPARISON, never an absolute threshold on `|diff|` alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProbeResult:
    target: str
    layer: int
    rho_pair: float          # Spearman on pair-averaged out-of-fold predictions
    rho_row: float           # ...and on rows, reported for comparison only
    ci_low: float
    ci_high: float
    n_pairs: int
    n_rows: int
    alphas: tuple[float, ...] = field(default=())

    def as_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "alphas"}


def item_folds(item1, item2, n_folds: int = 5, seed: int = 20261025):
    """Partition by ITEM. A pair is test if either of its items is held out.

    Returns a list of boolean test masks over rows. Deterministic given the seed; the item
    ordering is sorted first so the split does not depend on row order.
    """
    item1 = np.asarray(item1)
    item2 = np.asarray(item2)
    items = np.array(sorted(set(item1) | set(item2)))
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(items)) % n_folds
    fold_of = dict(zip(items, assignment))

    masks = []
    for k in range(n_folds):
        held = {i for i, f in fold_of.items() if f == k}
        # EITHER item held out -> the whole pair is test. Never "one item each side".
        masks.append(np.array([(a in held) or (b in held) for a, b in zip(item1, item2)]))
    return masks


def _ridge_fit(X, y, alpha):
    """Ridge in closed form on centred data. No intercept term to regularise."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    n_feat = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(n_feat, dtype=Xc.dtype)
    w = np.linalg.solve(A, Xc.T @ yc)
    return w, X.mean(0), y.mean()


def _predict(X, w, x_mean, y_mean):
    return (X - x_mean) @ w + y_mean


def _select_alpha(X, y, item1, item2, alphas, seed):
    """Alpha chosen on INNER folds that respect the same item split.

    Selecting it on folds that share items with the training set leaks the identity
    confound into the hyperparameter, which is the least visible way to get a good number.
    """
    from scipy import stats

    inner = item_folds(item1, item2, n_folds=3, seed=seed + 1)
    best, best_score = alphas[0], -np.inf
    for alpha in alphas:
        preds = np.full(len(y), np.nan)
        for mask in inner:
            if mask.all() or not mask.any():
                continue
            w, xm, ym = _ridge_fit(X[~mask], y[~mask], alpha)
            preds[mask] = _predict(X[mask], w, xm, ym)
        ok = ~np.isnan(preds)
        if ok.sum() < 10 or np.std(preds[ok]) == 0:
            continue
        score = stats.spearmanr(preds[ok], y[ok]).statistic
        if np.isfinite(score) and score > best_score:
            best, best_score = alpha, score
    return best


def run_probe(activations, y, pair_id, item1, item2, *, target: str, layer: int,
              alphas=(1e2, 1e3, 1e4, 1e5, 1e6), n_folds: int = 5,
              n_boot: int = 2000, seed: int = 20261025) -> ProbeResult:
    """One probe, one layer, item-held-out, scored on pair-averaged predictions."""
    from scipy import stats

    X = np.asarray(activations, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    pair_id = np.asarray(pair_id)

    preds = np.full(len(y), np.nan)
    chosen = []
    for mask in item_folds(item1, item2, n_folds, seed):
        if mask.all() or not mask.any():
            continue
        alpha = _select_alpha(X[~mask], y[~mask], item1[~mask], item2[~mask], alphas, seed)
        chosen.append(alpha)
        w, xm, ym = _ridge_fit(X[~mask], y[~mask], alpha)
        preds[mask] = _predict(X[mask], w, xm, ym)

    ok = ~np.isnan(preds)
    rho_row = float(stats.spearmanr(preds[ok], y[ok]).statistic)

    # The pair is the unit: ten rows share one target, so scoring rows would count each
    # pair ten times and the interval would be wrong by roughly sqrt(10).
    pairs = np.array(sorted(set(pair_id[ok])))
    pred_pair = np.array([preds[ok][pair_id[ok] == p].mean() for p in pairs])
    y_pair = np.array([y[ok][pair_id[ok] == p][0] for p in pairs])
    rho_pair = float(stats.spearmanr(pred_pair, y_pair).statistic)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        if np.std(y_pair[idx]) == 0:
            continue
        boot.append(stats.spearmanr(pred_pair[idx], y_pair[idx]).statistic)
    lo, hi = np.percentile([b for b in boot if np.isfinite(b)], [2.5, 97.5])

    return ProbeResult(target=target, layer=layer, rho_pair=rho_pair, rho_row=rho_row,
                       ci_low=float(lo), ci_high=float(hi), n_pairs=len(pairs),
                       n_rows=int(ok.sum()), alphas=tuple(chosen))


def within_item_control(activations, y, pair_id, item1, item2, *, layer: int,
                        seed: int = 20261025, **kw) -> ProbeResult:
    """The SAME probe with folds that ignore items -- rows split at random.

    If magnitude decodes here but not under the item split, the probe learned item
    identity. Reported alongside; it licenses nothing on its own.
    """
    rng = np.random.default_rng(seed)
    fake = np.array([f"row{i}" for i in rng.permutation(len(y))])
    return run_probe(activations, y, pair_id, fake, fake, target="within-item",
                     layer=layer, seed=seed, **kw)
