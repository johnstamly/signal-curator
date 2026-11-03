"""Scoring of pooled test-fold predictions. Labels: GOOD = 1, BAD = 0; p = p(GOOD)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

_NAN = float("nan")


def _validate(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError(f"y and p must be 1-D of equal length, got {y.shape} and {p.shape}")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("y must contain only 0 (BAD) and 1 (GOOD)")
    if np.any(np.isnan(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p must lie within [0, 1]")
    return y.astype(int), p


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of p(GOOD): 10 equal-width bins on [0, 1],
    last bin closed on the right, empty bins skipped."""
    n = len(y)
    if n == 0:
        return _NAN
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        total += (nb / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """Score predictions. Prediction is GOOD iff p >= thr.

    Returns acc, bal_acc, bad_recall, good_retention, auroc, ece, n, n_bad as
    plain Python floats/ints. Undefined quantities (no BAD / no GOOD) are nan.
    """
    y, p = _validate(y, p)
    pred = (p >= thr).astype(int)
    bad = y == 0
    good = y == 1
    n = int(len(y))
    n_bad = int(bad.sum())

    acc = float((pred == y).mean()) if n else _NAN
    bad_recall = float((p[bad] < thr).mean()) if bad.any() else _NAN
    good_retention = float((p[good] >= thr).mean()) if good.any() else _NAN
    if np.isnan(bad_recall) or np.isnan(good_retention):
        bal_acc = _NAN
    else:
        bal_acc = float((bad_recall + good_retention) / 2.0)
    auroc = float(roc_auc_score(y, p)) if bad.any() and good.any() else _NAN

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "bad_recall": bad_recall,
        "good_retention": good_retention,
        "auroc": auroc,
        "ece": _ece(y, p),
        "n": n,
        "n_bad": n_bad,
    }


def bootstrap_ci(
    values, n_boot: int = 10000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of `values`.

    The given values (one per seed) are resampled with replacement as iid
    units, i.e. seeds are treated as independent draws. A single value returns
    (v, v). Raises ValueError on empty input.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if v.size == 1:
        x = float(v[0])
        return (x, x)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))
