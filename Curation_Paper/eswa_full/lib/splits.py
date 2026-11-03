"""Rotating shared test folds and the per-pool early-stopping split."""
from __future__ import annotations

import random
from typing import Iterable


def shared_folds(ref: dict[str, int], k: int, seed: int,
                 exclude_ids: Iterable[str] = ()) -> list[list[str]]:
    """k folds over the shared signals, stratified by reference label_int.

    Excluded IDs are removed from every fold (never tested). Per class, IDs are
    sorted, shuffled with Random(seed) and dealt round-robin; the GOOD deal
    starts where the BAD deal stopped so fold sizes differ by at most one.
    """
    excl = set(exclude_ids)
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    offset = 0
    for cls in (0, 1):  # BAD first
        ids = sorted(s for s, y in ref.items() if y == cls and s not in excl)
        rng.shuffle(ids)
        for i, s in enumerate(ids):
            folds[(i + offset) % k].append(s)
        offset += len(ids)
    return [sorted(f) for f in folds]


def fold_split(pool: dict[str, dict], test_ids: list[str], seed: int,
               es_frac: float) -> tuple[list[dict], list[dict]]:
    """(train_pool, es_set) from the pool minus the test IDs.

    es_set is stratified by label_int (es_frac of each class, at least one
    BAD and one GOOD). Rows are shallow copies so callers may mutate labels.
    """
    test = set(test_ids)
    rng = random.Random(seed)
    train: list[dict] = []
    es: list[dict] = []
    for cls in (0, 1):
        rows = sorted((r for s, r in pool.items() if s not in test and r["label_int"] == cls),
                      key=lambda r: r["signal_id"])
        rng.shuffle(rows)
        n_es = min(max(1, round(es_frac * len(rows))), max(len(rows) - 1, 0)) if rows else 0
        es += [dict(r) for r in rows[:n_es]]
        train += [dict(r) for r in rows[n_es:]]
    key = lambda r: r["signal_id"]
    return sorted(train, key=key), sorted(es, key=key)
