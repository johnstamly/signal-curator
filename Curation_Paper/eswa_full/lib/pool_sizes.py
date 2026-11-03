"""Training-pool, early-stopping and test-fold sizes of the stored E1 loop cells (min and max over all
seeds and folds), read from the cells' own ``len_train_pool`` / ``len_es`` / ``len(test_ids)``.
Nothing is typed in; the manuscript quotes these as macros."""
from __future__ import annotations


def _span(values) -> dict:
    values = list(values)
    return {"min": int(min(values)), "max": int(max(values)), "n_cells": len(values)}


def build_pool_sizes(results: list[dict]) -> dict:
    """``results`` = the ``result`` dicts of the cells (one per seed x fold x configuration)."""
    if not results:
        raise ValueError("no cell results")
    for r in results:
        for k in ("len_train_pool", "len_es", "test_ids"):
            if k not in r:
                raise KeyError(f"cell result lacks {k!r}")
    return {
        "len_train_pool": _span(r["len_train_pool"] for r in results),
        "len_es": _span(r["len_es"] for r in results),
        "n_test": _span(len(r["test_ids"]) for r in results),
        "n_cells": len(results),
    }
