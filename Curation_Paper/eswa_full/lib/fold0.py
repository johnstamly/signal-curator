"""Fold-0 clean slice (Task 14, carry from Task 10 / R14).

E4 tuning (and so the E4 compare and E5 recipes) was done on fold 0 using rows of folds 1-4 for
training and early stopping; those rows are TEST rows of folds 1-4.  The only strictly clean
slice is therefore fold 0's own test set.  This module re-scores the stored per-cell test
predictions (R14) restricted to fold 0, per seed, and summarises over seeds like ``curve()``.

Reads cells only; refuses (raises) on missing / hash-mismatched / mixed-key cells.  Dirty cells
are reported in ``cells_dirty`` and refused later by the numbers generator's own check.
"""
from __future__ import annotations

import numpy as np

from . import aggregate as AG
from .cells import cell_id, load_cell
from .metrics import metrics
from .numbers_core import InputHashMismatch, MixedCodeKeys
from .protocol import input_hashes


def fold0_rows(cfg: dict, specs: list[dict], ref: dict, budgets: list[int]) -> list[dict]:
    cells = [s for s in specs if s["fold"] == 0]
    if not cells:
        raise ValueError("no fold-0 cells in the spec list")
    by_seed: dict[int, dict] = {}
    for s in cells:
        payload = load_cell(s, cfg)
        if payload is None:
            raise AG.MissingCells([cell_id(s)])
        by_seed[s["seed"]] = payload["result"]
    seeds = sorted(by_seed)
    rows = []
    for b in budgets:
        per_seed = []
        for sd in seeds:
            res = by_seed[sd]
            rec = next(r for r in res["budgets"] if r["budget"] == b)
            tp = rec["test_pred"]
            if set(tp) != set(res["test_ids"]):
                raise ValueError("test_pred keys differ from test_ids")
            ids = list(res["test_ids"])
            per_seed.append(metrics(np.array([ref[i] for i in ids]),
                                    np.array([tp[i] for i in ids])))
        rows.append({"budget": b, "n_seeds": len(seeds), "seeds": seeds,
                     "n_scored": per_seed[0]["n"],
                     "metrics": {k: {**AG._nanstat([m[k] for m in per_seed]),
                                     "values": [m[k] for m in per_seed]}
                                 for k in AG.METRIC_KEYS}})
    return rows


def fold0_study(cfg: dict, groups: dict[str, list[dict]], ref: dict, budgets: list[int], *,
                prefix: str, of: str, stage: str = "stage1") -> dict:
    """groups: {model: loop-cell specs}.  Returns a study dict shaped like the others."""
    allspecs = [s for ss in groups.values() for s in ss if s["fold"] == 0]
    try:
        prov = AG.require(allspecs, cfg, allow_dirty=True)
    except AG.InputHashMismatch as e:
        raise InputHashMismatch(str(e)) from None
    except AG.KeyMismatch as e:
        raise MixedCodeKeys(str(e)) from None
    out = {"of": of, "prefix": prefix, "budgets": list(budgets), "fold": 0, "stage": stage,
           "note": "fold-0 test set only (the strictly clean slice, R14 re-score)",
           "models": {m: {"rows": fold0_rows(cfg, ss, ref, budgets)} for m, ss in groups.items()},
           "cells_code_key": prov["cells_code_key"], "cells_key_counts": prov["key_counts"],
           "cells_dirty": prov["n_dirty"], "n_seeds": len({s["seed"] for s in next(iter(groups.values()))}),
           "meta": {"input_hashes": input_hashes(cfg)}}
    return out


def reference(cfg: dict) -> dict:
    """Reference labels (L_AL on the shared signals); tests monkeypatch this."""
    from .protocol import load_pools, reference_labels, verify_inputs
    verify_inputs(cfg)          # frozen hashes before any label read, as every aggregator does
    return reference_labels(*load_pools(cfg))


def load_e04(cfg: dict, results_dir) -> dict | None:
    """Fold-0 re-score of the E4 compare cells, or None while cells/frozen file are absent.

    Seeds follow the E4 file the numbers use: 20 (``seeds_stage2``) once ``e04_compare_s20.json``
    exists, else the Stage-1 seeds (Task 16).  If the Stage-2 file exists but its fold-0 cells do not,
    the study is absent (never a silent fall-back to 5 seeds beside a 20-seed E4)."""
    from pathlib import Path
    from . import arch_study as A
    fz = Path(results_dir) / "e04_frozen_configs.json"
    if not fz.is_file():
        return None
    stage = "stage2" if (Path(results_dir) / "e04_compare_s20.json").is_file() else "stage1"
    frozen = A.load_frozen(fz)
    groups = A.compare_specs(cfg, frozen, seeds=cfg["seeds_" + stage], budgets=cfg["budgets"])
    if any(load_cell(s, cfg) is None for ss in groups.values() for s in ss if s["fold"] == 0):
        return None
    ref = reference(cfg)
    return fold0_study(cfg, groups, ref, cfg["budgets"], prefix="EfourFoldZero", of="e04", stage=stage)


def load_e05(cfg: dict, results_dir) -> dict | None:
    import json
    from pathlib import Path
    from . import arch_study as A
    from .jobs import cell_specs, grid
    sp = Path(results_dir) / "e05_selection.json"
    if not sp.is_file():
        return None
    sel = json.loads(sp.read_text(encoding="utf-8"))
    groups = dict(A.e05_specs(cfg, sel["selected"], seeds=cfg["seeds_stage1"], budgets=cfg["budgets"]))
    base = grid(cfg, arch="microconv1d", recipe=sel["microconv1d"]["recipe"],
                seeds=cfg["seeds_stage1"], budgets=cfg["budgets"])
    key = "microconv1d" if "microconv1d" not in groups else "microconv1d_baseline"
    groups[key] = cell_specs(base, pool="al", strategy="unc_oracle", eta=0.0)
    if any(load_cell(s, cfg) is None for ss in groups.values() for s in ss if s["fold"] == 0):
        return None
    ref = reference(cfg)
    return fold0_study(cfg, groups, ref, cfg["budgets"], prefix="EfiveFoldZero", of="e05")
