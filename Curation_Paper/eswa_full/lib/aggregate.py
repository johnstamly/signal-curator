"""Study aggregation over stored cells (R15): pool test predictions over the folds per
(strategy, pool, eta, seed, budget), score them against the reference labels, then report
mean +/- s.d. and a bootstrap CI over seeds.  Cells are found by spec, never by file name.

Counters (loop cells, summed over the 5 folds of one seed, then mean/sd over seeds):
  recovery_rate    n_recovered / n_flipped      (nan when no label was flipped)
  false_flag_rate  n_false_flags / (n_labelled - n_flipped)   labelled rows are summed over
                   folds; a false flag is a flagged row whose label was never flipped
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from .cells import cell_id, load_cell
from .jobs import (add_grid_args, cell_specs, cfg_and_grid, e1_specs, e2_specs, e3_specs,
                   e7_budgets, e7_specs, E7_BASE)
from .metrics import bootstrap_ci, metrics
from .protocol import PKG_ROOT, input_hashes, load_pools, reference_labels, verify_inputs
from .runio import save_result

METRIC_KEYS = ("acc", "bal_acc", "bad_recall", "good_retention", "auroc", "ece")
COUNTER_KEYS = ("n_flipped", "n_flagged", "n_recovered", "n_ever_recovered", "n_false_flags",
                "n_residual_flips")


class MissingCells(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"{len(missing)} required cell(s) missing or invalid: "
                         + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))


class InputHashMismatch(RuntimeError):
    pass


class KeyMismatch(RuntimeError):
    """Cells of one study come from different code versions, or from a dirty tree."""


STALE_HINT = ("re-run scripts/run_cells.py with the same job file -- stale cells are "
              "recomputed automatically")


def require(specs: list[dict], cfg: dict, *, expect_key: str | None = None,
            allow_dirty: bool = False) -> dict:
    """Every spec must be stored, valid and spec-matching; all cells must share ONE code_key
    (content hash of the kind's dependency files; or start with ``expect_key``) and none may
    come from a dirty tree unless ``allow_dirty``.  Call ``verify_inputs(cfg)`` first: the
    input-hash check here compares the cells' recorded hashes with the (then verified) frozen
    config hashes.  Returns {"cells_code_key", "key_counts", "git_shas", "n_cells", "n_dirty"}."""
    seen, missing, bad = set(), [], []
    keys: dict[str, int] = {}
    git_shas: dict[str, int] = {}
    n_dirty = 0
    frozen = input_hashes(cfg)
    for s in specs:
        c = cell_id(s)
        if c in seen:
            continue
        seen.add(c)
        payload = load_cell(s, cfg)
        if payload is None:
            missing.append(c)
            continue
        meta = payload.get("meta", {})
        if meta.get("input_hashes") != frozen:
            bad.append(c)
        k = str(meta.get("code_key", "?"))
        keys[k] = keys.get(k, 0) + 1
        g = str(meta.get("git_sha", "?"))
        git_shas[g] = git_shas.get(g, 0) + 1
        n_dirty += bool(meta.get("dirty", True))
    if missing:
        raise MissingCells(missing)
    if bad:
        raise InputHashMismatch(f"{len(bad)} cell(s) carry input hashes differing from the "
                                f"frozen ones, first: {bad[0]}")
    listing = ", ".join(f"{k[:10]}: {v} cell(s)" for k, v in sorted(keys.items()))
    if expect_key is not None:
        if any(not k.startswith(expect_key) for k in keys):
            raise KeyMismatch(f"cells with a code key other than --expect-key {expect_key}: "
                              f"{listing}; {STALE_HINT}")
    elif len(keys) != 1:
        raise KeyMismatch(f"cells come from {len(keys)} different code keys ({listing}); "
                          f"{STALE_HINT}")
    if n_dirty and not allow_dirty:
        raise KeyMismatch(f"{n_dirty} cell(s) were computed from dirty dependency files; "
                          f"{STALE_HINT} (pass --allow-dirty for smoke runs only)")
    return {"cells_code_key": next(iter(keys)) if len(keys) == 1 else sorted(keys),
            "key_counts": keys, "git_shas": git_shas, "n_cells": len(seen), "n_dirty": n_dirty}


def _nanstat(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0}
    lo, hi = bootstrap_ci(v)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "ci_lo": lo, "ci_hi": hi, "n": int(v.size)}


def curve(cfg: dict, g: dict, ref: dict, *, pool: str, strategy: str, eta: float,
          seed_size: int | None = None, batch: int | None = None, budgets: list | None = None,
          max_budget: int | None = None, exclude=()) -> list[dict]:
    """One row per budget: pooled-fold metrics per seed, summarised over seeds."""
    excl = set(exclude)
    specs = cell_specs(g, pool=pool, strategy=strategy, eta=eta, seed_size=seed_size,
                       batch=batch, budgets=budgets)
    by_seed: dict[int, list[dict]] = {}
    for s in specs:
        payload = load_cell(s, cfg)
        if payload is None:
            raise MissingCells([cell_id(s)])
        by_seed.setdefault(s["seed"], []).append(payload["result"])
    seeds = sorted(by_seed)
    first = by_seed[seeds[0]][0]
    budget_list = [r["budget"] for r in first["budgets"]]
    if max_budget is not None:
        budget_list = [b for b in budget_list if b <= max_budget]
    rows = []
    for b in budget_list:
        per_seed, ok_all, min_pool = [], True, None
        ctr = {k: [] for k in COUNTER_KEYS + ("n_labelled",)}
        for seed in seeds:
            ys, ps, scored = [], [], []
            c = dict.fromkeys(ctr, 0)
            for res in by_seed[seed]:
                rec = next(r for r in res["budgets"] if r["budget"] == b)
                tp = rec["test_pred"]
                if set(tp) != set(res["test_ids"]):
                    raise ValueError("test_pred keys differ from test_ids")
                for sid in res["test_ids"]:
                    if sid in excl:
                        continue
                    if sid not in ref:
                        raise ValueError(f"pooled scored set contains {sid!r}, "
                                         f"which has no reference label")
                    ys.append(ref[sid])
                    ps.append(tp[sid])
                    scored.append(sid)
                ok_all &= bool(rec["selection_claim_ok"])
                min_pool = (res["len_train_pool"] if min_pool is None
                            else min(min_pool, res["len_train_pool"]))
                for k in COUNTER_KEYS:
                    c[k] += rec[k]
                c["n_labelled"] += b
            expected = set(ref) - excl
            if len(scored) != len(set(scored)) or set(scored) != expected:
                raise ValueError(
                    f"pooled scored set differs from the reference set minus exclusions "
                    f"(seed {seed}, budget {b}): {len(scored)} scored, {len(set(scored))} unique, "
                    f"{len(expected)} expected")
            per_seed.append(metrics(np.array(ys), np.array(ps)))
            for k in ctr:
                ctr[k].append(c[k])
        row = {"budget": b, "n_seeds": len(seeds), "seeds": seeds,
               "n_scored": per_seed[0]["n"],
               "selection_claim_ok": bool(ok_all), "min_train_pool": int(min_pool),
               "metrics": {k: {**_nanstat([m[k] for m in per_seed]),
                               "values": [m[k] for m in per_seed]} for k in METRIC_KEYS}}
        fl = np.array(ctr["n_flipped"], dtype=float)
        rec_rate = np.where(fl > 0, np.array(ctr["n_recovered"]) / np.where(fl > 0, fl, 1), np.nan)
        clean = np.array(ctr["n_labelled"], dtype=float) - fl
        ff_rate = np.array(ctr["n_false_flags"]) / np.where(clean > 0, clean, 1)
        row["counters"] = {k: _nanstat(ctr[k]) for k in COUNTER_KEYS}
        row["counters"]["recovery_rate"] = _nanstat(rec_rate)
        row["counters"]["false_flag_rate"] = _nanstat(ff_rate)
        rows.append(row)
    return rows


def _revised_ids() -> list[str]:
    d = json.loads((PKG_ROOT / "results" / "e00_inputs.json").read_text(encoding="utf-8"))
    ids = [x["signal_id"] for x in d["disagree"]]
    if len(ids) != 13:
        raise ValueError(f"expected 13 revised ids in e00_inputs.json, got {len(ids)}")
    return ids


def _meta(g: dict) -> dict:
    return {"seeds": g["seeds"], "n_seeds": len(g["seeds"]), "folds": g["folds"],
            "arch": g["arch"], "budgets": g["budgets"], "seed_size": g["seed_size"],
            "batch": g["batch"]}


def build_e1(cfg, g, ref):
    revised = _revised_ids()
    cfgs = {"unc_oracle_al": dict(pool="al", strategy="unc_oracle"),
            "random_rand": dict(pool="rand", strategy="random"),
            "random_al": dict(pool="al", strategy="random")}
    out = {**_meta(g), "configs": {}, "sensitivity_excluding_revised": {},
           "revised_ids": revised, "eta": 0.0}
    for name, c in cfgs.items():
        out["configs"][name] = {**c, "rows": curve(cfg, g, ref, eta=0.0, **c)}
        out["sensitivity_excluding_revised"][name] = {
            **c, "rows": curve(cfg, g, ref, eta=0.0, exclude=revised, **c)}
    return out


def build_e2(cfg, g, ref):
    out = {**_meta(g), "etas": g["e2_etas"], "cells": {}}
    for strat in ("uncertainty", "unc_oracle", "unc_selftrain"):
        for eta in g["e2_etas"]:
            out["cells"][f"{strat}|eta={eta:g}"] = {
                "strategy": strat, "eta": eta, "pool": "al",
                "rows": curve(cfg, g, ref, pool="al", strategy=strat, eta=eta)}
    return out


def build_e3(cfg, g, ref):
    out = {**_meta(g), "etas": g["etas"], "cells": {}}
    for strat in ("uncertainty", "unc_oracle"):
        for eta in g["etas"]:
            out["cells"][f"{strat}|eta={eta:g}"] = {
                "strategy": strat, "eta": eta, "pool": "al",
                "rows": curve(cfg, g, ref, pool="al", strategy=strat, eta=eta)}
    return out


def build_e7(cfg, g, ref):
    out = {**_meta(g), "max_budget": g["e7_max_budget"], "cells": {},
           "base_from_e3": {"seed_size": E7_BASE[0], "batch": E7_BASE[1]}}
    for s in g["e7_seed_sizes"]:
        for b in g["e7_batches"]:
            if (s, b) == E7_BASE:
                rows = curve(cfg, g, ref, pool="al", strategy="unc_oracle", eta=0.0,
                             max_budget=g["e7_max_budget"])
            else:
                rows = curve(cfg, g, ref, pool="al", strategy="unc_oracle", eta=0.0,
                             seed_size=s, batch=b, budgets=e7_budgets(g, s))
            out["cells"][f"seed_size={s}|batch={b}"] = {
                "seed_size": s, "batch": b, "strategy": "unc_oracle", "eta": 0.0, "pool": "al",
                "rows": rows}
    return out


STUDIES = {
    "e01": (e1_specs, build_e1, "e01_selection"),
    "e02": (e2_specs, build_e2, "e02_reinspection"),
    "e03": (e3_specs, build_e3, "e03_noise"),
    "e07": (e7_specs, build_e7, "e07_loop_settings"),
}


def cli(study: str, argv=None) -> int:
    from .protocol import load_config
    specs_fn, build, name = STUDIES[study]
    ap = argparse.ArgumentParser(description=f"aggregate {name} from stored cells")
    ap.add_argument("--seeds", choices=("stage1", "stage2"), default="stage1")
    add_grid_args(ap)
    ap.add_argument("--expect-key", default=None,
                    help="accept cells whose code_key starts with this instead of requiring one key")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="smoke runs only: accept cells computed from a dirty tree")
    args = ap.parse_args(argv)
    overrides = [o for o in ("seed_list", "fold_list", "budgets", "arch", "recipe_json",
                             "seed_size", "batch", "etas", "e2_etas", "e7_seed_sizes",
                             "e7_batches", "e7_max_budget") if getattr(args, o) is not None]
    if overrides and not args.results_dir:
        print(f"REFUSING: grid override(s) {overrides} need --results-dir so a smoke "
              f"invocation cannot overwrite a headline JSON", file=sys.stderr)
        return 4
    cfg, g = cfg_and_grid(args, load_config(), args.seeds)
    verify_inputs(cfg)  # frozen hashes, counts and HDF5 resolvability, before any label read
    try:
        prov = require(specs_fn(g), cfg, expect_key=args.expect_key,
                       allow_dirty=args.allow_dirty)
    except MissingCells as e:
        print(f"REFUSING to write {name}: {e}", file=sys.stderr)
        print("missing cell ids (all):", " ".join(e.missing), file=sys.stderr)
        return 2
    except InputHashMismatch as e:
        print(f"REFUSING to write {name}: {e}", file=sys.stderr)
        return 3
    except KeyMismatch as e:
        print(f"REFUSING to write {name}: {e}", file=sys.stderr)
        return 5
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    payload = build(cfg, g, ref)
    payload["stage"] = args.seeds
    payload["cells_code_key"] = prov["cells_code_key"]
    payload["cells_key_counts"] = prov["key_counts"]
    payload["cells_git_shas"] = prov["git_shas"]  # information only
    payload["cells_dirty"] = prov["n_dirty"]
    out = save_result(name + ("_s20" if args.seeds == "stage2" else ""), payload, cfg)
    print("->", out)
    return 0
