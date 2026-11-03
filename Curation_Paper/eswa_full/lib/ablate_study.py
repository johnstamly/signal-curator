"""E6 mechanism study: recipe ablations and warm start (ruling R23, on the cell runner).

Static variants (kind "ablate", lib/kind_ablate.py): per fold and seed one MicroConv1D fit on a
seeded stratified 600-row subsample of the "al" training pool with ONE recipe factor changed
from the default; the default-recipe cell is the reference.  Variants of one (fold, seed) share
test fold, subsample, early-stopping split and fit seed, so they are paired.  The reference is
therefore a random 600-row subsample, not the end point of a curated loop (stated in the paper).

Warm start (kind "loop"): unc_oracle, pool al, eta 0, budgets 100..600, seed size 100, batch 50,
recipe = default + ``"warm_start": true`` (popped by kind_loop, R19).  Compared with E3's
unc_oracle eta-0 cells (the scratch curve): the same specs but for the recipe, so the two curves
differ only by warm starting.  Those scratch cells are listed in the job file too (they dedup by
cell ID with the Stage-1 job list) so a smoke run is self-contained.

Statistics: pooled test metrics (5 folds pooled per seed), then mean, s.d. and bootstrap CI over
seeds; paired differences vs the default (per seed on the pooled metrics, and per (fold, seed)
on the cells' own metrics); the warm-start curve minus the scratch curve per budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .aggregate import (METRIC_KEYS, InputHashMismatch, KeyMismatch, MissingCells, STALE_HINT,
                        _nanstat, curve, require)
from .cells import canonical_spec, cell_id, load_cell
from .fit import DEFAULT_RECIPE
from .jobs import cell_specs, grid
from .kind_ablate import FOLD_SEED, N_TRAIN
from .metrics import metrics
from .models import build_model, n_params
from .protocol import load_config, load_pools, reference_labels, verify_inputs
from .runio import results_dir, save_result
from .splits import shared_folds

# (name, change from the default recipe); the first entry is the reference
VARIANTS = (
    ("default", {}),
    ("relu", {"activation": "relu"}),
    ("class_weight_none", {"class_weight": "none"}),
    ("class_weight_full", {"class_weight": "full"}),
    ("no_augment", {"augment": False}),
    ("window_250", {"window": 250}),
    ("window_1000", {"window": 1000}),
    ("no_zscore", {"zscore": False}),
)
VARIANT_NAMES = tuple(n for n, _ in VARIANTS)
PAIRED_CELL_KEYS = ("acc", "bal_acc", "bad_recall", "good_retention", "auroc")
WARM_COMPARE_KEYS = METRIC_KEYS

DESIGN_NOTE = (
    "Static variants are scored on a seeded random 600-row stratified subsample of the fold's "
    "training pool, not on a curated-loop end point; the warm-start variant runs inside the "
    "unc_oracle loop (eta 0) and is compared with E3's scratch cells.  Window variants change the "
    "MicroConv1D parameter count (input length enters the classifier).")


def variant_recipe(name: str, extra: dict | None = None) -> dict:
    change = dict(VARIANTS)[name]
    return {**json.loads(json.dumps(DEFAULT_RECIPE)), **change, **(extra or {})}


def ablate_spec(recipe: dict, fold: int, seed: int) -> dict:
    return canonical_spec({
        "study_source": "ablate", "pool": "al", "fold": fold, "seed": seed, "strategy": "ablate",
        "eta": 0.0, "arch": "microconv1d", "recipe": recipe, "budgets": [N_TRAIN],
        "seed_size": 0, "batch": 0})


def grid_for(cfg: dict, *, seeds=None, folds=None, variants=None, extra: dict | None = None,
             budgets=None) -> dict:
    """Study grid.  ``folds`` are the ablate folds (default all); the loop cells always use all
    folds (``loop_folds``).  ``extra`` is merged into every recipe (smoke runs: max_epochs)."""
    extra = extra or {}
    g = grid(cfg, recipe={**DEFAULT_RECIPE, **extra},
             seeds=list(seeds) if seeds is not None else None,
             budgets=list(budgets) if budgets is not None else None)
    g["loop_folds"] = list(range(cfg["n_folds"]))
    g["folds"] = list(folds) if folds is not None else list(g["loop_folds"])
    g["variants"] = list(variants) if variants is not None else list(VARIANT_NAMES)
    g["extra"] = extra
    unknown = set(g["variants"]) - set(VARIANT_NAMES)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; choose from {VARIANT_NAMES}")
    return g


def ablate_specs(g: dict, variants=None) -> list[dict]:
    return [ablate_spec(variant_recipe(v, g["extra"]), f, s)
            for v in (variants or g["variants"]) for s in g["seeds"] for f in g["folds"]]


def _loop_grid(g: dict, warm: bool) -> dict:
    recipe = {**g["recipe"], "warm_start": True} if warm else dict(g["recipe"])
    return {**g, "recipe": recipe, "folds": g["loop_folds"]}


def warm_specs(g: dict) -> list[dict]:
    return cell_specs(_loop_grid(g, True), pool="al", strategy="unc_oracle", eta=0.0)


def scratch_specs(g: dict) -> list[dict]:
    """E3's unc_oracle eta-0 cells (same IDs as the Stage-1 job list)."""
    return cell_specs(_loop_grid(g, False), pool="al", strategy="unc_oracle", eta=0.0)


def job_payload(cfg: dict, g: dict) -> dict:
    ab, warm, scratch = ablate_specs(g), warm_specs(g), scratch_specs(g)
    specs = ab + warm + scratch
    return {
        "study": "e06_ablations", "seeds": g["seeds"], "folds": g["folds"],
        "variants": {n: variant_recipe(n, g["extra"]) for n in g["variants"]},
        "warm_budgets": g["budgets"], "design_note": DESIGN_NOTE,
        "counts": {"ablate": len(ab), "warm_start_loop": len(warm), "scratch_shared": len(scratch),
                   "total_new": len(ab) + len(warm), "total": len(specs)},
        "specs": specs}


# ------------------------------------------------------------------ aggregation

def _pooled(results: list[dict], ref: dict, expected: set) -> dict:
    ys, ps, scored = [], [], []
    for res in results:
        tp = res["test_pred"]
        if set(tp) != set(res["test_ids"]):
            raise ValueError("test_pred keys differ from test_ids")
        for sid in res["test_ids"]:
            ys.append(ref[sid])
            ps.append(tp[sid])
            scored.append(sid)
    if len(scored) != len(set(scored)) or set(scored) != expected:
        raise ValueError(f"pooled scored set differs from the reference test folds: "
                         f"{len(scored)} scored, {len(set(scored))} unique, {len(expected)} expected")
    return metrics(np.array(ys), np.array(ps))


def _with_values(vals) -> dict:
    return {**_nanstat(vals), "values": [float(v) for v in vals]}


def build_ablate(cfg: dict, g: dict, ref: dict, fold_ids: dict, variants=None) -> dict:
    """Per variant: pooled metrics over seeds, paired differences vs the default."""
    variants = list(variants or g["variants"])
    expected = set()
    for f in g["folds"]:
        expected |= set(fold_ids[f])
    out = {"seeds": g["seeds"], "n_seeds": len(g["seeds"]), "folds": g["folds"],
           "partial_folds": expected != set(ref), "n_scored": len(expected),
           "n_train": N_TRAIN, "variants": {}}
    per_variant: dict[str, dict] = {}
    for v in variants:
        by_seed, cell_metrics, cell_train = {}, {}, []
        for s in ablate_specs(g, variants=(v,)):
            payload = load_cell(s, cfg)
            res = payload["result"]
            if set(res["test_ids"]) != set(fold_ids[s["fold"]]):
                raise ValueError(f"cell {cell_id(s)}: test_ids differ from fold {s['fold']}")
            by_seed.setdefault(s["seed"], []).append(res)
            cell_metrics[(s["fold"], s["seed"])] = res["metrics"]
            cell_train.append(res)
        seeds = sorted(by_seed)
        pooled = [_pooled(by_seed[sd], ref, expected) for sd in seeds]
        recipe = variant_recipe(v, g["extra"])
        window = int(recipe["window"] or cfg["window"])
        model = build_model("microconv1d", window, activation=recipe["activation"])
        per_variant[v] = {
            "seeds": seeds, "cell_metrics": cell_metrics,
            "entry": {
                "change": dict(VARIANTS)[v], "recipe": recipe, "window": window,
                "n_params": int(n_params(model)),
                "metrics": {k: _with_values([m[k] for m in pooled]) for k in METRIC_KEYS},
                "train": {k: _nanstat([r[k] for r in cell_train])
                          for k in ("epochs", "train_seconds", "es_loss")},
            }}
    ref_v = per_variant.get("default")
    for v in variants:
        e = per_variant[v]["entry"]
        if v == "default" or ref_v is None:
            e["paired_vs_default"] = e["paired_cells_vs_default"] = None
        else:
            if per_variant[v]["seeds"] != ref_v["seeds"]:
                raise ValueError(f"variant {v}: seeds differ from the default's")
            e["paired_vs_default"] = {
                k: _with_values(np.array(e["metrics"][k]["values"])
                                - np.array(ref_v["entry"]["metrics"][k]["values"]))
                for k in METRIC_KEYS}
            keys = sorted(per_variant[v]["cell_metrics"])
            if keys != sorted(ref_v["cell_metrics"]):
                raise ValueError(f"variant {v}: (fold, seed) cells differ from the default's")
            e["paired_cells_vs_default"] = {
                k: _nanstat([per_variant[v]["cell_metrics"][c][k] - ref_v["cell_metrics"][c][k]
                             for c in keys]) for k in PAIRED_CELL_KEYS}
        out["variants"][v] = e
    return out


def _mean_train_seconds(specs: list[dict], cfg: dict) -> dict:
    acc: dict[int, list[float]] = {}
    for s in specs:
        for rec in load_cell(s, cfg)["result"]["budgets"]:
            acc.setdefault(rec["budget"], []).append(float(rec["train_seconds"]))
    return {str(b): _nanstat(v) for b, v in sorted(acc.items())}


def build_warm(cfg: dict, g: dict, ref: dict) -> dict:
    """Warm-start curve vs the scratch (E3) curve, plus the paired per-budget differences."""
    gs, gw = _loop_grid(g, False), _loop_grid(g, True)
    scratch = curve(cfg, gs, ref, pool="al", strategy="unc_oracle", eta=0.0)
    warm = curve(cfg, gw, ref, pool="al", strategy="unc_oracle", eta=0.0)
    paired = []
    for rs, rw in zip(scratch, warm):
        if rs["budget"] != rw["budget"] or rs["seeds"] != rw["seeds"]:
            raise ValueError("warm and scratch curves differ in budgets or seeds")
        paired.append({"budget": rs["budget"], "n_seeds": rs["n_seeds"],
                       "warm_minus_scratch": {
                           k: _with_values(np.array(rw["metrics"][k]["values"])
                                           - np.array(rs["metrics"][k]["values"]))
                           for k in WARM_COMPARE_KEYS}})
    return {"strategy": "unc_oracle", "pool": "al", "eta": 0.0, "seed_size": gs["seed_size"],
            "batch": gs["batch"], "budgets": gs["budgets"], "seeds": g["seeds"],
            "scratch_rows": scratch, "warm_rows": warm, "paired": paired,
            "train_seconds_cumulative": {"scratch": _mean_train_seconds(scratch_specs(g), cfg),
                                         "warm": _mean_train_seconds(warm_specs(g), cfg)}}


def require_by_kind(specs: list[dict], cfg: dict, args) -> dict | tuple:
    """``require`` per cell kind (each kind has its own code key).  Returns the combined
    provenance, or (exit_code, message) for a refusal: 2 missing, 3 input hashes, 5 stale/dirty."""
    groups: dict[str, list[dict]] = {}
    for s in specs:
        groups.setdefault(s["study_source"], []).append(s)
    prov = {"cells_code_key": {}, "key_counts": {}, "git_shas": {}, "n_dirty": 0, "n_cells": 0}
    for kind, ss in sorted(groups.items()):
        try:
            p = require(ss, cfg, expect_key=getattr(args, "expect_key", None),
                        allow_dirty=getattr(args, "allow_dirty", False))
        except MissingCells as e:
            return 2, f"{e}\nmissing cell ids (all): {' '.join(e.missing)}\nhint: {_HINT}"
        except InputHashMismatch as e:
            return 3, str(e)
        except KeyMismatch as e:
            return 5, f"[{kind} cells] {e}"
        prov["cells_code_key"][kind] = p["cells_code_key"]
        prov["key_counts"][kind] = p["key_counts"]
        prov["git_shas"][kind] = p["git_shas"]
        prov["n_dirty"] += p["n_dirty"]
        prov["n_cells"] += p["n_cells"]
    return prov


_HINT = ("run scripts/run_cells.py with results/jobs/e06_ablations.json (and the Stage-1 job "
         "list for the E3 scratch cells); " + STALE_HINT)


# ------------------------------------------------------------------ command lines

def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seeds", choices=("stage1", "stage2"), default="stage1")
    ap.add_argument("--results-dir", default=None, help="override cfg results_dir (smoke runs)")
    ap.add_argument("--seed-list", default=None, help="comma list (smoke runs)")
    ap.add_argument("--fold-list", default=None, help="ablate folds, comma list (smoke runs)")
    ap.add_argument("--variants", default=None, help="comma list (smoke runs)")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only")
    ap.add_argument("--warm-budgets", default=None, help="comma list (smoke runs)")


_OVERRIDES = ("seed_list", "fold_list", "variants", "max_epochs", "warm_budgets")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x != ""]


def _setup(args) -> tuple[dict, dict] | None:
    over = [o for o in _OVERRIDES if getattr(args, o, None) is not None]
    if over and not args.results_dir:
        print(f"REFUSING: override(s) {over} need --results-dir so a smoke invocation cannot "
              f"overwrite a headline file", file=sys.stderr)
        return None
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    stage = "seeds_stage2" if args.seeds == "stage2" else "seeds_stage1"
    g = grid_for(cfg,
                 seeds=_ints(args.seed_list) if args.seed_list else cfg[stage],
                 folds=_ints(args.fold_list) if args.fold_list else None,
                 variants=args.variants.split(",") if args.variants else None,
                 extra={"max_epochs": args.max_epochs} if args.max_epochs else None,
                 budgets=_ints(args.warm_budgets) if args.warm_budgets else None)
    return cfg, g


def _suffix(args) -> str:
    return "_s20" if args.seeds == "stage2" else ""


def jobs_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the E6 job list")
    _add_common(ap)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    setup = _setup(args)
    if setup is None:
        return 4
    cfg, g = setup
    j = job_payload(cfg, g)
    j["stage"] = args.seeds
    print("cells:", j["counts"])
    out = Path(args.out) if args.out else results_dir(cfg) / "jobs" / f"e06_ablations{_suffix(args)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(j, indent=1), encoding="utf-8")
    print("->", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="aggregate e06_ablations from stored cells")
    _add_common(ap)
    ap.add_argument("--expect-key", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="smoke runs only: accept cells computed from a dirty tree")
    args = ap.parse_args(argv)
    setup = _setup(args)
    if setup is None:
        return 4
    cfg, g = setup
    verify_inputs(cfg)
    specs = ablate_specs(g) + warm_specs(g) + scratch_specs(g)
    prov = require_by_kind(specs, cfg, args)
    if isinstance(prov, tuple):
        print(f"REFUSING to write e06_ablations: {prov[1]}", file=sys.stderr)
        return prov[0]
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    folds = shared_folds(ref, cfg["n_folds"], FOLD_SEED)
    payload = {
        "stage": args.seeds, "seeds": g["seeds"], "design_note": DESIGN_NOTE,
        "ablate": build_ablate(cfg, g, ref, dict(enumerate(folds))),
        "warm_start": build_warm(cfg, g, ref),
        "cells_code_key": prov["cells_code_key"], "cells_key_counts": prov["key_counts"],
        "cells_git_shas": prov["git_shas"], "cells_dirty": prov["n_dirty"]}
    if payload["ablate"]["partial_folds"] and not args.results_dir:
        print("REFUSING: partial folds without --results-dir", file=sys.stderr)
        return 4
    out = save_result("e06_ablations" + _suffix(args), payload, cfg)
    print("->", out)
    return 0
