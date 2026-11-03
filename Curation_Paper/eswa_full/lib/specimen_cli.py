"""Command lines of E8, E9, E10 and the E10 f_amb aggregator (Task 12).

E8 and f_amb are aggregators over stored loop cells (refuse on missing / stale / dirty cells,
as ``aggregate.cli`` does).  E9 and E10 are not cells: they run directly, and record the git SHA
and the dirty state of the files that define them in the result ("code")."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .aggregate import InputHashMismatch, KeyMismatch, MissingCells, require
from .jobs import add_grid_args, cell_specs, cfg_and_grid
from .protocol import REPO_ROOT, load_config, load_pools, reference_labels, verify_inputs
from .runio import git_sha, save_result
from .splits import shared_folds
from .versioning import dirty_files

DEFAULT_BUDGET = 600
FOLD_SEED = 0  # == kind_loop.FOLD_SEED; imported values would pull torch into the aggregators

CODE_FILES = {
    "e08": ["Curation_Paper/eswa_full/lib/rules_study.py",
            "Curation_Paper/eswa_full/lib/specimen_cli.py",
            "Curation_Paper/eswa_full/scripts/e08_thresholds.py"],
    "e09": ["Curation_Paper/eswa_full/lib/rules_study.py",
            "Curation_Paper/eswa_full/lib/specimen_cli.py",
            "Curation_Paper/eswa_full/scripts/e09_rules.py"],
    "e10": ["Curation_Paper/eswa_full/lib/coverage_study.py",
            "Curation_Paper/eswa_full/lib/specimen_cli.py",
            "Curation_Paper/eswa_full/scripts/e10_coverage.py"],
}


def code_provenance(key: str) -> dict:
    files = CODE_FILES[key]
    return {"git_sha": git_sha(), "files": files, "dirty_files": dirty_files(files, REPO_ROOT)}


def _agg_args(description: str, argv):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--seeds", choices=("stage1", "stage2"), default="stage1")
    add_grid_args(ap)
    ap.add_argument("--expect-key", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="smoke runs only: accept cells computed from a dirty tree")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    return ap.parse_args(argv)


def _cells_setup(args, name: str):
    """(cfg, g) or an int exit code."""
    overrides = [o for o in ("seed_list", "fold_list", "budgets", "arch", "recipe_json",
                             "seed_size", "batch", "etas", "e2_etas", "e7_seed_sizes",
                             "e7_batches", "e7_max_budget") if getattr(args, o) is not None]
    if args.budget != DEFAULT_BUDGET:
        overrides.append("budget")
    if overrides and not args.results_dir:
        print(f"REFUSING: override(s) {overrides} need --results-dir so a smoke invocation "
              f"cannot overwrite a headline JSON", file=sys.stderr)
        return 4
    return cfg_and_grid(args, load_config(), args.seeds)


def _require(specs, cfg, args, name: str):
    try:
        return require(specs, cfg, expect_key=args.expect_key, allow_dirty=args.allow_dirty)
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


def _cells_meta(payload: dict, prov: dict, args) -> None:
    payload["stage"] = args.seeds
    payload["cells_code_key"] = prov["cells_code_key"]
    payload["cells_key_counts"] = prov["key_counts"]
    payload["cells_git_shas"] = prov["git_shas"]
    payload["cells_dirty"] = prov["n_dirty"]


def _suffix(args) -> str:
    return "_s20" if args.seeds == "stage2" else ""


# ------------------------------------------------------------------ E8

def e08_main(argv=None) -> int:
    from . import rules_study as R
    args = _agg_args("E8: p_accept threshold study from stored E3 unc_oracle eta-0 cells", argv)
    setup = _cells_setup(args, "e08_thresholds")
    if isinstance(setup, int):
        return setup
    cfg, g = setup
    verify_inputs(cfg)
    specs = cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    prov = _require(specs, cfg, args, "e08_thresholds")
    if isinstance(prov, int):
        return prov
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    try:
        pooled = R.pool_by_seed(cfg, g, ref, args.budget)
    except ValueError as e:
        print(f"REFUSING to write e08_thresholds: {e}", file=sys.stderr)
        return 6
    payload = {"source": "unc_oracle, pool al, eta 0 loop cells (the E3/E1 curated runs)",
               "budget": args.budget, "arch": g["arch"], **R.build_e8(pooled)}
    _cells_meta(payload, prov, args)
    payload["code"] = code_provenance("e08")
    out = save_result("e08_thresholds" + _suffix(args), payload, cfg)
    print("->", out)
    return 0


def e10_famb_main(argv=None) -> int:
    from . import rules_study as R
    args = _agg_args("E10: f_amb per budget from stored E3 unc_oracle eta-0 cells", argv)
    setup = _cells_setup(args, "e10_famb")
    if isinstance(setup, int):
        return setup
    cfg, g = setup
    verify_inputs(cfg)
    specs = cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    prov = _require(specs, cfg, args, "e10_famb")
    if isinstance(prov, int):
        return prov
    payload = {"source": "unc_oracle, pool al, eta 0 loop cells", "arch": g["arch"],
               **R.famb_by_budget(cfg, g)}
    _cells_meta(payload, prov, args)
    out = save_result("e10_famb" + _suffix(args), payload, cfg)
    print("->", out)
    return 0


# ------------------------------------------------------------------ E9

def e09_main(argv=None) -> int:
    from . import rules_study as R
    from .signals import signals_for
    ap = argparse.ArgumentParser(description="E9: explicit rules (3 filters + depth<=3 tree)")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    verify_inputs(cfg)
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    folds = shared_folds(ref, cfg["n_folds"], FOLD_SEED)
    ids = sorted(al)
    stats = R.statistics(signals_for(ids, cfg))     # first cfg["window"] = 500 raw samples
    stat = dict(zip(ids, stats))
    payload = R.build_e9(stat, al, ref, folds, es_frac=cfg["es_frac"])
    payload["window"] = cfg["window"]
    payload["normalisation"] = "none (raw first-500 samples, float32), as in the short paper"
    payload["fold_seed"] = FOLD_SEED
    payload["code"] = code_provenance("e09")
    out = save_result("e09_rules", payload, cfg)
    print("->", out)
    for n, d in payload["filters"].items():
        print(f"  {n:6s} BAD recall {d['bad_recall']:.3f}  GOOD kept {d['good_kept']:.3f}")
    t = payload["tree"]["pooled"]
    print(f"  tree   BAD recall {t['bad_recall']:.3f}  GOOD kept {t['good_kept']:.3f}")
    return 0


# ------------------------------------------------------------------ E10

def e10_main(argv=None) -> int:
    from . import coverage_study as C
    from .signals import load_universe
    ap = argparse.ArgumentParser(description="E10: per-panel x cycle GOOD coverage (R5)")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--seed-list", default=None, help="comma list (smoke runs)")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only")
    args = ap.parse_args(argv)
    if (args.seed_list or args.max_epochs) and not args.results_dir:
        print("REFUSING: --seed-list / --max-epochs need --results-dir so a smoke invocation "
              "cannot overwrite a headline JSON", file=sys.stderr)
        return 4
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    seeds = ([int(x) for x in args.seed_list.split(",") if x] if args.seed_list
             else list(cfg["seeds_stage1"]))
    verify_inputs(cfg)
    al, _rand = load_pools(cfg)
    ids, X = load_universe(cfg)
    if len(ids) != 12504 and not args.results_dir:
        print(f"REFUSING: universe has {len(ids)} signals, expected 12504", file=sys.stderr)
        return 6
    recipe = {"max_epochs": args.max_epochs} if args.max_epochs else {}
    from .fit import DEFAULT_RECIPE
    out = C.run_e10(al, ids, X, seeds=seeds, es_frac=cfg["es_frac"],
                    recipe={**DEFAULT_RECIPE, **recipe})
    cov, scores = out["coverage"], out["scores"]
    cov["recipe"] = {**DEFAULT_RECIPE, **recipe}
    cov["code"] = code_provenance("e10")
    scores["code"] = cov["code"]
    p1 = save_result("e10_coverage", cov, cfg)
    p2 = save_result("e10_scores", scores, cfg)
    print("->", p1, p2)
    for pn, d in cov["panels"].items():
        print(f"  {pn}: mean GOOD fraction {d['mean_good_fraction']:.3f}, "
              f"cycles<50% {d['n_cycles_below_50']}/{d['n_cycles']}, "
              f"zero-GOOD cycles {d['cycles_zero_good']}")
    return 0
