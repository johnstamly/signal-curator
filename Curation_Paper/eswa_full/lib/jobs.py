"""Cell specifications of the loop studies E1/E2/E3/E7 and the deduplicated Stage-1 job list.

R12: shared loop cells run once.  E3 is the source for ``uncertainty`` and ``unc_oracle`` on
the active-learning pool at every eta; E1, E2 and E7 read them.  The (seed_size 100,
batch 50) E7 configuration is NOT a separate cell: the E7 aggregator reads the E3 eta-0
``unc_oracle`` cells truncated at the E7 maximum budget.  Queries depend only on the
labelled count, so truncation changes nothing but the number of records.
"""
from __future__ import annotations

import argparse
import json

from .cells import canonical_spec, cell_id
from .fit import DEFAULT_RECIPE

E3_ETAS = (0.0, 0.02, 0.05, 0.10, 0.20)
E2_ETAS = (0.0, 0.10)
E7_SEED_SIZES = (50, 100, 200)
E7_BATCHES = (25, 50, 100)
E7_MAX_BUDGET = 400
E7_BASE = (100, 50)  # (seed_size, batch) served by the E3 eta-0 unc_oracle cells


def grid(cfg: dict, **over) -> dict:
    g = {
        "arch": "microconv1d", "recipe": dict(DEFAULT_RECIPE),
        "seeds": list(cfg["seeds_stage1"]), "folds": list(range(cfg["n_folds"])),
        "budgets": list(cfg["budgets"]), "seed_size": 100, "batch": 50,
        "etas": list(E3_ETAS), "e2_etas": list(E2_ETAS),
        "e7_seed_sizes": list(E7_SEED_SIZES), "e7_batches": list(E7_BATCHES),
        "e7_max_budget": E7_MAX_BUDGET, "study_source": "loop",
    }
    unknown = set(over) - set(g)
    if unknown:
        raise ValueError(f"unknown grid keys {sorted(unknown)}")
    g.update({k: v for k, v in over.items() if v is not None})
    return g


def cell_specs(g: dict, *, pool: str, strategy: str, eta: float, seed_size=None, batch=None,
           budgets=None) -> list[dict]:
    out = []
    for seed in g["seeds"]:
        for fold in g["folds"]:
            out.append(canonical_spec({
                "study_source": g["study_source"], "pool": pool, "fold": fold, "seed": seed,
                "strategy": strategy, "eta": eta, "arch": g["arch"], "recipe": g["recipe"],
                "budgets": g["budgets"] if budgets is None else budgets,
                "seed_size": g["seed_size"] if seed_size is None else seed_size,
                "batch": g["batch"] if batch is None else batch}))
    return out


def e3_specs(g: dict) -> list[dict]:
    return [c for s in ("uncertainty", "unc_oracle") for eta in g["etas"]
            for c in cell_specs(g, pool="al", strategy=s, eta=eta)]


def e2_specs(g: dict) -> list[dict]:
    return [c for s in ("uncertainty", "unc_oracle", "unc_selftrain") for eta in g["e2_etas"]
            for c in cell_specs(g, pool="al", strategy=s, eta=eta)]


def e1_specs(g: dict) -> list[dict]:
    return (cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
            + cell_specs(g, pool="rand", strategy="random", eta=0.0)
            + cell_specs(g, pool="al", strategy="random", eta=0.0))


def e7_budgets(g: dict, seed_size: int) -> list[int]:
    return [seed_size] + [b for b in (200, 300, 400, 500, 600)
                          if seed_size < b <= g["e7_max_budget"]]


def e7_extra_specs(g: dict) -> list[dict]:
    out = []
    for s in g["e7_seed_sizes"]:
        for b in g["e7_batches"]:
            if (s, b) == E7_BASE:
                continue
            out += cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0, seed_size=s, batch=b,
                          budgets=e7_budgets(g, s))
    return out


def e7_specs(g: dict) -> list[dict]:
    """Everything the E7 aggregator reads: the extra cells plus the E3 eta-0 base cells."""
    base = cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    return e7_extra_specs(g) + base


def _dedup(specs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in specs:
        c = cell_id(s)
        if c not in seen:
            seen.add(c)
            out.append(s)
    return out


def stage1_jobs(cfg: dict, *, include_e7: bool = True, **over) -> dict:
    """Deduplicated job list plus per-study counts (cells needed vs cells new)."""
    g = grid(cfg, **over)
    e3 = _dedup(e3_specs(g))
    have = {cell_id(s) for s in e3}
    e2_extra = [s for s in _dedup(e2_specs(g)) if cell_id(s) not in have]
    have |= {cell_id(s) for s in e2_extra}
    e1_extra = [s for s in _dedup(e1_specs(g)) if cell_id(s) not in have]
    have |= {cell_id(s) for s in e1_extra}
    e7_extra = ([s for s in _dedup(e7_extra_specs(g)) if cell_id(s) not in have]
                if include_e7 else [])
    allspecs = _dedup(e3 + e2_extra + e1_extra + e7_extra)
    return {
        "grid": g, "E3": e3, "E2_extra": e2_extra, "E1_extra": e1_extra, "E7_extra": e7_extra,
        "all": allspecs,
        "counts": {
            "E3": len(e3), "E2_extra": len(e2_extra), "E1_extra": len(e1_extra),
            "E7_extra": len(e7_extra), "total": len(allspecs),
            "needed": {"E1": len(_dedup(e1_specs(g))), "E2": len(_dedup(e2_specs(g))),
                       "E3": len(_dedup(e3_specs(g))), "E7": len(_dedup(e7_specs(g)))},
        },
    }


# ------------------------------------------------------------------ command line

def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x != ""]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x != ""]


def add_grid_args(ap: argparse.ArgumentParser) -> None:
    """Overrides used by the smoke test; defaults are the Stage-1 definition."""
    ap.add_argument("--results-dir", default=None, help="override cfg results_dir")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--recipe-json", default=None)
    ap.add_argument("--seed-list", default=None, help="comma list overriding --seeds")
    ap.add_argument("--fold-list", default=None)
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--seed-size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--etas", default=None)
    ap.add_argument("--e2-etas", default=None)
    ap.add_argument("--e7-seed-sizes", default=None)
    ap.add_argument("--e7-batches", default=None)
    ap.add_argument("--e7-max-budget", type=int, default=None)


def cfg_and_grid(args, cfg: dict, stage: str = "stage1") -> tuple[dict, dict]:
    cfg = dict(cfg)
    if getattr(args, "results_dir", None):
        cfg["results_dir"] = args.results_dir
    seeds = cfg["seeds_stage2" if stage == "stage2" else "seeds_stage1"]
    over = {
        "arch": args.arch,
        "recipe": json.loads(args.recipe_json) if args.recipe_json else None,
        "seeds": _ints(args.seed_list) if args.seed_list else list(seeds),
        "folds": _ints(args.fold_list) if args.fold_list else None,
        "budgets": _ints(args.budgets) if args.budgets else None,
        "seed_size": args.seed_size, "batch": args.batch,
        "etas": _floats(args.etas) if args.etas else None,
        "e2_etas": _floats(args.e2_etas) if args.e2_etas else None,
        "e7_seed_sizes": _ints(args.e7_seed_sizes) if args.e7_seed_sizes else None,
        "e7_batches": _ints(args.e7_batches) if args.e7_batches else None,
        "e7_max_budget": args.e7_max_budget,
    }
    return cfg, grid(cfg, **over)
