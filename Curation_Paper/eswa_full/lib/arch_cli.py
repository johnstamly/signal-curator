"""Command-line entry points of E4 / E5 (thin scripts call these).  Every command that reads
cells refuses on missing, stale, mixed-key or dirty cells with the aggregator's actionable hint.
Exit codes as ``aggregate.cli``: 2 missing cells, 3 input-hash mismatch, 5 code-key mismatch,
4 smoke override without --results-dir, 6 missing prerequisite file."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import arch_study as A
from .aggregate import InputHashMismatch, KeyMismatch, MissingCells, require
from .jobs import grid
from .protocol import REPO_ROOT, load_config, load_pools, reference_labels, verify_inputs
from .runio import results_dir, save_result


def _base_args(ap: argparse.ArgumentParser, *, seeds: bool = True) -> None:
    if seeds:
        ap.add_argument("--seeds", choices=("stage1", "stage2"), default="stage1")
    ap.add_argument("--results-dir", default=None, help="override cfg results_dir (smoke runs)")


def _grid_args(ap: argparse.ArgumentParser) -> None:
    """Smoke-run overrides (need --results-dir)."""
    ap.add_argument("--models", default=None, help="comma list (smoke runs)")
    ap.add_argument("--seed-list", default=None, help="comma list (smoke runs)")
    ap.add_argument("--budgets", default=None, help="comma list (smoke runs)")


def _untracked_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--allow-untracked", action="store_true",
                    help="smoke runs only: accept a frozen/selection file that is not committed")


def _cell_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--expect-key", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="smoke runs only: accept cells computed from a dirty tree")


def _cfg(args) -> dict:
    cfg = load_config()
    if getattr(args, "results_dir", None):
        cfg["results_dir"] = args.results_dir
    return cfg


def _suffix(args) -> str:
    return "_s20" if getattr(args, "seeds", "stage1") == "stage2" else ""


def _models(args) -> tuple:
    return tuple(args.models.split(",")) if getattr(args, "models", None) else A.MODELS


def _grid(cfg: dict, args) -> dict:
    stage = "seeds_stage2" if getattr(args, "seeds", "stage1") == "stage2" else "seeds_stage1"
    over = {"seeds": [int(x) for x in args.seed_list.split(",")] if args.seed_list
            else list(cfg[stage])}
    if args.budgets:
        over["budgets"] = [int(x) for x in args.budgets.split(",")]
    return grid(cfg, **over)


def _refuse_overrides(args, names=("models", "seed_list", "budgets")) -> bool:
    over = [o for o in names if getattr(args, o, None) is not None]
    if over and not args.results_dir:
        print(f"REFUSING: override(s) {over} need --results-dir so a smoke invocation cannot "
              f"overwrite a headline JSON", file=sys.stderr)
        return True
    return False


def _write_jobs(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("->", path)


def _read_cells(label: str, specs: list[dict], cfg: dict, args) -> dict | int:
    """require() with the standard refusals; returns provenance or an exit code."""
    verify_inputs(cfg)
    try:
        return require(specs, cfg, expect_key=args.expect_key, allow_dirty=args.allow_dirty)
    except MissingCells as e:
        print(f"REFUSING to write {label}: {e}", file=sys.stderr)
        print("missing cell ids (all):", " ".join(e.missing), file=sys.stderr)
        print(f"hint: {_HINT}", file=sys.stderr)
        return 2
    except InputHashMismatch as e:
        print(f"REFUSING to write {label}: {e}", file=sys.stderr)
        return 3
    except KeyMismatch as e:
        print(f"REFUSING to write {label}: {e}", file=sys.stderr)
        return 5


_HINT = ("run scripts/run_cells.py with the job file that lists these cells; stale cells "
         "are recomputed automatically")


def _stamp(payload: dict, prov: dict, args) -> dict:
    payload["stage"] = getattr(args, "seeds", "stage1")
    payload["cells_code_key"] = prov["cells_code_key"]
    payload["cells_key_counts"] = prov["key_counts"]
    payload["cells_git_shas"] = prov["git_shas"]
    payload["cells_dirty"] = prov["n_dirty"]
    return payload


def _git_tracked_clean(path: Path) -> bool:
    p = str(Path(path).resolve())
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", p], cwd=REPO_ROOT, check=True,
                       capture_output=True)
        out = subprocess.run(["git", "status", "--porcelain", "--", p], cwd=REPO_ROOT, check=True,
                             capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return False
    return not out.strip()


def _committed(path: Path, args, what: str) -> bool:
    """The file a downstream step reads must be committed and unmodified (the frozen configs are
    committed before any compare cell runs).  ``--allow-untracked`` is for smoke runs."""
    if getattr(args, "allow_untracked", False) or _git_tracked_clean(path):
        return True
    print(f"REFUSING: {what} {path} is not tracked in git or has uncommitted changes; commit it "
          f"first (smoke runs: --allow-untracked with --results-dir)", file=sys.stderr)
    return False


def _tuned_provenance_ok(frozen: dict, path: Path) -> bool:
    if frozen.get("cells_dirty") != 0 or not isinstance(frozen.get("cells_code_key"), str):
        print(f"REFUSING: {path} was frozen from dirty tune cells or from several code keys "
              f"(cells_dirty={frozen.get('cells_dirty')!r}, cells_code_key="
              f"{frozen.get('cells_code_key')!r}); re-run the tune cells and e04_freeze.py",
              file=sys.stderr)
        return False
    return True


def _need(path: Path, hint: str) -> bool:
    if not path.is_file():
        print(f"REFUSING: {path} does not exist; {hint}", file=sys.stderr)
        return False
    return True


# ------------------------------------------------------------------ e04_tune_jobs

def tune_jobs_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the E4 tuning job list")
    _base_args(ap, seeds=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--archs", default=None, help="comma list (smoke runs)")
    ap.add_argument("--seed-list", default=None, help="comma list (smoke runs)")
    ap.add_argument("--lrs", default=None, help="comma list (smoke runs)")
    ap.add_argument("--widths", default=None, help="comma list, torch archs (smoke runs)")
    ap.add_argument("--kernels", default=None, help="comma list, minirocket (smoke runs)")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only")
    args = ap.parse_args(argv)
    if _refuse_overrides(args, ("archs", "seed_list", "lrs", "widths", "kernels", "max_epochs")):
        return 4
    cfg = _cfg(args)
    kw = {}
    if args.archs:
        kw["archs"] = tuple(args.archs.split(","))
    if args.seed_list:
        kw["seeds"] = tuple(int(x) for x in args.seed_list.split(","))
    if args.lrs:
        kw["lrs"] = tuple(float(x) for x in args.lrs.split(","))
    if args.widths:
        kw["widths"] = tuple(float(x) for x in args.widths.split(","))
    if args.kernels:
        kw["kernels"] = tuple(int(x) for x in args.kernels.split(","))
    if args.max_epochs:
        kw["max_epochs"] = args.max_epochs
    j = A.tune_jobs(**kw)
    for a, c in j["configs"].items():
        print(f"{a:12s} {len(c)} config(s): " + ", ".join(
            f"{x['name']} ({x['n_params']})" for x in c))
    print("tune cells:", j["counts"]["total"])
    out = Path(args.out) if args.out else results_dir(cfg) / "jobs" / "e04_tune.json"
    _write_jobs(out, j)
    return 0


# ------------------------------------------------------------------ e04_freeze

def freeze_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="freeze one recipe per model from the tuning cells")
    _base_args(ap, seeds=False)
    _cell_args(ap)
    ap.add_argument("--jobs", default=None, help="tune job list (default results/jobs/e04_tune.json)")
    args = ap.parse_args(argv)
    cfg = _cfg(args)
    jobs = Path(args.jobs) if args.jobs else results_dir(cfg) / "jobs" / "e04_tune.json"
    if not _need(jobs, "run scripts/e04_tune_jobs.py first"):
        return 6
    spec_file = json.loads(jobs.read_text(encoding="utf-8"))
    specs = spec_file["specs"]
    prov = _read_cells("e04_frozen_configs", specs, cfg, args)
    if isinstance(prov, int):
        return prov
    names = {(a, json.dumps(c["recipe"], sort_keys=True)): c["name"]
             for a, cs in spec_file.get("configs", {}).items() for c in cs}
    payload = _stamp(A.freeze_payload(specs, cfg, names), prov, args)
    payload["tune_seeds"] = spec_file.get("seeds")
    for a, v in payload["models"].items():
        f = v["frozen"]
        print(f"{a:12s} -> {f['name']}  es_loss {f['es_loss_mean']:.4f} +/- {f['es_loss_sd']:.4f}"
              f"  es_auroc {f['es_auroc_mean']:.4f}  (ranked by {v['selection_metric']})")
    print("->", save_result("e04_frozen_configs", payload, cfg))
    return 0


# ------------------------------------------------------------------ e04_compare_jobs

def compare_jobs_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the E4 comparison job list from frozen configs")
    _base_args(ap)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--out", default=None)
    _grid_args(ap)
    _untracked_arg(ap)
    args = ap.parse_args(argv)
    if _refuse_overrides(args) or (args.allow_untracked and not args.results_dir):
        return 4
    cfg = _cfg(args)
    frozen_path = Path(args.frozen) if args.frozen else results_dir(cfg) / "e04_frozen_configs.json"
    if not _need(frozen_path, "run scripts/e04_freeze.py first"):
        return 6
    if not _committed(frozen_path, args, "frozen configs"):
        return 7
    models = _models(args)
    frozen = A.load_frozen(frozen_path, models)
    g = _grid(cfg, args)
    per_model = A.compare_specs(cfg, frozen, seeds=g["seeds"], models=models, budgets=g["budgets"])
    shared = A.shared_with_e1(cfg, per_model, seeds=g["seeds"], budgets=g["budgets"])
    specs = [s for ss in per_model.values() for s in ss]
    if "microconv1d" in per_model:
        mc = len(per_model["microconv1d"])
        print("MicroConv1D compare cells identical to E1 random/al cells (same IDs):",
              f"{shared['microconv1d']}/{mc}",
              "-> dedup" if shared["microconv1d"] == mc else "-> NOT deduplicated (tuned recipe differs)")
    out = Path(args.out) if args.out else results_dir(cfg) / "jobs" / f"e04_compare{_suffix(args)}.json"
    _write_jobs(out, {
        "stage": args.seeds, "frozen_from": str(frozen_path), "seeds": g["seeds"],
        "budgets": g["budgets"],
        "models": {m: frozen["models"][m]["frozen"]["recipe"] for m in models},
        "shared_with_e1": shared,
        "counts": {"total": len(specs), **{m: len(s) for m, s in per_model.items()}},
        "specs": specs})
    return 0


# ------------------------------------------------------------------ e04_compare

def compare_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="aggregate e04_compare from stored cells")
    _base_args(ap)
    _cell_args(ap)
    ap.add_argument("--frozen", default=None)
    _grid_args(ap)
    args = ap.parse_args(argv)
    if _refuse_overrides(args):
        return 4
    cfg = _cfg(args)
    frozen_path = Path(args.frozen) if args.frozen else results_dir(cfg) / "e04_frozen_configs.json"
    if not _need(frozen_path, "run scripts/e04_freeze.py first"):
        return 6
    models = _models(args)
    frozen = A.load_frozen(frozen_path, models)
    g = _grid(cfg, args)
    specs = [s for ss in A.compare_specs(cfg, frozen, seeds=g["seeds"], models=models,
                                         budgets=g["budgets"]).values() for s in ss]
    prov = _read_cells("e04_compare", specs, cfg, args)
    if isinstance(prov, int):
        return prov
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    payload = A.build_compare(cfg, g, ref, frozen, models)
    payload["frozen_from"] = str(frozen_path)
    payload["frozen_created"] = frozen.get("meta", {}).get("created")
    out = save_result("e04_compare" + _suffix(args), _stamp(payload, prov, args), cfg)
    print("->", out)
    return 0


# ------------------------------------------------------------------ e05_jobs

def e05_jobs_main(argv=None) -> int:
    """Stage 1: select the top 3 non-MicroConv1D models by es AUROC of their frozen config and
    write results/e05_selection.json (to be committed) plus the E5 job list.  Stage 2 reuses the
    committed Stage-1 selection and never re-selects."""
    ap = argparse.ArgumentParser(description="write the E5 job list (top 3 non-MicroConv1D models)")
    _base_args(ap)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=A.E5_TOP)
    ap.add_argument("--models", default=None, help="comma list (smoke runs)")
    ap.add_argument("--seed-list", default=None, help="comma list (smoke runs)")
    ap.add_argument("--budgets", default=None, help="comma list (smoke runs)")
    _untracked_arg(ap)
    args = ap.parse_args(argv)
    if _refuse_overrides(args, ("models", "seed_list", "budgets")) or (
            args.allow_untracked and not args.results_dir):
        return 4
    cfg = _cfg(args)
    sel_path = results_dir(cfg) / "e05_selection.json"
    if args.seeds == "stage2":
        if not _need(sel_path, "Stage 2 reuses the Stage-1 selection: run e05_jobs.py (stage1) "
                               "and commit results/e05_selection.json first"):
            return 6
        if not _committed(sel_path, args, "selection"):
            return 7
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
    else:
        frozen_path = Path(args.frozen) if args.frozen else results_dir(cfg) / "e04_frozen_configs.json"
        if not _need(frozen_path, "run scripts/e04_freeze.py first"):
            return 6
        if not _committed(frozen_path, args, "frozen configs"):
            return 7
        frozen = A.load_frozen(frozen_path, _models(args))
        if not _tuned_provenance_ok(frozen, frozen_path):
            return 5
        selection = A.selection_payload(frozen, args.top)
        selection["frozen_from"] = str(frozen_path)
        print("->", save_result("e05_selection", dict(selection), cfg))
    for s in selection["selected"]:
        print(f"{s['arch']:12s} es_auroc {s['es_auroc_mean']:.4f}")
    mc = selection["microconv1d"]
    print("MicroConv1D baseline recipe equals default:", mc["frozen_equals_default"])
    g = _grid(cfg, args)
    specs = A.e05_required_specs(cfg, selection, g["seeds"], g["budgets"])
    out = Path(args.out) if args.out else results_dir(cfg) / "jobs" / f"e05_arch_loop{_suffix(args)}.json"
    _write_jobs(out, {"stage": args.seeds, "selected": selection["selected"], "seeds": g["seeds"],
                      "budgets": g["budgets"],
                      "note": "includes MicroConv1D's baseline cells at its frozen recipe (the E3 "
                              "cells, skipped when stored, if that recipe is the default)",
                      "counts": {"total": len(specs)}, "specs": specs})
    return 0


# ------------------------------------------------------------------ e05_arch_loop

def e05_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="aggregate e05_arch_loop from stored cells")
    _base_args(ap)
    _cell_args(ap)
    ap.add_argument("--selection", default=None,
                    help="selection file (default results/e05_selection.json)")
    ap.add_argument("--seed-list", default=None, help="comma list (smoke runs)")
    ap.add_argument("--budgets", default=None, help="comma list (smoke runs)")
    _untracked_arg(ap)
    args = ap.parse_args(argv)
    if _refuse_overrides(args, ("seed_list", "budgets")) or (
            args.allow_untracked and not args.results_dir):
        return 4
    cfg = _cfg(args)
    sel_path = Path(args.selection) if args.selection else results_dir(cfg) / "e05_selection.json"
    if not _need(sel_path, "run scripts/e05_jobs.py first"):
        return 6
    if not _committed(sel_path, args, "selection"):
        return 7
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    g = _grid(cfg, args)
    prov = _read_cells("e05_arch_loop", A.e05_required_specs(cfg, selection, g["seeds"],
                                                              g["budgets"]), cfg, args)
    if isinstance(prov, int):
        return prov
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    payload = A.build_e05(cfg, g, ref, selection)
    out = save_result("e05_arch_loop" + _suffix(args), _stamp(payload, prov, args), cfg)
    print("->", out)
    return 0
