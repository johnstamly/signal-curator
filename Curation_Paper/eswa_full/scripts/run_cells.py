"""Resumable parallel cell runner.  Run from the repo root (background-safe, re-run to resume):
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/run_cells.py ^
        --jobs Curation_Paper/eswa_full/results/jobs/stage1_loop.json --workers 6

Exit code: 0 all cells present; 1 at least one cell failed; 3 only deferred (memory pressure,
re-run later).  Failures: results/cells/_errors/<id>.txt.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.cells import dirty_dependencies, run_cells  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import load_config, verify_inputs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--results-dir", default=None, help="override results dir (smoke runs)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="smoke tests only: run although dependency files have uncommitted changes")
    args = ap.parse_args(argv)
    specs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))["specs"]
    dirty = dirty_dependencies(specs)
    if dirty and not args.allow_dirty:
        print("REFUSING to start: uncommitted changes in dependency files of the cell kind(s): "
              + ", ".join(dirty) + "; commit them or pass --allow-dirty (smoke only)",
              file=sys.stderr)
        return 4
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    cfg["allow_dirty"] = bool(args.allow_dirty)
    print(f"verifying frozen inputs ...", flush=True)
    info = verify_inputs(cfg)
    print(f"inputs ok (union {info['n_union']}); {len(specs)} cells, {args.workers} workers",
          flush=True)
    counts = run_cells(specs, args.workers, cfg, log=lambda m: print(m, flush=True))
    if counts["failed"]:
        return 1
    return 3 if counts["deferred"] else 0


if __name__ == "__main__":
    sys.exit(main())
