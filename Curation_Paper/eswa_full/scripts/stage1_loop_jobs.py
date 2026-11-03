"""Write the deduplicated Stage-1 cell list for E1/E2/E3/E7 (R12).  Run from the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/stage1_loop_jobs.py
Default output: Curation_Paper/eswa_full/results/jobs/stage1_loop.json.  The grid overrides
(--arch, --seed-list, ...) exist for the smoke test; defaults are the Stage-1 definition.
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

from Curation_Paper.eswa_full.lib.jobs import add_grid_args, cfg_and_grid, stage1_jobs  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=("stage1", "stage2"), default="stage1")
    ap.add_argument("--out", default=None)
    add_grid_args(ap)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(PKG_ROOT / "results" / "jobs" /
                       ("stage2_loop.json" if args.seeds == "stage2" else "stage1_loop.json"))
    cfg, g = cfg_and_grid(args, load_config(), args.seeds)
    over = {k: v for k, v in g.items()}
    # Stage 2 = E1-E4 only: no E7 cells at 20 seeds
    j = stage1_jobs(cfg, include_e7=args.seeds != "stage2", **over)
    c = j["counts"]
    print(f"E3 source cells        : {c['E3']}")
    print(f"E2 extra (selftrain)   : {c['E2_extra']}   (E2 needs {c['needed']['E2']} in all)")
    print(f"E1 extra (random)      : {c['E1_extra']}   (E1 needs {c['needed']['E1']} in all)")
    print(f"E7 extra (loop settings): {c['E7_extra']}   (E7 needs {c['needed']['E7']} in all)")
    print(f"TOTAL unique cells     : {c['total']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"grid": j["grid"], "counts": c, "specs": j["all"]}, indent=1),
                   encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
