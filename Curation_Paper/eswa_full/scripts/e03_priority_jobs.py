"""R38 priority job: the loop cells E2/E3 must recompute first under the R37 loop key.

Selects from results/jobs/stage1_loop.json (same spec dicts, so the cell IDs are identical):
  - every loop spec with eta > 0 (E3 uncertainty / unc_oracle, E2 unc_selftrain eta 0.1);
  - the eta = 0 unc_oracle cells on pool al at the base setting (seed_size / batch / budgets of
    the grid), so that E2/E3 aggregate under one code key.
Writes results/jobs/e03_priority.json.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e03_priority_jobs.py
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.cells import cell_id  # noqa: E402
from Curation_Paper.eswa_full.lib.jobs import _dedup  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT  # noqa: E402

SRC = PKG_ROOT / "results" / "jobs" / "stage1_loop.json"
OUT = PKG_ROOT / "results" / "jobs" / "e03_priority.json"


def select(doc: dict) -> list[dict]:
    g = doc["grid"]
    picked = []
    for s in doc["specs"]:
        if s["eta"] > 0:
            picked.append(s)
        elif (s["strategy"] == "unc_oracle" and s["pool"] == "al" and s["eta"] == 0
              and s["seed_size"] == g["seed_size"] and s["batch"] == g["batch"]
              and s["budgets"] == g["budgets"]):
            picked.append(s)
    return _dedup(picked)


def main() -> int:
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    specs = select(doc)
    by = Counter((s["strategy"], s["eta"]) for s in specs)
    print(f"source: {SRC.name}  ({len(doc['specs'])} specs)")
    for (strat, eta), n in sorted(by.items()):
        print(f"  {strat:14s} eta={eta:<5} {n}")
    print(f"TOTAL priority cells: {len(specs)}")
    ids = {cell_id(s) for s in specs}
    assert len(ids) == len(specs)
    src_ids = {cell_id(s) for s in doc["specs"]}
    assert ids <= src_ids, "priority cell id not in stage1_loop.json"
    counts = {"total": len(specs),
              "by_strategy_eta": {f"{k[0]}|eta={k[1]}": v for k, v in sorted(by.items())}}
    OUT.write_text(json.dumps({"grid": doc["grid"], "counts": counts, "specs": specs}, indent=1),
                   encoding="utf-8")
    print("->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
