"""Per-fold training-pool / early-stopping / test sizes of the E1 loop cells.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e00_pool_sizes.py
Reads the stored cells by spec (one code key required, none dirty); writes results/e00_pool_sizes.json.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.aggregate import MissingCells, require  # noqa: E402
from Curation_Paper.eswa_full.lib.cells import load_cell  # noqa: E402
from Curation_Paper.eswa_full.lib.jobs import add_grid_args, cfg_and_grid, e1_specs  # noqa: E402
from Curation_Paper.eswa_full.lib.pool_sizes import build_pool_sizes  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import REPO_ROOT, load_config, verify_inputs  # noqa: E402
from Curation_Paper.eswa_full.lib.runio import git_sha, save_result  # noqa: E402
from Curation_Paper.eswa_full.lib.versioning import dirty_files  # noqa: E402

CODE_FILES = ["Curation_Paper/eswa_full/lib/pool_sizes.py",
              "Curation_Paper/eswa_full/scripts/e00_pool_sizes.py",
              "Curation_Paper/eswa_full/tests/test_pool_sizes.py"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E1 cell pool sizes")
    add_grid_args(ap)
    args = ap.parse_args(argv)
    cfg, g = cfg_and_grid(args, load_config(), "stage1")
    verify_inputs(cfg)
    specs = e1_specs(g)
    try:
        prov = require(specs, cfg)
    except MissingCells as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    results = [load_cell(s, cfg)["result"] for s in specs]
    payload = build_pool_sizes(results)
    payload.update({"source": "E1 loop cells (unc_oracle/L_AL, random/L_rand, random/L_AL; eta 0)",
                    "seeds": g["seeds"], "folds": g["folds"], "stage": "stage1",
                    "cells_code_key": prov["cells_code_key"], "cells_key_counts": prov["key_counts"],
                    "cells_git_shas": prov["git_shas"], "cells_dirty": prov["n_dirty"],
                    "code": {"git_sha": git_sha(), "files": CODE_FILES,
                             "dirty_files": dirty_files(CODE_FILES, REPO_ROOT)}})
    out = save_result("e00_pool_sizes", payload, cfg)
    print("->", out)
    for k in ("len_train_pool", "len_es", "n_test"):
        print(k, payload[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
