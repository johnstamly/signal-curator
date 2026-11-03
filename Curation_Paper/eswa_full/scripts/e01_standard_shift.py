"""E1 standard-shift evidence (R36): BAD rate, panel, receiver and cycle-0 mix and model disagreement
of the seed and of each session's additions.  Reads both label databases read-only (mode=ro URI,
through protocol.load_pools, after verify_inputs checked the frozen hashes) and results/e10_scores.json.
From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e01_standard_shift.py
Writes results/e01_standard_shift.json.
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

from Curation_Paper.eswa_full.lib.protocol import (  # noqa: E402
    REPO_ROOT, load_config, load_pools, sha256_file, verify_inputs)
from Curation_Paper.eswa_full.lib.runio import git_sha, results_dir, save_result  # noqa: E402
from Curation_Paper.eswa_full.lib.standard_shift import build_standard_shift  # noqa: E402
from Curation_Paper.eswa_full.lib.versioning import dirty_files  # noqa: E402

CODE_FILES = ["Curation_Paper/eswa_full/lib/standard_shift.py",
              "Curation_Paper/eswa_full/lib/live_record.py",
              "Curation_Paper/eswa_full/scripts/e01_standard_shift.py",
              "Curation_Paper/eswa_full/tests/test_standard_shift.py"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E1 standard-shift evidence")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    verify_inputs(cfg)
    al, rand = load_pools(cfg)
    scores_path = results_dir(cfg) / "e10_scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    payload = build_standard_shift(al, rand, scores["ids"], scores["mean_p"])
    payload["extra_input_hashes"] = {"e10_scores.json": sha256_file(scores_path)}
    payload["code"] = {"git_sha": git_sha(), "files": CODE_FILES,
                       "dirty_files": dirty_files(CODE_FILES, REPO_ROOT)}
    out = save_result("e01_standard_shift", payload, cfg)
    print("->", out)
    print("pools", payload["pools"], "| shared", payload["n_shared"], "seed", payload["n_seed"],
          "shared outside seed", payload["n_shared_outside_seed"])
    for name, g in payload["groups"].items():
        m = g["model"]
        print(f"{name:13s} n={g['n']:4d} BAD={g['n_bad']:3d} ({100 * g['bad_fraction']:.1f}%) "
              f"BAD->GOOD {m['n_bad_called_good']}/{g['n_bad']}  GOOD->BAD {m['n_good_called_bad']}/{g['n_good']}  "
              f"missing-from-scores {m['n_missing_from_scores']}  in-L_AL-training {m['n_in_L_AL_training']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
