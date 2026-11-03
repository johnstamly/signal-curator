"""E14: live-session record -- labels per day, seed/added counts, the 13 revised shared labels,
and the reconstructed f_amb trajectory.  Reads both label databases read-only.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e14_live_record.py
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.live_record import build_live_record  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import (  # noqa: E402
    REPO_ROOT, load_config, load_pools, repo_path, sha256_file, verify_inputs)
from Curation_Paper.eswa_full.lib.runio import git_sha, save_result  # noqa: E402
from Curation_Paper.eswa_full.lib.versioning import dirty_files  # noqa: E402

FAMB = "Curation_Paper/results/famb_trajectory.json"
CODE_FILES = ["Curation_Paper/eswa_full/lib/live_record.py",
              "Curation_Paper/eswa_full/lib/downstream.py",
              "Curation_Paper/eswa_full/scripts/e14_live_record.py",
              "Curation_Paper/eswa_full/tests/test_downstream.py"]


def sources(cfg) -> dict:
    out = {}
    for name, spec in cfg["pools"].items():
        con = sqlite3.connect(f"file:{repo_path(spec['path']).as_posix()}?mode=ro", uri=True)
        try:
            out[name] = {r[0]: r[1] for r in con.execute(
                "SELECT source, COUNT(*) FROM labels WHERE freq_khz = ? GROUP BY source",
                (cfg["freq_khz"],))}
        finally:
            con.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E14 live-session record")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    verify_inputs(cfg)
    al, rand = load_pools(cfg)
    famb = json.loads(repo_path(FAMB).read_text(encoding="utf-8"))
    payload = build_live_record(al, rand, famb)
    payload["label_source_values"] = sources(cfg)
    payload["extra_input_hashes"] = {"famb_trajectory.json": sha256_file(repo_path(FAMB))}
    payload["code"] = {"git_sha": git_sha(), "files": CODE_FILES,
                       "dirty_files": dirty_files(CODE_FILES, REPO_ROOT)}
    out = save_result("e14_live_record", payload, cfg)
    print("->", out)
    sa = payload["seed_and_added"]
    print(f"seed {sa['n_seed']} + added {sa['n_added']} = {sa['n_total']}; "
          f"cross-check {payload['cross_check_famb']}")
    print("per day L_AL ", payload["labels_per_day"]["L_AL"])
    print("per day L_rand", payload["labels_per_day"]["L_rand"])
    r = payload["revisions"]
    print(f"revisions {r['n_revised']}/{r['n_shared']} = {r['revision_rate']:.4f}; "
          f"rand->AL {r['direction_rand_to_AL']}; in time {r['direction_in_time']}; "
          f"earlier pool {r['earlier_pool']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
