"""E13: labelling throughput (seconds per label from ``labeled_at`` gaps) and the interface
screenshot record.  Reads both label databases read-only.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e13_throughput.py
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

from Curation_Paper.eswa_full.lib import throughput as T  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import (  # noqa: E402
    REPO_ROOT, load_config, load_pools, repo_path, sha256_file, verify_inputs)
from Curation_Paper.eswa_full.lib.live_record import seed_and_added  # noqa: E402
from Curation_Paper.eswa_full.lib.runio import git_sha, save_result  # noqa: E402
from Curation_Paper.eswa_full.lib.versioning import dirty_files  # noqa: E402

FIG_DIR = "Curation_Paper/eswa_full/figures"
MANIFEST = f"{FIG_DIR}/ui_capture_manifest.json"
CODE_FILES = ["Curation_Paper/eswa_full/lib/throughput.py",
              "Curation_Paper/eswa_full/scripts/e13_throughput.py",
              "Curation_Paper/eswa_full/tests/test_throughput.py",
              "Curation_Paper/eswa_full/tests/mutation_throughput.py",
              "Curation_Paper/eswa_full/lib/live_record.py",
              MANIFEST]
POOL_NAMES = {"al": "L_AL", "rand": "L_rand"}


def read_stamps(path: Path, freq: int) -> list[str]:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT labeled_at FROM labels WHERE freq_khz = ? ORDER BY labeled_at", (freq,))]
    finally:
        con.close()


def screenshots_record() -> dict:
    man = json.loads(repo_path(MANIFEST).read_text(encoding="utf-8"))
    for name, rec in man["panels"].items():
        png = repo_path(f"{FIG_DIR}/{rec['file']}")
        rec["sha256"] = sha256_file(png)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E13 throughput")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    verify_inputs(cfg)
    al, rand = load_pools(cfg)
    sa = seed_and_added(al, rand)
    seed_ids = set(sa["seed_ids"])
    al_stamps, al_n = T.pool_usage(al, [])
    rand_stamps, rand_n = T.pool_usage(rand, seed_ids)
    pools = {"L_AL": T.pool_stats(al_stamps), "L_rand": T.pool_stats(rand_stamps)}
    pools["L_AL"].update(al_n)
    pools["L_rand"].update(rand_n)
    pools["L_rand"]["seed_definition"] = sa["derivation"]
    payload = {"rule": T.RULE, "pools": pools,
               "screenshots": screenshots_record(),
               "code": {"git_sha": git_sha(), "files": CODE_FILES,
                        "dirty_files": dirty_files(CODE_FILES, REPO_ROOT)}}
    out = save_result("e13_throughput", payload, cfg)
    print("->", out)
    for name, p in pools.items():
        o = p["overall"]
        print(f"{name}: n={o['n_labels']} days={o['n_days']} breaks={o['n_breaks']} "
              f"median={o['median_s']} s  IQR={o['iqr_s']} s  active={o['active_s']/3600:.2f} h")
        for d, s in p["per_day"].items():
            print(f"   {d}: n={s['n_labels']} breaks={s['n_breaks']} median={s['median_s']} "
                  f"iqr={s['iqr_s']} active={s['active_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
