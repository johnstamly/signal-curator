"""E10: final MicroConv1D trained on all of L_AL (seeds 0-4, R5), scored on all 12,504 signals;
per panel x cycle GOOD coverage.  Writes e10_coverage.json and e10_scores.json (per-signal mean p,
the input of E11).  Run the memory preflight first.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e10_coverage.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.specimen_cli import e10_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(e10_main())
