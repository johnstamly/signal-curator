"""E10: f_amb per budget from the E3 unc_oracle eta-0 cells (mean +/- s.d. over folds and seeds).
Aggregator: run after Stage 1.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e10_famb.py [--seeds stage1|stage2]
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.specimen_cli import e10_famb_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(e10_famb_main())
