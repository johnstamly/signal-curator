"""E5 step 1: pick the top 3 non-MicroConv1D models of e04_compare by pooled balanced accuracy at
budget 600 and write results/jobs/e05_arch_loop.json (unc_oracle loop cells).  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e05_jobs.py [--seeds stage1|stage2]
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.arch_cli import e05_jobs_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(e05_jobs_main())
