"""E4 step 1: write results/jobs/e04_tune.json (tuning cells; run them with run_cells.py).  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e04_tune_jobs.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.arch_cli import tune_jobs_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(tune_jobs_main())
