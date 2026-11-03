"""E4 step 3: write results/jobs/e04_compare[_s20].json from e04_frozen_configs.json
(loop cells: strategy random, pool al, eta 0, budgets 100..600, 5 folds, 8 models).  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e04_compare_jobs.py [--seeds stage1|stage2]
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.arch_cli import compare_jobs_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(compare_jobs_main())
