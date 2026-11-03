"""E6 job list: write results/jobs/e06_ablations[_s20].json (ablate cells for 8 recipe variants,
warm-start loop cells, and the shared E3 scratch cells).  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e06_jobs.py [--seeds stage1|stage2]
Run the cells with scripts/run_cells.py --jobs <that file>.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.ablate_study import jobs_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(jobs_main())
