"""E4 step 4: aggregate results/e04_compare[_s20].json from stored cells.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e04_compare.py [--seeds stage1|stage2]
Refuses on missing / stale cells.  minirocket carries calibrated=false (ECE not comparable).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.arch_cli import compare_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(compare_main())
