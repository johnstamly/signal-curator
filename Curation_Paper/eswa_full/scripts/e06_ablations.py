"""E6: aggregate results/e06_ablations[_s20].json from stored cells.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e06_ablations.py [--seeds stage1|stage2]
Refuses on missing, stale (code key differs / mixed keys) or dirty cells.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.ablate_study import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
