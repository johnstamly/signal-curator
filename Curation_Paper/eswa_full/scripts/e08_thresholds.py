"""E8: p_accept threshold study on the pooled E3 unc_oracle eta-0 test predictions (budget 600).

Aggregator: refuses on missing, stale or dirty cells; run after Stage 1.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e08_thresholds.py [--seeds stage1|stage2]
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.specimen_cli import e08_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(e08_main())
