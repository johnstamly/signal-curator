"""E2: re-inspection (uncertainty, unc_oracle, unc_selftrain at eta 0 and 0.10).
Run from the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e02_reinspection.py [--seeds stage1|stage2]

Reads stored loop cells by spec (see lib/cells.py); refuses to write if any is missing.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.aggregate import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli("e02"))
