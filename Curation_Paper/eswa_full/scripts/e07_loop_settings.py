"""E7: loop settings (seed size x batch), unc_oracle up to budget 400.
Run from the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e07_loop_settings.py [--seeds stage1|stage2]

Reads stored loop cells by spec (see lib/cells.py); the (100, 50) setting is read from the E3
eta-0 unc_oracle cells truncated at 400 (R12).  Refuses to write if any cell is missing.
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
    sys.exit(cli("e07"))
