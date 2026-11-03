"""E9: the short paper's three classical filters (Table 3 settings) and a depth<=3 decision tree
on the same three statistics.  CPU only; independent of Stage 1.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e09_rules.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.specimen_cli import e09_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(e09_main())
