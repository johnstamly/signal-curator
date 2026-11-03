"""E4 step 2: freeze one recipe per model (lowest mean early-stopping loss) into
results/e04_frozen_configs.json.  Commit that file BEFORE any compare cell runs.  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e04_freeze.py
Refuses if any tune cell is missing, stale, from mixed code keys or from a dirty tree.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.arch_cli import freeze_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(freeze_main())
