"""E5: aggregate results/e05_arch_loop[_s20].json (unc_oracle curves of the selected architectures vs
MicroConv1D's E3 unc_oracle eta-0 cells).  From the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/e05_arch_loop.py [--seeds stage1|stage2]
Refuses on missing / stale cells.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from Curation_Paper.eswa_full.lib.arch_cli import e05_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(e05_main())
