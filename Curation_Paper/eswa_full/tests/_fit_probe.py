"""Subprocess probe for test_fit: trains a short fit and writes predictions to .npy.

Usage: python _fit_probe.py <out.npy> [arch]   (PYTHONPATH must contain the repo root)
"""
import sys

import numpy as np

from Curation_Paper.eswa_full.lib.fit import fit, predict
from Curation_Paper.eswa_full.tests.synth import make_pool, x_of


def main() -> None:
    out = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "microconv1d"
    train, es = make_pool(80, 0), make_pool(30, 1, prefix="e")
    model, _ = fit(arch, train, es, x_of, seed=3, recipe={"max_epochs": 5})
    ids = [r["signal_id"] for r in es]
    np.save(out, predict(model, x_of(ids)))


if __name__ == "__main__":
    main()
