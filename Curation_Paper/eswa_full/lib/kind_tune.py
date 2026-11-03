"""Cell kind "tune": one hyper-parameter candidate fitted once (E4 tuning, ruling R20).

A tune cell fits ``spec["arch"]`` with ``spec["recipe"]`` (the candidate configuration) on a
seeded, label-stratified subsample of ``TUNE_N`` training rows of fold 0's training pool and
records only early-stopping quantities: ``es_loss``, ``es_auroc`` (threshold-free, from the
fitted model on the early-stopping split; nan if that split has one class), ``epochs``,
``train_seconds``, ``n_params`` (torch models; ``None`` for the classical ones) and ``device``.

Leakage note (disclosed, not removable): fold 0 is the test fold of the loop, but tuning fits
and early-stops on rows that are TEST rows of folds 1-4.  The tuned hyper-parameters therefore
carry information from those rows' labels; results on fold 0 alone are the strictly clean slice.

The test fold is never scored and no test signal is ever read: ``fold_split`` removes the fold's
test IDs from the pool, and the only signals requested are those of the subsample and of the
early-stopping split (``test_ids`` is used for exactly that removal, nothing else).

Folds are built exactly as ``kind_loop`` builds them: ``shared_folds(reference_labels(al, rand),
n_folds, 0)``; ``FOLD_SEED`` below must equal ``kind_loop.FOLD_SEED`` (a test asserts it).

Import chain (verified by reading the imports):
  kind_tune -> protocol, signals, splits, fit, models
  fit       -> classic (sklearn, aeon: third-party), models
  models    -> app.model (MicroConv1D; torch)
  signals   -> protocol;   protocol -> stdlib, h5py
``loop.py``, ``metrics.py`` and ``Curation_Paper/lib/strategies.py`` are not imported, so they
are not dependencies of this kind.
"""
from __future__ import annotations

import random
import zlib

import numpy as np
from sklearn.metrics import roc_auc_score

from .fit import fit, predict
from .models import n_params
from .protocol import load_pools, reference_labels
from .signals import signals_for
from .splits import fold_split, shared_folds

KIND = "tune"
FOLD_SEED = 0       # == kind_loop.FOLD_SEED: the five test folds are fixed across seeds
TUNE_FOLD = 0
TUNE_N = 600        # training rows used per tuning fit (the largest E1 budget)

DEPENDENCY_FILES = (
    "Curation_Paper/eswa_full/lib/protocol.py",
    "Curation_Paper/eswa_full/lib/signals.py",
    "Curation_Paper/eswa_full/lib/splits.py",
    "Curation_Paper/eswa_full/lib/models.py",
    "Curation_Paper/eswa_full/lib/fit.py",
    "Curation_Paper/eswa_full/lib/classic.py",
    "Curation_Paper/eswa_full/lib/kind_tune.py",
    "Curation_Paper/eswa_full/config/protocol.json",
    "app/model.py",
    "app/__init__.py",
    "Curation_Paper/__init__.py",
    "Curation_Paper/eswa_full/__init__.py",
    "Curation_Paper/eswa_full/lib/__init__.py",
)

_WORLD: dict = {}


def _world(cfg: dict):
    """(al pool, fold-0 test ids) -- loaded once per process."""
    key = repr(sorted((k, repr(cfg[k])) for k in ("pools", "freq_khz", "n_folds")))
    if key not in _WORLD:
        al, rand = load_pools(cfg)
        ref = reference_labels(al, rand)
        folds = shared_folds(ref, cfg["n_folds"], FOLD_SEED)
        _WORLD.clear()
        _WORLD[key] = (al, folds[TUNE_FOLD])
    return _WORLD[key]


def subsample_stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` rows, both classes kept, class shares as in ``rows`` (seeded, order-independent)."""
    if n > len(rows):
        raise ValueError(f"cannot draw {n} rows from a training pool of {len(rows)}")
    bad = sorted((r for r in rows if r["label_int"] == 0), key=lambda r: r["signal_id"])
    good = sorted((r for r in rows if r["label_int"] == 1), key=lambda r: r["signal_id"])
    if not bad or not good:
        raise ValueError("training pool lacks one class")
    n_bad = min(max(1, round(n * len(bad) / len(rows))), len(bad))
    n_good = n - n_bad
    if n_good > len(good):
        n_good = len(good)
        n_bad = n - n_good
    rng = random.Random(zlib.crc32(f"{seed}|tune|subsample".encode()))
    pick = rng.sample(bad, n_bad) + rng.sample(good, n_good)
    return sorted(pick, key=lambda r: r["signal_id"])


def run(spec: dict, cfg: dict) -> dict:
    if spec["fold"] != TUNE_FOLD:
        raise ValueError(f"tune cells use fold {TUNE_FOLD}, got {spec['fold']}")
    if spec["pool"] != "al":
        raise ValueError(f"tune cells use the 'al' pool, got {spec['pool']!r}")
    pool, test_ids = _world(cfg)
    seed = spec["seed"]
    train_pool, es_set = fold_split(pool, test_ids, seed, cfg["es_frac"])
    test = set(test_ids)
    sub = subsample_stratified(train_pool, TUNE_N, seed)
    leaked = [r["signal_id"] for r in sub + es_set if r["signal_id"] in test]
    if leaked:  # fold_split removes them; a failure here would be a programming error
        raise RuntimeError(f"{len(leaked)} test signal(s) in tuning data, first: {leaked[0]}")

    def X_of(ids):
        return signals_for(list(ids), cfg)

    model, info = fit(spec["arch"], sub, es_set, X_of, seed=zlib.crc32(f"{seed}|{TUNE_N}|tune".encode()),
                      recipe=spec["recipe"])
    y_es = np.array([r["label_int"] for r in es_set], dtype=int)
    if len(set(y_es.tolist())) < 2:
        es_auroc = float("nan")
    else:
        es_auroc = float(roc_auc_score(y_es, predict(model, X_of([r["signal_id"] for r in es_set]))))
    try:
        import torch.nn as nn
        n_par = n_params(model) if isinstance(model, nn.Module) else None
    except ImportError:  # pragma: no cover
        n_par = None
    return {"es_loss": float(info["es_loss"]), "es_auroc": es_auroc, "epochs": int(info["epochs"]),
            "train_seconds": float(info["train_seconds"]), "n_params": n_par,
            "device": str(info["device"]), "n_train": len(sub), "n_es": len(es_set),
            "len_train_pool": len(train_pool)}
