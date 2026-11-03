"""Cell kind "ablate": one static MicroConv1D variant fitted once (E6, ruling R23).

A cell fits ``spec["arch"]`` with ``spec["recipe"]`` (the default recipe with one factor changed)
on a seeded, label-stratified 600-row subsample of fold ``f``'s training pool (the "al" pool,
fold ``f`` held out), early-stops on the fold's early-stopping split, and scores the fold's test
signals against the reference labels.  The default-recipe cell is the reference of the study.

Everything but the recipe is identical across the variants of one (fold, seed): the test folds
(built exactly as ``kind_loop`` builds them: ``shared_folds(reference_labels(al, rand), n_folds,
0)``; ``FOLD_SEED`` must equal ``kind_loop.FOLD_SEED``, a test asserts it), ``fold_split``, the
600-row subsample (``kind_tune.subsample_stratified``, a pure function, imported rather than
copied) and the fit seed.  The variants are therefore paired.

Window (E6 window ablation): ``fit`` never slices, so this kind builds ``X_of`` with the recipe's
window (``recipe["window"]`` or the config window) and ``checked_x_of`` asserts on every call that
the returned width equals that window (a Task 5 carry).

The test fold is read exactly once, for the final prediction; training and early stopping only
see rows ``fold_split`` kept out of the test fold (checked again below).

Import chain (verified by a test that imports this module in a fresh interpreter):
  kind_ablate -> protocol, signals, splits, fit, models, metrics, kind_tune
  kind_tune   -> protocol, signals, splits, fit, models  (no loop.py, no strategies.py)
  fit -> classic (sklearn, aeon), models;  models -> app.model;  signals -> protocol
"""
from __future__ import annotations

import functools
import zlib

import numpy as np

from .fit import fit, predict
from .kind_tune import TUNE_N, subsample_stratified
from .metrics import metrics
from .protocol import load_pools, reference_labels
from .signals import signals_for
from .splits import fold_split, shared_folds

KIND = "ablate"
FOLD_SEED = 0       # == kind_loop.FOLD_SEED: the five test folds are fixed across seeds
N_TRAIN = TUNE_N    # 600: the largest loop budget

DEPENDENCY_FILES = (
    "Curation_Paper/eswa_full/lib/protocol.py",
    "Curation_Paper/eswa_full/lib/signals.py",
    "Curation_Paper/eswa_full/lib/splits.py",
    "Curation_Paper/eswa_full/lib/models.py",
    "Curation_Paper/eswa_full/lib/fit.py",
    "Curation_Paper/eswa_full/lib/classic.py",
    "Curation_Paper/eswa_full/lib/metrics.py",
    "Curation_Paper/eswa_full/lib/kind_tune.py",
    "Curation_Paper/eswa_full/lib/kind_ablate.py",
    "Curation_Paper/eswa_full/config/protocol.json",
    "app/model.py",
    "app/__init__.py",
    "Curation_Paper/__init__.py",
    "Curation_Paper/eswa_full/__init__.py",
    "Curation_Paper/eswa_full/lib/__init__.py",
)

_WORLD: dict = {}


def _world(cfg: dict):
    """(al pool, reference labels, folds) -- loaded once per process."""
    key = repr(sorted((k, repr(cfg[k])) for k in ("pools", "freq_khz", "n_folds")))
    if key not in _WORLD:
        al, rand = load_pools(cfg)
        ref = reference_labels(al, rand)
        _WORLD.clear()
        _WORLD[key] = (al, ref, shared_folds(ref, cfg["n_folds"], FOLD_SEED))
    return _WORLD[key]


def checked_x_of(x_of, window: int):
    """``x_of`` that asserts every returned array has exactly ``window`` columns."""
    @functools.wraps(x_of)
    def wrapped(ids):
        X = x_of(ids)
        if X.ndim != 2 or X.shape[1] != window:
            raise AssertionError(f"X_of returned shape {tuple(X.shape)}, but the recipe "
                                 f"window is {window}")
        return X
    return wrapped


def run(spec: dict, cfg: dict) -> dict:
    if spec["pool"] != "al" or spec["strategy"] != "ablate" or spec["arch"] != "microconv1d":
        raise ValueError("ablate cells use pool 'al', strategy 'ablate' and arch 'microconv1d'")
    if spec["budgets"] != [N_TRAIN]:
        raise ValueError(f"ablate cells use budgets [{N_TRAIN}], got {spec['budgets']}")
    pool, ref, folds = _world(cfg)
    if not 0 <= spec["fold"] < len(folds):
        raise ValueError(f"fold {spec['fold']} outside 0..{len(folds) - 1}")
    test_ids = folds[spec["fold"]]
    test = set(test_ids)
    missing = sorted(test - set(ref))
    if missing:
        raise ValueError(f"{len(missing)} test ids have no reference label, first: {missing[0]}")
    seed, recipe = spec["seed"], spec["recipe"]
    train_pool, es_set = fold_split(pool, test_ids, seed, cfg["es_frac"])
    sub = subsample_stratified(train_pool, N_TRAIN, seed)
    leaked = [r["signal_id"] for r in sub + es_set if r["signal_id"] in test]
    if leaked:  # fold_split removes them; a failure here would be a programming error
        raise RuntimeError(f"{len(leaked)} test signal(s) in training data, first: {leaked[0]}")

    window = int(recipe.get("window") or cfg["window"])
    X_of = checked_x_of(functools.partial(signals_for, cfg=cfg, window=window), window)

    model, info = fit(spec["arch"], sub, es_set, X_of,
                      seed=zlib.crc32(f"{seed}|{N_TRAIN}|ablate".encode()), recipe=recipe)
    p = predict(model, X_of(list(test_ids)))
    y = np.array([ref[s] for s in test_ids], dtype=int)
    return {"metrics": metrics(y, p), "test_ids": list(test_ids),
            "test_pred": {s: round(float(v), 6) for s, v in zip(test_ids, p)},
            "es_loss": float(info["es_loss"]), "epochs": int(info["epochs"]),
            "train_seconds": float(info["train_seconds"]), "device": str(info["device"]),
            "n_train": len(sub), "n_es": len(es_set), "len_train_pool": len(train_pool),
            "window": window, "recipe": recipe}
