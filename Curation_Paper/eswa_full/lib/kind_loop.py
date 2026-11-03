"""Cell kind "loop": one ``loop.simulate`` call per cell (studies E1/E2/E3/E5/E6/E7).

Test folds: ``shared_folds(reference_labels(al, rand), 5, seed=FOLD_SEED)`` with FOLD_SEED = 0
for EVERY training seed: the five test folds are fixed across seeds and the seed only varies
training (seed set, early-stopping split, label noise, model init).

DEPENDENCY_FILES is the code key's input (see versioning.py).  Import chain reached from
``simulate`` (verified by reading the imports):
  kind_loop -> loop, protocol, splits
  loop      -> Curation_Paper.lib.strategies (random only), fit, metrics, signals, splits
  fit       -> classic (sklearn, aeon: third-party), models
  models    -> app.model (MicroConv1D; torch)
  signals   -> protocol
  protocol  -> stdlib, h5py
app/trainer.py is only mentioned in comments of fit.py and is not imported.  The package
__init__ files are empty and listed because they execute on import.
"""
from __future__ import annotations

from .loop import simulate
from .protocol import load_pools, reference_labels
from .splits import shared_folds

KIND = "loop"
FOLD_SEED = 0

DEPENDENCY_FILES = (
    "Curation_Paper/eswa_full/lib/protocol.py",
    "Curation_Paper/eswa_full/lib/signals.py",
    "Curation_Paper/eswa_full/lib/splits.py",
    "Curation_Paper/eswa_full/lib/models.py",
    "Curation_Paper/eswa_full/lib/fit.py",
    "Curation_Paper/eswa_full/lib/classic.py",
    "Curation_Paper/eswa_full/lib/metrics.py",
    "Curation_Paper/eswa_full/lib/loop.py",
    "Curation_Paper/eswa_full/lib/kind_loop.py",
    "Curation_Paper/eswa_full/config/protocol.json",
    "Curation_Paper/lib/strategies.py",
    "app/model.py",
    "app/__init__.py",
    "Curation_Paper/__init__.py",
    "Curation_Paper/lib/__init__.py",
    "Curation_Paper/eswa_full/__init__.py",
    "Curation_Paper/eswa_full/lib/__init__.py",
)

_WORLD: dict = {}


def _world(cfg: dict):
    """(pools {'al','rand'}, ref, folds) -- loaded once per process."""
    key = repr(sorted((k, repr(cfg[k])) for k in ("pools", "freq_khz", "n_folds")))
    if key not in _WORLD:
        al, rand = load_pools(cfg)
        ref = reference_labels(al, rand)
        _WORLD.clear()
        _WORLD[key] = ({"al": al, "rand": rand}, ref, shared_folds(ref, cfg["n_folds"], FOLD_SEED))
    return _WORLD[key]


def run(spec: dict, cfg: dict) -> dict:
    """Execute one (canonical) loop-cell spec; returns the ``simulate`` result.

    ``recipe["warm_start"]: true`` (E6) is popped from a copy of the recipe and passed to
    ``simulate(warm_start=True)``; ``fit`` never sees it.  Specs without the key keep their IDs."""
    pools, ref, folds = _world(cfg)
    if not 0 <= spec["fold"] < len(folds):
        raise ValueError(f"fold {spec['fold']} outside 0..{len(folds) - 1}")
    recipe = dict(spec["recipe"])
    warm_start = bool(recipe.pop("warm_start", False))  # R19: a loop option, not a fit option
    return simulate(pools[spec["pool"]], folds[spec["fold"]], ref, strategy=spec["strategy"],
                    budgets=spec["budgets"], seed=spec["seed"], seed_size=spec["seed_size"],
                    batch=spec["batch"], eta=spec["eta"], arch=spec["arch"],
                    recipe=recipe, cfg=cfg, warm_start=warm_start)
