"""Training engine: one ``fit`` / ``predict`` pair for the torch architectures and the
classical models.

Default torch recipe = the short paper's Table 1 recipe (Curation_Paper/lib/train.py
``train_with_fixed_val`` + app/trainer.py ``augment``): AdamW lr 3e-4 / wd 1e-3, batch 64,
max 300 epochs, patience 60 on the early-stopping BCE with best-checkpoint restore, grad-norm
clip 1.0, BCE with weight sqrt(n_neg/n_pos) on the GOOD (label 1) class, 4x augmentation
(original, roll +10, roll -10, Gaussian noise 0.02*std on the RAW signal), then a per-signal
z-score.

Window handling (E6): ``fit`` never reads the HDF5 itself; it calls ``X_of(ids)`` and builds
the model with ``input_len = X.shape[1]``.  For a window other than the config's, the CALLER
passes an ``X_of`` that returns that window (e.g. ``partial(signals_for, cfg=cfg, window=w)``)
and may also put ``window`` in the recipe so the choice is recorded; ``fit`` does not slice.

Determinism: ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` is set before CUDA initialises,
``torch.use_deterministic_algorithms(True)`` (strict), ``cudnn.benchmark = False``, torch /
numpy / random seeded, augmentation noise from a seeded ``numpy.random.Generator`` and the
epoch shuffle from a seeded ``torch.Generator``.  No global ``np.random`` state is used.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .classic import CLASSIC, ClassicModel, fit_classic, zscore_rows
from .models import ARCHS, build_model

DEFAULT_RECIPE = {
    "lr": 3e-4, "weight_decay": 1e-3, "batch_size": 64, "max_epochs": 300, "patience": 60,
    "grad_clip": 1.0,
    "activation": "leaky",       # microconv1d only: leaky | relu
    "class_weight": "sqrt",      # none | sqrt | full
    "augment": True, "zscore": True,
    "window": None,              # informational; the caller's X_of decides the window
    "model_cfg": {},             # extra build_model kwargs, e.g. {"width": 1.5}
    "minirocket_kernels": 10000,
}
_ES_BATCH = 256


def _recipe(recipe: dict) -> dict:
    bad = set(recipe) - set(DEFAULT_RECIPE)
    if bad:
        raise ValueError(f"unknown recipe keys {sorted(bad)}")
    r = {**DEFAULT_RECIPE, **recipe}
    if r["class_weight"] not in ("none", "sqrt", "full"):
        raise ValueError(f"class_weight must be none|sqrt|full, got {r['class_weight']!r}")
    return r


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _augment(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """4x: original, roll +10, roll -10, +noise (sigma = 0.02*(std+1e-8) of the raw signal),
    in the same order as app/trainer.py ``augment``; noise from ``rng``."""
    xs, ys = [], []
    for sig, lab in zip(X.astype(np.float32), y):
        noise = rng.normal(0, 0.02 * (sig.std() + 1e-8), sig.shape).astype(np.float32)
        xs += [sig.copy(), np.roll(sig, 10), np.roll(sig, -10), sig + noise]
        ys += [lab] * 4
    return np.stack(xs), np.array(ys, dtype=np.float32)


def _labels(rows: list[dict]) -> np.ndarray:
    return np.array([r["label_int"] for r in rows], dtype=int)


def _ids(rows: list[dict]) -> list[str]:
    return [r["signal_id"] for r in rows]


def fit(name: str, train: list[dict], es: list[dict],
        X_of: Callable[[list[str]], np.ndarray], *, seed: int, recipe: dict,
        init_state: dict | None = None) -> tuple[object, dict]:
    """Train ``name`` on ``train``; early stopping / model selection use ``es`` only.

    ``X_of(ids)`` -> (n, window) array (see module docstring for the window).  Returns
    (model, info) with info = {es_loss, epochs, train_seconds, device}."""
    if name not in ARCHS and name not in CLASSIC:
        raise ValueError(f"unknown model {name!r}; choose from {ARCHS + CLASSIC}")
    r = _recipe(recipe)
    if len(train) < 2 or len(es) < 2:
        raise ValueError("fit needs >= 2 rows in each of train and es")
    t0 = time.perf_counter()
    Xtr = np.array(X_of(_ids(train)), dtype=np.float32, copy=True)
    Xes = np.array(X_of(_ids(es)), dtype=np.float32, copy=True)
    ytr, yes = _labels(train), _labels(es)

    if name in CLASSIC:
        model, info = fit_classic(name, Xtr, ytr, Xes, yes, seed=seed, recipe=r)
        info["train_seconds"] = time.perf_counter() - t0
        return model, info

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)

    if r["augment"]:
        Xtr, ytr_f = _augment(Xtr, ytr, rng)
    else:
        ytr_f = ytr.astype(np.float32)
    if r["zscore"]:
        Xtr, Xes = zscore_rows(Xtr), zscore_rows(Xes)
    Xt = torch.from_numpy(Xtr).unsqueeze(1).to(device)
    yt = torch.from_numpy(ytr_f).to(device)
    Xe = torch.from_numpy(Xes).unsqueeze(1).to(device)
    ye = torch.from_numpy(yes.astype(np.float32)).to(device)

    cfg = dict(r["model_cfg"])
    if name == "microconv1d":
        cfg["activation"] = r["activation"]
    model = build_model(name, Xt.shape[-1], **cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)

    # Same weighting as train.py: pos_weight = sqrt(max(n_neg,1)/max(n_pos,1)) on label==1.
    n_pos = int((ytr == 1).sum())
    n_neg = len(ytr) - n_pos
    ratio = max(n_neg, 1) / max(n_pos, 1)
    pos_weight = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": float(ratio)}[r["class_weight"]]

    crit = nn.BCELoss(reduction="none")
    es_crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
    n, bs = Xt.shape[0], int(r["batch_size"])
    n_batches = n // bs if n > bs else 1          # drop_last only when more than one batch
    pw = torch.tensor(pos_weight, device=device)
    one = torch.tensor(1.0, device=device)

    best, best_state, stale, epochs_run = float("inf"), None, 0, 0
    for ep in range(int(r["max_epochs"])):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            xb, yb = Xt[idx], yt[idx]
            w = torch.where(yb > 0.5, pw, one)
            loss = (crit(model(xb), yb) * w).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), r["grad_clip"])
            opt.step()
        epochs_run = ep + 1

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, Xe.shape[0], _ES_BATCH):
                p = model(Xe[i:i + _ES_BATCH])
                tot += es_crit(p, ye[i:i + _ES_BATCH]).item() * p.shape[0]
        v = tot / Xe.shape[0]
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= r["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.eswa_zscore = bool(r["zscore"])
    return model, {"es_loss": best, "epochs": epochs_run,
                   "train_seconds": time.perf_counter() - t0, "device": str(device)}


def predict(model, X: np.ndarray) -> np.ndarray:
    """p(GOOD) for each row of the raw signals ``X`` (never modified)."""
    if isinstance(model, ClassicModel):
        return np.asarray(model.predict_proba_good(X), dtype=np.float64)
    X = np.array(X, dtype=np.float32, copy=True)
    if getattr(model, "eswa_zscore", True):
        X = zscore_rows(X)
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], _ES_BATCH):
            xb = torch.from_numpy(X[i:i + _ES_BATCH]).unsqueeze(1).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.array([])
