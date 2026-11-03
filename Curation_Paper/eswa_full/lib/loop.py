"""Retrospective curation-loop simulator (studies E1-E3, E5, E7).

One call simulates one (pool, test fold, strategy, seed, eta) cell: a seed set, then
rounds of ``batch`` queries until each budget is reached; a model is fitted at every
budget and scored on the test fold against the REFERENCE labels ``ref`` (never against the
pool's own, possibly noisy, labels).  Labels: GOOD = 1, BAD = 0.

Determinism (R8): every fit / query / noise / cleaning seed is a stable hash of
(seed, round, purpose); there is no shared advancing RNG, so ``uncertainty`` and
``unc_oracle`` at eta 0 make identical choices.  ``round`` is the number of labelled rows
when the step happens, so the budget model and the query model of the same state coincide.

Label noise (R37, supersedes R7): with eta > 0, ``inject_label_noise`` flips
floor(eta * n) labels of the TRAINING POOL rows only; the early-stopping split keeps its
stored labels.  Judging early stopping on labels that no strategy may correct stopped a model
trained on restored labels within a few epochs and so penalised re-inspection by construction.
The stored (pre-flip) label of each row is kept for the oracle and for the counters.  The
oracle restores flagged training labels.

Counters (by signal id, per budget record):
  n_flipped   labelled rows whose label was flipped at injection (fixed for a labelled set)
  n_flagged   rows ever flagged by the 3-fold out-of-fold check (cumulative)
  n_recovered LIVE state: flagged rows flipped at injection that carry their stored label NOW
              (can fall under unc_selftrain if a recovered row is flagged and relabelled again)
  n_ever_recovered cumulative: rows that were ever restored to their stored label
  n_false_flags flagged rows that were not flipped at injection (cumulative)
  n_residual_flips labelled rows whose CURRENT label differs from the stored one
  n_oof_folds_skipped OOF folds skipped for lacking a class in their training part (cumulative)
warm_start (R19): each round's fit (query and budget fits, not the out-of-fold fits) starts
from the previous round's weights instead of a fresh init; torch architectures only.

Realised noise (top level): n_flipped_total (training rows flipped), n_flipped_es (always 0
since R37; kept for schema stability).
"""
from __future__ import annotations

import random
import zlib

import numpy as np

from Curation_Paper.lib.strategies import inject_label_noise, random_query, uncertainty_query

from .classic import CLASSIC
from .fit import fit, predict
from .metrics import metrics
from .signals import signals_for
from .splits import fold_split

STRATEGIES = ("random", "uncertainty", "unc_oracle", "unc_selftrain")
FLAG_MARGIN = 0.3
AMBIGUOUS_BAND = 0.1
OOF_FOLDS = 3


class BudgetExceedsPool(ValueError):
    """A budget is below the seed size or larger than the training pool."""


class LeakageError(RuntimeError):
    """A test-fold signal appeared in the training or early-stopping data."""


def _sub(seed: int, rnd: int, purpose: str) -> int:
    return zlib.crc32(f"{seed}|{rnd}|{purpose}".encode())


def _guard(test: set[str], *row_lists: list[dict]) -> None:
    for rows in row_lists:
        leaked = sorted(r["signal_id"] for r in rows if r["signal_id"] in test)
        if leaked:
            raise LeakageError(f"{len(leaked)} test signal(s) in training data, first: {leaked[0]}")


def _seed_set(rows: list[dict], n: int, seed: int) -> list[str]:
    """n rows stratified by current label, both classes present (seeded)."""
    bad = sorted(r["signal_id"] for r in rows if r["label_int"] == 0)
    good = sorted(r["signal_id"] for r in rows if r["label_int"] == 1)
    if not bad or not good:
        raise ValueError("training pool lacks one class; cannot draw a two-class seed set")
    if n < 2 or n > len(rows):
        raise ValueError(f"seed_size {n} must be in [2, {len(rows)}]")
    n_bad = min(max(1, round(n * len(bad) / len(rows))), len(bad))
    n_good = n - n_bad
    if n_good > len(good):
        n_good = len(good)
        n_bad = n - n_good
    if n_good < 1:
        n_good, n_bad = 1, n - 1
    rng = random.Random(_sub(seed, 0, "seed_set"))
    return sorted(rng.sample(bad, n_bad) + rng.sample(good, n_good))


def _oof_folds(ids: list[str], rows: dict[str, dict], seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(OOF_FOLDS)]
    off = 0
    for cls in (0, 1):
        members = sorted(s for s in ids if rows[s]["label_int"] == cls)
        rng.shuffle(members)
        for i, s in enumerate(members):
            folds[(i + off) % OOF_FOLDS].append(s)
        off += len(members)
    return folds


def simulate(pool: dict, test_ids: list[str], ref: dict[str, int], *, strategy: str,
             budgets: list[int], seed: int, seed_size: int, batch: int, eta: float,
             arch: str, recipe: dict, cfg: dict, warm_start: bool = False) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if warm_start and arch in CLASSIC:
        raise ValueError(f"warm_start needs a torch architecture, not classic {arch!r}")
    test = set(test_ids)
    missing = sorted(test - set(ref))
    if missing:
        raise ValueError(f"{len(missing)} test ids have no reference label, first: {missing[0]}")
    if batch < 1:
        raise ValueError("batch must be >= 1")

    train_pool, es_set = fold_split(pool, test_ids, seed, cfg["es_frac"])
    _guard(test, train_pool, es_set)

    budgets_requested = [int(b) for b in budgets]
    if not budgets_requested:
        raise ValueError("budgets must not be empty")
    if len(set(budgets_requested)) != len(budgets_requested):
        raise ValueError(f"duplicate budgets: {budgets_requested}")
    budgets = sorted(budgets_requested)
    for b in budgets:
        if b < seed_size or b > len(train_pool):
            raise BudgetExceedsPool(
                f"budget {b} outside [seed_size {seed_size}, training pool {len(train_pool)}]")

    stored = {r["signal_id"]: r["label_int"] for r in train_pool + es_set}
    if eta > 0:
        noisy = {r["signal_id"]: r for r in
                 inject_label_noise(train_pool, eta, _sub(seed, 0, "noise"))}
        train_pool = [noisy[r["signal_id"]] for r in train_pool]
    rows = {r["signal_id"]: r for r in train_pool}
    flipped_at_injection = {s for s in rows if rows[s]["label_int"] != stored[s]}
    n_flipped_es = sum(1 for r in es_set if r["label_int"] != stored[r["signal_id"]])

    # One read of every signal this call can need; X_of slices the cache (values identical
    # to re-reading, since signals_for is a pure function of the id).
    all_ids = sorted({r["signal_id"] for r in train_pool} | {r["signal_id"] for r in es_set}
                     | test)
    cache = dict(zip(all_ids, signals_for(all_ids, cfg)))

    def X_of(ids):
        return np.stack([cache[i] for i in ids])

    y_test = np.array([ref[s] for s in test_ids], dtype=int)
    X_test = X_of(list(test_ids))
    seconds = [0.0]

    prev_state: list = [None]  # warm_start: weights of the previous round's fit

    def fit_on(ids: list[str], rnd: int, purpose: str):
        tr = [rows[s] for s in ids]
        _guard(test, tr, es_set)
        model, info = fit(arch, tr, es_set, X_of, seed=_sub(seed, rnd, purpose), recipe=recipe,
                          init_state=prev_state[0] if warm_start else None)
        seconds[0] += info["train_seconds"]
        if warm_start:
            prev_state[0] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return model

    labelled = _seed_set(train_pool, seed_size, seed)
    lab_set = set(labelled)
    flagged: set[str] = set()
    ever_recovered: set[str] = set()
    skipped = [0]
    out: list[dict] = []

    def clean(rnd: int) -> None:
        """3-fold OOF check on the labelled set; flag, then (oracle|selftrain) relabel."""
        folds = _oof_folds(sorted(lab_set), rows, _sub(seed, rnd, "oof_folds"))
        ids_all = sorted(lab_set)
        snapshot = {s: rows[s]["label_int"] for s in ids_all}
        for k, held in enumerate(folds):
            tr_ids = [s for s in ids_all if s not in set(held)]
            if not held or len({snapshot[s] for s in tr_ids}) < 2:
                skipped[0] += 1
                continue
            tr = [dict(rows[s], label_int=snapshot[s]) for s in tr_ids]
            _guard(test, tr, es_set)
            model, info = fit(arch, tr, es_set, X_of, seed=_sub(seed, rnd, f"oof{k}"),
                              recipe=recipe)
            seconds[0] += info["train_seconds"]
            p = predict(model, X_of(held))
            for s, pi in zip(held, p):
                pred = 1 if pi >= 0.5 else 0
                if pred != snapshot[s] and abs(pi - 0.5) > FLAG_MARGIN:
                    flagged.add(s)
                    if strategy == "unc_oracle":
                        rows[s]["label_int"] = stored[s]
                    else:  # unc_selftrain
                        rows[s]["label_int"] = pred
                    if s in flipped_at_injection and rows[s]["label_int"] == stored[s]:
                        ever_recovered.add(s)

    max_budget = budgets[-1]
    while True:
        n = len(labelled)
        model = None
        if strategy != "random" or n in budgets:
            model = fit_on(labelled, n, "fit")
        unl = sorted(s for s in rows if s not in lab_set)
        p_unl = predict(model, X_of(unl)) if (model is not None and unl) else np.array([])

        if n in budgets:
            p_test = predict(model, X_test)
            f_amb = (float(np.mean(np.abs(p_unl - 0.5) < AMBIGUOUS_BAND))
                     if strategy != "random" and len(unl) else float("nan"))
            lab_sorted = sorted(labelled)
            flipped = lab_set & flipped_at_injection
            out.append({
                "budget": n,
                "metrics": metrics(y_test, p_test),
                "test_pred": {s: round(float(v), 6) for s, v in zip(test_ids, p_test)},
                "train_ids": lab_sorted,
                "train_labels": [int(rows[s]["label_int"]) for s in lab_sorted],
                "n_flipped": len(flipped),
                "n_flagged": len(flagged),
                "n_recovered": sum(1 for s in flagged & flipped
                                   if rows[s]["label_int"] == stored[s]),
                "n_ever_recovered": len(ever_recovered),
                "n_false_flags": len(flagged - flipped_at_injection),
                "n_oof_folds_skipped": skipped[0],
                "n_residual_flips": sum(1 for s in lab_sorted
                                        if rows[s]["label_int"] != stored[s]),
                "f_amb": f_amb,
                "train_seconds": seconds[0],
                "selection_claim_ok": bool(n <= cfg["selection_claim_max_frac"] * len(train_pool)),
            })
        if n >= max_budget:
            break

        if not unl:  # unreachable once budgets are validated; never loop forever
            raise RuntimeError(f"no unlabelled rows left at {n} labelled, budgets {budgets}")
        nxt = min(b for b in budgets if b > n)
        take = min(batch, nxt - n)
        if strategy == "random":
            chosen = random_query([rows[s] for s in unl], take, _sub(seed, n, "query"))
        else:
            chosen = uncertainty_query([rows[s] for s in unl],
                                       dict(zip(unl, p_unl.tolist())), take)
        for r in chosen:
            labelled.append(r["signal_id"])
            lab_set.add(r["signal_id"])
        if strategy in ("unc_oracle", "unc_selftrain"):
            clean(len(labelled))

    result = {
        "strategy": strategy, "seed": int(seed), "eta": float(eta), "arch": arch,
        "seed_size": int(seed_size), "batch": int(batch),
        "len_train_pool": len(train_pool), "len_es": len(es_set),
        "test_ids": list(test_ids), "budgets": out,
        "budgets_requested": budgets_requested, "recipe": recipe,
        "es_frac": cfg["es_frac"], "flag_margin": FLAG_MARGIN,
        "ambiguous_band": AMBIGUOUS_BAND, "oof_folds": OOF_FOLDS,
        "n_flipped_total": len(flipped_at_injection) + n_flipped_es,
        "n_flipped_es": n_flipped_es,
    }
    if warm_start:  # key only present when used, so default results are unchanged
        result["warm_start"] = True
    return result
