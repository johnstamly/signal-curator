"""E8 (p_accept threshold study) and E9 (explicit-rule baselines), Task 12.

E8 reads stored loop cells (the curated-loop runs: pool al, strategy unc_oracle, eta 0 -- the E3
cells that E1 also reads, R12), pools the test-fold predictions of the five folds per seed at one
budget (default 600), and scores a grid of acceptance thresholds.  A signal is *kept* (accepted
as GOOD) iff p(GOOD) >= p_accept, the convention of ``metrics.metrics``.

E9 re-implements the short paper's three classical statistics and its three filter settings
(Table 3), and adds a depth-<=3 decision tree on the same statistics.  No cell is involved: the
statistics are deterministic functions of the raw first-500-sample signals.

Statistic definitions are those of ``Curation_Paper/scripts/classical_filter_sweep.py`` (the
script that owns the tight / mid / loose presets of Table 3):
  E_pp   = np.ptp(sig)                                  (reject when  E_pp  < thr)
  |s|    = |scipy.stats.skew(sig)|                      (reject when  |s|   > thr)
  f5     = sum of the 5 largest bins of |rfft(sig)|^2 / sum of all bins   (reject when f5 < thr)
``16_classical_filters.py`` removes the mean before the FFT; on the 1,011-signal pool both
variants give the same flags at all three settings (checked in the E9 report), so this is a
no-op for the recorded numbers.  A constant signal has an undefined skewness (nan); it is set to
0 here, which flags nothing on that rule exactly as ``nan > thr`` does in the sweep script
(a constant signal is caught by E_pp anyway).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew
from sklearn.tree import DecisionTreeClassifier, export_text

from .aggregate import MissingCells, _nanstat
from .cells import cell_id, load_cell
from .jobs import cell_specs
from .splits import fold_split

FEATURES = ("E_pp", "abs_skew", "f5")
# short paper Table 3: (E_pp <, |s| >, f5 <)
FILTER_SETTINGS = {
    "tight": (1e-3, 3.0, 0.10),
    "medium": (5e-3, 2.5, 0.15),
    "loose": (1e-2, 2.0, 0.20),
}
P_ACCEPT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
N_BINS = 10


# ------------------------------------------------------------------ E9: statistics and rules

def stat_ptp(sig) -> float:
    return float(np.ptp(sig))


def stat_abs_skew(sig) -> float:
    if np.ptp(sig) == 0:          # constant signal: skewness undefined (see module docstring)
        return 0.0
    v = float(abs(skew(sig)))
    return 0.0 if np.isnan(v) else v


def stat_top5(sig) -> float:
    p = np.abs(np.fft.rfft(sig)) ** 2
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    return float(np.sort(p)[-5:].sum())


def statistics(X: np.ndarray) -> np.ndarray:
    """(n, 3) float64 array of (E_pp, |skew|, f5), one row per signal."""
    return np.array([[stat_ptp(s), stat_abs_skew(s), stat_top5(s)] for s in X], dtype=float)


def flag_any(S: np.ndarray, setting: tuple[float, float, float]) -> np.ndarray:
    a, b, c = setting
    return (S[:, 0] < a) | (S[:, 1] > b) | (S[:, 2] < c)


def score_flags(y: np.ndarray, flagged: np.ndarray) -> dict:
    """y: 0 BAD / 1 GOOD; flagged: True = rejected as BAD."""
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(flagged, dtype=bool)
    bad, good = y == 0, y == 1
    nb, ng = int(bad.sum()), int(good.sum())
    nbf, ngf = int((flagged & bad).sum()), int((flagged & good).sum())
    return {"n": int(len(y)), "n_bad": nb, "n_good": ng,
            "n_bad_flagged": nbf, "n_good_flagged": ngf,
            "bad_recall": nbf / nb if nb else float("nan"),
            "good_kept": 1.0 - ngf / ng if ng else float("nan")}


def tree_nodes(clf: DecisionTreeClassifier) -> list[dict]:
    """Flat, full-precision dump of a fitted tree (node order = sklearn's)."""
    t = clf.tree_
    out = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        cls = int(np.argmax(t.value[i][0]))
        out.append({"node": i, "leaf": bool(leaf),
                    "feature": None if leaf else FEATURES[int(t.feature[i])],
                    "threshold": None if leaf else float(t.threshold[i]),
                    "left": None if leaf else int(t.children_left[i]),
                    "right": None if leaf else int(t.children_right[i]),
                    "n_samples": int(t.n_node_samples[i]),
                    "class": ("BAD", "GOOD")[clf.classes_[cls]]})
    return out


def fit_tree(S: np.ndarray, y: np.ndarray, max_depth: int = 3) -> DecisionTreeClassifier:
    if max_depth > 3:
        raise ValueError("E9 trees are limited to depth <= 3")
    return DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                  random_state=0).fit(S, y)


def tree_text(clf: DecisionTreeClassifier) -> str:
    return export_text(clf, feature_names=list(FEATURES), decimals=6)


def tree_cv(stat: dict[str, np.ndarray], pool: dict[str, dict], ref: dict[str, int],
            folds: list[list[str]], *, es_frac: float, split_seed: int = 0,
            max_depth: int = 3) -> dict:
    """Per fold: fit the tree on the fold's training pool (``fold_split`` minus the test IDs;
    the early-stopping rows are not used -- a tree has no use for them), predict the fold's
    test signals; pool the predictions over folds.  Deterministic, so no seeds."""
    pooled_ids, pooled_flag, per_fold = [], [], []
    for k, test in enumerate(folds):
        train, _es = fold_split(pool, test, split_seed, es_frac)
        if set(r["signal_id"] for r in train) & set(test):
            raise AssertionError("test signal in the tree's training pool")
        Str = np.stack([stat[r["signal_id"]] for r in train])
        ytr = np.array([r["label_int"] for r in train])
        clf = fit_tree(Str, ytr, max_depth)
        Ste = np.stack([stat[s] for s in test])
        pred = clf.predict(Ste)                       # 0 = BAD = rejected
        yte = np.array([ref[s] for s in test])
        pooled_ids += list(test)
        pooled_flag += list(pred == 0)
        per_fold.append({"fold": k, "n_train": len(train), "n_test": len(test),
                         **score_flags(yte, pred == 0),
                         "rules": tree_text(clf), "nodes": tree_nodes(clf)})
    y = np.array([ref[s] for s in pooled_ids])
    if len(set(pooled_ids)) != len(pooled_ids) or set(pooled_ids) != set(ref):
        raise ValueError("pooled tree predictions do not cover the shared set exactly once")
    return {"pooled": score_flags(y, np.array(pooled_flag)), "folds": per_fold}


def build_e9(stat: dict[str, np.ndarray], pool: dict[str, dict], ref: dict[str, int],
             folds: list[list[str]], *, es_frac: float) -> dict:
    """stat: signal_id -> (E_pp, |s|, f5) for every L_AL signal."""
    shared = sorted(ref)
    S = np.stack([stat[s] for s in shared])
    y = np.array([ref[s] for s in shared])
    filters = {}
    for name, thr in FILTER_SETTINGS.items():
        filters[name] = {"thresholds": {"E_pp_below": thr[0], "abs_skew_above": thr[1],
                                        "f5_below": thr[2]},
                         **score_flags(y, flag_any(S, thr))}
    # sanity: the same filters on the whole 1,011-signal L_AL pool (the short paper's Table 3
    # population), reported as a cross-check only, never as the E9 headline.
    al_ids = sorted(pool)
    Sa = np.stack([stat[s] for s in al_ids])
    ya = np.array([pool[s]["label_int"] for s in al_ids])
    sanity = {name: score_flags(ya, flag_any(Sa, thr)) for name, thr in FILTER_SETTINGS.items()}
    cv = tree_cv(stat, pool, ref, folds, es_frac=es_frac)
    full = fit_tree(Sa, ya)
    return {
        "n_shared": len(shared), "n_shared_bad": int((y == 0).sum()),
        "n_shared_good": int((y == 1).sum()),
        "features": list(FEATURES),
        "filters": filters,
        "sanity_al_pool": {"n": len(al_ids), "n_bad": int((ya == 0).sum()),
                           "settings": sanity},
        "tree": {"max_depth": 3, "class_weight": "balanced", "random_state": 0,
                 "split_seed": 0, "deterministic_no_seed_axis": True, **cv,
                 "descriptive_full_pool": {
                     "note": "fitted on all 1,011 L_AL signals, test signals included; for "
                             "the paper's rule listing only, never scored",
                     "rules": tree_text(full), "nodes": tree_nodes(full)}},
        "microconv1d": None,
        "microconv1d_note": "filled by the numbers generator from the E3 unc_oracle eta-0 "
                            "cells (budget 600), not computed here",
    }


# ------------------------------------------------------------------ E8: thresholds

def pool_by_seed(cfg: dict, g: dict, ref: dict[str, int], budget: int
                 ) -> dict[int, tuple[list[str], np.ndarray, np.ndarray]]:
    """seed -> (ids, y, p): the test predictions of that seed's folds at ``budget``, pooled.
    Raises MissingCells / ValueError rather than guess."""
    specs = cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    results: dict[int, list[dict]] = {}
    for s in specs:
        payload = load_cell(s, cfg)
        if payload is None:
            raise MissingCells([cell_id(s)])
        results.setdefault(s["seed"], []).append(payload["result"])
    out = {}
    for seed in sorted(results):
        ids, ps = [], []
        for res in results[seed]:
            rec = next((r for r in res["budgets"] if r["budget"] == budget), None)
            if rec is None:
                raise ValueError(f"seed {seed}: no record at budget {budget}")
            tp = rec["test_pred"]
            if set(tp) != set(res["test_ids"]):
                raise ValueError("test_pred keys differ from test_ids")
            for sid in res["test_ids"]:
                if sid not in ref:
                    raise ValueError(f"{sid!r} has no reference label")
                ids.append(sid)
                ps.append(tp[sid])
        if len(ids) != len(set(ids)) or set(ids) != set(ref):
            raise ValueError(f"seed {seed}: pooled set differs from the reference set "
                             f"({len(ids)} scored, {len(set(ids))} unique, {len(ref)} expected)")
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        out[seed] = ([ids[i] for i in order],
                     np.array([ref[ids[i]] for i in order], dtype=int),
                     np.array([ps[i] for i in order], dtype=float))
    return out


def threshold_row(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Keep iff p >= t.  precision = GOOD share of the kept; contamination = BAD share."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    keep = p >= t
    nk = int(keep.sum())
    bad, good = y == 0, y == 1
    nbk, ngk = int((keep & bad).sum()), int((keep & good).sum())
    nan = float("nan")
    return {"p_accept": float(t), "n_kept": nk, "n_good_kept": ngk, "n_bad_kept": nbk,
            "fraction_kept": nk / len(y) if len(y) else nan,
            "precision": ngk / nk if nk else nan,
            "contamination": nbk / nk if nk else nan,
            "bad_recall": 1.0 - nbk / int(bad.sum()) if bad.any() else nan,
            "good_retention": ngk / int(good.sum()) if good.any() else nan}


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Equal-width bins on [0, 1], last closed on the right (same indexing as metrics._ece)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n,
                    "mean_p": float(p[m].mean()) if n else float("nan"),
                    "obs_good_rate": float(y[m].mean()) if n else float("nan")})
    return out


ROW_KEYS = ("n_kept", "fraction_kept", "precision", "contamination", "bad_recall",
            "good_retention")


def build_e8(pooled: dict[int, tuple], thresholds=P_ACCEPT_GRID, n_bins: int = N_BINS) -> dict:
    seeds = sorted(pooled)
    per_seed_rows = {s: [threshold_row(pooled[s][1], pooled[s][2], t) for t in thresholds]
                     for s in seeds}
    rows = []
    for j, t in enumerate(thresholds):
        row = {"p_accept": float(t), "n_seeds": len(seeds)}
        for k in ROW_KEYS:
            vals = [per_seed_rows[s][j][k] for s in seeds]
            row[k] = {**_nanstat(vals), "values": vals}
        rows.append(row)
    per_seed_bins = {s: reliability_bins(pooled[s][1], pooled[s][2], n_bins) for s in seeds}
    bins = []
    y_all = np.concatenate([pooled[s][1] for s in seeds])
    p_all = np.concatenate([pooled[s][2] for s in seeds])
    pooled_bins = reliability_bins(y_all, p_all, n_bins)
    for b in range(n_bins):
        per = [per_seed_bins[s][b] for s in seeds]
        bins.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins,
                     "n": {**_nanstat([x["n"] for x in per]), "values": [x["n"] for x in per]},
                     "mean_p": {**_nanstat([x["mean_p"] for x in per]),
                                "values": [x["mean_p"] for x in per]},
                     "obs_good_rate": {**_nanstat([x["obs_good_rate"] for x in per]),
                                       "values": [x["obs_good_rate"] for x in per]},
                     "pooled_over_seeds": {"n": pooled_bins[b]["n"],
                                           "mean_p": pooled_bins[b]["mean_p"],
                                           "obs_good_rate": pooled_bins[b]["obs_good_rate"]}})
    y0 = pooled[seeds[0]][1]
    return {"seeds": seeds, "n_seeds": len(seeds), "n_scored": int(len(y0)),
            "n_bad": int((y0 == 0).sum()), "n_good": int((y0 == 1).sum()),
            "keep_rule": "kept (accepted as GOOD) iff p(GOOD) >= p_accept",
            "thresholds": rows, "reliability": bins}


# ------------------------------------------------------------------ E10 f_amb (loop cells)

def famb_by_budget(cfg: dict, g: dict) -> dict:
    """f_amb (share of the unlabelled pool with |p - 0.5| < 0.1 after training on the labelled
    set) per budget, from the unc_oracle eta-0 loop cells.  One value per (seed, fold) cell;
    summarised over cells (mean, s.d. over seeds x folds) and over seeds (fold-mean per seed)."""
    specs = cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    by_cell: dict[tuple, dict[int, float]] = {}
    for s in specs:
        payload = load_cell(s, cfg)
        if payload is None:
            raise MissingCells([cell_id(s)])
        by_cell[(s["seed"], s["fold"])] = {r["budget"]: float(r["f_amb"])
                                           for r in payload["result"]["budgets"]}
    budgets = sorted(next(iter(by_cell.values())))
    seeds = sorted({k[0] for k in by_cell})
    rows = []
    for b in budgets:
        cell_vals = [by_cell[k][b] for k in sorted(by_cell)]
        seed_means = []
        for sd in seeds:
            v = np.array([by_cell[(sd, f)][b] for (s2, f) in sorted(by_cell) if s2 == sd])
            seed_means.append(float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan"))
        rows.append({"budget": b, "n_cells": len(cell_vals),
                     "over_cells": {**_nanstat(cell_vals), "values": cell_vals},
                     "over_seeds": {**_nanstat(seed_means), "values": seed_means}})
    return {"seeds": seeds, "ambiguous_band": 0.1, "rows": rows,
            "definition": "f_amb = mean(|p - 0.5| < 0.1) over the unlabelled pool rows"}
