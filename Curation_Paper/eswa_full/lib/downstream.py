"""E11 downstream value of curation: pure functions (rulings R24, R25).

Curated set, damage index (DI) per path and cycle, per-panel monotonicity and trendability.
Everything here is CPU numpy on in-memory arrays; no model is trained and no file is read.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from .signals import parse_id

MIN_CYCLE = 200          # baseline = first cycle >= 200; cycles below 200 are not part of a DI
P_ACCEPT = 0.5
MIN_POINTS = 3           # a correlation over fewer than 3 points is undefined here


# ---------------------------------------------------------------- curated set

def reference_label(sid: str, al: dict, rand: dict) -> int | None:
    """Label_int used for curation: the L_AL label if L_AL holds the signal, else L_rand's."""
    if sid in al:
        return al[sid]["label_int"]
    if sid in rand:
        return rand[sid]["label_int"]
    return None


def curated_mask(ids, mean_p, al: dict, rand: dict, thr: float = P_ACCEPT) -> np.ndarray:
    """Boolean keep-mask over ``ids`` (R25a).

    Labelled BAD (reference label) -> removed; labelled GOOD -> always kept;
    unlabelled -> kept iff mean p >= thr."""
    mean_p = np.asarray(mean_p, dtype=float)
    if len(mean_p) != len(ids):
        raise ValueError("ids and mean_p differ in length")
    keep = np.zeros(len(ids), dtype=bool)
    for i, sid in enumerate(ids):
        lab = reference_label(sid, al, rand)
        keep[i] = bool(lab == 1) if lab is not None else bool(mean_p[i] >= thr)
    return keep


def labels_only_mask(ids, al: dict, rand: dict) -> np.ndarray:
    """Keep-mask that removes ONLY signals whose reference label is BAD (R27); no model removal.
    Unlabelled and GOOD signals are kept."""
    return np.array([reference_label(s, al, rand) != 0 for s in ids], dtype=bool)


def restrict_to_common(di: dict, ref_di: dict) -> dict:
    """Drop from ``di`` every path-cycle that is absent in ``ref_di`` (and every path absent from
    it), so ``di`` is scored on exactly the reference condition's path-cycles (R27)."""
    return {p: {c: v for c, v in t.items() if c in ref_di[p]}
            for p, t in di.items() if p in ref_di}


def count_path_cycles(di: dict) -> int:
    return sum(len(t) for t in di.values())


# ---------------------------------------------------------------- correlations

def spearman(x, y) -> float:
    """Spearman rho; nan when fewer than MIN_POINTS points or either side is constant."""
    # the undefined cases (constant side, too few points) are guarded once, in pearson, on the
    # ranks: a constant series has constant ranks
    return pearson(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < MIN_POINTS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------- damage index

def damage_indices(ids, X: np.ndarray, keep: np.ndarray, min_cycle: int = MIN_CYCLE):
    """DI per path (panel, actuator, receiver) and cycle (R25b).

    ``X`` rows are the raw protocol window (no z-scoring); ``keep`` selects the repetitions that
    exist under the condition.  Baseline cycle = the path's first cycle >= ``min_cycle`` among
    ALL its signals in ``ids`` (condition-independent); b = mean of the kept repetitions there.

    Returns (di, counts); di[path][cycle] = mean over kept reps of RMS(x - b)."""
    X = np.asarray(X, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    if X.shape[0] != len(ids) or keep.shape[0] != len(ids):
        raise ValueError("ids, X and keep differ in length")
    groups: dict[tuple, dict[int, list[int]]] = {}
    n_below = 0
    for i, sid in enumerate(ids):
        d = parse_id(sid)
        if d["cycle"] < min_cycle:
            n_below += 1
            continue
        groups.setdefault((d["panel"], d["actuator"], d["receiver"]), {}) \
            .setdefault(d["cycle"], []).append(i)
    di: dict[tuple, dict[int, float]] = {}
    counts = {"n_signals_below_min_cycle": n_below, "n_paths": len(groups),
              "n_paths_dropped_no_kept_baseline": 0, "n_path_cycles_total": 0,
              "n_path_cycles_missing": 0, "n_path_cycles_present": 0,
              "n_path_cycles_in_dropped_paths": 0}
    for path, by_cycle in groups.items():
        base_rows = [i for i in by_cycle[min(by_cycle)] if keep[i]]
        if not base_rows:
            counts["n_paths_dropped_no_kept_baseline"] += 1
            counts["n_path_cycles_in_dropped_paths"] += len(by_cycle)
            continue
        b = X[base_rows].mean(axis=0)
        traj: dict[int, float] = {}
        for cyc in sorted(by_cycle):
            counts["n_path_cycles_total"] += 1
            rows = [i for i in by_cycle[cyc] if keep[i]]
            if not rows:
                counts["n_path_cycles_missing"] += 1
                continue
            counts["n_path_cycles_present"] += 1
            traj[cyc] = float(np.sqrt(((X[rows] - b) ** 2).mean(axis=1)).mean())
        di[path] = traj
    return di, counts


# ---------------------------------------------------------------- panel metrics

def _nanmean(v):
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def path_monotonicity(di: dict[int, float]) -> float:
    """|Spearman(DI, cycle)| of one path trajectory (nan if undefined)."""
    cyc = sorted(di)
    r = spearman(cyc, [di[c] for c in cyc])
    return abs(r)


def panel_di(paths: dict[tuple, dict[int, float]]) -> dict[int, float]:
    """Median over paths of DI at each cycle (over the paths that have that cycle)."""
    by_cycle: dict[int, list[float]] = {}
    for traj in paths.values():
        for c, v in traj.items():
            by_cycle.setdefault(c, []).append(v)
    return {c: float(np.median(v)) for c, v in sorted(by_cycle.items())}


def trendability(paths: dict[tuple, dict[int, float]]):
    """Coble-Hines form: pairwise Pearson of the paths' DI trajectories over common cycles.

    Returns (min, mean, n_pairs_defined, n_pairs_total); nan pairs are skipped and counted."""
    keys = sorted(paths)
    rs, total = [], 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            total += 1
            ta, tb = paths[keys[a]], paths[keys[b]]
            common = sorted(set(ta) & set(tb))
            r = pearson([ta[c] for c in common], [tb[c] for c in common])
            if not math.isnan(r):
                rs.append(r)
    if not rs:
        return float("nan"), float("nan"), 0, total
    return float(min(rs)), float(np.mean(rs)), len(rs), total


def panel_metrics(paths: dict[tuple, dict[int, float]]) -> dict:
    """Metrics for ONE panel's paths (R25c)."""
    mono = {p: path_monotonicity(t) for p, t in paths.items()}
    pdi = panel_di(paths)
    cyc = sorted(pdi)
    tmin, tmean, n_def, n_tot = trendability(paths)
    return {"n_paths": len(paths),
            "n_paths_mono_nan": sum(1 for v in mono.values() if math.isnan(v)),
            "monotonicity": _nanmean(list(mono.values())),
            "panel_di_abs_spearman": abs(spearman(cyc, [pdi[c] for c in cyc])),
            "trend_min": tmin, "trend_mean": tmean,
            "n_trend_pairs_defined": n_def, "n_trend_pairs_total": n_tot,
            "n_cycles": len(cyc)}, mono


def split_by_panel(di: dict[tuple, dict[int, float]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path, traj in di.items():
        out.setdefault(path[0], {})[path] = traj
    return out


def bootstrap_mean_ci(diffs, seed: int = 0, n_boot: int = 2000):
    """Percentile bootstrap CI of the mean of path-level differences (nan-free input)."""
    d = np.asarray([x for x in diffs if not math.isnan(x)], dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(d)


def compare_conditions(di_cur, di_unc) -> dict:
    """Per-panel metrics of both conditions plus the paired difference (curated - uncurated)."""
    cur_p, unc_p = split_by_panel(di_cur), split_by_panel(di_unc)
    out = {}
    for panel in sorted(set(cur_p) | set(unc_p)):
        mc, mono_c = panel_metrics(cur_p.get(panel, {}))
        mu, mono_u = panel_metrics(unc_p.get(panel, {}))
        diff = {k: mc[k] - mu[k] for k in
                ("monotonicity", "panel_di_abs_spearman", "trend_min", "trend_mean")}
        common = sorted(set(mono_c) & set(mono_u))
        mean_d, lo, hi, n = bootstrap_mean_ci([mono_c[p] - mono_u[p] for p in common])
        out[panel] = {"curated": mc, "uncurated": mu, "difference": diff,
                      "path_paired_monotonicity": {"n_paths_paired": n, "n_paths_common": len(common),
                                                   "mean_diff": mean_d, "ci95": [lo, hi],
                                                   "ci_method": "percentile bootstrap over paths, "
                                                                "2000 resamples, seed 0"}}
    return out


# ---------------------------------------------------------------- E14 helper

def revised_signals(al: dict, rand: dict) -> list[dict]:
    """Shared signals whose L_AL and L_rand labels differ, with both timestamps and the
    direction in label terms (L_rand -> L_AL) and in time order (earlier -> later)."""
    rows = []
    for sid in sorted(set(al) & set(rand)):
        a, r = al[sid], rand[sid]
        if a["label_int"] == r["label_int"]:
            continue
        earlier_is_rand = r["labeled_at"] <= a["labeled_at"]
        first, second = (r, a) if earlier_is_rand else (a, r)
        rows.append({"signal_id": sid, "L_rand_label": r["label"], "L_AL_label": a["label"],
                     "L_AL_labeled_at": a["labeled_at"], "L_rand_labeled_at": r["labeled_at"],
                     "direction_rand_to_AL": f"{r['label']}->{a['label']}",
                     "earlier_pool": "L_rand" if earlier_is_rand else "L_AL",
                     "direction_in_time": f"{first['label']}->{second['label']}"})
    return rows


def labels_per_day(pool: dict[str, dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in pool.values():
        day = str(r["labeled_at"])[:10]
        out[day] = out.get(day, 0) + 1
    return dict(sorted(out.items()))
