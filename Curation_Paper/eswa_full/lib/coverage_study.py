"""E10: specimen-level GOOD coverage of the final MicroConv1D (ruling R5), Task 12.

Per seed 0-4: a stratified 15 % early-stopping split of ALL of L_AL (no test fold -- the model
is only used to score the 12,504-signal universe, never to report accuracy), a MicroConv1D fit
with the default recipe, p(GOOD) for every signal of the universe.  Scores are averaged over
seeds; a signal is GOOD iff the mean p >= 0.5 (the short paper used p > 0.5; the two differ only
at exactly 0.5).  The model fit here is the one E11 re-uses (R4).

Seeds feed both the split and the fit directly (``fit_all_lal`` is not a loop cell, so the
R8 (seed, round, purpose) derivation does not apply).
"""
from __future__ import annotations

import numpy as np

from .fit import DEFAULT_RECIPE, fit, predict
from .signals import parse_id
from .splits import fold_split

THRESHOLD = 0.5


def fit_all_lal(pool: dict[str, dict], X_of, *, seed: int, es_frac: float,
                recipe: dict | None = None, arch: str = "microconv1d"):
    """Fit on L_AL minus a stratified ``es_frac`` early-stopping split (no test IDs)."""
    train, es = fold_split(pool, [], seed, es_frac)
    if len(train) + len(es) != len(pool):
        raise AssertionError("train + es must cover the whole pool")
    model, info = fit(arch, train, es, X_of, seed=seed,
                      recipe=dict(DEFAULT_RECIPE if recipe is None else recipe))
    info = {**info, "n_train": len(train), "n_es": len(es),
            "n_es_bad": sum(1 for r in es if r["label_int"] == 0)}
    return model, info


def coverage_table(ids, p_good, thr: float = THRESHOLD) -> dict:
    """Per panel x cycle (n acquired, n GOOD, GOOD fraction) and per-panel summaries."""
    cells: dict[tuple[str, int], list[float]] = {}
    for sid, p in zip(ids, p_good):
        d = parse_id(sid)
        cells.setdefault((d["panel"], d["cycle"]), []).append(float(p))
    panels: dict[str, dict] = {}
    for (panel, cyc), ps in sorted(cells.items()):
        ps = np.asarray(ps)
        n_good = int((ps >= thr).sum())
        panels.setdefault(panel, {"cycles": []})["cycles"].append(
            {"cycle": cyc, "n": int(len(ps)), "n_good": n_good,
             "good_fraction": n_good / len(ps)})
    for panel, d in panels.items():
        fr = np.array([c["good_fraction"] for c in d["cycles"]])
        n = sum(c["n"] for c in d["cycles"])
        ng = sum(c["n_good"] for c in d["cycles"])
        d.update({"n_cycles": len(fr), "n_signals": n, "n_good": ng,
                  "pooled_good_fraction": ng / n,
                  "mean_good_fraction": float(fr.mean()),
                  "n_cycles_below_50": int((fr < 0.5).sum()),
                  "cycles_zero_good": [c["cycle"] for c in d["cycles"] if c["n_good"] == 0]})
    return panels


def run_e10(pool: dict[str, dict], ids, X: np.ndarray, *, seeds, es_frac: float,
            recipe: dict | None = None) -> dict:
    """Core of E10 on in-memory inputs.  Returns {"coverage": ..., "scores": ...}."""
    ids = list(ids)
    index = {s: i for i, s in enumerate(ids)}
    missing = [s for s in pool if s not in index]
    if missing:
        raise KeyError(f"{len(missing)} L_AL signals are not in the universe, e.g. {missing[0]}")
    X_of = lambda rows: X[[index[s] for s in rows]]
    P, fits = [], []
    for seed in seeds:
        model, info = fit_all_lal(pool, X_of, seed=seed, es_frac=es_frac, recipe=recipe)
        P.append(predict(model, X))
        fits.append({"seed": int(seed), **{k: info[k] for k in
                                           ("es_loss", "epochs", "train_seconds", "device",
                                            "n_train", "n_es", "n_es_bad")}})
    P = np.stack(P)
    mean_p = P.mean(axis=0)
    cov = coverage_table(ids, mean_p)
    per_seed = {}
    for k, seed in enumerate(seeds):
        t = coverage_table(ids, P[k])
        per_seed[str(seed)] = {pn: {"pooled_good_fraction": d["pooled_good_fraction"],
                                    "mean_good_fraction": d["mean_good_fraction"],
                                    "n_cycles_below_50": d["n_cycles_below_50"]}
                               for pn, d in t.items()}
    return {
        "coverage": {"threshold": THRESHOLD, "rule": "GOOD iff seed-averaged p >= 0.5",
                     "seeds": [int(s) for s in seeds], "n_signals": len(ids),
                     "n_good": int((mean_p >= THRESHOLD).sum()),
                     "panels": cov, "per_seed_panel_summary": per_seed, "fits": fits},
        "scores": {"ids": ids, "mean_p": [round(float(v), 6) for v in mean_p],
                   "sd_p": [round(float(v), 6) for v in P.std(axis=0, ddof=1)]
                   if len(seeds) > 1 else None,
                   "seeds": [int(s) for s in seeds]},
    }
