"""E4 (architecture tuning and comparison) and E5 (architectures in the loop), on the cell runner.

Pipeline (rulings R20-R22):
  1. ``tune_jobs``    -> results/jobs/e04_tune.json     (kind "tune" cells, fold 0, seeds 0-2)
  2. run_cells.py on that file
  3. ``freeze_payload`` -> results/e04_frozen_configs.json: one recipe per model, chosen on
                         early-stopping quantities only (calibrated models: lowest mean es_loss;
                         minirocket: highest mean es AUROC; within one standard error of the best
                         the default config, then the smaller model).  Committed BEFORE any
                         compare cell runs.
  4. ``compare_specs``-> results/jobs/e04_compare.json  (loop cells, strategy random, pool al, eta 0)
  5. run_cells.py, then ``build_compare`` -> results/e04_compare.json
  6. ``select_top``   -> results/e05_selection.json + results/jobs/e05_arch_loop.json: the top 3
                         non-MicroConv1D models by mean es AUROC of their frozen config (tune
                         cells; the compare/test results are never read), fixed at Stage 1 and
                         reused by Stage 2
  7. run_cells.py, then ``build_e05``      -> results/e05_arch_loop.json

Only early-stopping information enters steps 1-3 and 6.  Disclosed: tuning is done on fold 0 and
trains/early-stops on rows that are test rows of folds 1-4 (TUNING_LEAKAGE_NOTE).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .aggregate import _nanstat, curve
from .cells import canonical_spec, cell_id, load_cell
from .classic import CLASSIC, FEATURE_NAMES
from .fit import DEFAULT_RECIPE, _recipe
from .jobs import cell_specs, grid
from .kind_tune import TUNE_FOLD, TUNE_N
from .models import ARCHS, build_model, n_params

MODELS = ARCHS + CLASSIC            # the 8 compared models
TUNE_SEEDS = (0, 1, 2)
LRS = (1e-3, 3e-4, 1e-4)
ACTIVATIONS = ("leaky", "relu")     # microconv1d only (same 2529 parameters)
HGB_LRS = (0.05, 0.1)
HGB_LEAVES = (15, 31, 63)
LOGREG_CS = (0.01, 0.1, 1.0, 10.0, 100.0)
_HGB_DEFAULT = {"learning_rate": 0.1, "max_leaf_nodes": 31}
_LOGREG_DEFAULT = {"C": 1.0}
# Task 4 parameter bands (tests/test_models.py); microconv1d is fixed at 2529
BANDS = {"microconv1d": (2529, 2529), "cnn_wide": (20_000, 40_000),
         "resnet1d": (70_000, 130_000), "bigru": (10_000, 60_000),
         "transformer": (30_000, 80_000)}
# second width per torch architecture (the first is 1.0); both widths must sit inside the band
ALT_WIDTH = {"cnn_wide": 1.15, "resnet1d": 0.85, "bigru": 0.7, "transformer": 1.3}
MINIROCKET_KERNELS = (2500, 5000, 10000)
INPUT_LEN = 500
E5_TOP = 3
# model whose configs are ranked by es AUROC instead of es_loss (its p(GOOD) is a sigmoid of a
# ridge score, so its log-loss is not a calibrated loss)
AUROC_RANKED = ("minirocket",)

TUNING_LEAKAGE_NOTE = (
    "Tuning on fold 0 trains and early-stops on rows that are TEST rows of folds 1-4, so the "
    "frozen hyper-parameters carry information from those rows' labels. Results on fold 0 alone "
    "are the strictly clean slice (re-score from the stored predictions).")
SELECTION_RULE = (
    "per model: lowest mean es_loss over tune seeds (minirocket: highest mean es AUROC); every "
    "config within one standard error (over seeds, of the best config) of the best is eligible; "
    "among the eligible the default config, else the smallest (parameters; hgb max_leaf_nodes; "
    "logreg C; minirocket kernels), else the best.")


def candidate_recipe(*, lr=None, width=None, kernels=None, model_cfg=None, **extra) -> dict:
    """A full recipe (DEFAULT_RECIPE plus the candidate's changes).  The default values leave
    the recipe identical to DEFAULT_RECIPE, so a default winner dedups with E1's cells."""
    r = json.loads(json.dumps(DEFAULT_RECIPE))
    if lr is not None:
        r["lr"] = float(lr)
    if width is not None and float(width) != 1.0:
        r["model_cfg"] = {"width": float(width)}
    if model_cfg:
        r["model_cfg"] = {**r["model_cfg"], **model_cfg}
    if kernels is not None:
        r["minirocket_kernels"] = int(kernels)
    r.update(extra)
    _recipe(r)  # unknown keys raise
    return r


def model_size(arch: str, recipe: dict, n_par: int | None) -> tuple[int, str]:
    """(size, unit) reported for every model under one key: parameters (torch), input
    features (hgb, logreg) or kernels (minirocket)."""
    if arch in ARCHS:
        return int(n_par), "parameters"
    if arch == "minirocket":
        return int(recipe["minirocket_kernels"]), "kernels"
    return len(FEATURE_NAMES), "features"


def complexity(arch: str, recipe: dict, n_par: int | None) -> float:
    """Ordering used to prefer the smaller model among tied configurations."""
    if arch in ARCHS:
        return float(n_par)
    cfg = recipe.get("model_cfg") or {}
    if arch == "hgb":
        return float(cfg.get("max_leaf_nodes", _HGB_DEFAULT["max_leaf_nodes"]))
    if arch == "logreg":
        return float(cfg.get("C", _LOGREG_DEFAULT["C"]))   # smaller C = stronger regularisation
    return float(recipe["minirocket_kernels"])


def candidate_configs(arch: str, *, lrs=LRS, widths=None, kernels=MINIROCKET_KERNELS,
                      activations=ACTIVATIONS, hgb_lrs=HGB_LRS, leaves=HGB_LEAVES, cs=LOGREG_CS,
                      log=print, **extra) -> list[dict]:
    """The tuning configurations of ``arch``: [{name, recipe, n_params}], every torch one built
    and checked against the parameter band (out-of-band ones are dropped and logged).

    torch: lr x width (microconv1d ignores width and varies the activation instead);
    hgb: learning_rate x max_leaf_nodes; logreg: C; minirocket: kernel count.  Knob values equal to
    the library default are NOT written to the recipe, so the default configuration's recipe is
    DEFAULT_RECIPE."""
    if arch not in MODELS:
        raise ValueError(f"unknown model {arch!r}; choose from {MODELS}")
    out: list[dict] = []
    if arch == "microconv1d":
        lo, hi = BANDS[arch]
        for act in activations:
            n = n_params(build_model(arch, INPUT_LEN, activation=act))
            if not lo <= n <= hi:
                log(f"DROPPED out-of-band config {arch} activation={act}: {n} params")
                continue
            for lr in lrs:
                ex = {} if act == "leaky" else {"activation": act}
                out.append({"name": f"lr={lr:g}|act={act}",
                            "recipe": candidate_recipe(lr=lr, **ex, **extra), "n_params": n})
    elif arch in ARCHS:
        lo, hi = BANDS[arch]
        for w in tuple(widths or (1.0, ALT_WIDTH[arch])):
            kw = {} if float(w) == 1.0 else {"width": float(w)}
            n = n_params(build_model(arch, INPUT_LEN, **kw))
            if not lo <= n <= hi:
                log(f"DROPPED out-of-band config {arch} width={w}: {n} params not in [{lo}, {hi}]")
                continue
            for lr in lrs:
                out.append({"name": f"lr={lr:g}|width={w:g}",
                            "recipe": candidate_recipe(lr=lr, width=w, **extra), "n_params": n})
    elif arch == "hgb":
        for lr in hgb_lrs:
            for lv in leaves:
                cfg = {k: v for k, v in (("learning_rate", lr), ("max_leaf_nodes", lv))
                       if v != _HGB_DEFAULT[k]}
                out.append({"name": f"learning_rate={lr:g}|max_leaf_nodes={lv}",
                            "recipe": candidate_recipe(model_cfg=cfg, **extra), "n_params": None})
    elif arch == "logreg":
        for c in cs:
            cfg = {} if c == _LOGREG_DEFAULT["C"] else {"C": float(c)}
            out.append({"name": f"C={c:g}", "recipe": candidate_recipe(model_cfg=cfg, **extra),
                        "n_params": None})
    else:  # minirocket
        for k in kernels:
            out.append({"name": f"kernels={k}", "recipe": candidate_recipe(kernels=k, **extra),
                        "n_params": None})
    return out


def tune_spec(arch: str, recipe: dict, seed: int) -> dict:
    return canonical_spec({
        "study_source": "tune", "pool": "al", "fold": TUNE_FOLD, "seed": seed,
        "strategy": "tune", "eta": 0.0, "arch": arch, "recipe": recipe,
        "budgets": [TUNE_N], "seed_size": 0, "batch": 0})


def tune_jobs(*, archs=MODELS, seeds=TUNE_SEEDS, log=print, **kw) -> dict:
    """{"configs": {arch: [{name, recipe, n_params}]}, "specs": [...], "counts": {...}}"""
    configs, specs = {}, []
    for a in archs:
        configs[a] = candidate_configs(a, log=log, **kw)
        for c in configs[a]:
            specs += [tune_spec(a, c["recipe"], s) for s in seeds]
    ids = [cell_id(s) for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate tune cells in the job list")
    return {"seeds": list(seeds), "configs": configs, "specs": specs,
            "counts": {"total": len(specs), **{a: len(c) * len(seeds) for a, c in configs.items()}}}


# ----------------------------------------------------------------------------- freeze

def _group_tune(specs: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """{arch: {recipe-json: [specs of that configuration]}}"""
    out: dict[str, dict[str, list[dict]]] = {}
    for s in specs:
        c = canonical_spec(s)
        if c["study_source"] != "tune":
            raise ValueError(f"not a tune spec: {cell_id(c)}")
        out.setdefault(c["arch"], {}).setdefault(json.dumps(c["recipe"], sort_keys=True), []).append(c)
    return out


def _stats(vals) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def pick_config(arch: str, cands: list[dict]) -> dict:
    """The frozen candidate under the SELECTION_RULE (``cands`` carry es_loss / es_auroc stats,
    recipe, n_params and ``_order``)."""
    by_auroc = arch in AUROC_RANKED
    key = "es_auroc" if by_auroc else "es_loss"
    sign = -1.0 if by_auroc else 1.0          # sign * mean: smaller is better
    best = min(cands, key=lambda c: (round(sign * c[key + "_mean"], 9), c["_order"]))
    n = best["n_seeds"]
    se = best[key + "_sd"] / math.sqrt(n) if n > 1 else 0.0
    eligible = [c for c in cands
                if sign * c[key + "_mean"] <= sign * best[key + "_mean"] + se + 1e-12]
    default = [c for c in eligible if c["recipe"] == DEFAULT_RECIPE]
    if default:
        return default[0]
    return min(eligible, key=lambda c: (complexity(arch, c["recipe"], c["n_params"]),
                                        round(sign * c[key + "_mean"], 9), c["_order"]))


def freeze_payload(specs: list[dict], cfg: dict, names: dict | None = None) -> dict:
    """Per model, the frozen configuration (SELECTION_RULE) and every candidate's statistics.
    All cells must be present."""
    names = names or {}
    out = {"tune_fold": TUNE_FOLD, "tune_n_train": TUNE_N, "selection_rule": SELECTION_RULE,
           "tuning_leakage_note": TUNING_LEAKAGE_NOTE, "models": {}}
    for arch, by_recipe in _group_tune(specs).items():
        cands = []
        for order, (rkey, ss) in enumerate(by_recipe.items()):
            res = [load_cell(s, cfg)["result"] for s in ss]
            nps = {r["n_params"] for r in res}
            if len(nps) != 1:
                raise ValueError(f"{arch}: inconsistent n_params across seeds {sorted(map(str, nps))}")
            for k in ("es_loss", "es_auroc"):
                if not np.all(np.isfinite([r[k] for r in res])):
                    raise ValueError(f"{arch}: non-finite {k} in config {rkey}")
            recipe = json.loads(rkey)
            n_par = next(iter(nps))
            size, unit = model_size(arch, recipe, n_par)
            c = {"name": names.get((arch, rkey), rkey), "recipe": recipe, "n_params": n_par,
                 "size": size, "size_unit": unit, "n_seeds": len(res),
                 "seeds": [s["seed"] for s in ss], "_order": order,
                 "epochs_mean": float(np.mean([r["epochs"] for r in res])),
                 "train_seconds_mean": float(np.mean([r["train_seconds"] for r in res]))}
            for k in ("es_loss", "es_auroc"):
                vals = [r[k] for r in res]
                c[k + "_mean"], c[k + "_sd"] = _stats(vals)
                c[k + "_values"] = [float(x) for x in vals]
            cands.append(c)
        if len({c["n_seeds"] for c in cands}) != 1:
            raise ValueError(f"{arch}: configurations differ in their number of seeds")
        best = pick_config(arch, cands)
        for c in cands:
            c.pop("_order")
        out["models"][arch] = {
            "selection_metric": "es_auroc" if arch in AUROC_RANKED else "es_loss",
            "frozen": {**{k: best[k] for k in ("name", "recipe", "n_params", "size", "size_unit",
                                               "es_loss_mean", "es_loss_sd", "es_auroc_mean",
                                               "es_auroc_sd")},
                       "equals_default": best["recipe"] == DEFAULT_RECIPE},
            "candidates": cands}
    return out


# ----------------------------------------------------------------------------- compare / E5

def load_frozen(path, models=MODELS) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [m for m in models if m not in d.get("models", {})]
    if missing:
        raise ValueError(f"frozen configs lack model(s) {missing}")
    for a, v in d["models"].items():
        _recipe(v["frozen"]["recipe"])
    return d


def compare_specs(cfg: dict, frozen: dict, *, seeds=None, models=MODELS,
                  budgets=None) -> dict[str, list[dict]]:
    """{model: loop cell specs}: strategy random, pool al, eta 0, budgets 100..600, 5 folds."""
    out = {}
    for m in models:
        g = grid(cfg, arch=m, recipe=frozen["models"][m]["frozen"]["recipe"], seeds=seeds,
                 budgets=budgets)
        out[m] = cell_specs(g, pool="al", strategy="random", eta=0.0)
    return out


def e1_random_al_ids(cfg: dict, seeds=None, budgets=None) -> set[str]:
    g = grid(cfg, seeds=seeds, budgets=budgets)
    return {cell_id(s) for s in cell_specs(g, pool="al", strategy="random", eta=0.0)}


def shared_with_e1(cfg: dict, per_model: dict[str, list[dict]], seeds=None, budgets=None) -> dict:
    e1 = e1_random_al_ids(cfg, seeds, budgets)
    return {m: sum(cell_id(s) in e1 for s in ss) for m, ss in per_model.items()}


def _fit_seconds(res: dict) -> list[float]:
    """Per-fit seconds at each budget (random strategy fits exactly once per budget; the stored
    train_seconds are cumulative)."""
    cum = [r["train_seconds"] for r in res["budgets"]]
    return [cum[0]] + [b - a for a, b in zip(cum, cum[1:])]


def build_compare(cfg: dict, g_base: dict, ref: dict, frozen: dict, models=MODELS) -> dict:
    out = {"strategy": "random", "pool": "al", "eta": 0.0, "seeds": g_base["seeds"],
           "n_seeds": len(g_base["seeds"]), "folds": g_base["folds"], "budgets": g_base["budgets"],
           "seed_size": g_base["seed_size"], "batch": g_base["batch"],
           "tuning_leakage_note": TUNING_LEAKAGE_NOTE, "models": {}}
    for m in models:
        fz = frozen["models"][m]["frozen"]
        g = {**g_base, "arch": m, "recipe": fz["recipe"]}
        rows = curve(cfg, g, ref, pool="al", strategy="random", eta=0.0)
        per_budget: dict[int, list[float]] = {}
        for s in cell_specs(g, pool="al", strategy="random", eta=0.0):
            res = load_cell(s, cfg)["result"]
            for r, sec in zip(res["budgets"], _fit_seconds(res)):
                per_budget.setdefault(r["budget"], []).append(sec)
        for row in rows:
            row["train_seconds_per_fit"] = _nanstat(per_budget[row["budget"]])
        allsec = [x for v in per_budget.values() for x in v]
        out["models"][m] = {
            "arch": m, "recipe": fz["recipe"], "frozen_name": fz["name"], "n_params": fz["n_params"],
            "size": fz["size"], "size_unit": fz["size_unit"],
            "calibrated": m != "minirocket",
            "calibration_note": (None if m != "minirocket" else
                                 "p(GOOD) is a sigmoid of a ridge score, compressed to ~0.27-0.73: "
                                 "ECE and log-loss are not comparable with the other models; "
                                 "compare AUROC and 0.5-threshold metrics"),
            "train_seconds_per_fit_mean": float(np.mean(allsec)), "rows": rows}
    return out


def select_top(frozen: dict, k: int = E5_TOP) -> list[dict]:
    """Top ``k`` non-MicroConv1D models by the mean early-stopping AUROC of their frozen
    configuration (R22; tune cells only -- no test-fold result is read).  nan ranks last, ties
    by model name."""
    ranked = []
    for m, v in frozen["models"].items():
        if m == "microconv1d":
            continue
        a = float(v["frozen"]["es_auroc_mean"])
        ranked.append((-(a if not math.isnan(a) else -1.0), m, a))
    ranked.sort()
    return [{"arch": m, "es_auroc_mean": a, "recipe": frozen["models"][m]["frozen"]["recipe"],
             "n_params": frozen["models"][m]["frozen"]["n_params"],
             "size": frozen["models"][m]["frozen"]["size"],
             "size_unit": frozen["models"][m]["frozen"]["size_unit"]}
            for _, m, a in ranked[:k]]


def selection_payload(frozen: dict, k: int = E5_TOP) -> dict:
    """What E5 runs: the selected architectures and MicroConv1D's FROZEN recipe as baseline."""
    mc = frozen["models"]["microconv1d"]["frozen"]["recipe"]
    return {"selected": select_top(frozen, k),
            "ranking_basis": "mean es AUROC of the frozen config over tune seeds (fold 0)",
            "microconv1d": {"recipe": mc, "frozen_equals_default": mc == DEFAULT_RECIPE},
            "tuning_leakage_note": TUNING_LEAKAGE_NOTE}


def e05_specs(cfg: dict, selected: list[dict], *, seeds=None,
              budgets=None) -> dict[str, list[dict]]:
    out = {}
    for s in selected:
        g = grid(cfg, arch=s["arch"], recipe=s["recipe"], seeds=seeds, budgets=budgets)
        out[s["arch"]] = cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    return out


def e05_required_specs(cfg: dict, selection: dict, seeds=None, budgets=None) -> list[dict]:
    """E5 cells plus MicroConv1D's baseline cells at its frozen recipe (the E3 cells when that
    recipe is the default, otherwise E5's own cells)."""
    specs = [s for ss in e05_specs(cfg, selection["selected"], seeds=seeds,
                                   budgets=budgets).values() for s in ss]
    base = grid(cfg, arch="microconv1d", recipe=selection["microconv1d"]["recipe"], seeds=seeds,
                budgets=budgets)
    return specs + cell_specs(base, pool="al", strategy="unc_oracle", eta=0.0)


def build_e05(cfg: dict, g_base: dict, ref: dict, selection: dict) -> dict:
    out = {"strategy": "unc_oracle", "pool": "al", "eta": 0.0, "seeds": g_base["seeds"],
           "n_seeds": len(g_base["seeds"]), "folds": g_base["folds"], "budgets": g_base["budgets"],
           "seed_size": g_base["seed_size"], "batch": g_base["batch"],
           "selected": selection["selected"], "tuning_leakage_note": TUNING_LEAKAGE_NOTE,
           "models": {}}
    for s in selection["selected"]:
        g = {**g_base, "arch": s["arch"], "recipe": s["recipe"]}
        out["models"][s["arch"]] = {
            "arch": s["arch"], "recipe": s["recipe"], "n_params": s["n_params"],
            "size": s.get("size"), "size_unit": s.get("size_unit"),
            "rows": curve(cfg, g, ref, pool="al", strategy="unc_oracle", eta=0.0)}
    mc = selection["microconv1d"]
    base = {**g_base, "arch": "microconv1d", "recipe": mc["recipe"]}
    out["microconv1d_baseline"] = {
        "note": "MicroConv1D at its frozen recipe, unc_oracle eta 0 ("
                + ("the E3 cells: frozen recipe is the default)" if mc["frozen_equals_default"]
                   else "E5's own cells: frozen recipe differs from the default)"),
        "arch": "microconv1d", "recipe": mc["recipe"],
        "frozen_equals_default": mc["frozen_equals_default"],
        "rows": curve(cfg, base, ref, pool="al", strategy="unc_oracle", eta=0.0)}
    return out
