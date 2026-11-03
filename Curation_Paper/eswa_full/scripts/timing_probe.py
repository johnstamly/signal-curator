"""Task 8: timing probe and Stage-1 compute estimate.  Run from the repo root:
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/timing_probe.py

Measures (fold 0, seed 0): one full-recipe fit per architecture on the first 600 rows of the
L_AL training pool (GPU; MicroConv1D also CPU), the strict-deterministic CUDA check per torch
architecture, and the real cost of one ``simulate`` call (unc_oracle on L_AL, random on L_rand).
Then it combines these with the study definitions of the plan into a Stage-1 estimate.
Partial results are printed as they arrive.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Curation_Paper.eswa_full.lib.fit import DEFAULT_RECIPE, fit  # noqa: E402
from Curation_Paper.eswa_full.lib.loop import simulate  # noqa: E402
from Curation_Paper.eswa_full.lib.models import ARCHS as _ARCHS, build_model  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import (  # noqa: E402
    load_config, load_pools, reference_labels)
from Curation_Paper.eswa_full.lib.runio import save_result  # noqa: E402
from Curation_Paper.eswa_full.lib.signals import signals_for  # noqa: E402
from Curation_Paper.eswa_full.lib.splits import fold_split, shared_folds  # noqa: E402

ARCHS = list(_ARCHS)
CLASSIC_MODELS = ["hgb", "logreg", "minirocket"]
BUDGETS = [100, 200, 300, 400, 500, 600]
SEED_SIZE, BATCH = 100, 50
N_CELLS = 25  # 5 folds x 5 seeds (Stage 1)


def p(*a):
    print(*a, flush=True)


def time_fit(name, train, es, X_of, device):
    cuda_ok = None
    err = None
    t0 = time.perf_counter()
    try:
        if device == "cpu":
            real = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            try:
                _, info = fit(name, train, es, X_of, seed=0, recipe=dict(DEFAULT_RECIPE))
            finally:
                torch.cuda.is_available = real
        else:
            _, info = fit(name, train, es, X_of, seed=0, recipe=dict(DEFAULT_RECIPE))
            cuda_ok = str(info["device"]).startswith("cuda") and \
                torch.are_deterministic_algorithms_enabled()
    except Exception:
        err = traceback.format_exc(limit=4)
        info = {"epochs": None, "device": device, "train_seconds": None}
        if device == "cuda":
            cuda_ok = False
    wall = time.perf_counter() - t0
    return {"arch": name, "device": info.get("device", device),
            "seconds": info.get("train_seconds"), "wall_seconds": wall,
            "epochs": info.get("epochs"), "strict_cuda_ok": cuda_ok, "error": err}


def fit_units(strategy, seed_size, batch, bmax, budgets):
    """Sum of training-set sizes over every fit one simulate call performs (cost proxy,
    assumes fit time ~ linear in training-set size).  Mirrors loop.simulate: a fit at every
    round (every budget for 'random'); unc_oracle adds 3 out-of-fold fits at 2/3 size."""
    units, n, fits = 0.0, seed_size, 0
    while True:
        if strategy != "random" or n in budgets:
            units += n
            fits += 1
        if n >= bmax:
            break
        nxt = min(b for b in budgets if b > n)
        n += min(batch, nxt - n)
        if strategy in ("unc_oracle", "unc_selftrain"):
            units += 3 * (2 / 3) * n
            fits += 3
    return units, fits


def main():
    cfg = load_config()
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    test_ids = shared_folds(ref, cfg["n_folds"], 0)[0]
    train_pool, es = fold_split(al, test_ids, 0, cfg["es_frac"])
    train = train_pool[:600]
    ids = sorted({r["signal_id"] for r in train + es})
    cache = dict(zip(ids, signals_for(ids, cfg)))
    X_of = lambda i: np.stack([cache[s] for s in i])  # noqa: E731
    p(f"fold0 test={len(test_ids)} train_pool={len(train_pool)} es={len(es)} train_used={len(train)}")

    fits = {}
    # warm-up (CUDA init, cudnn) so the first timed fit is not penalised
    fit("microconv1d", train[:100], es, X_of, seed=0, recipe={**DEFAULT_RECIPE, "max_epochs": 2})
    for name in ARCHS + CLASSIC_MODELS:
        r = time_fit(name, train, es, X_of, "cuda")
        if name in CLASSIC_MODELS:
            r["strict_cuda_ok"] = None
        try:
            r["n_params"] = (sum(q.numel() for q in build_model(name, cfg["window"]).parameters())
                             if name in ARCHS else None)
        except Exception:
            r["n_params"] = None
        fits[name] = r
        p(name, {k: r[k] for k in ("device", "seconds", "epochs", "strict_cuda_ok", "n_params")},
          "ERROR" if r["error"] else "")
    cpu = time_fit("microconv1d", train, es, X_of, "cpu")
    cpu["n_params"] = fits["microconv1d"]["n_params"]
    p("microconv1d cpu", cpu["seconds"], cpu["epochs"])

    # real simulate cost
    sims = {}
    for key, pool, strat in (("unc_oracle_al", al, "unc_oracle"), ("random_rand", rand, "random")):
        t0 = time.perf_counter()
        out = simulate(pool, test_ids, ref, strategy=strat, budgets=BUDGETS, seed=0,
                       seed_size=SEED_SIZE, batch=BATCH, eta=0.0, arch="microconv1d",
                       recipe=dict(DEFAULT_RECIPE), cfg=cfg)
        wall = time.perf_counter() - t0
        u, nf = fit_units(strat, SEED_SIZE, BATCH, BUDGETS[-1], BUDGETS)
        sims[key] = {"strategy": strat, "wall_seconds": wall,
                     "fit_seconds_reported": out["budgets"][-1]["train_seconds"],
                     "len_train_pool": out["len_train_pool"], "fit_units": u, "n_fits": nf}
        p("simulate", key, wall, out["budgets"][-1]["train_seconds"])

    # ---------------- Stage-1 estimate ----------------
    t_fit = {k: v["seconds"] for k, v in fits.items()}
    torch_archs = [a for a in ARCHS]
    S_or = sims["unc_oracle_al"]["wall_seconds"]
    S_rd = sims["random_rand"]["wall_seconds"]
    U_or, U_rd = sims["unc_oracle_al"]["fit_units"], sims["random_rand"]["fit_units"]
    mc = t_fit["microconv1d"]

    def sim_time(strategy, seed_size=SEED_SIZE, batch=BATCH, bmax=600, budgets=BUDGETS,
                 arch="microconv1d"):
        u, _ = fit_units(strategy, seed_size, batch, bmax, budgets)
        base = S_or / U_or if strategy != "random" else S_rd / U_rd
        ratio = (t_fit[arch] / mc) if arch != "microconv1d" else 1.0
        return u * base * ratio

    studies = {}

    def add(name, items):
        """items: list of (description, count, unit_seconds)."""
        rows = [{"what": d, "count": c, "unit_seconds": s, "seconds": c * s} for d, c, s in items]
        studies[name] = {"items": rows, "hours": sum(r["seconds"] for r in rows) / 3600.0}

    # E1: 3 configs (unc_oracle/L_AL, random/L_rand, random/L_AL) x 25 cells, doubled for the
    # 13-excluded sensitivity variant (new fold set => recomputed, no reuse assumed).
    e1 = [("unc_oracle on L_AL", 2 * N_CELLS, sim_time("unc_oracle")),
          ("random on L_rand", 2 * N_CELLS, sim_time("random")),
          ("random on L_AL", 2 * N_CELLS, sim_time("random"))]
    add("E1", e1)
    add("E2", [("unc_oracle x 2 eta", 2 * N_CELLS, sim_time("unc_oracle")),
               ("unc_selftrain x 2 eta", 2 * N_CELLS, sim_time("unc_oracle")),
               ("uncertainty x 2 eta", 2 * N_CELLS, sim_time("uncertainty"))])
    add("E3", [("unc_oracle x 5 eta", 5 * N_CELLS, sim_time("unc_oracle")),
               ("uncertainty x 5 eta", 5 * N_CELLS, sim_time("uncertainty"))])
    # E4 tuning: <=6 configs x fold 0 x seeds 0-2, full-pool fits (probe is at 600 rows;
    # the full training pool is ~1.2x larger -> use the 600-row time, noted as slight under-estimate)
    tune = [(f"{a} tuning (6 cfg x 3 seeds)", 18, t_fit[a]) for a in torch_archs]
    tune += [(f"{a} tuning (6 cfg x 3 seeds)", 18, t_fit[a]) for a in CLASSIC_MODELS]
    add("E4_tune", tune)
    # E4 compare: 8 models x 6 budgets x 25 cells; fit time ~ linear in n => mean fraction of 600
    frac = float(np.mean(BUDGETS)) / 600.0
    add("E4_compare", [(f"{a} x 6 budgets x 25 cells (x{frac:.3f} of 600-row time)",
                        6 * N_CELLS, t_fit[a] * frac) for a in torch_archs + CLASSIC_MODELS])
    # E5: top 3 torch archs by accuracy unknown -> upper bound = 3 slowest non-microconv torch
    # archs (microconv is already in E1); lower bound = 3 fastest.
    others = sorted([a for a in torch_archs if a != "microconv1d"], key=lambda a: t_fit[a])
    e5_hi = [(f"{a} in unc_oracle loop (upper bound: 3 slowest)", N_CELLS,
              sim_time("unc_oracle", arch=a)) for a in others[-3:]]
    e5_lo = [(f"{a} in unc_oracle loop (lower bound: 3 fastest)", N_CELLS,
              sim_time("unc_oracle", arch=a)) for a in others[:3]]
    add("E5", e5_hi)
    e5_low_hours = sum(c * s for _, c, s in e5_lo) / 3600.0
    # E6: 7 one-factor fits + 1 default control at budget 600 (microconv) x 25 cells;
    # + warm-start loop variant (unc_oracle, 25 cells). Window 1000 costs more (~2x input) -> x2.
    add("E6", [("one-factor fits (relu, cw none, cw full, no aug, win250, no zscore)", 6 * N_CELLS, mc),
               ("window 1000 fit (x2 input length)", N_CELLS, 2 * mc),
               ("default control fit", N_CELLS, mc),
               ("warm-start loop variant", N_CELLS, sim_time("unc_oracle"))])
    # E7: 3 batch x 3 seed-size, unc_oracle up to budget 400
    b7 = [b for b in BUDGETS if b <= 400]
    add("E7", [(f"unc_oracle seed={s} batch={b} (to 400)", N_CELLS,
                sim_time("unc_oracle", s, b, 400, [x for x in b7 if x >= s]
                         if s > 100 else b7))
               for b in (25, 50, 100) for s in (50, 100, 200)])
    # E8/E9: predictions pooled from E1 / cheap sklearn -> negligible, 0 fits.
    add("E8_E9", [("reuse E1 predictions / sklearn", 0, 0.0)])
    # E10: final MicroConv1D on all of L_AL minus 15% ES, seeds 0-4; ~1.7x the 600-row set
    add("E10", [("microconv1d on all L_AL, seeds 0-4 (x1.7 of 600 rows)", 5, 1.7 * mc)])
    add("E11", [("reuses E10 model (R4)", 0, 0.0)])

    total_h = sum(s["hours"] for s in studies.values())
    est = {
        "formula": "Stage1 = sum_studies sum_items count * unit_seconds. Simulate unit_seconds = "
                   "fit_units(config) * (measured simulate wall seconds / fit_units of the "
                   "measured config), fit_units = sum of training-set sizes over all fits "
                   "(round fit + 3 OOF fits at 2/3 size for oracle/selftrain strategies). "
                   "Torch archs in simulate (E5) are scaled by t_fit(arch)/t_fit(microconv1d).",
        "assumptions": [
            "fit time ~ linear in training-set size",
            "5 folds x 5 seeds = 25 cells per config; E1 doubled for the 13-excluded sensitivity run",
            "E1/E2/E3 overlaps are not de-duplicated (conservative)",
            "E2 unc_selftrain costs the same as unc_oracle; 'uncertainty' = one fit per round, no OOF",
            "E4 tuning fits use the 600-row time (full pool ~1.2x larger: slight under-estimate)",
            "E5 headline uses the 3 slowest non-microconv torch archs (upper bound); lower bound listed",
            "E6 window-1000 fit costs 2x the default; other ablations 1x",
            "E7 uses unc_oracle, budgets <= 400; seed sizes above 100 start at that size",
            "E10 fit uses ~1.7x the rows of the 600-row probe; E8/E9/E11 add no fits",
            "classic models time includes feature extraction as measured by fit()",
            "probe is a single sample on one fold and seed; epochs vary by seed/arch",
        ],
        "per_study_hours": {k: v["hours"] for k, v in studies.items()},
        "studies": studies,
        "e5_lower_bound_hours": e5_low_hours,
        "stage1_total_hours": total_h,
        "stage1_total_hours_with_e5_lower_bound": total_h - studies["E5"]["hours"] + e5_low_hours,
        "exceeds_24h": total_h > 24.0,
    }
    p("\nStage-1 estimate (GPU-hours):")
    for k, v in sorted(est["per_study_hours"].items(), key=lambda kv: -kv[1]):
        p(f"  {k:12s} {v:8.2f}")
    p(f"  TOTAL        {total_h:8.2f}  (E5 lower bound variant: "
      f"{est['stage1_total_hours_with_e5_lower_bound']:.2f})  exceeds_24h={est['exceeds_24h']}")

    payload = {"fold": 0, "seed": 0, "n_train_rows": len(train), "n_es": len(es),
               "len_train_pool": len(train_pool), "n_test": len(test_ids),
               "recipe": DEFAULT_RECIPE, "fits": fits, "microconv1d_cpu": cpu,
               "strict_cuda": {a: fits[a]["strict_cuda_ok"] for a in ARCHS},
               "simulate": sims, "estimate": est,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    path = save_result("timing_probe", payload, cfg)
    p("->", path)


if __name__ == "__main__":
    main()
