"""Synthetic study JSONs in the aggregators' real schema (Task 14).

Synthetic loop / ablate cells are written to a temp results dir and the REAL aggregation
functions run on them (aggregate.build_e1/2/3/7, arch_study.build_compare/build_e05,
ablate_study.build_ablate/build_warm, rules_study.build_e8/famb_by_budget); the results are
wrapped with the same provenance fields the CLIs add (save_result meta + cells_* keys).  The six
real study JSONs are copied unchanged.  Nothing here touches the real results/ or cells/.
"""
from __future__ import annotations

import json
import shutil
import zlib
from pathlib import Path

import numpy as np

from Curation_Paper.eswa_full.lib import ablate_study as AB
from Curation_Paper.eswa_full.lib import aggregate as AG
from Curation_Paper.eswa_full.lib import arch_study as A
from Curation_Paper.eswa_full.lib import jobs
from Curation_Paper.eswa_full.lib import rules_study as R
from Curation_Paper.eswa_full.lib.cells import canonical_spec, cell_id, cell_path
from Curation_Paper.eswa_full.lib.metrics import metrics
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT, input_hashes, load_config
from Curation_Paper.eswa_full.lib.runio import save_result

LOOP_KEY = "a" * 64
ABLATE_KEY = "b" * 64
REAL = ("e00_inputs", "e04_frozen_configs", "e09_rules", "e10_coverage", "e10_scores", "e01_standard_shift",
        "e11_downstream", "e13_throughput", "e14_live_record")


def fake_ref(n=50):
    return {f"r{i}": (0 if i % 4 == 0 else 1) for i in range(n)}


def fold_ids(ref, n_folds=5):
    ids = sorted(ref)
    return [ids[i::n_folds] for i in range(n_folds)]


def _store(cfg, spec, result, key, kind, dirty=False):
    cell_path(spec, cfg).write_text(json.dumps({
        "spec": canonical_spec(spec), "result": result,
        "meta": {"code_key": key, "dirty": dirty, "kind": kind, "git_sha": "x",
                 "input_hashes": input_hashes(cfg)}}))


def loop_result(spec, ref, folds, *, dirty_unused=False):
    rng = np.random.default_rng(zlib.crc32(cell_id(spec).encode()))
    test_ids = folds[spec["fold"]]
    eta = spec["eta"]
    strat = spec["strategy"]
    base = {"unc_oracle": 0.55, "uncertainty": 0.45, "unc_selftrain": 0.5, "random": 0.35}[strat]
    if spec["pool"] == "rand":
        base -= 0.05
    recs = []
    for i, b in enumerate(spec["budgets"]):
        q = base + 0.04 * i - 0.5 * eta
        p = {s: float(np.clip(0.5 + (0.5 if ref[s] else -0.5) * min(q, 0.95)
                              + rng.normal(0, 0.45), 0.01, 0.99)) for s in test_ids}
        n_fl = int(round(eta * b))
        recs.append({"budget": b, "metrics": {}, "test_pred": {s: round(v, 6) for s, v in p.items()},
                     "n_flipped": n_fl, "n_flagged": n_fl + 1, "n_recovered": n_fl // 2,
                     "n_ever_recovered": n_fl // 2, "n_false_flags": 1, "n_residual_flips": n_fl // 2,
                     "selection_claim_ok": b <= 450, "train_seconds": 2.0 * (i + 1),
                     "f_amb": float(0.1 + 0.02 * i)})
    return {"test_ids": list(test_ids), "len_train_pool": 600, "budgets": recs}


def ablate_result(ref, test_ids, quality):
    rng = np.random.default_rng(len(test_ids) + int(quality * 1000))
    p = {s: float(np.clip(0.5 + (0.5 if ref[s] else -0.5) * quality + rng.normal(0, 0.1), 0.01, 0.99))
         for s in test_ids}
    m = metrics(np.array([ref[s] for s in test_ids]), np.array([p[s] for s in test_ids]))
    return {"test_ids": list(test_ids), "test_pred": {s: round(v, 6) for s, v in p.items()},
            "metrics": m, "epochs": 5, "es_loss": 0.4, "train_seconds": 1.0, "n_train": 600,
            "n_es": 10, "len_train_pool": 700, "window": 500}


def cells_provenance(key, kind="loop"):
    k = key if isinstance(key, dict) else key
    counts = ({kk: {v: 10} for kk, v in key.items()} if isinstance(key, dict) else {key: 10})
    return {"stage": "stage1", "cells_code_key": k, "cells_key_counts": counts,
            "cells_git_shas": {"x": 10}, "cells_dirty": 0}


def build_stage2(world: dict, seeds=tuple(range(20))) -> dict:
    """Add the Stage-2 (20-seed) E1-E4 studies to a ``build_all`` world, as the aggregators do for
    ``--seeds stage2``: synthetic loop cells for every seed, then the REAL builders (aggregate.build_e1/2/3,
    arch_study.build_compare) written as ``<name>_s20.json`` with ``stage: stage2``.  Cells of seeds the
    world already holds are kept (same cell ids).  Returns {"paths": {name: path}, "g": grid}."""
    cfg = dict(world["cfg"])
    ref, folds, frozen = world["ref"], world["folds"], world["frozen"]
    assert list(seeds) == list(cfg["seeds_stage2"]), "the _s20 files carry the config's Stage-2 seeds"
    g = jobs.grid(cfg, seeds=list(seeds))
    for s in (jobs.e3_specs(g) + jobs.e2_specs(g) + jobs.e1_specs(g)
              + [x for ss in A.compare_specs(cfg, frozen, seeds=list(seeds), budgets=g["budgets"]).values()
                 for x in ss]):
        if not cell_path(s, cfg).exists():
            _store(cfg, s, loop_result(s, ref, folds), LOOP_KEY, "loop")
    prov = {**cells_provenance(LOOP_KEY), "stage": "stage2"}
    paths = {}
    for name, payload in (("e01_selection_s20", AG.build_e1(cfg, g, ref)),
                          ("e02_reinspection_s20", AG.build_e2(cfg, g, ref)),
                          ("e03_noise_s20", AG.build_e3(cfg, g, ref)),
                          ("e04_compare_s20", A.build_compare(cfg, g, ref, frozen))):
        paths[name] = save_result(name, {**payload, **prov}, cfg)
    return {"paths": paths, "g": g}


def build_all(out_dir: Path, seeds=(0, 1)) -> dict:
    """Write every study JSON (synthetic + copied real ones) into out_dir; returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["results_dir"] = str(out_dir)
    (out_dir / "cells").mkdir(exist_ok=True)
    ref = fake_ref()
    folds = fold_ids(ref)
    g = jobs.grid(cfg, seeds=list(seeds))

    def store_loop(specs):
        for s in specs:
            if not cell_path(s, cfg).exists():
                _store(cfg, s, loop_result(s, ref, folds), LOOP_KEY, "loop")

    store_loop(jobs.e3_specs(g) + jobs.e2_specs(g) + jobs.e1_specs(g) + jobs.e7_specs(g))
    paths = {}

    def save(name, payload, prov):
        payload = {**payload, **prov}
        paths[name] = save_result(name, payload, cfg)

    prov = cells_provenance(LOOP_KEY)
    save("e01_selection", AG.build_e1(cfg, g, ref), prov)
    save("e02_reinspection", AG.build_e2(cfg, g, ref), prov)
    save("e03_noise", AG.build_e3(cfg, g, ref), prov)
    save("e07_loop_settings", AG.build_e7(cfg, g, ref), prov)
    # E8 / f_amb
    save("e08_thresholds", {"source": "synthetic", "budget": 600, "arch": g["arch"],
                            **R.build_e8(R.pool_by_seed(cfg, g, ref, 600))}, prov)
    save("e10_famb", {"source": "synthetic", "arch": g["arch"], **R.famb_by_budget(cfg, g)}, prov)
    # E4 compare / E5 (frozen configs are the real committed ones)
    frozen = A.load_frozen(PKG_ROOT / "results" / "e04_frozen_configs.json")
    for m, specs in A.compare_specs(cfg, frozen, seeds=list(seeds), budgets=g["budgets"]).items():
        store_loop([{**s} for s in specs])
    save("e04_compare", A.build_compare(cfg, g, ref, frozen), prov)
    sel = A.selection_payload(frozen)
    store_loop(A.e05_required_specs(cfg, sel, list(seeds), g["budgets"]))
    save("e05_arch_loop", A.build_e05(cfg, g, ref, sel), prov)
    (out_dir / "e05_selection.json").write_text(json.dumps(sel))
    # E6
    ge = AB.grid_for(cfg, seeds=list(seeds))
    for v, q in zip(AB.VARIANT_NAMES, (0.30, 0.34, 0.28, 0.31, 0.27, 0.25, 0.33, 0.20)):
        for s in AB.ablate_specs(ge, variants=(v,)):
            _store(cfg, s, ablate_result(ref, folds[s["fold"]], q + 0.01 * s["seed"]), ABLATE_KEY, "ablate")
    store_loop(AB.warm_specs(ge) + AB.scratch_specs(ge))
    e6 = {"stage": "stage1", "seeds": ge["seeds"], "design_note": AB.DESIGN_NOTE,
          "ablate": AB.build_ablate(cfg, ge, ref, dict(enumerate(folds))),
          "warm_start": AB.build_warm(cfg, ge, ref)}
    save("e06_ablations", e6, cells_provenance({"ablate": ABLATE_KEY, "loop": LOOP_KEY}))
    # pool sizes of the E1 cells (the fixture's cells carry len_train_pool; add len_es for this study)
    from Curation_Paper.eswa_full.lib.cells import load_cell
    from Curation_Paper.eswa_full.lib.pool_sizes import build_pool_sizes
    res = []
    for s in jobs.e1_specs(g):
        r = dict(load_cell(s, cfg)["result"])
        r.setdefault("len_es", 100)
        res.append(r)
    save("e00_pool_sizes", {**build_pool_sizes(res), "seeds": list(seeds), "folds": g["folds"],
                            "code": {"git_sha": "x", "files": [], "dirty_files": []}}, prov)
    # real study JSONs, copied unchanged
    for name in REAL:
        shutil.copy(PKG_ROOT / "results" / f"{name}.json", out_dir / f"{name}.json")
        paths[name] = out_dir / f"{name}.json"
    # the real tune cells were run with their own key; align loop keys for the cross-study check
    return {"paths": paths, "cfg": cfg, "ref": ref, "folds": folds, "g": g, "frozen": frozen}
