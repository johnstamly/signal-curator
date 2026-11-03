"""E6: the "ablate" cell kind, the variant/warm-start job list and the aggregator (R23).
No real data; MicroConv1D fits run on synthetic signals for 2 epochs."""
import json
import math
import subprocess
import sys
import zlib

import numpy as np
import pytest

from Curation_Paper.eswa_full.lib import (ablate_study as AB, aggregate, cells, jobs, kind_ablate,
                                          kind_loop)
from Curation_Paper.eswa_full.lib.fit import DEFAULT_RECIPE, _recipe
from Curation_Paper.eswa_full.lib.protocol import REPO_ROOT, input_hashes, load_config
from Curation_Paper.eswa_full.lib.versioning import code_key

KEY = "k" * 64


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["results_dir"] = str(tmp_path)
    return c


@pytest.fixture(autouse=True)
def fixed_version(monkeypatch):
    monkeypatch.setattr(cells, "_compute_version", lambda kind: (KEY, []))
    monkeypatch.setattr(cells, "_live_key", lambda kind: KEY)
    cells.reset_versions()
    yield
    cells.reset_versions()


def _sig(sid: str, label: int, length: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    t = np.arange(length)
    return (np.sin(t / (6 if label else 11)) + 0.3 * rng.normal(size=length)).astype(np.float32)


@pytest.fixture
def world(monkeypatch):
    pool = {f"p{i}": {"signal_id": f"p{i}", "label_int": 0 if i % 9 == 0 else 1}
            for i in range(900)}
    test_ids = [f"p{i}" for i in range(0, 900, 9)][:50] + [f"p{i}" for i in range(1, 900, 9)][:50]
    ref = {k: v["label_int"] for k, v in pool.items()}
    monkeypatch.setattr(kind_ablate, "_world", lambda c: (pool, ref, [test_ids] + [[]] * 4))
    seen = []

    def spy(ids, cfg, window=None):
        w = window or cfg["window"]
        seen.append((list(ids), w))
        return np.stack([_sig(i, pool[i]["label_int"])[:w] for i in ids])
    monkeypatch.setattr(kind_ablate, "signals_for", spy)
    return pool, set(test_ids), seen


def _spec(name="default", fold=0, seed=0, extra=None):
    return AB.ablate_spec({**AB.variant_recipe(name), **(extra or {"max_epochs": 2})}, fold, seed)


# ------------------------------------------------------------------ the kind

def test_ablate_never_trains_on_test_ids(world, cfg, monkeypatch):
    pool, test, seen = world
    got = {}
    real_fit = kind_ablate.fit

    def spy_fit(name, train, es, X_of, **kw):
        got["train"], got["es"] = list(train), list(es)
        return real_fit(name, train, es, X_of, **kw)
    monkeypatch.setattr(kind_ablate, "fit", spy_fit)
    res = kind_ablate.run(_spec(), cfg)
    assert len(got["train"]) == 600 and got["es"]
    assert not ({r["signal_id"] for r in got["train"] + got["es"]} & test)
    # the only signals that include test rows are the final test-fold prediction
    for ids, _ in seen:
        if set(ids) & test:
            assert set(ids) == test
    assert set(res["test_pred"]) == test and set(res["test_ids"]) == test
    assert all(round(v, 6) == v and 0.0 <= v <= 1.0 for v in res["test_pred"].values())
    assert set(res["metrics"]) >= {"acc", "bal_acc", "bad_recall", "good_retention", "auroc", "ece"}
    assert res["epochs"] <= 2 and math.isfinite(res["es_loss"]) and res["train_seconds"] >= 0
    assert res["n_train"] == 600


def test_window_assert_fires_on_mismatched_xof(world, cfg, monkeypatch):
    _, _, seen = world

    def ignores_window(ids, cfg, window=None):          # always the config window (500)
        return np.stack([_sig(i, 1)[:500] for i in ids])
    monkeypatch.setattr(kind_ablate, "signals_for", ignores_window)
    with pytest.raises(AssertionError, match="window"):
        kind_ablate.run(_spec("window_250"), cfg)


def test_window_variants_use_their_window(world, cfg):
    _, _, seen = world
    r250 = kind_ablate.run(_spec("window_250"), cfg)
    assert {w for _, w in seen} == {250} and r250["window"] == 250
    seen.clear()
    kind_ablate.run(_spec("default"), cfg)
    assert {w for _, w in seen} == {500}


def test_check_window_helper():
    ok = kind_ablate.checked_x_of(lambda ids: np.zeros((len(ids), 250), np.float32), 250)
    assert ok(["a"]).shape == (1, 250)
    bad = kind_ablate.checked_x_of(lambda ids: np.zeros((len(ids), 500), np.float32), 250)
    with pytest.raises(AssertionError, match="window"):
        bad(["a"])


def test_ablate_kind_is_deterministic_and_variants_share_data(world, cfg, monkeypatch):
    rows = []
    real_fit = kind_ablate.fit

    def spy_fit(name, train, es, X_of, **kw):
        rows.append(([r["signal_id"] for r in train], [r["signal_id"] for r in es], kw["seed"]))
        return real_fit(name, train, es, X_of, **kw)
    monkeypatch.setattr(kind_ablate, "fit", spy_fit)
    a = kind_ablate.run(_spec("default"), cfg)
    b = kind_ablate.run(_spec("relu"), cfg)
    c = kind_ablate.run(_spec("default"), cfg)
    assert rows[0] == rows[1] == rows[2]            # same rows and fit seed: paired variants
    assert a["test_pred"] == c["test_pred"]
    assert a["test_pred"] != b["test_pred"]


def test_fold_seed_matches_kind_loop():
    assert kind_ablate.FOLD_SEED == kind_loop.FOLD_SEED == 0


def test_ablate_dependency_list_is_complete():
    code = ("import sys, json; import Curation_Paper.eswa_full.lib.kind_ablate; "
            "print(json.dumps(sorted({m.__file__ for m in list(sys.modules.values()) "
            "if getattr(m, '__file__', None)})))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout
    files = json.loads(out.strip().splitlines()[-1])
    root = str(REPO_ROOT).replace("\\", "/").lower() + "/"
    local = {f.replace("\\", "/")[len(root):] for f in files
             if f.replace("\\", "/").lower().startswith(root) and "site-packages" not in f}
    deps = set(kind_ablate.DEPENDENCY_FILES)
    assert local <= deps, sorted(local - deps)
    for d in deps:
        assert (REPO_ROOT / d).is_file(), d
    assert "Curation_Paper/eswa_full/lib/kind_loop.py" not in deps  # not a loop dependent


def test_ablate_registered_with_its_own_key():
    assert cells.KINDS["ablate"] is kind_ablate
    assert cells.kind_of({"study_source": "ablate"}) is kind_ablate
    assert kind_ablate.KIND == "ablate"


# ------------------------------------------------------------------ variants and specs

def test_variants_are_one_factor_changes_and_pass_recipe_validation():
    names = [n for n, _ in AB.VARIANTS]
    assert names == ["default", "relu", "class_weight_none", "class_weight_full", "no_augment",
                     "window_250", "window_1000", "no_zscore"]
    for name, change in AB.VARIANTS:
        r = AB.variant_recipe(name)
        assert set(r) == set(DEFAULT_RECIPE)             # no unknown or missing keys
        _recipe(r)                                       # accepted by fit
        diff = {k for k in r if r[k] != DEFAULT_RECIPE[k]}
        assert diff == set(change), name
        assert len(change) <= 1
    assert AB.variant_recipe("default") == DEFAULT_RECIPE
    assert AB.variant_recipe("relu")["activation"] == "relu"
    assert AB.variant_recipe("window_1000")["window"] == 1000
    assert AB.variant_recipe("no_zscore")["zscore"] is False


def test_ablate_spec_shape_and_default_id_stable():
    s = AB.ablate_spec(AB.variant_recipe("default"), 0, 0)
    assert s["study_source"] == "ablate" and s["pool"] == "al" and s["strategy"] == "ablate"
    assert s["eta"] == 0.0 and s["arch"] == "microconv1d" and s["budgets"] == [600]
    assert s["seed_size"] == 0 and s["batch"] == 0
    assert cells.cell_id(s) == "239706d6f6d8fe3e"
    assert cells.cell_id(s) == cells.cell_id(AB.ablate_spec(dict(DEFAULT_RECIPE), 0, 0))


def test_cell_counts_and_unique_ids():
    cfg = load_config()
    g = AB.grid_for(cfg)
    ab = AB.ablate_specs(g)
    assert len(ab) == 8 * 5 * 5 == 200
    warm, scratch = AB.warm_specs(g), AB.scratch_specs(g)
    assert len(warm) == len(scratch) == 25
    ids = [cells.cell_id(s) for s in ab + warm]
    assert len(ids) == len(set(ids))
    assert all(s["study_source"] == "loop" for s in warm)
    j = AB.job_payload(cfg, g)
    assert j["counts"] == {"ablate": 200, "warm_start_loop": 25, "scratch_shared": 25,
                           "total_new": 225, "total": 250}
    assert len(j["specs"]) == 250


def test_warm_specs_differ_from_e3_only_by_the_recipe():
    cfg = load_config()
    g = AB.grid_for(cfg)
    e3 = {(s["seed"], s["fold"]): s for s in jobs.e3_specs(jobs.grid(cfg))
          if s["strategy"] == "unc_oracle" and s["eta"] == 0.0}
    assert len(e3) == 25
    for w in AB.warm_specs(g):
        base = e3[(w["seed"], w["fold"])]
        assert {k for k in w if w[k] != base[k]} == {"recipe"}
        assert w["recipe"] == {**DEFAULT_RECIPE, "warm_start": True}
        assert w["strategy"] == "unc_oracle" and w["pool"] == "al" and w["eta"] == 0.0
        assert w["budgets"] == [100, 200, 300, 400, 500, 600]
        assert w["seed_size"] == 100 and w["batch"] == 50
    for sc in AB.scratch_specs(g):
        assert cells.cell_id(sc) == cells.cell_id(e3[(sc["seed"], sc["fold"])])


def test_e06_job_file_is_committed_and_current():
    cfg = load_config()
    stored = json.loads((REPO_ROOT / "Curation_Paper/eswa_full/results/jobs/e06_ablations.json")
                        .read_text(encoding="utf-8"))
    now = AB.job_payload(cfg, AB.grid_for(cfg))
    assert [cells.cell_id(s) for s in stored["specs"]] == [cells.cell_id(s) for s in now["specs"]]
    assert stored["counts"] == now["counts"]


def test_loop_code_key_unchanged_by_task_11():
    """Task 11 must not touch the loop dependency files (R17/R18).  R37 (loop.py) moved the key
    from 193b514622b7 to 84a02678827a."""
    assert code_key(kind_loop.DEPENDENCY_FILES, REPO_ROOT).startswith("84a02678827a")


# ------------------------------------------------------------------ E7 verification

def test_e07_reads_the_documented_cells():
    cfg = load_config()
    g = jobs.grid(cfg)
    specs = jobs.e7_specs(g)
    stored = {cells.cell_id(s) for s in json.loads(
        (REPO_ROOT / "Curation_Paper/eswa_full/results/jobs/stage1_loop.json")
        .read_text(encoding="utf-8"))["specs"]}
    assert {cells.cell_id(s) for s in specs} <= stored
    assert {(s["seed_size"], s["batch"]) for s in specs} == {
        (a, b) for a in (50, 100, 200) for b in (25, 50, 100)}
    assert all(s["strategy"] == "unc_oracle" and s["eta"] == 0.0 and s["pool"] == "al"
               and s["arch"] == "microconv1d" and s["recipe"] == DEFAULT_RECIPE for s in specs)
    for s in specs:
        if (s["seed_size"], s["batch"]) != (100, 50):
            assert max(s["budgets"]) == 400 and s["budgets"][0] == s["seed_size"]
        else:                                   # the E3 eta-0 cells, truncated at 400 on reading
            assert s["budgets"] == [100, 200, 300, 400, 500, 600]
    assert aggregate.STUDIES["e07"][0] is jobs.e7_specs
    assert jobs.E7_MAX_BUDGET == 400 and jobs.E7_BASE == (100, 50)


# ------------------------------------------------------------------ aggregation

def _store(cfg, spec, result, key=KEY, kind="ablate", dirty=False):
    cells.cell_path(spec, cfg).write_text(json.dumps({
        "spec": cells.canonical_spec(spec), "result": result,
        "meta": {"code_key": key, "dirty": dirty, "kind": kind, "git_sha": "x",
                 "input_hashes": input_hashes(cfg)}}))


def _fake_ref(n=40):
    return {f"r{i}": (0 if i % 4 == 0 else 1) for i in range(n)}


def _ablate_result(ref, test_ids, quality):
    rng = np.random.default_rng(len(test_ids) + int(quality * 1000))
    p = {s: float(np.clip(0.5 + (0.5 if ref[s] else -0.5) * quality + rng.normal(0, 0.1), 0.01, 0.99))
         for s in test_ids}
    from Curation_Paper.eswa_full.lib.metrics import metrics
    m = metrics(np.array([ref[s] for s in test_ids]), np.array([p[s] for s in test_ids]))
    return {"test_ids": list(test_ids), "test_pred": {s: round(v, 6) for s, v in p.items()},
            "metrics": m, "epochs": 5, "es_loss": 0.4, "train_seconds": 1.0, "n_train": 600,
            "n_es": 10, "len_train_pool": 700, "window": 500}


def test_build_ablate_pooled_and_paired_differences(cfg):
    ref = _fake_ref()
    ids = sorted(ref)
    folds = [ids[i::2] for i in range(2)]
    g = AB.grid_for(cfg, seeds=[0, 1, 2], folds=[0, 1], variants=("default", "relu"))
    for v, q in (("default", 0.30), ("relu", 0.45)):
        for s in AB.ablate_specs(g, variants=(v,)):
            _store(cfg, s, _ablate_result(ref, folds[s["fold"]], q + 0.01 * s["seed"]))
    out = AB.build_ablate(cfg, g, ref, {0: folds[0], 1: folds[1]}, variants=("default", "relu"))
    d, r = out["variants"]["default"], out["variants"]["relu"]
    assert d["metrics"]["auroc"]["n"] == 3 and len(d["metrics"]["auroc"]["values"]) == 3
    assert r["metrics"]["bal_acc"]["mean"] >= d["metrics"]["bal_acc"]["mean"]
    diff = r["paired_vs_default"]["auroc"]
    manual = [a - b for a, b in zip(r["metrics"]["auroc"]["values"], d["metrics"]["auroc"]["values"])]
    assert diff["values"] == pytest.approx(manual) and diff["mean"] == pytest.approx(np.mean(manual))
    assert "paired_vs_default" not in d or d["paired_vs_default"] is None
    assert r["paired_cells_vs_default"]["auroc"]["n"] == 6      # 2 folds x 3 seeds
    assert out["n_scored"] == len(ref) and out["partial_folds"] is False


def test_build_ablate_refuses_incomplete_coverage(cfg):
    ref = _fake_ref()
    ids = sorted(ref)
    g = AB.grid_for(cfg, seeds=[0], folds=[0], variants=("default",))
    s = AB.ablate_specs(g, variants=("default",))[0]
    _store(cfg, s, _ablate_result(ref, ids[:10], 0.3))
    with pytest.raises(ValueError, match="test_ids differ"):
        AB.build_ablate(cfg, g, ref, {0: ids[:20]}, variants=("default",))


def test_require_by_kind_refuses_missing_and_stale(cfg):
    g = AB.grid_for(cfg, seeds=[0], folds=[0], variants=("default", "relu"))
    specs = AB.ablate_specs(g, variants=("default", "relu"))
    class A:           # argparse stand-in
        expect_key = None
        allow_dirty = False
    out = AB.require_by_kind(specs, cfg, A)
    assert isinstance(out, tuple) and out[0] == 2
    for s in specs:
        _store(cfg, s, {"test_ids": [], "test_pred": {}}, key="a" * 64)
    _store(cfg, specs[1], {"test_ids": [], "test_pred": {}}, key="b" * 64)
    out = AB.require_by_kind(specs, cfg, A)
    assert out[0] == 5 and "re-run" in out[1]
    _store(cfg, specs[1], {"test_ids": [], "test_pred": {}}, key="a" * 64, dirty=True)
    out = AB.require_by_kind(specs, cfg, A)
    assert out[0] == 5 and "dirty" in out[1]
