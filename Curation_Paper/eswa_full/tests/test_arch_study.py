"""E4 tuning kind, freeze, compare job list, E5 selection (no real simulate calls)."""
import json
import math
import zlib

import numpy as np
import pytest

from Curation_Paper.eswa_full.lib import (arch_cli, arch_study as A, cells, classic, jobs,
                                          kind_loop, kind_tune)
from Curation_Paper.eswa_full.lib.fit import DEFAULT_RECIPE
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


# ---------------------------------------------------------------- synthetic world

def _sig(sid: str, label: int) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    t = np.arange(500)
    return (np.sin(t / (6 if label else 11)) + 0.3 * rng.normal(size=500)).astype(np.float32)


@pytest.fixture
def world(monkeypatch, cfg):
    pool = {f"p{i}": {"signal_id": f"p{i}", "label_int": 0 if i % 9 == 0 else 1}
            for i in range(900)}
    test_ids = [f"p{i}" for i in range(0, 900, 9)][:50] + [f"p{i}" for i in range(1, 900, 9)][:50]
    monkeypatch.setattr(kind_tune, "_world", lambda c: (pool, test_ids))
    seen = []

    def spy(ids, c, window=None):
        seen.append(list(ids))
        return np.stack([_sig(i, pool[i]["label_int"]) for i in ids])
    monkeypatch.setattr(kind_tune, "signals_for", spy)
    return pool, set(test_ids), seen


def _spec(arch="logreg", recipe=None, seed=0):
    return A.tune_spec(arch, recipe if recipe is not None else A.candidate_recipe(), seed)


# ---------------------------------------------------------------- tune kind

def test_tune_never_reads_test_fold_signals_and_records_es_quantities(world, cfg):
    pool, test, seen = world
    res = kind_tune.run(_spec("logreg"), cfg)
    asked = {i for ids in seen for i in ids}
    assert asked and not (asked & test), "a test-fold signal was requested"
    assert math.isfinite(res["es_loss"]) and res["es_loss"] > 0
    assert 0.5 < res["es_auroc"] <= 1.0          # the synthetic classes are separable
    assert res["n_train"] == kind_tune.TUNE_N == 600
    assert res["n_params"] is None and res["device"] == "cpu"
    assert set(res) >= {"es_loss", "es_auroc", "epochs", "train_seconds", "n_params", "device"}


def test_tune_torch_cell_reports_params(world, cfg):
    _, test, seen = world
    r = A.candidate_recipe(max_epochs=2)
    res = kind_tune.run(_spec("microconv1d", r), cfg)
    assert res["n_params"] == 2529 and res["epochs"] <= 2 and 0.0 <= res["es_auroc"] <= 1.0
    assert not ({i for ids in seen for i in ids} & test)


def test_tune_subsample_is_seeded_and_stratified(world):
    pool, _, _ = world
    rows = list(pool.values())
    a = kind_tune.subsample_stratified(rows, 600, 0)
    assert a == kind_tune.subsample_stratified(list(reversed(rows)), 600, 0)
    assert a != kind_tune.subsample_stratified(rows, 600, 1)
    labels = [r["label_int"] for r in a]
    assert len(a) == 600 and 0 in labels and 1 in labels
    assert abs(labels.count(0) / 600 - sum(1 for r in rows if r["label_int"] == 0) / len(rows)) < 0.01


def test_tune_rejects_wrong_fold_and_oversized_pool(world, cfg):
    with pytest.raises(ValueError):
        kind_tune.run({**_spec(), "fold": 1}, cfg)
    with pytest.raises(ValueError):
        kind_tune.subsample_stratified([{"signal_id": "a", "label_int": 0},
                                        {"signal_id": "b", "label_int": 1}], 600, 0)


def test_tune_fold_zero_equals_loop_fold_zero():
    cfg = load_config()
    assert kind_tune.FOLD_SEED == kind_loop.FOLD_SEED
    _, test_ids = kind_tune._world(cfg)
    assert test_ids == kind_loop._world(cfg)[2][0]


def test_tune_kind_is_registered_and_runs_through_the_runner(world, cfg, monkeypatch):
    assert cells.KINDS["tune"] is kind_tune
    p = cells.run_cell(_spec("logreg"), cfg)
    payload = json.loads(p.read_text())
    assert payload["meta"]["kind"] == "tune" and payload["result"]["es_loss"] > 0
    assert cells.run_cell(_spec("logreg"), cfg) == p  # skipped when present


def test_tune_dependencies_are_loop_list_minus_loop_files():
    deps = set(kind_tune.DEPENDENCY_FILES)
    assert "Curation_Paper/eswa_full/lib/kind_tune.py" in deps
    assert not deps & {"Curation_Paper/eswa_full/lib/loop.py",
                       "Curation_Paper/eswa_full/lib/kind_loop.py"}
    assert deps - {"Curation_Paper/eswa_full/lib/kind_tune.py"} <= set(kind_loop.DEPENDENCY_FILES)
    code_key(kind_tune.DEPENDENCY_FILES, REPO_ROOT)  # every file exists


def test_leakage_note_is_in_kind_docstring_and_frozen_payload(cfg):
    assert "TEST rows of folds 1-4" in kind_tune.__doc__
    assert "TEST rows of folds 1-4" in A.freeze_payload([], cfg)["tuning_leakage_note"]


# ---------------------------------------------------------------- classic knobs (R21)

def _classic_data():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=300) < 0.7).astype(int)
    X = np.stack([_sig(f"c{i}", int(l)) for i, l in enumerate(y)])
    X = X + 0.8 * rng.normal(size=X.shape).astype(np.float32)
    return X[:200], y[:200], X[200:], y[200:]


@pytest.mark.parametrize("name,explicit,changed", [
    ("hgb", {"learning_rate": 0.1, "max_leaf_nodes": 31}, {"learning_rate": 0.02, "max_leaf_nodes": 4}),
    ("logreg", {"C": 1.0}, {"C": 0.001})])
def test_classic_knobs_default_unchanged_and_non_default_changes_model(name, explicit, changed):
    Xtr, ytr, Xes, yes = _classic_data()

    def pred(cfg_):
        m, _ = classic.fit_classic(name, Xtr, ytr, Xes, yes, seed=0, recipe={"model_cfg": cfg_})
        return m.predict_proba_good(Xes)
    base = pred({})
    assert np.array_equal(base, pred(explicit))
    assert not np.allclose(base, pred(changed))


def test_classic_rejects_unknown_knobs():
    Xtr, ytr, Xes, yes = _classic_data()
    with pytest.raises(ValueError):
        classic.fit_classic("logreg", Xtr, ytr, Xes, yes, seed=0, recipe={"model_cfg": {"width": 2}})


def test_loop_code_key_value_recorded():
    # R21 changed classic.py once, pre-launch: the key moved from 90105f510bd9 to this value.
    assert code_key(kind_loop.DEPENDENCY_FILES, REPO_ROOT).startswith(NEW_LOOP_KEY)


NEW_LOOP_KEY = "84a02678827a"  # R37 (loop.py) moved it from 193b514622b7


# ---------------------------------------------------------------- grid

def test_default_grid_sizes_and_bands():
    logs = []
    j = A.tune_jobs(log=logs.append)
    assert not logs, f"default grid dropped a config: {logs}"
    n = {a: len(c) for a, c in j["configs"].items()}
    assert n == {"microconv1d": 6, "cnn_wide": 6, "resnet1d": 6, "bigru": 6, "transformer": 6,
                 "hgb": 6, "logreg": 5, "minirocket": 3}
    assert j["counts"]["total"] == 3 * sum(n.values()) == 132
    assert all(len(c) <= 6 for c in j["configs"].values())
    for a, cs in j["configs"].items():
        if a in A.BANDS:
            lo, hi = A.BANDS[a]
            assert all(lo <= c["n_params"] <= hi for c in cs)
    ids = [cells.cell_id(s) for s in j["specs"]]
    assert len(ids) == len(set(ids))
    assert {s["seed"] for s in j["specs"]} == {0, 1, 2}
    assert all(s["fold"] == 0 and s["pool"] == "al" and s["budgets"] == [600]
               and s["strategy"] == "tune" and s["eta"] == 0 and s["seed_size"] == 0
               and s["batch"] == 0 for s in j["specs"])


def test_out_of_band_configs_are_dropped_and_logged():
    logs = []
    cs = A.candidate_configs("cnn_wide", widths=(1.0, 1.5), log=logs.append)
    assert len(cs) == 3 and all(c["n_params"] == 28273 for c in cs)
    assert len(logs) == 1 and "width=1.5" in logs[0] and "DROPPED" in logs[0]


def test_microconv_grid_is_lr_by_activation_with_equal_params():
    mc = A.candidate_configs("microconv1d")
    assert [c["name"] for c in mc] == [
        "lr=0.001|act=leaky", "lr=0.0003|act=leaky", "lr=0.0001|act=leaky",
        "lr=0.001|act=relu", "lr=0.0003|act=relu", "lr=0.0001|act=relu"]
    assert {c["n_params"] for c in mc} == {2529}
    assert mc[1]["recipe"] == DEFAULT_RECIPE
    assert mc[4]["recipe"]["activation"] == "relu" and mc[4]["recipe"]["lr"] == 3e-4


def test_default_candidates_equal_default_recipe_for_every_model():
    for arch in ("hgb", "logreg", "minirocket"):
        recipes = [c["recipe"] for c in A.candidate_configs(arch)]
        assert recipes.count(DEFAULT_RECIPE) == 1, arch
    hg = {c["name"]: c["recipe"]["model_cfg"] for c in A.candidate_configs("hgb")}
    assert hg["learning_rate=0.05|max_leaf_nodes=15"] == {"learning_rate": 0.05, "max_leaf_nodes": 15}
    assert {c["name"] for c in A.candidate_configs("logreg")} == {
        f"C={c:g}" for c in (0.01, 0.1, 1.0, 10.0, 100.0)}


def test_stage1_loop_job_ids_unchanged_by_the_new_knobs():
    j = jobs.stage1_jobs(load_config())
    assert j["counts"]["total"] == 550
    stored = json.loads((REPO_ROOT / "Curation_Paper/eswa_full/results/jobs/stage1_loop.json")
                        .read_text(encoding="utf-8"))["specs"]
    assert {cells.cell_id(s) for s in stored} == {cells.cell_id(s) for s in j["all"]}


# ---------------------------------------------------------------- freeze

def _store(cfg, spec, result, key=KEY):
    p = cells.cell_path(spec, cfg)
    p.write_text(json.dumps({"spec": cells.canonical_spec(spec), "result": result,
                             "meta": {"code_key": key, "dirty": False, "kind": "tune",
                                      "git_sha": "x", "input_hashes": input_hashes(cfg)}}))


def _tune_result(es, n_params=None, auroc=0.9):
    return {"es_loss": es, "es_auroc": auroc, "epochs": 10, "train_seconds": 1.0,
            "n_params": n_params, "device": "cpu"}


def _small_tune_job(cfg, losses, aurocs=None):
    """losses {(arch, config index): [es per seed]} -> jobs dict, cells stored.
    cnn_wide: 4 configs (lr 1e-3/3e-4 x width 1/1.15; index 1 is the default); logreg: C 1 and 10."""
    j = A.tune_jobs(archs=("cnn_wide", "logreg"), seeds=(0, 1, 2), lrs=(1e-3, 3e-4),
                    widths=(1.0, 1.15), cs=(1.0, 10.0), log=lambda m: None)
    for arch, cs in j["configs"].items():
        for ci, c in enumerate(cs):
            for si, seed in enumerate((0, 1, 2)):
                au = (aurocs or {}).get((arch, ci), [0.9] * 3)[si]
                _store(cfg, A.tune_spec(arch, c["recipe"], seed),
                       _tune_result(losses[(arch, ci)][si], c["n_params"], au))
    return j


def _flat(best_cnn=None):
    losses = {("logreg", 0): [0.3] * 3, ("logreg", 1): [0.35] * 3}
    for ci in range(4):
        losses[("cnn_wide", ci)] = [0.5 + 0.1 * ci] * 3
    losses.update(best_cnn or {})
    return losses


def test_freeze_picks_lowest_mean_es_loss_and_keeps_all_candidates(cfg):
    j = _small_tune_job(cfg, _flat({("cnn_wide", 2): [0.20, 0.60, 0.10]}))  # mean 0.30
    out = A.freeze_payload(j["specs"], cfg)
    fz = out["models"]["cnn_wide"]
    assert len(fz["candidates"]) == 4 and fz["selection_metric"] == "es_loss"
    assert fz["frozen"]["recipe"] == j["configs"]["cnn_wide"][2]["recipe"]
    assert fz["frozen"]["es_loss_mean"] == pytest.approx(0.30)
    assert fz["frozen"]["es_loss_sd"] == pytest.approx(np.std([0.2, 0.6, 0.1], ddof=1))
    assert all(c["n_seeds"] == 3 and len(c["es_loss_values"]) == 3
               and len(c["es_auroc_values"]) == 3 for c in fz["candidates"])
    assert fz["frozen"]["size"] == 36930 and fz["frozen"]["size_unit"] == "parameters"
    assert out["models"]["logreg"]["frozen"]["recipe"] == DEFAULT_RECIPE
    assert out["models"]["logreg"]["frozen"]["size_unit"] == "features"


def test_freeze_tie_prefers_default_then_smaller_model(cfg):
    flat = {("cnn_wide", ci): [0.4] * 3 for ci in range(4)}
    j = _small_tune_job(cfg, _flat(flat))
    fz = A.freeze_payload(j["specs"], cfg)["models"]["cnn_wide"]["frozen"]
    assert fz["recipe"] == DEFAULT_RECIPE and fz["equals_default"] is True
    # default clearly worse: among the tied others the smaller model (28,273 < 36,930 params)
    flat[("cnn_wide", 1)] = [0.9] * 3
    j = _small_tune_job(cfg, _flat(flat))
    fz = A.freeze_payload(j["specs"], cfg)["models"]["cnn_wide"]["frozen"]
    assert fz["n_params"] == 28273 and fz["equals_default"] is False
    assert fz["recipe"] == j["configs"]["cnn_wide"][0]["recipe"]


def test_freeze_one_standard_error_rule_prefers_default(cfg):
    # best mean 0.30 (sd 0.1, SE 0.0577); default 0.34 is within one SE -> default wins
    flat = {("cnn_wide", 0): [0.2, 0.3, 0.4], ("cnn_wide", 1): [0.34] * 3,
            ("cnn_wide", 2): [0.6] * 3, ("cnn_wide", 3): [0.7] * 3}
    j = _small_tune_job(cfg, _flat(flat))
    assert A.freeze_payload(j["specs"], cfg)["models"]["cnn_wide"]["frozen"]["recipe"] == DEFAULT_RECIPE
    flat[("cnn_wide", 1)] = [0.45] * 3     # outside one SE -> the best config wins
    j = _small_tune_job(cfg, _flat(flat))
    assert A.freeze_payload(j["specs"], cfg)["models"]["cnn_wide"]["frozen"]["recipe"] == \
        j["configs"]["cnn_wide"][0]["recipe"]


def test_minirocket_is_ranked_by_es_auroc_not_es_loss(cfg):
    j = A.tune_jobs(archs=("minirocket",), seeds=(0, 1, 2), log=lambda m: None)
    cs = j["configs"]["minirocket"]       # kernels 2500, 5000, 10000
    loss = {0: 0.9, 1: 0.8, 2: 0.2}       # es_loss prefers 10000 (also the default)
    auc = {0: 0.95, 1: 0.80, 2: 0.70}     # es AUROC prefers 2500
    for ci, c in enumerate(cs):
        for seed in (0, 1, 2):
            _store(cfg, A.tune_spec("minirocket", c["recipe"], seed),
                   _tune_result(loss[ci], None, auc[ci]))
    out = A.freeze_payload(j["specs"], cfg)["models"]["minirocket"]
    assert out["selection_metric"] == "es_auroc"
    assert out["frozen"]["recipe"]["minirocket_kernels"] == 2500
    assert out["frozen"]["size"] == 2500 and out["frozen"]["size_unit"] == "kernels"


def test_freeze_refuses_when_a_tune_cell_is_missing(cfg, tmp_path, capsys):
    j = _small_tune_job(cfg, _flat())
    jobs_file = tmp_path / "tune.json"
    jobs_file.write_text(json.dumps(j))
    cells.cell_path(j["specs"][0], cfg).unlink()
    rc = arch_cli.freeze_main(["--results-dir", str(tmp_path), "--jobs", str(jobs_file),
                               "--allow-dirty"])
    err = capsys.readouterr().err
    assert rc == 2 and "REFUSING" in err and cells.cell_id(j["specs"][0]) in err
    assert "run_cells.py" in err
    assert not (tmp_path / "e04_frozen_configs.json").exists()


def test_freeze_refuses_mixed_code_keys(cfg, tmp_path, capsys):
    j = _small_tune_job(cfg, _flat())
    _store(cfg, j["specs"][0], _tune_result(0.3), key="o" * 64)
    jobs_file = tmp_path / "tune.json"
    jobs_file.write_text(json.dumps(j))
    rc = arch_cli.freeze_main(["--results-dir", str(tmp_path), "--jobs", str(jobs_file)])
    assert rc == 5 and "different code keys" in capsys.readouterr().err


def test_freeze_writes_file_via_cli(cfg, tmp_path):
    flat = {("cnn_wide", ci): [0.4 + 0.01 * ci] * 3 for ci in range(4)}
    j = _small_tune_job(cfg, _flat(flat))
    jobs_file = tmp_path / "tune.json"
    jobs_file.write_text(json.dumps(j))
    assert arch_cli.freeze_main(["--results-dir", str(tmp_path), "--jobs", str(jobs_file),
                                 "--allow-dirty"]) == 0
    d = json.loads((tmp_path / "e04_frozen_configs.json").read_text())
    assert set(d["models"]) == {"cnn_wide", "logreg"} and d["meta"]["git_sha"]
    assert d["models"]["cnn_wide"]["frozen"]["name"] == "lr=0.001|width=1"
    assert "TEST rows of folds 1-4" in d["tuning_leakage_note"] and d["selection_rule"]


# ---------------------------------------------------------------- compare job list

def _frozen(microconv_lr=None, auroc=None, dirty=0):
    models = {}
    for m in A.MODELS:
        r = A.candidate_recipe(lr=microconv_lr) if (m == "microconv1d" and microconv_lr) \
            else A.candidate_recipe()
        size, unit = A.model_size(m, r, 2529 if m == "microconv1d" else 1000)
        models[m] = {"frozen": {"name": "x", "recipe": r, "n_params": None, "size": size,
                                "size_unit": unit, "es_loss_mean": 0.1, "es_loss_sd": 0.0,
                                "es_auroc_mean": (auroc or {}).get(m, 0.5), "es_auroc_sd": 0.0,
                                "equals_default": r == DEFAULT_RECIPE}}
    return {"models": models, "cells_dirty": dirty, "cells_code_key": KEY}


def test_compare_microconv_dedups_with_e1_when_recipe_is_default(cfg):
    per = A.compare_specs(cfg, _frozen())
    assert set(per) == set(A.MODELS) and len(A.MODELS) == 8
    shared = A.shared_with_e1(cfg, per)
    assert shared["microconv1d"] == len(per["microconv1d"]) == 25
    assert all(shared[m] == 0 for m in A.MODELS if m != "microconv1d")
    e1 = {cells.cell_id(s) for s in jobs.e1_specs(jobs.grid(cfg)) if s["pool"] == "al"
          and s["strategy"] == "random"}
    assert {cells.cell_id(s) for s in per["microconv1d"]} == e1
    for ss in per.values():
        assert all(s["strategy"] == "random" and s["pool"] == "al" and s["eta"] == 0
                   and s["budgets"] == [100, 200, 300, 400, 500, 600] and s["seed_size"] == 100
                   and s["batch"] == 50 for s in ss)
        assert {s["fold"] for s in ss} == set(range(5)) and {s["seed"] for s in ss} == set(range(5))


def test_compare_microconv_does_not_dedup_when_tuned(cfg):
    per = A.compare_specs(cfg, _frozen(microconv_lr=1e-3))
    assert A.shared_with_e1(cfg, per)["microconv1d"] == 0


def test_compare_jobs_cli_writes_list_and_reports_dedup(cfg, tmp_path, capsys):
    (tmp_path / "e04_frozen_configs.json").write_text(json.dumps(_frozen()))
    args = ["--results-dir", str(tmp_path), "--allow-untracked"]
    assert arch_cli.compare_jobs_main(args) == 0
    out = json.loads((tmp_path / "jobs" / "e04_compare.json").read_text())
    assert out["counts"]["total"] == 8 * 25 and out["shared_with_e1"]["microconv1d"] == 25
    assert "-> dedup" in capsys.readouterr().out
    assert arch_cli.compare_jobs_main(args + ["--seeds", "stage2"]) == 0
    s2 = json.loads((tmp_path / "jobs" / "e04_compare_s20.json").read_text())
    assert s2["counts"]["total"] == 8 * 100


def test_compare_jobs_refuses_untracked_frozen_file(tmp_path, capsys):
    (tmp_path / "e04_frozen_configs.json").write_text(json.dumps(_frozen()))
    assert arch_cli.compare_jobs_main(["--results-dir", str(tmp_path)]) == 7
    assert "not tracked in git" in capsys.readouterr().err
    assert not (tmp_path / "jobs").exists()


def test_git_tracked_clean_helper(tmp_path):
    assert arch_cli._git_tracked_clean(REPO_ROOT / "Curation_Paper/eswa_full/lib/versioning.py")
    f = tmp_path / "x.json"
    f.write_text("{}")
    assert not arch_cli._git_tracked_clean(f)


def test_compare_jobs_refuses_without_frozen_file(tmp_path, capsys):
    assert arch_cli.compare_jobs_main(["--results-dir", str(tmp_path)]) == 6
    assert "e04_freeze.py" in capsys.readouterr().err


def test_smoke_overrides_need_results_dir(capsys):
    assert arch_cli.compare_jobs_main(["--seed-list", "0"]) == 4
    assert arch_cli.compare_jobs_main(["--allow-untracked"]) == 4
    assert arch_cli.tune_jobs_main(["--max-epochs", "2"]) == 4


# ---------------------------------------------------------------- compare aggregation

IDS = [f"t{i}" for i in range(10)]
REF = {s: (0 if i < 3 else 1) for i, s in enumerate(IDS)}


def _loop_result(test_ids, budgets, secs, seed):
    rng = np.random.default_rng(seed)
    recs = []
    for b, cum in zip(budgets, secs):
        recs.append({"budget": b, "metrics": {}, "test_pred": {s: float(rng.uniform()) for s in test_ids},
                     "n_flipped": 0, "n_flagged": 0, "n_recovered": 0, "n_ever_recovered": 0,
                     "n_false_flags": 0, "n_residual_flips": 0, "selection_claim_ok": True,
                     "train_seconds": cum})
    return {"test_ids": list(test_ids), "len_train_pool": 700, "budgets": recs}


def _store_loop(cfg, spec, result):
    p = cells.cell_path(spec, cfg)
    p.write_text(json.dumps({"spec": cells.canonical_spec(spec), "result": result,
                             "meta": {"code_key": KEY, "dirty": False, "kind": "loop",
                                      "git_sha": "x", "input_hashes": input_hashes(cfg)}}))


def test_build_compare_flags_calibration_size_and_fit_seconds(cfg):
    g = jobs.grid(cfg, seeds=[0, 1], budgets=[100, 200])
    frozen = _frozen()
    for m in ("microconv1d", "minirocket"):
        for s in A.compare_specs(cfg, frozen, seeds=[0, 1], models=(m,), budgets=[100, 200])[m]:
            _store_loop(cfg, s, _loop_result(IDS[2 * s["fold"]:2 * s["fold"] + 2], [100, 200],
                                             [3.0, 8.0], s["seed"] * 10 + s["fold"]))
    out = A.build_compare(cfg, g, REF, frozen, models=("microconv1d", "minirocket"))
    assert set(out["models"]) == {"microconv1d", "minirocket"}
    assert out["models"]["microconv1d"]["calibrated"] is True
    mr = out["models"]["minirocket"]
    assert mr["calibrated"] is False and "not comparable" in mr["calibration_note"]
    assert (mr["size"], mr["size_unit"]) == (10000, "kernels")
    mcv = out["models"]["microconv1d"]
    assert (mcv["size"], mcv["size_unit"]) == (2529, "parameters")
    rows = mcv["rows"]
    assert [r["budget"] for r in rows] == [100, 200] and rows[0]["n_seeds"] == 2
    assert rows[0]["train_seconds_per_fit"]["mean"] == pytest.approx(3.0)
    assert rows[1]["train_seconds_per_fit"]["mean"] == pytest.approx(5.0)   # 8 - 3
    assert mcv["train_seconds_per_fit_mean"] == pytest.approx(4.0)
    assert A.model_size("hgb", DEFAULT_RECIPE, None) == (18, "features")


# ---------------------------------------------------------------- E5 selection (es AUROC only)

def test_e05_selection_by_es_auroc_excludes_microconv_and_never_reads_test_metrics():
    au = {"microconv1d": 0.99, "cnn_wide": 0.80, "resnet1d": 0.91, "bigru": 0.85,
          "transformer": 0.70, "hgb": 0.88, "logreg": 0.60, "minirocket": 0.89}
    fz = _frozen(auroc=au)
    for v in fz["models"].values():           # the selection must not need any compare data
        v.pop("rows", None)
    sel = A.select_top(fz)
    assert [s["arch"] for s in sel] == ["resnet1d", "minirocket", "hgb"]
    assert sel[0]["es_auroc_mean"] == 0.91 and sel[0]["recipe"] == DEFAULT_RECIPE


def test_e05_selection_nan_last_and_ties_by_name():
    au = {"microconv1d": 0.9, "cnn_wide": float("nan"), "resnet1d": 0.8, "bigru": 0.8,
          "transformer": 0.8, "hgb": 0.7, "logreg": 0.6, "minirocket": 0.5}
    assert [s["arch"] for s in A.select_top(_frozen(auroc=au))] == ["bigru", "resnet1d", "transformer"]
    au["cnn_wide"] = 0.95
    assert A.select_top(_frozen(auroc=au))[0]["arch"] == "cnn_wide"


def test_e05_baseline_uses_frozen_microconv_recipe(cfg):
    sel = A.selection_payload(_frozen())
    assert sel["microconv1d"]["frozen_equals_default"] is True
    sel2 = A.selection_payload(_frozen(microconv_lr=1e-3))
    assert sel2["microconv1d"]["frozen_equals_default"] is False
    assert sel2["microconv1d"]["recipe"]["lr"] == 1e-3

    e3 = {cells.cell_id(s) for s in jobs.e3_specs(jobs.grid(cfg))}
    base_default = {cells.cell_id(s) for s in A.e05_required_specs(cfg, sel)
                    if s["arch"] == "microconv1d"}
    assert len(base_default) == 25 and base_default <= e3
    base_tuned = {cells.cell_id(s) for s in A.e05_required_specs(cfg, sel2)
                  if s["arch"] == "microconv1d"}
    assert len(base_tuned) == 25 and not base_tuned & e3


def test_e05_specs_are_unc_oracle_loop_cells(cfg):
    sel = A.selection_payload(_frozen(auroc={"bigru": 0.9, "hgb": 0.8, "logreg": 0.7}))
    assert [s["arch"] for s in sel["selected"]] == ["bigru", "hgb", "logreg"]
    per = A.e05_specs(cfg, sel["selected"])
    for ss in per.values():
        assert len(ss) == 25
        assert all(s["strategy"] == "unc_oracle" and s["pool"] == "al" and s["eta"] == 0
                   and s["study_source"] == "loop" and s["seed_size"] == 100 and s["batch"] == 50
                   and s["budgets"] == [100, 200, 300, 400, 500, 600] for s in ss)
    assert len(A.e05_required_specs(cfg, sel)) == 100


def test_e05_jobs_cli_writes_selection_and_stage2_reuses_it(cfg, tmp_path, capsys):
    au = {"bigru": 0.9, "hgb": 0.8, "logreg": 0.7}
    (tmp_path / "e04_frozen_configs.json").write_text(json.dumps(_frozen(auroc=au)))
    flags = ["--results-dir", str(tmp_path), "--allow-untracked"]
    assert arch_cli.e05_jobs_main(flags) == 0
    sel = json.loads((tmp_path / "e05_selection.json").read_text())
    assert [s["arch"] for s in sel["selected"]] == ["bigru", "hgb", "logreg"]
    assert sel["microconv1d"]["frozen_equals_default"] is True and sel["meta"]["git_sha"]
    # Stage 2 re-reads the stored selection even if the frozen file changes afterwards
    (tmp_path / "e04_frozen_configs.json").write_text(
        json.dumps(_frozen(auroc={"cnn_wide": 0.99, "resnet1d": 0.98, "transformer": 0.97})))
    assert arch_cli.e05_jobs_main(flags + ["--seeds", "stage2"]) == 0
    s2 = json.loads((tmp_path / "jobs" / "e05_arch_loop_s20.json").read_text())
    assert [s["arch"] for s in s2["selected"]] == ["bigru", "hgb", "logreg"]
    assert s2["counts"]["total"] == 4 * 100
    assert arch_cli.e05_jobs_main(["--results-dir", str(tmp_path), "--seeds", "stage2"]) == 7


def test_e05_jobs_refuses_dirty_frozen_and_missing_stage1_selection(tmp_path, capsys):
    (tmp_path / "e04_frozen_configs.json").write_text(json.dumps(_frozen(dirty=3)))
    assert arch_cli.e05_jobs_main(["--results-dir", str(tmp_path), "--allow-untracked"]) == 5
    assert "dirty tune cells" in capsys.readouterr().err
    assert arch_cli.e05_jobs_main(["--results-dir", str(tmp_path), "--allow-untracked",
                                   "--seeds", "stage2"]) == 6


def test_e05_aggregator_refuses_without_selection_file(tmp_path, capsys):
    assert arch_cli.e05_main(["--results-dir", str(tmp_path)]) == 6
    assert arch_cli.e05_jobs_main(["--results-dir", str(tmp_path)]) == 6
