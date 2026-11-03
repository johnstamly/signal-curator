"""Content-hash cell versioning, stale-cell recompute, aggregator guards (R18)."""
import importlib
import json
import subprocess
import sys

import pytest

from Curation_Paper.eswa_full.lib import aggregate, cells, jobs, kind_loop, versioning
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT, REPO_ROOT, load_config

SPEC = {"study_source": "loop", "pool": "al", "fold": 0, "seed": 0, "strategy": "unc_oracle",
        "eta": 0.0, "arch": "logreg", "recipe": {}, "budgets": [100, 200],
        "seed_size": 100, "batch": 50}


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["results_dir"] = str(tmp_path)
    return c


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    folds = [[f"s{i}", f"t{i}"] for i in range(5)]
    monkeypatch.setattr(kind_loop, "_world", lambda cfg: ({"al": {}, "rand": {}}, {}, folds))
    cells.reset_versions()
    yield
    cells.reset_versions()


def set_version(monkeypatch, key, dirty=()):
    monkeypatch.setattr(cells, "_compute_version", lambda kind: (key, list(dirty)))
    monkeypatch.setattr(cells, "_live_key", lambda kind: key)
    cells.reset_versions()


# ------------------------------------------------------------------ code key

def test_key_ignores_unrelated_files_and_tracks_dependencies(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "latex").mkdir()
    (tmp_path / "lib/dep.py").write_text("x = 1\n")
    files = ["lib/dep.py"]
    k0 = versioning.code_key(files, tmp_path)
    (tmp_path / "latex/paper.tex").write_text("unrelated")
    (tmp_path / "lib/new_module_not_listed.py").write_text("y = 2\n")
    assert versioning.code_key(files, tmp_path) == k0
    (tmp_path / "lib/dep.py").write_text("x = 2\n")
    assert versioning.code_key(files, tmp_path) != k0


def test_key_is_line_ending_independent(tmp_path):
    p = tmp_path / "a.py"
    p.write_bytes(b"a = 1\nb = 2\n")
    k_lf = versioning.code_key(["a.py"], tmp_path)
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert versioning.code_key(["a.py"], tmp_path) == k_lf


def test_key_depends_on_path_set_and_fails_on_missing(tmp_path):
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "b.py").write_text("1")
    assert versioning.code_key(["a.py"], tmp_path) != versioning.code_key(["b.py"], tmp_path)
    with pytest.raises(FileNotFoundError):
        versioning.code_key(["nope.py"], tmp_path)


def test_dirty_files_only_lists_dependency_files(tmp_path):
    for d in ("lib", "other"):
        (tmp_path / d).mkdir()
    (tmp_path / "lib/a.py").write_text("x = 1\n")
    (tmp_path / "other/b.txt").write_text("b\n")

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path,
                       check=True, capture_output=True)
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert versioning.dirty_files(["lib/a.py"], tmp_path) == []
    (tmp_path / "other/b.txt").write_text("changed\n")
    assert versioning.dirty_files(["lib/a.py"], tmp_path) == []
    (tmp_path / "lib/a.py").write_text("x = 2\n")
    assert versioning.dirty_files(["lib/a.py"], tmp_path) == ["lib/a.py"]


def test_loop_dependency_list_is_complete():
    """Every repo module imported by kind_loop (fresh interpreter) is a listed dependency."""
    code = ("import sys, json; import Curation_Paper.eswa_full.lib.kind_loop; "
            "print(json.dumps(sorted({m.__file__ for m in list(sys.modules.values()) "
            "if getattr(m, '__file__', None)})))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout
    files = json.loads(out.strip().splitlines()[-1])
    root = str(REPO_ROOT).replace("\\", "/").lower() + "/"
    local = {f.replace("\\", "/")[len(root):] for f in files
             if f.replace("\\", "/").lower().startswith(root) and "site-packages" not in f}
    deps = set(kind_loop.DEPENDENCY_FILES)
    assert local <= deps, sorted(local - deps)
    for d in deps:
        assert (REPO_ROOT / d).is_file(), d


def test_cell_ids_unchanged_for_stage1_job_file():
    stored = json.loads((PKG_ROOT / "results/jobs/stage1_loop.json").read_text())["specs"]
    now = jobs.stage1_jobs(load_config())["all"]
    assert [cells.cell_id(s) for s in stored] == [cells.cell_id(s) for s in now]


# ------------------------------------------------- stale handling in the runner

def fake_sim(calls):
    def sim(pool, test_ids, ref, **kw):
        calls.append(kw)
        return {"test_ids": list(test_ids), "budgets": []}
    return sim


def test_runner_recomputes_stale_and_skips_current(monkeypatch, cfg):
    calls, logs = [], []
    monkeypatch.setattr(kind_loop, "simulate", fake_sim(calls))
    set_version(monkeypatch, "a" * 64)
    cells.run_cells([SPEC], 0, cfg, log=logs.append)
    assert len(calls) == 1
    cells.run_cells([SPEC], 0, cfg, log=logs.append)  # same key -> skipped
    assert len(calls) == 1
    set_version(monkeypatch, "b" * 64)  # a dependency file changed
    summary = cells.run_cells([SPEC], 0, cfg, log=logs.append)
    assert len(calls) == 2 and summary["done"] == 1 and summary["stale"] == 1
    assert any(l.startswith("stale -> recompute") and "old key aaaaaaaaaa" in l for l in logs)
    stored = cells.load_cell(SPEC, cfg)
    assert stored["meta"]["code_key"] == "b" * 64 and stored["meta"]["dirty"] is False
    cells.run_cells([SPEC], 0, cfg, log=logs.append)
    assert len(calls) == 2


def test_dirty_cell_is_stale_unless_allow_dirty(monkeypatch, cfg):
    calls = []
    monkeypatch.setattr(kind_loop, "simulate", fake_sim(calls))
    set_version(monkeypatch, "a" * 64, dirty=["Curation_Paper/eswa_full/lib/loop.py"])
    cells.run_cells([SPEC], 0, {**cfg, "allow_dirty": True}, log=lambda m: None)
    assert len(calls) == 1 and cells.load_cell(SPEC, cfg)["meta"]["dirty"] is True
    cells.run_cells([SPEC], 0, {**cfg, "allow_dirty": True}, log=lambda m: None)
    assert len(calls) == 1  # allowed (smoke): same key is current
    summary = cells.run_cells([SPEC], 0, cfg, log=lambda m: None)
    assert len(calls) == 2 and summary["stale"] == 1  # strict: dirty cells are recomputed


def test_dependency_change_during_run_refuses_to_write(monkeypatch, cfg):
    monkeypatch.setattr(kind_loop, "simulate", fake_sim([]))
    set_version(monkeypatch, "a" * 64)
    monkeypatch.setattr(cells, "_live_key", lambda kind: "z" * 64)
    with pytest.raises(RuntimeError, match="changed while this process"):
        cells.run_cell(SPEC, cfg)
    assert not cells.cell_path(SPEC, cfg).exists()


def test_unknown_kind_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown cell kind"):
        cells.run_cell({**SPEC, "study_source": "nope"}, cfg)


def test_runner_refuses_to_start_on_dirty_dependencies(monkeypatch, capsys, tmp_path):
    mod = importlib.import_module("Curation_Paper.eswa_full.scripts.run_cells")
    monkeypatch.setattr(mod, "dirty_dependencies", lambda specs: ["app/model.py"])
    jf = tmp_path / "j.json"
    jf.write_text(json.dumps({"specs": [SPEC]}))
    assert mod.main(["--jobs", str(jf), "--results-dir", str(tmp_path)]) == 4
    assert "app/model.py" in capsys.readouterr().err


# ------------------------------------------------------------------ aggregators

def _store(cfg, spec, key, dirty=False, result=None):
    cells._atomic_write(cells.cell_path(spec, cfg), {
        "spec": cells.canonical_spec(spec), "result": result or {},
        "meta": {"code_key": key, "git_sha": "g" * 40, "dirty": dirty,
                 "input_hashes": cells.input_hashes(cfg)}})


def _many(cfg, keys):
    specs = [{**SPEC, "seed": i} for i in range(len(keys))]
    for s, k in zip(specs, keys):
        _store(cfg, s, k)
    return specs


def test_one_key_passes_and_is_reported(cfg):
    prov = aggregate.require(_many(cfg, ["a" * 64] * 3), cfg)
    assert prov["cells_code_key"] == "a" * 64 and prov["n_cells"] == 3


def test_mixed_keys_refuse_with_counts_and_actionable_message(cfg):
    specs = _many(cfg, ["a" * 64, "a" * 64, "b" * 64])
    with pytest.raises(aggregate.KeyMismatch) as ei:
        aggregate.require(specs, cfg)
    msg = str(ei.value)
    assert "aaaaaaaaaa: 2 cell(s)" in msg and "bbbbbbbbbb: 1 cell(s)" in msg
    assert ("re-run scripts/run_cells.py with the same job file -- stale cells are "
            "recomputed automatically") in msg


def test_expect_key(cfg):
    specs = _many(cfg, ["a" * 64] * 2)
    assert aggregate.require(specs, cfg, expect_key="aaaa")["cells_code_key"] == "a" * 64
    with pytest.raises(aggregate.KeyMismatch):
        aggregate.require(specs, cfg, expect_key="bbbb")


def test_dirty_cells_refused_unless_allowed(cfg):
    s = {**SPEC, "seed": 7}
    _store(cfg, s, "c" * 64, dirty=True)
    with pytest.raises(aggregate.KeyMismatch, match="dirty"):
        aggregate.require([s], cfg)
    assert aggregate.require([s], cfg, allow_dirty=True)["n_dirty"] == 1


def _rec(budget, preds):
    return {"budget": budget, "test_pred": preds, "selection_claim_ok": True,
            "n_flipped": 0, "n_flagged": 0, "n_recovered": 0, "n_ever_recovered": 0,
            "n_false_flags": 0, "n_residual_flips": 0}


def test_curve_asserts_pooled_set_equals_reference(cfg):
    g = jobs.grid(cfg, arch="logreg", seeds=[0], folds=[0], budgets=[100])
    spec = jobs.cell_specs(g, pool="al", strategy="random", eta=0.0)[0]
    res = {"test_ids": ["a", "b"], "len_train_pool": 500,
           "budgets": [_rec(100, {"a": 0.9, "b": 0.2})]}
    _store(cfg, spec, "d" * 64, result=res)
    kw = dict(pool="al", strategy="random", eta=0.0)
    assert aggregate.curve(cfg, g, {"a": 1, "b": 0}, **kw)[0]["n_scored"] == 2
    with pytest.raises(ValueError, match="pooled scored set"):
        aggregate.curve(cfg, g, {"a": 1, "b": 0, "c": 1}, **kw)
    with pytest.raises(ValueError, match="pooled scored set"):
        aggregate.curve(cfg, g, {"a": 1}, **kw)


def test_aggregator_refuses_grid_override_without_results_dir(capsys):
    assert aggregate.cli("e03", ["--budgets", "100,200"]) == 4
    assert "--results-dir" in capsys.readouterr().err


def test_stage2_job_list_has_no_e7():
    cfg = load_config()
    j = jobs.stage1_jobs(cfg, include_e7=False, seeds=list(range(20)))
    assert j["counts"]["E7_extra"] == 0
    assert j["counts"]["E3"] == 20 * 5 * 2 * 5


# ------------------------------------------------------------------ R19

def test_kind_loop_pops_warm_start_before_simulate(monkeypatch, cfg):
    seen = {}

    def sim(pool, test_ids, ref, **kw):
        seen.update(kw)
        return {"test_ids": list(test_ids), "budgets": []}
    monkeypatch.setattr(kind_loop, "simulate", sim)
    spec = cells.canonical_spec({**SPEC, "recipe": {"max_epochs": 3, "warm_start": True}})
    kind_loop.run(spec, cfg)
    assert seen["warm_start"] is True and "warm_start" not in seen["recipe"]
    assert seen["recipe"] == {"max_epochs": 3}
    assert spec["recipe"]["warm_start"] is True  # the spec itself is not mutated
    seen.clear()
    kind_loop.run(cells.canonical_spec(SPEC), cfg)
    assert seen["warm_start"] is False


def test_fit_never_receives_warm_start(monkeypatch):
    """A recipe key unknown to fit raises; popping it is what keeps warm cells runnable."""
    from Curation_Paper.eswa_full.lib import fit as fitmod
    with pytest.raises(ValueError, match="unknown recipe keys"):
        fitmod._recipe({"warm_start": True})


def test_warm_start_spec_has_its_own_id():
    warm = {**SPEC, "recipe": {"warm_start": True}}
    assert cells.cell_id(warm) != cells.cell_id(SPEC)


def test_package_versions_in_meta(monkeypatch, cfg):
    monkeypatch.setattr(kind_loop, "simulate", fake_sim([]))
    set_version(monkeypatch, "a" * 64)
    cells.run_cell(SPEC, cfg)
    pk = cells.load_cell(SPEC, cfg)["meta"]["packages"]
    assert set(pk) == {"numpy", "scikit-learn", "aeon", "h5py", "torch"}
    assert all(pk[k] for k in ("numpy", "scikit-learn", "h5py", "torch"))


def test_kind_keys_are_frozen_by_freeze_versions(monkeypatch):
    monkeypatch.undo()
    cells.reset_versions()
    cells._freeze_versions()  # what runs at import of cells.py
    key, dirty = cells._VERSION["loop"]
    assert len(key) == 64 and isinstance(dirty, list)
