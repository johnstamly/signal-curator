"""Cell runner, job lists and aggregation refusal (no real simulate calls)."""
import copy
import json

import pytest

from Curation_Paper.eswa_full.lib import cells, jobs, kind_loop
from Curation_Paper.eswa_full.lib.protocol import load_config

SPEC = {"study_source": "loop", "pool": "al", "fold": 0, "seed": 0, "strategy": "unc_oracle",
        "eta": 0.0, "arch": "logreg", "recipe": {"a": 1}, "budgets": [100, 200],
        "seed_size": 100, "batch": 50}


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["results_dir"] = str(tmp_path)
    return c


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    ids = [f"s{i}" for i in range(10)]
    folds = [ids[i::5] for i in range(5)]
    monkeypatch.setattr(kind_loop, "_world", lambda cfg: ({"al": {}, "rand": {}}, {}, folds))
    monkeypatch.setattr(cells, "_compute_version", lambda kind: ("k" * 64, []))
    monkeypatch.setattr(cells, "_live_key", lambda kind: "k" * 64)
    cells.reset_versions()
    yield
    cells.reset_versions()


def fake_sim(calls=None):
    def sim(pool, test_ids, ref, **kw):
        if calls is not None:
            calls.append(kw)
        return {"strategy": kw["strategy"], "test_ids": list(test_ids), "budgets": []}
    return sim


def test_cell_id_is_stable_and_order_independent():
    a = cells.cell_id(SPEC)
    assert a == cells.cell_id(dict(reversed(list(SPEC.items()))))
    assert a == cells.cell_id(copy.deepcopy(SPEC))
    assert len(a) == 16
    # int and float eta describe the same cell
    assert cells.cell_id({**SPEC, "eta": 0}) == a


@pytest.mark.parametrize("key,val", [
    ("study_source", "other"), ("pool", "rand"), ("fold", 1), ("seed", 1),
    ("strategy", "uncertainty"), ("eta", 0.1), ("arch", "bigru"), ("recipe", {"a": 2}),
    ("budgets", [100, 300]), ("seed_size", 50), ("batch", 25)])
def test_cell_id_sensitive_to_every_field(key, val):
    assert cells.cell_id({**SPEC, key: val}) != cells.cell_id(SPEC)


def test_spec_requires_exact_fields():
    with pytest.raises(ValueError):
        cells.cell_id({k: v for k, v in SPEC.items() if k != "batch"})
    with pytest.raises(ValueError):
        cells.cell_id({**SPEC, "extra": 1})


def test_fold_assignment_uses_seed_zero_for_every_training_seed(monkeypatch, cfg):
    got = []
    monkeypatch.setattr(kind_loop, "simulate", fake_sim(got))
    cells.run_cell({**SPEC, "seed": 3, "fold": 2}, cfg)
    cells.run_cell({**SPEC, "seed": 4, "fold": 2}, cfg)
    assert got[0]["seed"] == 3 and got[1]["seed"] == 4


def test_skip_if_present(monkeypatch, cfg):
    calls = []
    monkeypatch.setattr(kind_loop, "simulate", fake_sim(calls))
    p1 = cells.run_cell(SPEC, cfg)
    assert p1.exists() and len(calls) == 1
    p2 = cells.run_cell(SPEC, cfg)
    assert p2 == p1 and len(calls) == 1
    payload = json.loads(p1.read_text())
    assert payload["spec"] == cells.canonical_spec(SPEC)
    assert payload["meta"]["input_hashes"] and "git_sha" in payload["meta"]


def test_corrupt_or_mismatched_file_is_recomputed(monkeypatch, cfg):
    calls = []
    monkeypatch.setattr(kind_loop, "simulate", fake_sim(calls))
    p = cells.cell_path(SPEC, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    cells.run_cell(SPEC, cfg)
    assert len(calls) == 1
    p.write_text(json.dumps({"spec": {**SPEC, "seed": 9}, "result": {}}))
    cells.run_cell(SPEC, cfg)
    assert len(calls) == 2


def test_atomic_write_leaves_no_final_file_on_failure(monkeypatch, cfg):
    def boom(*a, **k):
        raise RuntimeError("mid-run")
    monkeypatch.setattr(kind_loop, "simulate", boom)
    with pytest.raises(RuntimeError):
        cells.run_cell(SPEC, cfg)
    assert not cells.cell_path(SPEC, cfg).exists()


def test_atomic_write_failure_during_dump_keeps_no_final(monkeypatch, cfg):
    monkeypatch.setattr(kind_loop, "simulate", fake_sim())
    real = cells.os.replace

    def fail(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(cells.os, "replace", fail)
    with pytest.raises(OSError):
        cells.run_cell(SPEC, cfg)
    monkeypatch.setattr(cells.os, "replace", real)
    assert not cells.cell_path(SPEC, cfg).exists()


def test_stale_tmp_is_overwritten(monkeypatch, cfg):
    monkeypatch.setattr(kind_loop, "simulate", fake_sim())
    p = cells.cell_path(SPEC, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("garbage from a killed run")
    cells.run_cell(SPEC, cfg)
    assert json.loads(p.read_text())["spec"] == cells.canonical_spec(SPEC)
    assert not tmp.exists()


def test_ordering_longest_first():
    specs = [{**SPEC, "strategy": s, "seed": i} for i, s in
             enumerate(["random", "uncertainty", "unc_selftrain", "unc_oracle"])]
    order = [s["strategy"] for s in cells.order_longest_first(specs)]
    assert order[:2] == sorted(order[:2]) and set(order[:2]) == {"unc_oracle", "unc_selftrain"}
    assert order[2] == "uncertainty" and order[3] == "random"


def test_run_cells_in_process_counts_and_errors(monkeypatch, cfg):
    def sim(pool, test_ids, ref, **kw):
        if kw["seed"] == 1:
            raise RuntimeError("bad cell")
        return {"test_ids": list(test_ids), "budgets": []}
    monkeypatch.setattr(kind_loop, "simulate", sim)
    specs = [{**SPEC, "seed": s} for s in (0, 1, 2)]
    summary = cells.run_cells(specs, 0, cfg)  # workers=0 -> in-process (tests)
    assert summary == {"done": 2, "skipped": 0, "failed": 1, "deferred": 0, "total": 3,
                       "stale": 0}
    err = cells.cells_dir(cfg) / "_errors" / f"{cells.cell_id(specs[1])}.txt"
    assert "bad cell" in err.read_text()
    again = cells.run_cells(specs, 0, cfg)
    assert again["skipped"] == 2 and again["failed"] == 1 and again["done"] == 0


def test_duplicate_specs_run_once(monkeypatch, cfg):
    calls = []
    monkeypatch.setattr(kind_loop, "simulate", fake_sim(calls))
    cells.run_cells([SPEC, dict(SPEC)], 0, cfg)
    assert len(calls) == 1


# ---------------------------------------------------------------- job lists

def test_stage1_job_counts_and_dedup():
    cfg = load_config()
    j = jobs.stage1_jobs(cfg)
    ids = [cells.cell_id(s) for s in j["all"]]
    assert len(ids) == len(set(ids))
    assert j["counts"]["E3"] == 250
    assert j["counts"]["E2_extra"] == 50
    assert j["counts"]["E1_extra"] == 50
    assert j["counts"]["E7_extra"] == 200
    assert j["counts"]["total"] == len(j["all"]) == 550
    # (s=100, b=50) E7 configuration is not a separate cell
    assert not any(s["seed_size"] == 100 and s["batch"] == 50 and s["strategy"] == "unc_oracle"
                   and s["eta"] == 0 and s["budgets"] != [100, 200, 300, 400, 500, 600]
                   for s in j["all"])
    e7 = [s for s in j["E7_extra"]]
    assert {(s["seed_size"], s["batch"]) for s in e7} == {
        (a, b) for a in (50, 100, 200) for b in (25, 50, 100)} - {(100, 50)}
    assert {tuple(s["budgets"]) for s in e7 if s["seed_size"] == 200} == {(200, 300, 400)}
    assert {tuple(s["budgets"]) for s in e7 if s["seed_size"] == 50} == {(50, 200, 300, 400)}


def test_study_specs_are_covered_by_job_list():
    cfg = load_config()
    j = jobs.stage1_jobs(cfg)
    have = {cells.cell_id(s) for s in j["all"]}
    g = jobs.grid(cfg)
    for fn in (jobs.e1_specs, jobs.e2_specs, jobs.e3_specs, jobs.e7_specs):
        need = {cells.cell_id(s) for s in fn(g)}
        assert need <= have, fn.__name__


# ------------------------------------------------------------- aggregation

def test_aggregator_refuses_on_missing_cell(cfg):
    from Curation_Paper.eswa_full.lib import aggregate
    g = jobs.grid(cfg, arch="logreg", seeds=[0], budgets=[100, 200])
    with pytest.raises(aggregate.MissingCells) as ei:
        aggregate.require(jobs.e3_specs(g), cfg)
    assert len(ei.value.missing) == len(jobs.e3_specs(g))
    assert ei.value.missing[0] in str(ei.value)
