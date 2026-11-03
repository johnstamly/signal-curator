"""Task 12: E8 thresholds, E9 explicit rules, E10 coverage + f_amb.  Synthetic data only."""
import json
import math

import numpy as np
import pytest

from Curation_Paper.eswa_full.lib import (cells, coverage_study as C, kind_loop, rules_study as R,
                                          specimen_cli as CLI)
from Curation_Paper.eswa_full.lib.protocol import REPO_ROOT, input_hashes, load_config
from Curation_Paper.eswa_full.lib.signals import make_id
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


# ------------------------------------------------------------------ E9 statistics

def test_ptp_hand_value():
    assert R.stat_ptp(np.array([1.0, -2.0, 3.0, 0.0])) == 5.0


def test_abs_skew_hand_values():
    # [0,0,0,4]: mean 1, m2 = 3, m3 = 6  ->  skew = 6 / 3**1.5
    want = 6 / 3 ** 1.5
    assert R.stat_abs_skew(np.array([0.0, 0, 0, 4])) == pytest.approx(want)
    assert R.stat_abs_skew(np.array([0.0, 4, 4, 4])) == pytest.approx(want)   # sign dropped


def test_abs_skew_of_constant_signal_is_zero_not_nan():
    assert R.stat_abs_skew(np.full(10, 2.5)) == 0.0


def test_top5_hand_values():
    delta = np.zeros(16)
    delta[0] = 1.0                            # flat spectrum: 9 equal bins -> 5/9
    assert R.stat_top5(delta) == pytest.approx(5 / 9)
    t = np.arange(64)
    two_tones = np.cos(2 * np.pi * 4 * t / 64) + np.cos(2 * np.pi * 8 * t / 64)
    assert R.stat_top5(two_tones) == pytest.approx(1.0)       # only two non-zero bins
    assert R.stat_top5(np.zeros(32)) == 0.0                   # silent signal


def test_statistics_stacks_in_documented_order():
    X = np.array([[1.0, -2.0, 3.0, 0.0] * 4, [0.0, 0.0, 0.0, 4.0] * 4])
    S = R.statistics(X)
    assert S.shape == (2, 3) and R.FEATURES == ("E_pp", "abs_skew", "f5")
    assert S[0, 0] == 5.0 and S[1, 0] == 4.0


def test_filter_settings_are_the_short_paper_table3():
    assert R.FILTER_SETTINGS == {"tight": (1e-3, 3.0, 0.10), "medium": (5e-3, 2.5, 0.15),
                                 "loose": (1e-2, 2.0, 0.20)}


def test_flag_any_is_an_or_with_documented_directions():
    S = np.array([[5e-4, 0.0, 0.5],      # energy rule only (E_pp < 1e-3)
                  [1.0, 3.5, 0.5],       # skew rule only (|s| > 3)
                  [1.0, 0.0, 0.05],      # spectral rule only (f5 < 0.10)
                  [1.0, 3.0, 0.10],      # boundary: nothing fires (strict inequalities)
                  [1.0, 0.0, 0.5]])
    assert R.flag_any(S, R.FILTER_SETTINGS["tight"]).tolist() == [True, True, True, False, False]


def test_score_flags_hand_counts():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    fl = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    d = R.score_flags(y, fl)
    assert d["n_bad_flagged"] == 1 and d["n_good_flagged"] == 1
    assert d["bad_recall"] == pytest.approx(1 / 3) and d["good_kept"] == pytest.approx(4 / 5)


# ------------------------------------------------------------------ E9 tree

def _tree_world(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pool, stat = {}, {}
    for i in range(n):
        bad = i % 4 == 0
        sid = f"s{i:03d}"
        pool[sid] = {"signal_id": sid, "label_int": 0 if bad else 1}
        stat[sid] = np.array([rng.uniform(0, 1e-3) if bad else rng.uniform(0.5, 2.0),
                              rng.uniform(0, 4), rng.uniform(0.1, 0.9)])
    ref = {s: r["label_int"] for s, r in pool.items()}
    ids = sorted(ref)
    folds = [ids[k::5] for k in range(5)]
    return pool, stat, ref, folds


def test_tree_depth_limit_is_enforced():
    with pytest.raises(ValueError, match="depth"):
        R.fit_tree(np.zeros((4, 3)), np.array([0, 1, 0, 1]), max_depth=4)


def test_tree_cv_separable_problem_and_exactly_once_pooling():
    pool, stat, ref, folds = _tree_world()
    cv = R.tree_cv(stat, pool, ref, folds, es_frac=0.15)
    assert cv["pooled"]["n"] == len(ref)
    assert cv["pooled"]["bad_recall"] == 1.0 and cv["pooled"]["good_kept"] == 1.0
    assert len(cv["folds"]) == 5 and all("E_pp" in f["rules"] for f in cv["folds"])
    root = cv["folds"][0]["nodes"][0]
    assert root["feature"] == "E_pp" and 1e-3 < root["threshold"] < 0.5   # learned, full precision


def test_tree_cv_never_trains_on_test_ids_and_is_deterministic():
    pool, stat, ref, folds = _tree_world()
    a = R.tree_cv(stat, pool, ref, folds, es_frac=0.15)
    b = R.tree_cv(stat, pool, ref, folds, es_frac=0.15)
    assert [f["nodes"] for f in a["folds"]] == [f["nodes"] for f in b["folds"]]
    for f, test in zip(a["folds"], folds):
        assert f["n_test"] == len(test)
        assert f["n_train"] <= len(pool) - len(test)          # es rows are held out of training


def test_tree_cv_refuses_when_pooled_set_is_not_the_reference_set():
    pool, stat, ref, folds = _tree_world()
    with pytest.raises(ValueError, match="exactly once"):
        R.tree_cv(stat, pool, ref, folds[:4], es_frac=0.15)


def test_build_e9_shape_and_microconv_placeholder():
    pool, stat, ref, folds = _tree_world()
    out = R.build_e9(stat, pool, ref, folds, es_frac=0.15)
    assert set(out["filters"]) == {"tight", "medium", "loose"}
    assert out["microconv1d"] is None and "E3" in out["microconv1d_note"]
    assert out["tree"]["deterministic_no_seed_axis"] is True
    assert out["tree"]["descriptive_full_pool"]["rules"].startswith("|---")
    assert out["sanity_al_pool"]["n"] == len(pool)
    f = out["filters"]["tight"]
    assert f["n_bad_flagged"] + (f["n_bad"] - f["n_bad_flagged"]) == f["n_bad"]


# ------------------------------------------------------------------ E8 arithmetic

def test_threshold_row_hand_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.45, 0.5, 0.7, 0.3, 0.5, 0.6, 0.8, 0.9, 0.99])
    r = R.threshold_row(y, p, 0.5)             # kept: p >= 0.5
    assert r["n_kept"] == 7 and r["n_bad_kept"] == 2 and r["n_good_kept"] == 5
    assert r["precision"] == pytest.approx(5 / 7) and r["contamination"] == pytest.approx(2 / 7)
    assert r["precision"] + r["contamination"] == pytest.approx(1.0)
    assert r["fraction_kept"] == pytest.approx(0.7)
    assert r["bad_recall"] == pytest.approx(2 / 4)            # BAD with p < 0.5
    assert r["good_retention"] == pytest.approx(5 / 6)
    hi = R.threshold_row(y, p, 0.95)
    assert hi["n_kept"] == 1 and hi["precision"] == 1.0 and hi["bad_recall"] == 1.0
    empty = R.threshold_row(y, p, 1.0)
    assert empty["n_kept"] == 0 and math.isnan(empty["precision"])


def test_threshold_row_matches_metrics_at_half():
    from Curation_Paper.eswa_full.lib.metrics import metrics
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.7).astype(int)
    p = np.clip(rng.random(200) * 0.5 + 0.4 * y, 0, 1)
    m, r = metrics(y, p), R.threshold_row(y, p, 0.5)
    assert r["bad_recall"] == pytest.approx(m["bad_recall"])
    assert r["good_retention"] == pytest.approx(m["good_retention"])


def test_reliability_bins_hand_values():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.05, 0.95, 1.0, 0.12, 0.99])       # 1.0 falls in the last (closed) bin
    b = R.reliability_bins(y, p)
    assert len(b) == 10 and sum(x["n"] for x in b) == 5
    assert b[0]["n"] == 1 and b[0]["obs_good_rate"] == 0.0
    assert b[1]["n"] == 1 and b[1]["mean_p"] == pytest.approx(0.12)
    assert b[9]["n"] == 3 and b[9]["obs_good_rate"] == pytest.approx(1.0)
    assert math.isnan(b[4]["mean_p"]) and b[4]["n"] == 0


def test_build_e8_over_seeds():
    rng = np.random.default_rng(1)
    pooled = {}
    for s in (0, 1, 2):
        y = (rng.random(100) < 0.8).astype(int)
        p = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, 100), 0, 1)
        pooled[s] = ([f"x{i}" for i in range(100)], y, p)
    out = R.build_e8(pooled)
    assert [r["p_accept"] for r in out["thresholds"]] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    row = out["thresholds"][2]
    vals = [R.threshold_row(pooled[s][1], pooled[s][2], 0.5)["precision"] for s in (0, 1, 2)]
    assert row["precision"]["values"] == pytest.approx(vals)
    assert row["precision"]["mean"] == pytest.approx(np.mean(vals))
    assert row["precision"]["sd"] == pytest.approx(np.std(vals, ddof=1))
    assert row["precision"]["ci_lo"] <= row["precision"]["mean"] <= row["precision"]["ci_hi"]
    assert len(out["reliability"]) == 10
    assert sum(b["pooled_over_seeds"]["n"] for b in out["reliability"]) == 300


# ------------------------------------------------------------------ E8 / f_amb over stored cells

def _fake_ref(n=40):
    return {f"r{i:02d}": (0 if i % 4 == 0 else 1) for i in range(n)}


def _store(cfg, spec, result, key=KEY, dirty=False):
    cells.cell_path(spec, cfg).write_text(json.dumps({
        "spec": cells.canonical_spec(spec), "result": result,
        "meta": {"code_key": key, "dirty": dirty, "kind": "loop", "git_sha": "x",
                 "input_hashes": input_hashes(cfg)}}))


def _loop_result(ref, test_ids, seed, budgets=(100, 200)):
    rng = np.random.default_rng(seed * 100 + len(test_ids))
    recs = []
    for b in budgets:
        tp = {s: float(np.clip(0.5 + (0.4 if ref[s] else -0.4) + rng.normal(0, 0.2), 0.01, 0.99))
              for s in test_ids}
        recs.append({"budget": b, "test_pred": tp, "f_amb": 0.1 * seed + b / 1000})
    return {"test_ids": list(test_ids), "budgets": recs}


def _grid(cfg, seeds=(0, 1), folds=(0, 1), budgets=(100, 200)):
    from Curation_Paper.eswa_full.lib.jobs import grid
    return grid(cfg, seeds=list(seeds), folds=list(folds), budgets=list(budgets))


def _populate(cfg, g, ref, **kw):
    ids = sorted(ref)
    folds = {0: ids[0::2], 1: ids[1::2]}
    for s in CLI.cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0):
        _store(cfg, s, _loop_result(ref, folds[s["fold"]], s["seed"]), **kw)
    return folds


def test_pool_by_seed_pools_the_folds_and_checks_coverage(cfg):
    ref = _fake_ref()
    g = _grid(cfg)
    _populate(cfg, g, ref)
    out = R.pool_by_seed(cfg, g, ref, 200)
    assert sorted(out) == [0, 1]
    ids, y, p = out[0]
    assert ids == sorted(ref) and len(p) == len(ref)
    assert y.tolist() == [ref[s] for s in ids]
    with pytest.raises(ValueError, match="no record at budget"):
        R.pool_by_seed(cfg, g, ref, 600)
    with pytest.raises(ValueError, match="pooled set differs"):
        R.pool_by_seed(cfg, g, {**ref, "extra": 1}, 200)


def test_pool_by_seed_refuses_missing_cells(cfg):
    from Curation_Paper.eswa_full.lib.aggregate import MissingCells
    g = _grid(cfg)
    with pytest.raises(MissingCells):
        R.pool_by_seed(cfg, g, _fake_ref(), 200)


def test_famb_by_budget_summaries(cfg):
    ref = _fake_ref()
    g = _grid(cfg)
    _populate(cfg, g, ref)
    out = R.famb_by_budget(cfg, g)
    r100 = next(r for r in out["rows"] if r["budget"] == 100)
    assert r100["n_cells"] == 4
    assert sorted(r100["over_cells"]["values"]) == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert r100["over_cells"]["mean"] == pytest.approx(0.15)
    assert r100["over_seeds"]["values"] == pytest.approx([0.1, 0.2])


def _patch_world(monkeypatch, ref):
    monkeypatch.setattr(CLI, "verify_inputs", lambda cfg: {})
    monkeypatch.setattr(CLI, "load_pools", lambda cfg: ({}, {}))
    monkeypatch.setattr(CLI, "reference_labels", lambda al, rand: ref)


SMOKE = ["--seed-list", "0,1", "--fold-list", "0,1", "--budgets", "100,200", "--budget", "200"]


def test_e08_cli_smoke_writes_study_json(cfg, tmp_path, monkeypatch, capsys):
    ref = _fake_ref()
    _patch_world(monkeypatch, ref)
    g = _grid(cfg)
    _populate(cfg, g, ref)
    rc = CLI.e08_main(["--results-dir", str(tmp_path), *SMOKE])
    assert rc == 0, capsys.readouterr().err
    out = json.loads((tmp_path / "e08_thresholds.json").read_text())
    assert out["budget"] == 200 and out["n_seeds"] == 2 and out["n_scored"] == 40
    assert out["cells_code_key"] == KEY and out["cells_dirty"] == 0
    assert out["code"]["git_sha"] and out["meta"]["git_sha"]
    assert len(out["thresholds"]) == 8 and len(out["reliability"]) == 10
    assert out["thresholds"][0]["p_accept"] == 0.3


def test_e08_cli_refuses_missing_stale_dirty_and_headline_overrides(cfg, tmp_path, monkeypatch,
                                                                    capsys):
    ref = _fake_ref()
    _patch_world(monkeypatch, ref)
    g = _grid(cfg)
    assert CLI.e08_main(["--results-dir", str(tmp_path), *SMOKE]) == 2          # missing
    specs = CLI.cell_specs(g, pool="al", strategy="unc_oracle", eta=0.0)
    _populate(cfg, g, ref)
    _store(cfg, specs[0], _loop_result(ref, sorted(ref)[0::2], 0), key="z" * 64)
    assert CLI.e08_main(["--results-dir", str(tmp_path), *SMOKE]) == 5          # mixed keys
    assert "re-run" in capsys.readouterr().err
    _populate(cfg, g, ref, dirty=True)
    assert CLI.e08_main(["--results-dir", str(tmp_path), *SMOKE]) == 5          # dirty
    assert CLI.e08_main(["--seed-list", "0"]) == 4                              # no --results-dir
    assert not (tmp_path / "e08_thresholds.json").exists()


def test_e10_famb_cli_smoke(cfg, tmp_path, monkeypatch, capsys):
    ref = _fake_ref()
    _patch_world(monkeypatch, ref)
    g = _grid(cfg)
    _populate(cfg, g, ref)
    rc = CLI.e10_famb_main(["--results-dir", str(tmp_path), *SMOKE])
    assert rc == 0, capsys.readouterr().err
    out = json.loads((tmp_path / "e10_famb.json").read_text())
    assert [r["budget"] for r in out["rows"]] == [100, 200]
    assert out["rows"][0]["over_cells"]["n"] == 4


# ------------------------------------------------------------------ E10

def test_coverage_table_hand_values():
    ids = [make_id("FOD3", 1, 2, 200, r) for r in range(4)] + \
          [make_id("FOD3", 1, 2, 300, r) for r in range(4)] + \
          [make_id("FOD4", 1, 2, 200, r) for r in range(2)]
    p = [0.9, 0.5, 0.49, 0.1,   0.0, 0.1, 0.2, 0.3,   0.8, 0.7]
    t = C.coverage_table(ids, p)
    f3 = t["FOD3"]
    assert [c["cycle"] for c in f3["cycles"]] == [200, 300]
    assert f3["cycles"][0] == {"cycle": 200, "n": 4, "n_good": 2, "good_fraction": 0.5}
    assert f3["cycles"][1]["n_good"] == 0
    assert f3["mean_good_fraction"] == pytest.approx(0.25)
    assert f3["pooled_good_fraction"] == pytest.approx(2 / 8)
    assert f3["n_cycles_below_50"] == 1 and f3["cycles_zero_good"] == [300]
    assert t["FOD4"]["mean_good_fraction"] == 1.0 and t["FOD4"]["n_cycles_below_50"] == 0


def _universe(n_per=30):
    ids, X, y = [], [], {}
    rng = np.random.default_rng(0)
    t = np.arange(500)
    for panel in ("FOD3", "FOD4"):
        for cyc in (200, 300):
            for r in range(n_per):
                sid = make_id(panel, 1, 2, cyc, r)
                good = (r % 5) != 0
                f = 0.05 if good else 0.09
                ids.append(sid)
                X.append((np.sin(2 * np.pi * f * t) + 0.3 * rng.standard_normal(500)
                          ).astype(np.float32))
                y[sid] = int(good)
    return ids, np.stack(X), y


def test_fit_all_lal_covers_the_pool_with_no_test_ids(monkeypatch):
    ids, X, y = _universe()
    pool = {s: {"signal_id": s, "label_int": y[s]} for s in ids[:80]}
    seen = {}

    def spy(name, train, es, X_of, *, seed, recipe, init_state=None):
        seen.update(train=train, es=es, name=name, seed=seed)
        raise StopIteration
    monkeypatch.setattr(C, "fit", spy)
    with pytest.raises(StopIteration):
        C.fit_all_lal(pool, lambda r: X[:len(r)], seed=3, es_frac=0.15)
    tr, es = {r["signal_id"] for r in seen["train"]}, {r["signal_id"] for r in seen["es"]}
    assert tr | es == set(pool) and not tr & es
    assert seen["name"] == "microconv1d" and seen["seed"] == 3
    n_bad = sum(1 for r in pool.values() if r["label_int"] == 0)
    assert sum(1 for r in seen["es"] if r["label_int"] == 0) == max(1, round(0.15 * n_bad))


def test_run_e10_end_to_end_on_synthetic_universe():
    ids, X, y = _universe()
    pool = {s: {"signal_id": s, "label_int": y[s]} for s in ids if s.endswith(("_r0", "_r1", "_r2"))}
    out = C.run_e10(pool, ids, X, seeds=[0, 1], es_frac=0.15,
                    recipe={"max_epochs": 2})
    cov, sc = out["coverage"], out["scores"]
    assert cov["n_signals"] == len(ids) == len(sc["ids"]) == len(sc["mean_p"])
    assert set(cov["panels"]) == {"FOD3", "FOD4"}
    assert all(0.0 <= v <= 1.0 for v in sc["mean_p"]) and len(sc["sd_p"]) == len(ids)
    assert sum(c["n"] for c in cov["panels"]["FOD3"]["cycles"]) == 60
    assert [f["seed"] for f in cov["fits"]] == [0, 1]
    assert set(cov["per_seed_panel_summary"]) == {"0", "1"}
    with pytest.raises(KeyError):
        C.run_e10({**pool, "FOD9_50kHz_act1_recv2_c1_r0": {"signal_id": "x", "label_int": 1}},
                  ids, X, seeds=[0], es_frac=0.15, recipe={"max_epochs": 2})


def test_loop_code_key_unchanged_by_task_12():
    assert code_key(kind_loop.DEPENDENCY_FILES, REPO_ROOT).startswith("84a02678827a")  # R37 (loop.py) moved it from 193b514622b7
