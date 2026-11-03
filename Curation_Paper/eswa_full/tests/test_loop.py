import json
import math

import pytest

from Curation_Paper.eswa_full.lib import loop
from Curation_Paper.eswa_full.lib.loop import BudgetExceedsPool, LeakageError, simulate
from Curation_Paper.eswa_full.lib.protocol import load_config, load_pools, reference_labels
from Curation_Paper.eswa_full.lib.splits import shared_folds

BUDGETS = [100, 200]
R8_BUDGETS = [100, 150, 200]
TORCH = {"arch": "microconv1d", "recipe": {"max_epochs": 3, "patience": 2, "augment": False}}


@pytest.fixture(scope="module")
def world():
    cfg = load_config()
    al, rand = load_pools(cfg)
    ref = reference_labels(al, rand)
    test_ids = shared_folds(ref, 5, 0)[0]
    return cfg, al, ref, test_ids


def run(world, strategy="uncertainty", eta=0.0, seed=0, budgets=BUDGETS, arch="logreg", recipe=None, **kw):
    cfg, al, ref, test_ids = world
    return simulate(al, test_ids, ref, strategy=strategy, budgets=budgets, seed=seed,
                    seed_size=kw.pop("seed_size", 100), batch=kw.pop("batch", 50), eta=eta,
                    arch=arch, recipe=recipe or {}, cfg=cfg, **kw)


def _strip(res):
    s = json.loads(json.dumps(res))
    for b in s["budgets"]:
        b.pop("train_seconds", None)
    return json.dumps(s, sort_keys=True)


def test_eta0_oracle_equals_uncertainty(world):
    # seed-sensitive arch: logreg ignores the fit seed, so it could not catch an R8 violation
    a = run(world, "uncertainty", budgets=R8_BUDGETS, **TORCH)
    b = run(world, "unc_oracle", budgets=R8_BUDGETS, **TORCH)
    assert len(a["budgets"]) == len(b["budgets"]) == 3
    for x, y in zip(a["budgets"], b["budgets"]):
        assert x["train_ids"] == y["train_ids"]
        assert x["metrics"] == y["metrics"] or json.dumps(x["metrics"]) == json.dumps(y["metrics"])


def test_oracle_restores_only_flagged_flips(world):
    cfg, al, ref, test_ids = world
    res = run(world, "unc_oracle", eta=0.10)
    for b in res["budgets"]:
        assert b["n_recovered"] <= min(b["n_flipped"], b["n_flagged"])
        assert b["n_false_flags"] + b["n_recovered"] <= b["n_flagged"] + 0
        # every row whose label differs from the stored one is an unrecovered flip
        wrong = sum(1 for sid, lab in zip(b["train_ids"], b["train_labels"])
                    if lab != al[sid]["label_int"])
        assert wrong == b["n_flipped"] - b["n_recovered"]
    assert res["budgets"][-1]["n_flipped"] > 0


def test_budget_exceeding_pool_raises(world):
    with pytest.raises(BudgetExceedsPool, match="2000"):
        run(world, budgets=[100, 2000])
    with pytest.raises(BudgetExceedsPool):
        run(world, budgets=[50, 200])  # below seed_size


def test_leakage_guard(world, monkeypatch):
    cfg, al, ref, test_ids = world
    real = loop.fold_split

    def leaky(pool, tids, seed, es_frac):
        train, es = real(pool, tids, seed, es_frac)
        row = dict(al[test_ids[0]])
        return train + [row], es

    monkeypatch.setattr(loop, "fold_split", leaky)
    with pytest.raises(LeakageError):
        run(world)


def test_budgets_are_nested(world):
    for strat in ("random", "uncertainty"):
        res = run(world, strat)
        ids = [set(b["train_ids"]) for b in res["budgets"]]
        assert [len(i) for i in ids] == BUDGETS
        assert ids[0] < ids[1]


def test_seed_set_has_both_classes(world):
    cfg, al, ref, test_ids = world
    for seed in range(3):
        res = run(world, "random", seed=seed, budgets=[100])
        labs = res["budgets"][0]["train_labels"]
        assert 0 in labs and 1 in labs and len(labs) == 100


def test_deterministic_same_seed(world):
    for strat in ("random", "unc_selftrain"):
        a = run(world, strat, eta=0.1, seed=3, budgets=R8_BUDGETS, **TORCH)
        b = run(world, strat, eta=0.1, seed=3, budgets=R8_BUDGETS, **TORCH)
        assert _strip(a) == _strip(b)


def test_budget_list_validation(world):
    with pytest.raises(ValueError, match="empty"):
        run(world, budgets=[])
    with pytest.raises(ValueError, match="duplicate"):
        run(world, budgets=[100, 100])


def test_bookkeeping_fields(world):
    res = run(world, "unc_selftrain", eta=0.1, budgets=R8_BUDGETS)
    for k in ("budgets_requested", "recipe", "es_frac", "flag_margin", "ambiguous_band",
              "oof_folds", "n_flipped_total", "n_flipped_es"):
        assert k in res
    assert res["budgets_requested"] == R8_BUDGETS
    last = res["budgets"][-1]
    assert res["n_flipped_total"] >= last["n_flipped"] + res["n_flipped_es"]
    assert last["n_ever_recovered"] >= last["n_recovered"]
    assert last["n_oof_folds_skipped"] >= 0


def test_result_is_self_describing_and_json(world):
    res = run(world, "random")
    json.dumps(res)
    for k in ("strategy", "seed", "eta", "arch", "seed_size", "batch", "len_train_pool",
              "len_es", "test_ids"):
        assert k in res
    b = res["budgets"][0]
    assert math.isnan(b["f_amb"]) and b["selection_claim_ok"] is True


def test_unknown_strategy(world):
    with pytest.raises(ValueError):
        run(world, "bogus")


def test_budget_records_store_test_predictions(world):
    """R14: every budget record carries p(GOOD) per test id, rounded to 6 decimals."""
    cfg, al, ref, test_ids = world
    res = run(world, "random")
    for b in res["budgets"]:
        tp = b["test_pred"]
        assert set(tp) == set(test_ids)
        assert all(0.0 <= v <= 1.0 and round(v, 6) == v for v in tp.values())
    json.dumps(res)


def test_warm_start_changes_result_and_is_deterministic(world):
    kw = dict(budgets=R8_BUDGETS, **TORCH)
    cold = run(world, "uncertainty", **kw)
    warm1 = run(world, "uncertainty", warm_start=True, **kw)
    warm2 = run(world, "uncertainty", warm_start=True, **kw)
    assert warm1["warm_start"] is True and "warm_start" not in cold
    assert _strip(warm1) == _strip(warm2)
    # the first budget point is a from-scratch fit in both modes; later ones differ
    assert warm1["budgets"][0]["metrics"] == cold["budgets"][0]["metrics"]
    assert any(w["test_pred"] != c["test_pred"]
               for w, c in zip(warm1["budgets"][1:], cold["budgets"][1:]))


def test_warm_start_default_false_is_identity(world):
    kw = dict(budgets=R8_BUDGETS, **TORCH)
    assert _strip(run(world, "unc_oracle", **kw)) == _strip(
        run(world, "unc_oracle", warm_start=False, **kw))


def test_warm_start_rejects_classic_arch(world):
    with pytest.raises(ValueError, match="warm_start"):
        run(world, "uncertainty", arch="logreg", warm_start=True)


# ---- R37: eta noise on training-pool rows only; the early-stopping split keeps stored labels

@pytest.mark.parametrize("strategy", ["uncertainty", "unc_oracle"])
def test_es_labels_are_stored_labels_under_noise(world, monkeypatch, strategy):
    """Every fit early-stops on the STORED es labels at eta > 0: a noisy es split stops a model
    trained on restored labels within a few epochs (debug-e3-report.md)."""
    cfg, al, ref, test_ids = world
    seen, real_fit = [], loop.fit

    def spy(arch, tr, es, X_of, **kw):
        seen.append([(r["signal_id"], r["label_int"]) for r in es])
        return real_fit(arch, tr, es, X_of, **kw)

    monkeypatch.setattr(loop, "fit", spy)
    res = run(world, strategy, eta=0.10)
    assert seen
    assert sum(1 for c in seen for sid, lab in c if lab != al[sid]["label_int"]) == 0
    assert res["n_flipped_es"] == 0
    assert res["budgets"][-1]["n_flipped"] > 0  # training-pool noise is still injected


def test_training_pool_gets_floor_eta_n_flips(world, monkeypatch):
    cfg, al, ref, test_ids = world
    calls, real = [], loop.inject_label_noise

    def spy(buf, eta, seed):
        out = real(buf, eta, seed)
        calls.append((len(buf), sum(1 for a, b in zip(buf, out) if a["label_int"] != b["label_int"]),
                      {r["signal_id"] for r in buf}))
        return out

    es_ids, real_fit = set(), loop.fit

    def fit_spy(arch, tr, es, X_of, **kw):
        es_ids.update(r["signal_id"] for r in es)
        return real_fit(arch, tr, es, X_of, **kw)

    monkeypatch.setattr(loop, "inject_label_noise", spy)
    monkeypatch.setattr(loop, "fit", fit_spy)
    res = run(world, "random", eta=0.10)
    assert len(calls) == 1
    n, flips, ids = calls[0]
    assert es_ids and not ids & es_ids  # the noise buffer excludes the early-stopping split
    assert flips == int(0.10 * n) > 0
    assert res["n_flipped_total"] == flips
    assert not ids & set(test_ids)
    # the noise buffer is the training pool: every labelled row of the last budget is in it
    assert set(res["budgets"][-1]["train_ids"]) <= ids


def test_eta0_never_calls_noise_injection(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("inject_label_noise called at eta = 0")

    monkeypatch.setattr(loop, "inject_label_noise", boom)
    res = run(world, "unc_oracle", eta=0.0)
    assert res["n_flipped_total"] == 0 and res["n_flipped_es"] == 0
