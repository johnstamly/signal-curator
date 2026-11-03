import pytest

from Curation_Paper.eswa_full.lib.protocol import load_config, load_pools, reference_labels
from Curation_Paper.eswa_full.lib.splits import fold_split, shared_folds

K = 5
ES_FRAC = 0.15


@pytest.fixture(scope="module")
def data():
    al, rand = load_pools(load_config())
    return al, rand, reference_labels(al, rand)


def test_every_shared_signal_tested_exactly_once(data):
    _, _, ref = data
    for seed in range(5):
        folds = shared_folds(ref, K, seed)
        assert len(folds) == K
        assert sorted(sum(folds, [])) == sorted(ref)
        sets = [set(f) for f in folds]
        for i in range(K):
            for j in range(i + 1, K):
                assert not (sets[i] & sets[j])


def test_folds_stratified_by_reference_label(data):
    _, _, ref = data
    for seed in range(5):
        for f in shared_folds(ref, K, seed):
            n_bad = sum(1 - ref[s] for s in f)
            assert n_bad in (12, 13)
            assert len(f) in (149, 150)


def test_deterministic_and_seed_dependent(data):
    al, rand, ref = data
    assert shared_folds(ref, K, 3) == shared_folds(ref, K, 3)
    assert shared_folds(ref, K, 0) != shared_folds(ref, K, 1)
    test = shared_folds(ref, K, 0)[0]
    a = fold_split(al, test, 2, ES_FRAC)
    b = fold_split(al, test, 2, ES_FRAC)
    assert [r["signal_id"] for r in a[0]] == [r["signal_id"] for r in b[0]]
    assert [r["signal_id"] for r in a[1]] == [r["signal_id"] for r in b[1]]


def test_no_leakage_and_es_has_bad(data):
    al, rand, ref = data
    for seed in range(5):
        folds = shared_folds(ref, K, seed)
        for pool in (al, rand):
            for test in folds:
                tr, es = fold_split(pool, test, seed, ES_FRAC)
                tr_ids = {r["signal_id"] for r in tr}
                es_ids = {r["signal_id"] for r in es}
                assert not (set(test) & (tr_ids | es_ids))
                assert not (tr_ids & es_ids)
                assert len(tr_ids) == len(tr) and len(es_ids) == len(es)
                assert tr_ids | es_ids == set(pool) - set(test)
                assert sum(1 - r["label_int"] for r in es) >= 1


def test_fold_split_returns_copies(data):
    al, _, ref = data
    test = shared_folds(ref, K, 0)[0]
    tr, es = fold_split(al, test, 0, ES_FRAC)
    for r in tr + es:
        assert r is not al[r["signal_id"]]
        r["label_int"] = 1 - r["label_int"]
    assert all(al[r["signal_id"]]["label_int"] != r["label_int"] for r in tr + es)


def test_exclude_ids_never_tested_but_trainable(data):
    al, rand, ref = data
    revised = sorted(s for s in ref if al[s]["label_int"] != rand[s]["label_int"])
    assert len(revised) == 13
    folds = shared_folds(ref, K, 0, exclude_ids=revised)
    flat = sum(folds, [])
    assert not (set(flat) & set(revised))
    assert sorted(flat) == sorted(set(ref) - set(revised))
    for pool in (al, rand):
        tr, es = fold_split(pool, folds[0], 0, ES_FRAC)
        ids = {r["signal_id"] for r in tr + es}
        assert set(revised) <= ids
