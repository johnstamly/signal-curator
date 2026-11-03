"""lib/pool_sizes.py on synthetic cell results."""
import pytest

from Curation_Paper.eswa_full.lib.pool_sizes import build_pool_sizes


def res(train, es, n_test):
    return {"len_train_pool": train, "len_es": es, "test_ids": [f"s{i}" for i in range(n_test)]}


def test_min_and_max_over_cells():
    out = build_pool_sizes([res(732, 129, 150), res(733, 129, 149), res(732, 130, 149)])
    assert out["len_train_pool"] == {"min": 732, "max": 733, "n_cells": 3}
    assert out["len_es"]["min"] == 129 and out["len_es"]["max"] == 130
    assert out["n_test"]["min"] == 149 and out["n_test"]["max"] == 150
    assert out["n_cells"] == 3


def test_refuses_empty_and_incomplete():
    with pytest.raises(ValueError):
        build_pool_sizes([])
    with pytest.raises(KeyError):
        build_pool_sizes([{"len_train_pool": 1, "test_ids": []}])
