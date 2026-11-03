import random

import h5py
import numpy as np
import pytest

from Curation_Paper.eswa_full.lib import signals
from Curation_Paper.eswa_full.lib.protocol import (
    REPO_ROOT, h5_key, load_config, load_pools)
from Curation_Paper.eswa_full.lib.signals import load_universe, signals_for


@pytest.fixture(scope="module")
def universe():
    return load_universe(load_config())


def test_universe_shape(universe):
    ids, X = universe
    assert len(ids) == 12504
    assert X.shape == (12504, 500)
    assert X.dtype == np.float32
    assert len(set(ids)) == len(ids)


def test_ids_match_label_format(universe):
    ids, _ = universe
    al, rand = load_pools(load_config())
    idset = set(ids)
    assert set(al) <= idset and set(rand) <= idset


def test_universe_is_deterministic_and_cached(universe):
    ids, X = universe
    ids2, X2 = load_universe(load_config())
    assert ids2 == ids and X2 is X


def test_cached_universe_is_immutable(universe):
    ids, X = universe
    assert isinstance(ids, tuple)
    with pytest.raises(ValueError):
        X[0, 0] = 1.0
    with pytest.raises(ValueError):
        X -= 1.0
    with pytest.raises(AttributeError):
        ids.sort()
    with pytest.raises(TypeError):
        ids[0] = "x"


def test_load_universe_rejects_short_dataset(tmp_path):
    p = tmp_path / "short.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("FOD4/50kHz/act1/recv0/cycle200/data",
                         data=np.zeros((2, 100), dtype=np.float32))
    cfg = load_config()
    cfg["signals_h5"] = str(p)
    with pytest.raises(ValueError, match="cycle200"):
        load_universe(cfg)


@pytest.mark.skipif(not (REPO_ROOT / "dataset.h5").is_file(),
                    reason="the raw research-repository HDF5 is not published")
def test_first_500_equal_old_dataset(universe):
    ids, X = universe
    cfg = load_config()
    al, rand = load_pools(cfg)
    rows = {**rand, **al}
    pick = random.Random(0).sample(sorted(rows), 10)
    index = {s: i for i, s in enumerate(ids)}
    with h5py.File(REPO_ROOT / "dataset.h5", "r") as old:
        for sid in pick:
            r = rows[sid]
            ref = old[h5_key(r)][r["rep_idx"], :500]
            assert np.array_equal(X[index[sid]], ref), sid


def test_signals_for_matches_universe_and_window(universe):
    ids, X = universe
    cfg = load_config()
    pick = [ids[5], ids[9000], ids[12503]]
    out = signals_for(pick, cfg)
    assert out.shape == (3, 500)
    assert np.array_equal(out, X[[5, 9000, 12503]])
    wide = signals_for(pick, cfg, window=1000)
    assert wide.shape == (len(pick), 1000)
    assert np.array_equal(wide[:, :500], out)
    assert signals_for(pick, cfg, window=250).shape == (3, 250)


def test_signals_for_unknown_id_raises():
    with pytest.raises(KeyError):
        signals_for(["FOD9_50kHz_act1_recv0_c0_r0"], load_config())


def test_parse_id_roundtrip(universe):
    ids, _ = universe
    p = signals.parse_id(ids[0])
    assert signals.make_id(**p) == ids[0]
