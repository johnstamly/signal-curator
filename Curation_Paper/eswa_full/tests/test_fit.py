import copy
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from Curation_Paper.eswa_full.lib.fit import fit, predict
from Curation_Paper.eswa_full.tests.synth import make_pool, x_of

REPO = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).with_name("_fit_probe.py")
FAST = {"max_epochs": 4}


def _pools():
    return make_pool(80, 0), make_pool(30, 1, prefix="e")


def _weights(model):
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])


def test_fit_uses_only_es_for_stopping():
    train, es = _pools()
    test = make_pool(40, 2, prefix="x")
    seen = []

    def spy(ids):
        seen.extend(ids)
        return x_of(ids)

    fit("microconv1d", train, es, spy, seed=0, recipe=FAST)
    test_ids = {r["signal_id"] for r in test}
    assert not (set(seen) & test_ids)
    assert set(seen) == {r["signal_id"] for r in train + es}


def test_info_and_prediction_shape():
    train, es = _pools()
    model, info = fit("microconv1d", train, es, x_of, seed=0, recipe=FAST)
    assert set(info) == {"es_loss", "epochs", "train_seconds", "device"}
    assert 1 <= info["epochs"] <= 4 and info["es_loss"] > 0
    p = predict(model, x_of([r["signal_id"] for r in es]))
    assert p.shape == (len(es),) and ((p >= 0) & (p <= 1)).all()


def test_predict_does_not_mutate_input():
    train, es = _pools()
    model, _ = fit("microconv1d", train, es, x_of, seed=0, recipe=FAST)
    X = x_of([r["signal_id"] for r in es])
    X0 = X.copy()
    X.setflags(write=False)
    predict(model, X)
    assert np.array_equal(X, X0)


def test_same_seed_same_weights_in_process():
    train, es = _pools()
    a, _ = fit("microconv1d", train, es, x_of, seed=0, recipe=FAST)
    b, _ = fit("microconv1d", train, es, x_of, seed=0, recipe=FAST)
    c, _ = fit("microconv1d", train, es, x_of, seed=1, recipe=FAST)
    assert torch.equal(_weights(a), _weights(b))
    assert not torch.equal(_weights(a), _weights(c))


@pytest.mark.parametrize("arch", ["microconv1d", "bigru"])
def test_same_seed_same_predictions_separate_process(tmp_path, arch):
    outs = []
    for hs in ("0", "1"):
        out = tmp_path / f"p{hs}.npy"
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(REPO))
        r = subprocess.run([sys.executable, str(PROBE), str(out), arch], env=env,
                           cwd=str(REPO), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(np.load(out))
    assert np.max(np.abs(outs[0] - outs[1])) < 1e-6


def test_init_state_warm_start_is_used():
    train, es = _pools()
    base, _ = fit("microconv1d", train, es, x_of, seed=0, recipe=FAST)
    warm, _ = fit("microconv1d", train, es, x_of, seed=0, recipe={"max_epochs": 1},
                  init_state=copy.deepcopy(base.state_dict()))
    cold, _ = fit("microconv1d", train, es, x_of, seed=0, recipe={"max_epochs": 1})
    assert not torch.equal(_weights(warm), _weights(cold))


E6_KNOBS = [
    {"activation": "relu"},
    {"class_weight": "none"},
    {"class_weight": "full"},
    {"augment": False},
    {"zscore": False},
    {"window": 250},
]


@pytest.mark.parametrize("knob", E6_KNOBS, ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()))
def test_recipe_knobs_change_training(knob):
    train, es = _pools()
    base, _ = fit("microconv1d", train, es, x_of, seed=0, recipe=FAST)
    xo = partial(x_of, window=knob["window"]) if "window" in knob else x_of
    other, _ = fit("microconv1d", train, es, xo, seed=0, recipe={**FAST, **knob})
    wa, wb = _weights(base), _weights(other)
    assert wa.shape != wb.shape or not torch.equal(wa, wb)


def test_window_sets_input_len():
    train, es = _pools()
    model, _ = fit("microconv1d", train, es, partial(x_of, window=250), seed=0,
                   recipe={**FAST, "window": 250})
    p = predict(model, x_of([r["signal_id"] for r in es], window=250))
    assert p.shape == (len(es),)


def test_unknown_recipe_key_rejected():
    train, es = _pools()
    with pytest.raises(ValueError):
        fit("microconv1d", train, es, x_of, seed=0, recipe={"max_epoch": 3})


def test_unknown_model_rejected():
    train, es = _pools()
    with pytest.raises(ValueError):
        fit("nope", train, es, x_of, seed=0, recipe=FAST)
