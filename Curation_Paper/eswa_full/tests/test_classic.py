import numpy as np
import pytest

from Curation_Paper.eswa_full.lib.classic import features
from Curation_Paper.eswa_full.lib.fit import fit, predict
from Curation_Paper.eswa_full.tests.synth import make_pool, x_of


def test_features_shape_and_finite():
    X = x_of([r["signal_id"] for r in make_pool(12, 5)])
    F = features(X)
    assert F.shape == (12, 18) and np.isfinite(F).all()


def test_features_constant_signal_finite():
    F = features(np.ones((2, 500), dtype=np.float32))
    assert np.isfinite(F).all()


def test_features_band_energies_sum_to_one():
    X = x_of([r["signal_id"] for r in make_pool(6, 6)])
    assert np.allclose(features(X)[:, 10:18].sum(axis=1), 1.0, atol=1e-5)


def test_features_do_not_modify_input():
    X = x_of([r["signal_id"] for r in make_pool(4, 7)])
    X0 = X.copy()
    features(X)
    assert np.array_equal(X, X0)


@pytest.mark.parametrize("name", ["hgb", "logreg", "minirocket"])
def test_classic_models_fit_and_predict(name):
    train, es = make_pool(40, 8), make_pool(20, 9, prefix="e")
    model, info = fit(name, train, es, x_of, seed=0, recipe={})
    assert set(info) == {"es_loss", "epochs", "train_seconds", "device"}
    p = predict(model, x_of([r["signal_id"] for r in es]))
    assert p.shape == (20,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name", ["hgb", "logreg", "minirocket"])
def test_classic_models_deterministic(name):
    train, es = make_pool(40, 8), make_pool(20, 9, prefix="e")
    X = x_of([r["signal_id"] for r in es])
    a = predict(fit(name, train, es, x_of, seed=2, recipe={})[0], X)
    b = predict(fit(name, train, es, x_of, seed=2, recipe={})[0], X)
    assert np.allclose(a, b, atol=1e-9)
