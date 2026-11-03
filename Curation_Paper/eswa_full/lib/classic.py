"""Classical (non-neural) baselines on the same per-signal z-scored signals as the CNNs.

* ``hgb``        HistGradientBoostingClassifier on ``features``; ``max_iter`` chosen from
                 {100, 200, 400} by log-loss on the early-stopping set.
* ``logreg``     StandardScaler + LogisticRegression on ``features``.
* ``minirocket`` aeon MiniRocket transform + RidgeClassifierCV on the raw signal;
                 p(GOOD) = sigmoid(ridge decision function) (a monotone map, not a
                 calibrated probability).
All use class_weight="balanced"; the early-stopping set is used only to pick ``max_iter``
(hgb) and to report ``es_loss`` (log-loss of the final model).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSIC = ("hgb", "logreg", "minirocket")
HGB_ITERS = (100, 200, 400)
_EPS = 1e-8
N_BANDS = 8
FEATURE_NAMES = (
    "mean", "std", "skewness", "kurtosis", "peak_to_peak", "energy", "rms", "crest_factor",
    "top5_fft_power_fraction", "spectral_centroid",
    *[f"band_energy_{i}" for i in range(N_BANDS)],
)


def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Per-signal z-score, same formula as Curation_Paper/lib/train.py (std + 1e-8)."""
    X = np.asarray(X, dtype=np.float32)
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + _EPS)


def features(X: np.ndarray) -> np.ndarray:
    """(n, L) -> (n, 18).  Statistics: mean, std, skewness, kurtosis, peak-to-peak,
    energy (sum of squares), RMS, crest factor (8).  Spectral on the one-sided rfft power
    spectrum without DC: top-5 bin power fraction, spectral centroid (cycles/sample),
    fraction of power in 8 equal-width bands (10).  Never modifies ``X``."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    c = X - mu[:, None]
    m3 = (c ** 3).mean(axis=1)
    m4 = (c ** 4).mean(axis=1)
    skew = m3 / (sd ** 3 + _EPS)
    kurt = m4 / (sd ** 4 + _EPS)
    ptp = X.max(axis=1) - X.min(axis=1)
    energy = (X ** 2).sum(axis=1)
    rms = np.sqrt((X ** 2).mean(axis=1))
    crest = np.abs(X).max(axis=1) / (rms + _EPS)
    P = np.abs(np.fft.rfft(X, axis=1)) ** 2
    P = P[:, 1:]                                   # drop DC
    tot = P.sum(axis=1) + _EPS
    top5 = np.sort(P, axis=1)[:, -5:].sum(axis=1) / tot
    freqs = np.fft.rfftfreq(X.shape[1])[1:]
    centroid = (P * freqs).sum(axis=1) / tot
    edges = np.linspace(0, P.shape[1], N_BANDS + 1).astype(int)
    bands = np.stack([P[:, edges[i]:edges[i + 1]].sum(axis=1) for i in range(N_BANDS)], axis=1) / tot[:, None]
    return np.column_stack([mu, sd, skew, kurt, ptp, energy, rms, crest, top5, centroid, bands])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class ClassicModel:
    """Fitted classical model; ``predict_proba_good`` maps raw signals to p(GOOD)."""

    def __init__(self, name: str, clf, zscore: bool, transform=None) -> None:
        self.name, self.clf, self.zscore, self.transform = name, clf, zscore, transform

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        X = zscore_rows(X) if self.zscore else np.array(X, dtype=np.float32, copy=True)
        if self.name == "minirocket":
            return self.transform.transform(X[:, None, :])
        return features(X)

    def predict_proba_good(self, X: np.ndarray) -> np.ndarray:
        Z = self._inputs(X)
        if self.name == "minirocket":
            return _sigmoid(self.clf.decision_function(Z))
        return self.clf.predict_proba(Z)[:, 1]


#: classical-model knobs settable through ``recipe["model_cfg"]`` (R21); an absent knob keeps the
#: library default (hgb learning_rate 0.1, max_leaf_nodes 31; logreg C 1.0), so recipes without
#: ``model_cfg`` entries fit exactly what they fitted before.
CLASSIC_KNOBS = {"hgb": ("learning_rate", "max_leaf_nodes"), "logreg": ("C",), "minirocket": ()}


def _knobs(name: str, recipe: dict) -> dict:
    cfg = dict(recipe.get("model_cfg") or {})
    allowed = CLASSIC_KNOBS.get(name, ())
    bad = set(cfg) - set(allowed)
    if bad:
        raise ValueError(f"{name} takes model_cfg keys {allowed}, got {sorted(bad)}")
    return cfg


def fit_classic(name: str, Xtr: np.ndarray, ytr: np.ndarray, Xes: np.ndarray,
                yes: np.ndarray, *, seed: int, recipe: dict) -> tuple[ClassicModel, dict]:
    """Xtr / Xes are raw signals; the z-score (recipe["zscore"], default True) is applied here."""
    zs = bool(recipe.get("zscore", True))
    tr = zscore_rows(Xtr) if zs else np.array(Xtr, dtype=np.float32, copy=True)
    es = zscore_rows(Xes) if zs else np.array(Xes, dtype=np.float32, copy=True)
    ytr, yes = np.asarray(ytr, dtype=int), np.asarray(yes, dtype=int)
    labels = [0, 1]
    knobs = _knobs(name, recipe)
    if name == "hgb":
        Ftr, Fes = features(tr), features(es)
        best = None
        for it in HGB_ITERS:
            clf = HistGradientBoostingClassifier(max_iter=it, class_weight="balanced",
                                                 early_stopping=False, random_state=seed,
                                                 **knobs)
            clf.fit(Ftr, ytr)
            ll = float(log_loss(yes, clf.predict_proba(Fes)[:, 1], labels=labels))
            if best is None or ll < best[0] - 1e-12:
                best = (ll, it, clf)
        es_loss, epochs, clf = best
        return ClassicModel(name, clf, zs), {"es_loss": es_loss, "epochs": epochs,
                                             "train_seconds": 0.0, "device": "cpu"}
    if name == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed, **knobs))
        clf.fit(features(tr), ytr)
        model = ClassicModel(name, clf, zs)
        es_loss = float(log_loss(yes, model.predict_proba_good(Xes), labels=labels))
        return model, {"es_loss": es_loss, "epochs": 0, "train_seconds": 0.0, "device": "cpu"}
    if name == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        tf = MiniRocket(n_kernels=int(recipe.get("minirocket_kernels", 10000)),
                        random_state=seed, n_jobs=1)
        Ztr = tf.fit_transform(tr[:, None, :])
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")
        clf.fit(Ztr, ytr)
        model = ClassicModel(name, clf, zs, transform=tf)
        p = np.clip(model.predict_proba_good(Xes), 1e-7, 1 - 1e-7)
        return model, {"es_loss": float(log_loss(yes, p, labels=labels)), "epochs": 0,
                       "train_seconds": 0.0, "device": "cpu"}
    raise ValueError(f"unknown classic model {name!r}; choose from {CLASSIC}")
