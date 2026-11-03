import json
import math

import numpy as np
import pytest

from Curation_Paper.eswa_full.lib.metrics import bootstrap_ci, metrics

# Fixture: y = [1,1,1,0,0,0], p = [0.9, 0.6, 0.4, 0.5, 0.2, 0.05], thr = 0.5
#   pred GOOD iff p >= 0.5 -> [1,1,0,1,0,0]; matches y at idx 0,1,4,5 -> acc = 4/6
#   bad_recall = mean(p[y==0] < 0.5) = [0.5<0.5 F, 0.2 T, 0.05 T] = 2/3
#   good_retention = mean(p[y==1] >= 0.5) = [T, T, F] = 2/3
#   bal_acc = (2/3 + 2/3)/2 = 2/3
#   auroc: pos {0.9,0.6,0.4} vs neg {0.5,0.2,0.05}: 9 pairs
#     0.9 beats all 3; 0.6 beats all 3; 0.4 beats 0.2,0.05 (loses to 0.5) -> 8/9
Y = np.array([1, 1, 1, 0, 0, 0])
P = np.array([0.9, 0.6, 0.4, 0.5, 0.2, 0.05])


def test_basic_metrics_hand_computed():
    m = metrics(Y, P)
    assert m["acc"] == pytest.approx(4 / 6)
    assert m["bad_recall"] == pytest.approx(2 / 3)
    assert m["good_retention"] == pytest.approx(2 / 3)
    assert m["bal_acc"] == pytest.approx(2 / 3)
    assert m["auroc"] == pytest.approx(8 / 9)
    assert m["n"] == 6
    assert m["n_bad"] == 3
    assert set(m) == {"acc", "bal_acc", "bad_recall", "good_retention", "auroc", "ece", "n", "n_bad"}


def test_threshold_changes_predictions():
    # thr=0.3: pred GOOD iff p>=0.3 -> [1,1,1,1,0,0]; bad_recall = 2/3 (0.2,0.05),
    # good_retention = 3/3 = 1, acc = 5/6
    m = metrics(Y, P, thr=0.3)
    assert m["bad_recall"] == pytest.approx(2 / 3)
    assert m["good_retention"] == pytest.approx(1.0)
    assert m["acc"] == pytest.approx(5 / 6)
    assert m["bal_acc"] == pytest.approx(5 / 6)


def test_ece_hand_computed():
    # y=[1,0,1,0], p=[0.95,0.05,0.55,0.15]; 10 bins width 0.1
    #   bin 9 [0.9,1.0]: p=0.95 y=1 -> |1-0.95| = 0.05, weight 1/4
    #   bin 0 [0,0.1):   p=0.05 y=0 -> 0.05, weight 1/4
    #   bin 5 [0.5,0.6): p=0.55 y=1 -> 0.45, weight 1/4
    #   bin 1 [0.1,0.2): p=0.15 y=0 -> 0.15, weight 1/4
    #   ECE = (0.05+0.05+0.45+0.15)/4 = 0.175
    y = np.array([1, 0, 1, 0])
    p = np.array([0.95, 0.05, 0.55, 0.15])
    assert metrics(y, p)["ece"] == pytest.approx(0.175)


def test_ece_multi_sample_bin_and_closed_last_bin():
    # p=1.0 lands in the last bin (closed on right). y=[1,0,1], p=[1.0,0.92,0.0]
    #   bin 9: p={1.0,0.92} y={1,0}: |0.5 - 0.96| = 0.46, weight 2/3
    #   bin 0: p=0.0 y=1: |1-0| = 1, weight 1/3
    #   ECE = 0.46*2/3 + 1/3 = 0.30667 + 0.33333 = 0.64
    y = np.array([1, 0, 1])
    p = np.array([1.0, 0.92, 0.0])
    assert metrics(y, p)["ece"] == pytest.approx(0.64)


def test_no_bad_gives_nan_not_exception():
    m = metrics(np.array([1, 1, 1]), np.array([0.9, 0.6, 0.2]))
    assert math.isnan(m["bad_recall"])
    assert math.isnan(m["bal_acc"])
    assert math.isnan(m["auroc"])
    assert m["good_retention"] == pytest.approx(2 / 3)
    assert m["n_bad"] == 0
    assert m["acc"] == pytest.approx(2 / 3)


def test_no_good_gives_nan():
    m = metrics(np.array([0, 0]), np.array([0.1, 0.8]))
    assert math.isnan(m["good_retention"])
    assert math.isnan(m["bal_acc"])
    assert math.isnan(m["auroc"])
    assert m["bad_recall"] == pytest.approx(0.5)
    assert m["n_bad"] == 2


def test_returns_python_scalars_and_json_roundtrip():
    m = metrics(Y, P)
    for k, v in m.items():
        assert type(v) in (float, int), (k, type(v))
    assert json.loads(json.dumps(m)) == pytest.approx(m)
    # nan case still serialisable by python json (emits NaN)
    m2 = metrics(np.array([1, 1]), np.array([0.5, 0.6]))
    for k, v in m2.items():
        assert type(v) in (float, int), (k, type(v))
    json.dumps(m2)


@pytest.mark.parametrize(
    "y,p",
    [
        (np.array([1, 0]), np.array([0.5])),
        (np.array([1, 2]), np.array([0.5, 0.5])),
        (np.array([1, 0]), np.array([1.1, 0.5])),
        (np.array([1, 0]), np.array([0.5, -0.1])),
    ],
)
def test_input_validation(y, p):
    with pytest.raises(ValueError):
        metrics(y, p)


def test_bootstrap_ci_constant_and_single():
    lo, hi = bootstrap_ci([0.7, 0.7, 0.7, 0.7], n_boot=200)
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)
    assert bootstrap_ci([0.42]) == (0.42, 0.42)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = bootstrap_ci(v, n_boot=2000, seed=3)
    b = bootstrap_ci(v, n_boot=2000, seed=3)
    assert a == b
    assert a[0] <= np.mean(v) <= a[1]
    assert 0.1 <= a[0] and a[1] <= 0.5
    assert all(type(x) is float for x in a)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])
