"""Tests of lib/standard_shift.py on a synthetic two-pool fixture with hand-counted expectations."""
import json

import pytest

from Curation_Paper.eswa_full.lib import standard_shift as SS
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT


def sid(panel, act, recv, cyc, rep):
    return f"FOD{panel}_50kHz_act{act}_recv{recv}_c{cyc}_r{rep}"


def row(s, label, stamp):
    return {"signal_id": s, "label": "GOOD" if label else "BAD", "label_int": label, "labeled_at": stamp}


# seed: 8 signals stamped on the first day of L_rand; labels differ on one (BAD in L_AL, GOOD in L_rand)
SEED = [(sid(3, 1, 1, 0, 1), 1), (sid(3, 1, 2, 200, 1), 0), (sid(4, 1, 1, 0, 2), 1), (sid(4, 2, 3, 400, 1), 1),
        (sid(5, 1, 1, 600, 1), 0), (sid(5, 2, 2, 0, 3), 1), (sid(6, 1, 1, 800, 1), 1), (sid(6, 3, 3, 0, 4), 1)]
AL_ADDED = [(sid(3, 2, 1, 1000, 1), 0), (sid(4, 3, 2, 1200, 1), 0), (sid(5, 3, 3, 1400, 1), 1),
            (sid(6, 1, 2, 1600, 1), 1), (sid(7, 1, 1, 1800, 1), 0)]
RAND_ONLY = [(sid(3, 3, 3, 2000, 1), 0), (sid(4, 4, 1, 2200, 1), 0), (sid(5, 4, 2, 2400, 1), 0),
             (sid(7, 2, 2, 0, 5), 1)]
SHARED_OUTSIDE = AL_ADDED[2]          # in both pools, not in the seed


def pools():
    al, rand = {}, {}
    for s, lab in SEED:
        al[s] = row(s, lab, "2026-05-03 10:00:00")          # relabel moved the stamp: still the seed
        rand[s] = row(s, lab, "2026-05-01 09:00:00")
    flip = SEED[1][0]
    rand[flip] = row(flip, 1, "2026-05-01 09:00:00")        # BAD in L_AL, GOOD in L_rand
    for s, lab in AL_ADDED:
        al[s] = row(s, lab, "2026-05-04 10:00:00")
    for s, lab in RAND_ONLY:
        rand[s] = row(s, lab, "2026-05-07 10:00:00")
    s, lab = SHARED_OUTSIDE
    rand[s] = row(s, lab, "2026-05-07 11:00:00")
    return al, rand


def scores_for(al, rand):
    """GOOD-labelled (L_AL) signals get p=0.9 except the seed GOOD at index 0 (p=0.2); BAD get 0.1
    except the seed BAD at index 4 (p=0.8) and the L_rand-only BADs (0.7, 0.7, 0.1)."""
    p = {}
    for s, r in al.items():
        p[s] = 0.9 if r["label_int"] else 0.1
    p[SEED[0][0]] = 0.2
    p[SEED[4][0]] = 0.8
    p[RAND_ONLY[0][0]], p[RAND_ONLY[1][0]], p[RAND_ONLY[2][0]], p[RAND_ONLY[3][0]] = 0.7, 0.7, 0.1, 0.9
    return p


@pytest.fixture(scope="module")
def payload():
    al, rand = pools()
    p = scores_for(al, rand)
    ids = sorted(p)
    return SS.build_standard_shift(al, rand, ids, [p[s] for s in ids])


def test_groups_are_derived_from_the_stamps_not_typed():
    al, rand = pools()
    g = SS.group_ids(al, rand)
    assert len(g["seed_L_AL"]) == len(g["seed_L_rand"]) == 8
    assert len(g["added_L_AL"]) == 5 and len(g["added_L_rand"]) == 5   # 4 rand-only + 1 shared
    assert SHARED_OUTSIDE[0] in g["added_L_rand"] and SHARED_OUTSIDE[0] in g["added_L_AL"]


def test_seed_counts_differ_by_labelling_pool(payload):
    g = payload["groups"]
    assert (g["seed_L_AL"]["n"], g["seed_L_AL"]["n_bad"]) == (8, 2)
    assert (g["seed_L_rand"]["n"], g["seed_L_rand"]["n_bad"]) == (8, 1)
    assert g["seed_L_AL"]["bad_fraction"] == pytest.approx(0.25)


def test_added_counts_and_panels(payload):
    a, r = payload["groups"]["added_L_AL"], payload["groups"]["added_L_rand"]
    assert (a["n"], a["n_bad"], a["n_good"]) == (5, 3, 2)
    assert (r["n"], r["n_bad"], r["n_good"]) == (5, 3, 2)
    assert a["per_panel"]["FOD7"] == {"n": 1, "n_bad": 1, "bad_fraction": 1.0}
    assert sum(d["n"] for d in a["per_panel"].values()) == 5


def test_receiver_and_cycle_zero_shares(payload):
    s = payload["groups"]["seed_L_AL"]
    assert s["cycle_zero"] == {"n": 4, "share": 0.5}
    assert s["receiver"]["1"]["n"] == 4 and s["receiver"]["3"]["n"] == 2
    assert sum(v["share"] for v in s["receiver"].values()) == pytest.approx(1.0)


def test_model_disagreement_counts_both_directions(payload):
    g = payload["groups"]
    m = g["seed_L_AL"]["model"]
    assert m["n_bad_called_good"] == 1 and m["n_good_called_bad"] == 1     # BAD@0.8, GOOD@0.2
    m = g["seed_L_rand"]["model"]                      # labels from L_rand: the flipped one is GOOD@0.1
    assert m["n_bad_called_good"] == 1 and m["n_good_called_bad"] == 2
    m = g["added_L_rand"]["model"]
    assert m["n_bad_called_good"] == 2 and m["n_good_called_bad"] == 0     # two BADs at 0.7
    assert m["n_in_L_AL_training"] == 1                # only the shared signal was in L_AL


def test_threshold_is_inclusive_at_half():
    al, rand = pools()
    p = scores_for(al, rand)
    s = SEED[1][0]                                     # BAD in L_AL
    p[s] = 0.5
    ids = sorted(p)
    out = SS.build_standard_shift(al, rand, ids, [p[x] for x in ids])
    assert out["groups"]["seed_L_AL"]["model"]["n_bad_called_good"] == 2   # p >= 0.5 is GOOD


def test_missing_scores_are_reported_not_dropped_silently():
    al, rand = pools()
    p = scores_for(al, rand)
    gone = RAND_ONLY[3][0]
    del p[gone]
    ids = sorted(p)
    out = SS.build_standard_shift(al, rand, ids, [p[x] for x in ids])
    assert out["groups"]["added_L_rand"]["model"]["n_missing_from_scores"] == 1
    assert out["scores"]["n_group_ids_missing"] == 1


def test_pool_counts_and_shared_outside_seed(payload):
    assert payload["pools"]["L_AL"] == {"n": 13, "n_good": 8, "n_bad": 5}
    assert payload["pools"]["L_rand"] == {"n": 13, "n_good": 9, "n_bad": 4}
    assert payload["n_shared"] == 9 and payload["n_seed"] == 8 and payload["n_shared_outside_seed"] == 1
    assert payload["seed_ids_in_both_pools"] is True


def test_refuses_duplicate_or_misaligned_scores():
    al, rand = pools()
    with pytest.raises(ValueError):
        SS.build_standard_shift(al, rand, ["a", "a"], [0.1, 0.2])
    with pytest.raises(ValueError):
        SS.build_standard_shift(al, rand, ["a"], [0.1, 0.2])


def test_refuses_when_rand_first_day_ids_are_missing_from_al():
    al, rand = pools()
    extra = sid(9, 1, 1, 0, 9)
    rand[extra] = row(extra, 1, "2026-05-01 09:30:00")
    with pytest.raises(ValueError):
        SS.group_ids(al, rand)


def test_real_json_is_internally_consistent_when_present():
    p = PKG_ROOT / "results" / "e01_standard_shift.json"
    if not p.is_file():
        pytest.skip("real e01_standard_shift.json not generated yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    for g in d["groups"].values():
        assert g["n"] == g["n_bad"] + g["n_good"]
        assert sum(v["n"] for v in g["per_panel"].values()) == g["n"]
        assert g["model"]["n_scored"] + g["model"]["n_missing_from_scores"] == g["n"]
    assert d["groups"]["seed_L_AL"]["n"] == d["groups"]["seed_L_rand"]["n"] == d["n_seed"]
    assert d["pools"]["L_AL"]["n"] == sum(d["groups"][k]["n"] for k in ("seed_L_AL", "added_L_AL"))
    assert d["pools"]["L_rand"]["n"] == sum(d["groups"][k]["n"] for k in ("seed_L_rand", "added_L_rand"))
