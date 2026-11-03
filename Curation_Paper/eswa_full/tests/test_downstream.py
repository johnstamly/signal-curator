"""Task 13: E11 (damage index, curated set, monotonicity, trendability) and E14 helpers.
Synthetic fixtures with hand-computable answers; no real data, CPU only."""
import math

import numpy as np
import pytest

from Curation_Paper.eswa_full.lib import downstream as D
from Curation_Paper.eswa_full.lib.signals import make_id

L = 4  # window length of the toy signals


def sid(panel, act, rec, cyc, rep):
    return make_id(panel, act, rec, cyc, rep)


def sig(v):
    return [float(v)] * L


def build(rows):
    """rows: (panel, act, rec, cyc, rep, value) -> ids, X."""
    ids = [sid(*r[:5]) for r in rows]
    return ids, np.array([sig(r[5]) for r in rows])


# Path P: cycle 0 (ignored), then 200/400/600 with two repetitions each.
P_ROWS = [("F", 1, 1, 0, 0, 100.0), ("F", 1, 1, 0, 1, 100.0),
          ("F", 1, 1, 200, 0, 1.0), ("F", 1, 1, 200, 1, 3.0),
          ("F", 1, 1, 400, 0, 4.0), ("F", 1, 1, 400, 1, 6.0),
          ("F", 1, 1, 600, 0, 7.0), ("F", 1, 1, 600, 1, 9.0)]
P = ("F", 1, 1)


def test_di_uncurated_hand_values():
    ids, X = build(P_ROWS)
    di, c = D.damage_indices(ids, X, np.ones(len(ids), bool))
    # b = mean(1,3) = 2: RMS = 1,1 -> 1 ; (2,4) -> 3 ; (5,7) -> 6
    assert di[P] == {200: pytest.approx(1.0), 400: pytest.approx(3.0), 600: pytest.approx(6.0)}
    assert c["n_signals_below_min_cycle"] == 2          # cycle 0 is not part of the DI
    assert c["n_path_cycles_total"] == 3 and c["n_path_cycles_missing"] == 0


def test_di_curated_baseline_uses_kept_reps_only():
    ids, X = build(P_ROWS)
    keep = np.ones(len(ids), bool)
    keep[3] = False   # c200 r1
    keep[6] = False   # c600 r0
    di, c = D.damage_indices(ids, X, keep)
    # b = 1.0 (rep 0 only): DI200 = 0, DI400 = mean(3, 5) = 4, DI600 = 8 (rep 1 only)
    assert di[P] == {200: pytest.approx(0.0), 400: pytest.approx(4.0), 600: pytest.approx(8.0)}


def test_missing_path_cycle_is_missing_and_counted():
    ids, X = build(P_ROWS)
    keep = np.ones(len(ids), bool)
    keep[4] = keep[5] = False   # both c400 reps removed
    di, c = D.damage_indices(ids, X, keep)
    assert 400 not in di[P] and set(di[P]) == {200, 600}
    assert c["n_path_cycles_missing"] == 1 and c["n_path_cycles_present"] == 2


def test_path_without_kept_baseline_is_dropped_and_counted():
    rows = P_ROWS + [("F", 2, 1, 200, 0, 1.0), ("F", 2, 1, 200, 1, 2.0),
                     ("F", 2, 1, 400, 0, 5.0), ("F", 2, 1, 400, 1, 6.0)]
    ids, X = build(rows)
    keep = np.ones(len(ids), bool)
    keep[8] = keep[9] = False   # path (F,2,1): all baseline reps removed, later cycle kept
    di, c = D.damage_indices(ids, X, keep)
    assert ("F", 2, 1) not in di and P in di
    assert c["n_paths_dropped_no_kept_baseline"] == 1
    assert c["n_path_cycles_in_dropped_paths"] == 2


def test_baseline_cycle_is_first_cycle_at_least_200_of_all_signals():
    # the path's cycles are 0, 100, 300: the baseline cycle is 300 even if 300 is sparse
    rows = [("F", 1, 1, 0, 0, 50.0), ("F", 1, 1, 100, 0, 60.0),
            ("F", 1, 1, 300, 0, 2.0), ("F", 1, 1, 500, 0, 5.0)]
    ids, X = build(rows)
    di, _ = D.damage_indices(ids, X, np.ones(4, bool))
    assert di[P] == {300: pytest.approx(0.0), 500: pytest.approx(3.0)}


def test_curated_mask_and_conflicting_labels_resolve_to_al():
    ids = ["a", "b", "c", "d", "e", "f"]
    al = {"a": {"label_int": 1}, "b": {"label_int": 0}, "c": {"label_int": 1}}
    rand = {"a": {"label_int": 0},      # AL GOOD, rand BAD -> GOOD (AL wins): kept
            "b": {"label_int": 1},      # AL BAD, rand GOOD -> BAD (AL wins): removed
            "d": {"label_int": 0},      # rand-only BAD: removed
            "e": {"label_int": 1}}      # rand-only GOOD: kept even though p is low
    p = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    p[3] = 0.9                          # d: BAD label beats a high p
    p[4] = 0.1                          # e: GOOD label beats a low p
    p[5] = 0.1                          # f unlabelled, p < 0.5 -> removed
    keep = D.curated_mask(ids, p, al, rand)
    assert keep.tolist() == [True, False, True, False, True, False]
    p2 = list(p)
    p2[5] = 0.5                         # unlabelled with p == 0.5 is kept (>=)
    assert D.curated_mask(ids, p2, al, rand)[5]


def test_spearman_and_nan_guards():
    assert D.spearman([1, 2, 3, 4], [10, 20, 15, 40]) == pytest.approx(0.8)
    assert D.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(D.spearman([1, 2], [1, 2]))            # too few points


@pytest.mark.filterwarnings("error")   # an unguarded constant input would warn in numpy
def test_constant_trajectory_gives_nan_counted_not_crash():
    paths = {("F", 1, 1): {200: 0.0, 400: 0.0, 600: 0.0},     # constant
             ("F", 2, 1): {200: 1.0, 400: 2.0, 600: 3.0},
             ("F", 3, 1): {200: 1.0, 400: 3.0, 600: 9.0}}
    m, mono = D.panel_metrics(paths)
    assert math.isnan(mono[("F", 1, 1)])
    assert m["n_paths_mono_nan"] == 1
    assert m["monotonicity"] == pytest.approx(1.0)           # nan path left out of the mean
    # pairs with the constant path are undefined, the (2,3) pair is perfectly correlated
    assert m["n_trend_pairs_total"] == 3 and m["n_trend_pairs_defined"] == 1
    assert m["trend_min"] == pytest.approx(np.corrcoef([1, 2, 3], [1, 3, 9])[0, 1])


def test_trendability_hand_values():
    paths = {("F", 1, 1): {200: 1.0, 400: 2.0, 600: 3.0},
             ("F", 2, 1): {200: 2.0, 400: 4.0, 600: 6.0},
             ("F", 3, 1): {200: 3.0, 400: 2.0, 600: 1.0}}
    tmin, tmean, nd, nt = D.trendability(paths)
    assert (nd, nt) == (3, 3)
    assert tmin == pytest.approx(-1.0)
    assert tmean == pytest.approx((1.0 - 1.0 - 1.0) / 3)


def test_trendability_uses_common_cycles_only():
    paths = {("F", 1, 1): {200: 1.0, 400: 2.0, 600: 3.0, 800: 100.0},
             ("F", 2, 1): {200: 2.0, 400: 4.0, 600: 6.0}}
    tmin, tmean, nd, nt = D.trendability(paths)
    assert tmin == pytest.approx(1.0) and (nd, nt) == (1, 1)


def test_panel_di_is_median_over_paths_per_cycle():
    paths = {("F", 1, 1): {200: 1.0, 400: 10.0}, ("F", 2, 1): {200: 3.0, 400: 20.0},
             ("F", 3, 1): {200: 2.0}}
    assert D.panel_di(paths) == {200: 2.0, 400: 15.0}


def test_compare_conditions_paired_difference_sign():
    unc = {("F", 1, 1): {200: 1.0, 400: 3.0, 600: 2.0}, ("F", 2, 1): {200: 1.0, 400: 3.0, 600: 2.0}}
    cur = {("F", 1, 1): {200: 1.0, 400: 2.0, 600: 3.0}, ("F", 2, 1): {200: 1.0, 400: 2.0, 600: 3.0}}
    out = D.compare_conditions(cur, unc)["F"]
    assert out["curated"]["monotonicity"] == pytest.approx(1.0)
    assert out["uncurated"]["monotonicity"] == pytest.approx(0.5)
    assert out["difference"]["monotonicity"] == pytest.approx(0.5)
    assert out["path_paired_monotonicity"]["n_paths_paired"] == 2


def test_revised_signals_extraction_and_direction():
    def r(label, ts):
        return {"label": label, "label_int": int(label == "GOOD"), "labeled_at": ts}
    al = {"s1": r("BAD", "2026-05-07 12:00:00"), "s2": r("GOOD", "2026-05-04 10:00:00"),
          "s3": r("GOOD", "2026-05-01 09:00:00"), "only_al": r("BAD", "2026-05-02 09:00:00")}
    rand = {"s1": r("GOOD", "2026-05-01 09:00:00"), "s2": r("BAD", "2026-05-07 06:00:00"),
            "s3": r("GOOD", "2026-05-01 09:00:00")}
    rows = D.revised_signals(al, rand)
    assert [x["signal_id"] for x in rows] == ["s1", "s2"]
    s1, s2 = rows
    assert (s1["L_rand_label"], s1["L_AL_label"], s1["L_AL_labeled_at"]) == \
        ("GOOD", "BAD", "2026-05-07 12:00:00")
    assert s1["direction_rand_to_AL"] == "GOOD->BAD" and s1["direction_in_time"] == "GOOD->BAD"
    assert s1["earlier_pool"] == "L_rand"
    assert s2["earlier_pool"] == "L_AL" and s2["direction_in_time"] == "GOOD->BAD"
    assert s2["direction_rand_to_AL"] == "BAD->GOOD"


def test_labels_per_day():
    pool = {"a": {"labeled_at": "2026-05-01 09:17:27"}, "b": {"labeled_at": "2026-05-01 18:00:00"},
            "c": {"labeled_at": "2026-05-02 01:00:00"}}
    assert D.labels_per_day(pool) == {"2026-05-01": 2, "2026-05-02": 1}


def _row(label, ts):
    return {"label": label, "label_int": int(label == "GOOD"), "labeled_at": ts}


def test_seed_and_added_derived_from_random_pool_first_day():
    from Curation_Paper.eswa_full.lib import live_record as LR
    d1, d7 = "2026-05-01 09:00:00", "2026-05-07 09:00:00"
    al = {"a": _row("GOOD", d1), "b": _row("BAD", "2026-05-07 12:00:00"),   # b relabelled later
          "c": _row("GOOD", "2026-05-02 10:00:00"), "d": _row("GOOD", d7)}
    rand = {"a": _row("GOOD", d1), "b": _row("GOOD", d1),
            "d": _row("GOOD", d7), "e": _row("BAD", d7)}
    sa = LR.seed_and_added(al, rand)
    assert sa["seed_ids"] == ["a", "b"] and sa["added_ids"] == ["c", "d"]
    assert sa["n_seed_relabelled_in_L_AL"] == 1
    famb = {"iterations": [{"n_labels": 2}, {"n_labels": 4}], "band": 0.1}
    rec = LR.build_live_record(al, rand, famb)
    assert rec["cross_check_famb"]["seed_matches_first"] and rec["cross_check_famb"]["total_matches_last"]
    assert rec["revisions"]["n_revised"] == 1
    assert rec["revisions"]["direction_rand_to_AL"] == {"GOOD->BAD": 1, "BAD->GOOD": 0}
    assert "not logged live" in rec["famb_trajectory"]["provenance"]
    # restored per-day puts the relabelled seed row back on the first day
    assert rec["labels_per_day_L_AL_relabels_restored_INFERRED"]["2026-05-01"] == 2
    assert "copied seed" in rec["labels_per_day_note"] and "2026-05-01 (2)" in rec["labels_per_day_note"]


def test_labels_only_mask_removes_only_reference_bad():
    ids = ["a", "b", "c", "d", "e"]
    al = {"a": {"label_int": 1}, "b": {"label_int": 0}}
    rand = {"a": {"label_int": 0},      # conflict: AL GOOD wins -> kept
            "b": {"label_int": 1},      # conflict: AL BAD wins -> removed
            "c": {"label_int": 0}}      # rand-only BAD -> removed
    # d, e unlabelled: kept regardless (no model removal exists in this condition)
    assert D.labels_only_mask(ids, al, rand).tolist() == [True, False, False, True, True]


def test_restrict_to_common_drops_cycles_and_paths_absent_in_reference():
    ref = {("F", 1, 1): {200: 1.0, 600: 3.0}}
    unc = {("F", 1, 1): {200: 1.0, 400: 9.0, 600: 3.0}, ("F", 2, 1): {200: 1.0}}
    out = D.restrict_to_common(unc, ref)
    assert out == {("F", 1, 1): {200: 1.0, 600: 3.0}}
    assert D.count_path_cycles(out) == 2


def test_common_cycle_control_changes_the_comparison():
    ref = {("F", 1, 1): {200: 1.0, 400: 2.0, 800: 3.0}, ("F", 2, 1): {200: 1.0, 400: 2.0, 800: 3.0}}
    unc = {p: {200: 1.0, 400: 2.0, 600: 0.0, 800: 3.0} for p in ref}   # extra cycle breaks the trend
    full = D.compare_conditions(ref, unc)["F"]["difference"]["monotonicity"]
    common = D.compare_conditions(ref, D.restrict_to_common(unc, ref))["F"]["difference"]["monotonicity"]
    assert full > 0 and common == pytest.approx(0.0)
