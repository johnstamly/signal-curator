"""Task 15 (E13): gap and break logic on synthetic stamps with hand-computable answers."""
import pytest

from Curation_Paper.eswa_full.lib import throughput as T

D = "2026-05-01 "


def t(hms):
    return D + hms


def test_basic_gaps_median_iqr_and_break():
    # stamps 09:00:00, :10, :30, :40, 09:46:40 (gap 360 s -> break), 09:46:50
    st = [t("09:00:00"), t("09:00:10"), t("09:00:30"), t("09:00:40"), t("09:06:40"),
          t("09:06:50")]
    r = T.day_stats(st)
    # gaps: 10, 20, 10, 360, 10  -> kept 10,20,10,10
    assert r["n_labels"] == 6 and r["n_gaps"] == 5
    assert r["n_breaks"] == 1 and r["n_kept_gaps"] == 4
    assert r["active_s"] == 50.0
    assert r["median_s"] == 10.0
    # sorted kept 10,10,10,20 -> q25 = 10, q75 = 12.5 (linear interpolation)
    assert r["q25_s"] == 10.0 and r["q75_s"] == 12.5 and r["iqr_s"] == 2.5


def test_gap_of_exactly_300_s_is_kept_301_is_a_break():
    r = T.day_stats([t("10:00:00"), t("10:05:00")])  # exactly 300 s
    assert r["n_breaks"] == 0 and r["n_kept_gaps"] == 1 and r["median_s"] == 300.0
    r = T.day_stats([t("10:00:00"), t("10:05:01")])  # 301 s
    assert r["n_breaks"] == 1 and r["n_kept_gaps"] == 0 and r["median_s"] is None
    assert r["active_s"] == 0.0


def test_one_label_day_has_no_gaps():
    r = T.day_stats([t("12:00:00")])
    assert r["n_labels"] == 1 and r["n_gaps"] == 0 and r["n_breaks"] == 0
    assert r["median_s"] is None and r["iqr_s"] is None and r["active_s"] == 0.0


def test_identical_timestamps_give_zero_gaps_that_are_kept():
    r = T.day_stats([t("08:00:00")] * 3 + [t("08:00:04")])
    # gaps 0, 0, 4
    assert r["n_gaps"] == 3 and r["n_zero_gaps"] == 2 and r["n_breaks"] == 0
    assert r["median_s"] == 0.0 and r["active_s"] == 4.0


def test_input_order_does_not_matter():
    a = [t("09:00:30"), t("09:00:00"), t("09:00:10")]
    assert T.day_stats(a)["active_s"] == 30.0


def test_pool_per_day_split_and_no_gap_across_midnight():
    st = ["2026-05-01 23:59:50", "2026-05-02 00:00:05",  # different days: no gap
         "2026-05-02 00:00:15", "2026-05-02 00:00:45",
         "2026-05-03 09:00:00"]
    p = T.pool_stats(st)
    assert list(p["per_day"]) == ["2026-05-01", "2026-05-02", "2026-05-03"]
    assert p["per_day"]["2026-05-01"]["n_gaps"] == 0
    assert p["per_day"]["2026-05-03"]["n_gaps"] == 0
    d2 = p["per_day"]["2026-05-02"]
    assert d2["n_labels"] == 3 and d2["median_s"] == 20.0 and d2["active_s"] == 40.0
    ov = p["overall"]
    assert ov["n_labels"] == 5 and ov["n_days"] == 3 and ov["n_kept_gaps"] == 2
    assert ov["median_s"] == 20.0  # pooled gaps 10, 30 -> median 20
    assert ov["active_s"] == 40.0
    assert "kept_gaps_s" not in d2


def test_overall_median_is_over_pooled_gaps_not_median_of_day_medians():
    # day A gaps 10,10,10 (median 10); day B one gap 100 (median 100)
    st = [t("09:00:00"), t("09:00:10"), t("09:00:20"), t("09:00:30"),
          "2026-05-02 09:00:00", "2026-05-02 09:01:40"]
    ov = T.pool_stats(st)["overall"]
    assert ov["median_s"] == 10.0  # pooled 10,10,10,100


def test_overall_breaks_counted_and_rule_is_stated():
    st = [t("09:00:00"), t("09:10:00"), t("09:20:00"), t("09:20:05")]
    ov = T.pool_stats(st)["overall"]
    assert ov["n_breaks"] == 2 and ov["n_kept_gaps"] == 1 and ov["median_s"] == 5.0
    assert "exactly 300 s is kept" in T.RULE["break_rule"]
    assert T.RULE["break_threshold_s"] == 300


def test_empty_pool():
    p = T.pool_stats([])
    assert p["overall"]["n_labels"] == 0 and p["overall"]["median_s"] is None


def test_session_stamps_excludes_copied_seed_rows_and_changes_the_median():
    # 4 seed rows carry copied stamps at a 1 s cadence; 3 session rows are 20 s apart.
    seed = {f"s{i}": {"labeled_at": f"2026-05-01 09:00:0{i}"} for i in range(4)}
    sess = {f"a{i}": {"labeled_at": f"2026-05-07 10:00:{20 * i:02d}"} for i in range(3)}
    pool = {**seed, **sess}
    with_seed = T.pool_stats([r["labeled_at"] for r in pool.values()])["overall"]
    assert with_seed["median_s"] == 1.0 and with_seed["n_labels"] == 7  # copied stamps leak in
    used = T.session_stamps(pool, seed.keys())
    assert len(used) == 3
    ov = T.pool_stats(used)["overall"]
    assert ov["n_labels"] == 3 and ov["median_s"] == 20.0 and ov["n_days"] == 1


def test_session_stamps_ignores_unknown_ids_and_keeps_everything_when_none_excluded():
    pool = {"x": {"labeled_at": t("09:00:00")}, "y": {"labeled_at": t("09:00:05")}}
    assert sorted(T.session_stamps(pool, ["nope"])) == [t("09:00:00"), t("09:00:05")]
    assert T.session_stamps(pool, ["x"]) == [t("09:00:05")]


def test_rule_states_the_seed_exclusion():
    assert "seed" in T.RULE["l_rand_seed_exclusion"]


def test_pool_usage_counts_add_up_and_ignore_foreign_ids():
    pool = {f"s{i}": {"labeled_at": f"2026-05-01 09:00:0{i}"} for i in range(4)}
    pool.update({f"a{i}": {"labeled_at": f"2026-05-07 10:00:{20 * i:02d}"} for i in range(3)})
    stamps, n = T.pool_usage(pool, [f"s{i}" for i in range(4)] + ["not_in_pool"])
    assert n == {"n_rows_in_pool": 7, "n_used": 3, "n_excluded_seed_rows": 4}
    assert len(stamps) == 3
    assert n["n_used"] + n["n_excluded_seed_rows"] == n["n_rows_in_pool"]
    _, n0 = T.pool_usage(pool, [])
    assert n0 == {"n_rows_in_pool": 7, "n_used": 7, "n_excluded_seed_rows": 0}
