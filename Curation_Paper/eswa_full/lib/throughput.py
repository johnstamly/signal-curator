"""E13 labelling throughput: per-day seconds-per-label from ``labeled_at`` gaps.

Rule (stated in the output JSON): within one calendar day (the first 10 characters of
``labeled_at``), sort the stamps and take the gap between each pair of consecutive labels.  A gap
strictly greater than ``BREAK_S`` seconds is a break and is excluded; a gap of exactly
``BREAK_S`` is kept.  Identical stamps give a gap of 0 s and are kept.  The first label of a day
has no preceding label, so a day with n labels has n-1 gaps.  Gaps never span midnight.
Quantiles use numpy's default linear interpolation; the IQR is q75 - q25."""
from __future__ import annotations

import datetime

import numpy as np

BREAK_S = 300
FMT = "%Y-%m-%d %H:%M:%S"

RULE = {
    "break_threshold_s": BREAK_S,
    "break_rule": "gap > 300 s between consecutive labels of the same day is a break and is "
                  "excluded; a gap of exactly 300 s is kept",
    "gap_definition": "difference of consecutive sorted labeled_at stamps within one calendar "
                      "day (first 10 characters of the stamp); gaps never span midnight; "
                      "identical stamps give a 0 s gap that is kept",
    "active_time_s": "sum of the kept gaps (the first label of a day starts the clock)",
    "quantiles": "numpy percentile, linear interpolation; IQR = q75 - q25",
    "overall_median": "median of all kept gaps of the pool, pooled over days",
    "l_rand_seed_exclusion": "L_rand rows that are seed rows (L_AL IDs stamped by L_rand on its first day, "
                             "the E14 seed definition, lib/live_record.seed_and_added) carry stamps copied "
                             "from L_AL when the random pool was created, not labelling events of the "
                             "random session; they are excluded from L_rand throughput. L_AL uses all rows.",
}


def _parse(ts: str) -> datetime.datetime:
    return datetime.datetime.strptime(str(ts), FMT)


def day_stats(stamps: list[str], break_s: float = BREAK_S) -> dict:
    """Statistics for the stamps of ONE day (any order)."""
    ts = sorted(_parse(s) for s in stamps)
    gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    kept = [g for g in gaps if g <= break_s]  # MUT:break
    out = {"n_labels": len(ts), "n_gaps": len(gaps), "n_breaks": len(gaps) - len(kept),
           "n_kept_gaps": len(kept), "n_zero_gaps": sum(g == 0 for g in kept),
           "active_s": float(sum(kept)), "kept_gaps_s": kept}
    if kept:
        q25, med, q75 = (float(x) for x in np.percentile(kept, [25, 50, 75]))
        out.update({"median_s": med, "q25_s": q25, "q75_s": q75, "iqr_s": q75 - q25})
    else:
        out.update({"median_s": None, "q25_s": None, "q75_s": None, "iqr_s": None})
    return out


def pool_stats(stamps: list[str], break_s: float = BREAK_S) -> dict:
    """Per-day and overall statistics for one pool's ``labeled_at`` stamps."""
    by_day: dict[str, list[str]] = {}
    for s in stamps:
        by_day.setdefault(str(s)[:10], []).append(s)
    days = {d: day_stats(v, break_s) for d, v in sorted(by_day.items())}
    pooled = [g for d in days.values() for g in d["kept_gaps_s"]]
    q = ([float(x) for x in np.percentile(pooled, [25, 50, 75])] if pooled else [None] * 3)
    overall = {"n_labels": len(stamps), "n_days": len(days),
               "n_kept_gaps": len(pooled),
               "n_breaks": sum(d["n_breaks"] for d in days.values()),
               "active_s": float(sum(pooled)), "median_s": q[1], "q25_s": q[0], "q75_s": q[2],
               "iqr_s": (q[2] - q[0]) if pooled else None}
    for d in days.values():  # per-gap lists stay in memory only
        d.pop("kept_gaps_s")
    return {"overall": overall, "per_day": days}


def session_stamps(pool: dict[str, dict], exclude_ids) -> list[str]:
    """``labeled_at`` stamps of the pool rows that are NOT in ``exclude_ids`` (rows whose stamp
    is a copy made when the pool was created rather than a labelling event of its own)."""
    excl = set(exclude_ids)
    return [r["labeled_at"] for sid, r in pool.items() if sid not in excl]  # MUT:seed


def pool_usage(pool: dict[str, dict], exclude_ids) -> tuple[list[str], dict]:
    """Stamps used for throughput plus the counts the JSON reports: rows in the pool, rows
    used, and excluded rows (excluded IDs that are actually in the pool)."""
    stamps = session_stamps(pool, exclude_ids)
    n_excl = sum(1 for sid in pool if sid in set(exclude_ids))
    return stamps, {"n_rows_in_pool": len(pool), "n_used": len(stamps),
                    "n_excluded_seed_rows": n_excl}
