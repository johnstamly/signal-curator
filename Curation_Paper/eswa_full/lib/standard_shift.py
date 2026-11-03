"""E1 standard-shift evidence (ruling R36): pure builders over the two label pools and the E10 scores.

The cross-pool E1 comparison (L_AL session vs L_rand session) is confounded by a difference in
labelling standard: the random session marked a similar share of the seed BAD but a different share
of its own additions.  This module measures that difference and nothing else; it assigns no cause.

Groups (every signal belongs to exactly one of the four, and the seed to two of them under the two
sets of labels):
  seed_L_AL    the seed, labels as stamped in the L_AL session
  seed_L_rand  the same seed signals, labels as copied into / stamped in the L_rand session
  added_L_AL   labels the L_AL session added after the seed (queried by the loop)
  added_L_rand labels the L_rand session added after the seed (random order)

Seed = ``live_record.seed_and_added``.  The E10 scores are the mean p(GOOD) of a MicroConv1D trained
on ALL of L_AL, so the L_AL groups are in-sample for that model and the L_rand additions are
(except for signals also in L_AL) out-of-sample; ``model.n_in_L_AL_training`` makes that visible.
Nothing here reads a file: the script passes dicts in.
"""
from __future__ import annotations

from collections import Counter

from .live_record import seed_and_added
from .signals import parse_id

THRESHOLD = 0.5          # p(GOOD) >= 0.5 -> the model calls the signal GOOD (as in E10)
GROUPS = ("seed_L_AL", "seed_L_rand", "added_L_AL", "added_L_rand")
NOTE = ("The seed is defined by live_record.seed_and_added (L_AL IDs stamped on the first day of the "
        "L_rand session). 'model' columns use the E10 scores: a MicroConv1D trained on all of L_AL, "
        "so the two L_AL groups are in-sample for it. No cause of any difference is established.")


def group_ids(al: dict, rand: dict) -> dict[str, list[str]]:
    """The four groups' signal IDs (derived, never typed)."""
    sa = seed_and_added(al, rand)
    seed = set(sa["seed_ids"])
    rand_first = {s for s, r in rand.items()
                  if str(r["labeled_at"])[:10] == sa["first_day"]}
    if rand_first - set(al):
        raise ValueError("L_rand first-day IDs absent from L_AL: the seed is not defined by "
                         "seed_and_added alone")
    added_rand = sorted(s for s in rand if s not in seed)
    if added_rand != sorted(s for s in rand if s not in rand_first):
        raise ValueError("L_rand additions differ between the seed-based and the stamp-based definition")
    return {"seed_L_AL": sa["seed_ids"], "seed_L_rand": sa["seed_ids"],
            "added_L_AL": sa["added_ids"], "added_L_rand": added_rand}


def _fraction(num: int, den: int) -> float:
    return num / den if den else float("nan")


def group_stats(ids, labels: dict[str, dict], scores: dict[str, float], al_ids) -> dict:
    """Counts for one group.  ``labels`` = the pool whose labels define the group."""
    ids = list(ids)
    n = len(ids)
    bad = [s for s in ids if labels[s]["label_int"] == 0]
    good = [s for s in ids if labels[s]["label_int"] == 1]
    parsed = {s: parse_id(s) for s in ids}
    per_panel = {}
    for s in ids:
        pn = parsed[s]["panel"]
        d = per_panel.setdefault(pn, {"n": 0, "n_bad": 0})
        d["n"] += 1
        d["n_bad"] += labels[s]["label_int"] == 0
    for d in per_panel.values():
        d["bad_fraction"] = _fraction(d["n_bad"], d["n"])
    recv = Counter(parsed[s]["receiver"] for s in ids)
    cyc0 = sum(parsed[s]["cycle"] == 0 for s in ids)
    scored = [s for s in ids if s in scores]
    bad_scored = [s for s in scored if labels[s]["label_int"] == 0]
    good_scored = [s for s in scored if labels[s]["label_int"] == 1]
    bad_called_good = sum(scores[s] >= THRESHOLD for s in bad_scored)
    good_called_bad = sum(scores[s] < THRESHOLD for s in good_scored)
    return {
        "n": n, "n_bad": len(bad), "n_good": len(good), "bad_fraction": _fraction(len(bad), n),
        "per_panel": dict(sorted(per_panel.items())),
        "receiver": {str(r): {"n": c, "share": _fraction(c, n)} for r, c in sorted(recv.items())},
        "cycle_zero": {"n": cyc0, "share": _fraction(cyc0, n)},
        "model": {
            "threshold": THRESHOLD, "n_scored": len(scored), "n_missing_from_scores": n - len(scored),
            "n_bad_scored": len(bad_scored), "n_good_scored": len(good_scored),
            "n_bad_called_good": int(bad_called_good), "n_good_called_bad": int(good_called_bad),
            "n_disagree": int(bad_called_good + good_called_bad),
            "n_called_bad": sum(scores[s] < THRESHOLD for s in scored),
            "n_in_L_AL_training": sum(s in al_ids for s in ids)},
    }


def pool_counts(pool: dict) -> dict:
    n_good = sum(r["label_int"] for r in pool.values())
    return {"n": len(pool), "n_good": n_good, "n_bad": len(pool) - n_good}


def build_standard_shift(al: dict, rand: dict, score_ids, score_p) -> dict:
    """The whole payload (before provenance fields)."""
    if len(score_ids) != len(score_p):
        raise ValueError("e10_scores ids and mean_p differ in length")
    scores = {s: float(p) for s, p in zip(score_ids, score_p)}
    if len(scores) != len(score_ids):
        raise ValueError("e10_scores ids are not unique")
    gi = group_ids(al, rand)
    al_ids = set(al)
    groups = {}
    for name in GROUPS:
        labels = rand if name.endswith("L_rand") else al
        groups[name] = group_stats(gi[name], labels, scores, al_ids)
        groups[name]["pool_labels"] = "L_rand" if name.endswith("L_rand") else "L_AL"
    shared = sorted(set(al) & set(rand))
    seed = set(gi["seed_L_AL"])
    return {
        "note": NOTE, "groups": groups,
        "pools": {"L_AL": pool_counts(al), "L_rand": pool_counts(rand)},
        "n_shared": len(shared), "n_seed": len(seed),
        "n_shared_outside_seed": len(set(shared) - seed),
        "seed_ids_in_both_pools": all(s in al and s in rand for s in seed),
        "scores": {"n_ids": len(scores), "threshold": THRESHOLD,
                   "n_group_ids_missing": sum(g["model"]["n_missing_from_scores"] for g in groups.values())},
    }
