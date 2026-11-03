"""E14 live-session record: pure builders over the two label pools and the f_amb trajectory file.

Nothing is typed in: seed and added counts are derived from the pools' ``labeled_at`` stamps and
cross-checked against the first and last entries of ``famb_trajectory.json``."""
from __future__ import annotations

from .downstream import labels_per_day, revised_signals

LABELS_PER_DAY_NOTE = ("L_rand count on {first_day} ({n}) is the copied seed (made in the L_AL session), "
                      "not random-session labelling; the random session labelled the remaining days")
FAMB_NOTE = ("reconstructed afterwards (each iteration's classifier retrained once), "
             "not logged live")


def seed_and_added(al: dict, rand: dict) -> dict:
    """Derive the seed set and the added labels of the active-learning session.

    The random pool was started from the same seed labels (same IDs, same first-day stamps), so
    its first-day IDs identify the seed inside L_AL even where a later relabel moved the L_AL
    stamp.  Seed = L_AL IDs that L_rand stamped on its first day; added = the rest of L_AL."""
    first_day = min(str(r["labeled_at"])[:10] for r in rand.values())
    rand_first = {s for s, r in rand.items() if str(r["labeled_at"])[:10] == first_day}
    seed = sorted(s for s in al if s in rand_first)
    added = sorted(s for s in al if s not in rand_first)
    same_stamp = sum(1 for s in seed if al[s]["labeled_at"] == rand[s]["labeled_at"])
    return {"first_day": first_day, "seed_ids": seed, "added_ids": added,
            "n_seed": len(seed), "n_added": len(added), "n_total": len(al),
            "n_rand_first_day": len(rand_first),
            "n_rand_first_day_missing_from_L_AL": len(rand_first - set(al)),
            "n_seed_with_identical_stamp_in_both_pools": same_stamp,
            "n_seed_relabelled_in_L_AL": len(seed) - same_stamp,
            "derivation": "seed = L_AL IDs that L_rand stamped on its first day "
                          f"({first_day}); added = remaining L_AL IDs"}


def al_per_day_first_stamp(al: dict, rand: dict) -> dict[str, int]:
    """INFERRED: L_AL labels per day with each relabelled seed row put back on the earlier
    L_rand stamp (a relabel only moves ``labeled_at``)."""
    stamps = {}
    for s, r in al.items():
        ts = str(r["labeled_at"])
        if s in rand and r["label_int"] != rand[s]["label_int"]:
            ts = min(ts, str(rand[s]["labeled_at"]))
        stamps[s] = {"labeled_at": ts}
    return labels_per_day(stamps)


def build_live_record(al: dict, rand: dict, famb: dict) -> dict:
    sa = seed_and_added(al, rand)
    rev = revised_signals(al, rand)
    shared = sorted(set(al) & set(rand))
    its = famb["iterations"]
    cum = [it["n_labels"] for it in its]
    out = {
        "n_L_AL": len(al), "n_L_rand": len(rand), "n_shared": len(shared),
        "labels_per_day": {"L_AL": labels_per_day(al), "L_rand": labels_per_day(rand)},
        "labels_per_day_note": LABELS_PER_DAY_NOTE.format(first_day=sa["first_day"],
                                                           n=sa["n_rand_first_day"]),
        "labels_per_day_L_AL_relabels_restored_INFERRED": al_per_day_first_stamp(al, rand),
        "seed_and_added": {k: v for k, v in sa.items() if k not in ("seed_ids", "added_ids")},
        "seed_ids": sa["seed_ids"], "added_ids": sa["added_ids"],
        "cross_check_famb": {
            "first_iteration_n_labels": cum[0], "last_iteration_n_labels": cum[-1],
            "increments": [b - a for a, b in zip(cum, cum[1:])],
            "seed_matches_first": sa["n_seed"] == cum[0],
            "total_matches_last": sa["n_total"] == cum[-1],
            "added_equals_sum_of_increments": sa["n_added"] == cum[-1] - cum[0]},
        "revisions": {
            "n_shared": len(shared), "n_revised": len(rev),
            "revision_rate": len(rev) / len(shared) if shared else float("nan"),
            "direction_rand_to_AL": {
                "GOOD->BAD": sum(r["direction_rand_to_AL"] == "GOOD->BAD" for r in rev),
                "BAD->GOOD": sum(r["direction_rand_to_AL"] == "BAD->GOOD" for r in rev)},
            "direction_in_time": {
                "GOOD->BAD": sum(r["direction_in_time"] == "GOOD->BAD" for r in rev),
                "BAD->GOOD": sum(r["direction_in_time"] == "BAD->GOOD" for r in rev)},
            "earlier_pool": {"L_rand": sum(r["earlier_pool"] == "L_rand" for r in rev),
                             "L_AL": sum(r["earlier_pool"] == "L_AL" for r in rev)},
            "note": "reference label = L_AL; 'direction_rand_to_AL' reads L_rand label -> L_AL "
                    "label; 'direction_in_time' reads earlier stamp -> later stamp. A relabel "
                    "only moves labeled_at, so which pool holds the earlier verdict is read "
                    "from the stamps.",
            "signals": rev},
        "famb_trajectory": {"provenance": FAMB_NOTE, "band": famb.get("band"),
                            "source": famb.get("source"), "iterations": its},
    }
    return out
