from __future__ import annotations
import random
from typing import Mapping


def random_query(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n items uniformly without replacement. Deterministic by seed."""
    rng = random.Random(seed)
    pool_sorted = sorted(pool, key=lambda L: L["signal_id"])
    return rng.sample(pool_sorted, min(n, len(pool_sorted)))


def uncertainty_query(
    pool: list[dict], probs: Mapping[str, float], n: int,
) -> list[dict]:
    """Return the n entries from pool whose probs[signal_id] is closest to 0.5."""
    def key(L):
        p = probs.get(L["signal_id"])
        return abs(p - 0.5) if p is not None else 1e9
    return sorted(pool, key=key)[:n]


def fpfn_relabel(
    val: list[dict], probs: Mapping[str, float], threshold: float = 0.5,
) -> list[dict]:
    """Return validation entries where (p > threshold) disagrees with label_int."""
    flagged = []
    for L in val:
        p = probs.get(L["signal_id"])
        if p is None:
            continue
        pred = 1 if p > threshold else 0
        if pred != L["label_int"]:
            flagged.append(L)
    return flagged


def inject_label_noise(buf: list[dict], eta: float, seed: int) -> list[dict]:
    """Flip exactly floor(eta * len(buf)) labels uniformly at random.

    Returns a new list; original is unchanged.
    """
    if eta <= 0:
        return [dict(b) for b in buf]
    rng = random.Random(seed)
    out = [dict(b) for b in buf]
    n_flip = int(eta * len(out))
    indices = sorted(range(len(out)), key=lambda i: out[i]["signal_id"])
    rng.shuffle(indices)
    for i in indices[:n_flip]:
        out[i]["label_int"] = 1 - out[i]["label_int"]
    return out
