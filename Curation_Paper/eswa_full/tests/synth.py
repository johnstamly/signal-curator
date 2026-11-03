"""Synthetic signals + pools for fast, disk-free tests of the training engine."""
from __future__ import annotations

import zlib

import numpy as np

_LABELS: dict[str, int] = {}


def _sig(sid: str, label: int, length: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    t = np.arange(length)
    f = 0.05 if label == 1 else 0.08
    base = np.sin(2 * np.pi * f * t) * np.exp(-t / 300.0) + 0.3 * rng.standard_normal(length)
    return (base * (1 + 3 * rng.random())).astype(np.float32)


def make_pool(n: int, seed: int, prefix: str = "t") -> list[dict]:
    """Pool of n rows (about 75 % GOOD); registers each id's label for ``x_of``."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.75).astype(int)
    labels[:2] = [0, 1]
    rows = [{"signal_id": f"{prefix}{seed}_{i}", "label_int": int(l)}
            for i, l in enumerate(labels)]
    for r in rows:
        _LABELS[r["signal_id"]] = r["label_int"]
    return rows


def x_of(ids: list[str], window: int = 500) -> np.ndarray:
    return np.stack([_sig(i, _LABELS[i])[:window] for i in ids])
