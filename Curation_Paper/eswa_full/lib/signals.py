"""Signal store: reads the Zenodo 50 kHz HDF5 once, in memory only (no writes)."""
from __future__ import annotations

import re

import h5py
import numpy as np

from .protocol import h5_key, load_config, repo_path

_ID_RE = re.compile(r"^(?P<panel>[^_]+)_50kHz_act(?P<actuator>\d+)_recv(?P<receiver>\d+)"
                    r"_c(?P<cycle>\d+)_r(?P<rep_idx>\d+)$")

_CACHE: dict = {}


def make_id(panel: str, actuator: int, receiver: int, cycle: int, rep_idx: int) -> str:
    return f"{panel}_50kHz_act{actuator}_recv{receiver}_c{cycle}_r{rep_idx}"


def parse_id(sid: str) -> dict:
    m = _ID_RE.match(sid)
    if not m:
        raise KeyError(f"malformed signal id: {sid!r}")
    d = m.groupdict()
    return {"panel": d["panel"], **{k: int(d[k]) for k in ("actuator", "receiver", "cycle", "rep_idx")}}


def _num(prefix: str, name: str) -> int:
    return int(name[len(prefix):])


def _window(cfg: dict, window: int | None) -> int:
    return cfg["window"] if window is None else window


def load_universe(cfg: dict) -> tuple[tuple[str, ...], np.ndarray]:
    """Returns an immutable id tuple and a read-only array (copy before modifying).

    All signals of the HDF5 in deterministic order (panel, actuator, receiver, cycle, rep),
    first ``cfg["window"]`` samples, float32. Cached in memory per (file, window)."""
    path = repo_path(cfg["signals_h5"])
    w = cfg["window"]
    key = (str(path), w)
    if key in _CACHE:
        return _CACHE[key]
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    with h5py.File(path, "r") as h5:
        for panel in sorted(h5.keys()):
            fgrp = h5[panel]["50kHz"]
            for act in sorted(fgrp.keys(), key=lambda n: _num("act", n)):
                for recv in sorted(fgrp[act].keys(), key=lambda n: _num("recv", n)):
                    for cyc in sorted(fgrp[act][recv].keys(), key=lambda n: _num("cycle", n)):
                        ds = fgrp[act][recv][cyc]["data"]
                        if ds.shape[1] < w:
                            raise ValueError(f"{ds.name}: signal length {ds.shape[1]} "
                                             f"< window {w}")
                        blocks.append(np.asarray(ds[:, :w], dtype=np.float32))
                        a, r, c = _num("act", act), _num("recv", recv), _num("cycle", cyc)
                        ids.extend(make_id(panel, a, r, c, i) for i in range(ds.shape[0]))
    X = np.concatenate(blocks, axis=0)
    X.setflags(write=False)  # shared cache: callers must copy before modifying
    out = (tuple(ids), X)
    _CACHE[key] = out
    return out


def signals_for(ids: list[str], cfg: dict, window: int | None = None) -> np.ndarray:
    """Rows for ``ids`` (in the given order), first ``window`` samples (default cfg window)."""
    w = _window(cfg, window)
    out = np.empty((len(ids), w), dtype=np.float32)
    with h5py.File(repo_path(cfg["signals_h5"]), "r") as h5:
        for i, sid in enumerate(ids):
            p = parse_id(sid)
            key = h5_key(p)
            if key not in h5:
                raise KeyError(f"{sid}: no dataset {key}")
            ds = h5[key]
            if p["rep_idx"] >= ds.shape[0] or ds.shape[1] < w:
                raise KeyError(f"{sid}: row {p['rep_idx']} / window {w} outside shape {ds.shape}")
            out[i] = ds[p["rep_idx"], :w]
    return out
