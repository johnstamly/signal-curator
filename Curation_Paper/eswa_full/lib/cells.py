"""Resumable cell runner for the loop studies (ruling R15).

A *cell* is exactly one ``loop.simulate`` call, fully described by a spec dict::

    study_source, pool ("al"|"rand"), fold (0-4), seed, strategy, eta, arch, recipe,
    budgets, seed_size, batch

The cell ID is a stable hash of the canonical JSON of the spec (sorted keys, eta as float),
so the same spec always lands in ``<results_dir>/cells/<id>.json`` and studies find their
cells by spec, not by file name.

Test folds (kind_loop): ``shared_folds(reference_labels(al, rand), 5, seed=0)``, seed 0 for
EVERY training seed: the test folds are fixed and the seed only varies training.

Each cell is written atomically (``<id>.json.tmp`` then ``os.replace``), so a killed run
never leaves a partial file under the final name; a stale ``.tmp`` is simply overwritten.
A present cell is skipped only if it is *current*: its meta.code_key equals the content hash of
its kind's dependency files (versioning.py) and it was not computed from a dirty dependency
tree.  Stale cells are recomputed.  Execution lives in kind modules (``kind_loop``) registered
in ``KINDS`` by ``spec["study_source"]``.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from . import kind_ablate, kind_loop, kind_tune
from .protocol import REPO_ROOT, input_hashes, load_config, verify_inputs
from .runio import git_sha, results_dir
from .versioning import code_key as _code_key, dirty_files as _dirty_files

SPEC_FIELDS = ("study_source", "pool", "fold", "seed", "strategy", "eta", "arch", "recipe",
               "budgets", "seed_size", "batch")
_RANK = {"unc_oracle": 0, "unc_selftrain": 0, "uncertainty": 1, "random": 2}

# Cell kinds, keyed by spec["study_source"].  A kind module exposes KIND, DEPENDENCY_FILES
# (files whose content can change its results) and run(spec, cfg) -> result dict.  Task 10
# registers further kinds here without touching the loop kind's code key.
KINDS = {"loop": kind_loop, "tune": kind_tune, "ablate": kind_ablate}


def canonical_spec(spec: dict) -> dict:
    got, want = set(spec), set(SPEC_FIELDS)
    if got != want:
        raise ValueError(f"cell spec fields differ: missing {sorted(want - got)}, "
                         f"unexpected {sorted(got - want)}")
    if spec["pool"] not in ("al", "rand"):
        raise ValueError(f"pool must be 'al' or 'rand', got {spec['pool']!r}")
    out = {
        "study_source": str(spec["study_source"]), "pool": spec["pool"],
        "fold": int(spec["fold"]), "seed": int(spec["seed"]), "strategy": str(spec["strategy"]),
        "eta": float(spec["eta"]), "arch": str(spec["arch"]),
        "recipe": json.loads(json.dumps(spec["recipe"])),
        "budgets": [int(b) for b in spec["budgets"]],
        "seed_size": int(spec["seed_size"]), "batch": int(spec["batch"]),
    }
    return out


def cell_id(spec: dict) -> str:
    blob = json.dumps(canonical_spec(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def cells_dir(cfg: dict | None = None) -> Path:
    d = results_dir(cfg) / "cells"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_path(spec: dict, cfg: dict | None = None) -> Path:
    return cells_dir(cfg) / f"{cell_id(spec)}.json"


def load_cell(spec: dict, cfg: dict | None = None) -> dict | None:
    """The stored payload if present, valid JSON and spec-matching, else None."""
    p = cell_path(spec, cfg)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("spec") != canonical_spec(spec) \
            or "result" not in payload:
        return None
    return payload


def kind_of(spec: dict):
    src = spec["study_source"]
    if src not in KINDS:
        raise ValueError(f"unknown cell kind {src!r}; registered: {sorted(KINDS)}")
    return KINDS[src]


_VERSION: dict = {}


def _live_key(kind) -> str:
    return _code_key(kind.DEPENDENCY_FILES, REPO_ROOT)


def _compute_version(kind) -> tuple[str, list[str]]:
    return _live_key(kind), _dirty_files(kind.DEPENDENCY_FILES, REPO_ROOT)


def kind_version(kind) -> tuple[str, list[str]]:
    """(code_key, dirty dependency files) of a kind, fixed at first use in this process:
    long-lived workers have imported that code, so their cells carry that key."""
    if kind.KIND not in _VERSION:
        _VERSION[kind.KIND] = _compute_version(kind)
    return _VERSION[kind.KIND]


def reset_versions() -> None:
    _VERSION.clear()


def _freeze_versions() -> None:
    """Fix every registered kind's code key at import time (before any task runs)."""
    for kind in KINDS.values():
        kind_version(kind)


def dirty_dependencies(specs: list[dict]) -> list[str]:
    """Dirty dependency files of every kind used by ``specs`` (fresh git query)."""
    out: set[str] = set()
    for src in {s["study_source"] for s in specs}:
        kind = kind_of({"study_source": src})
        out |= set(_dirty_files(kind.DEPENDENCY_FILES, REPO_ROOT))
    return sorted(out)


def staleness(payload: dict | None, spec: dict, cfg: dict) -> str | None:
    """None when the stored cell is current; else the reason it must be recomputed."""
    if payload is None:
        return "missing"
    key, _ = kind_version(kind_of(spec))
    meta = payload.get("meta", {})
    if meta.get("code_key") != key:
        return f"old key {str(meta.get('code_key'))[:10]} != {key[:10]}"
    if meta.get("dirty", True) and not cfg.get("allow_dirty", False):
        return f"dirty (key {key[:10]})"
    return None


def package_versions() -> dict:
    out = {}
    for name in ("numpy", "scikit-learn", "aeon", "h5py", "torch"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _provenance(seconds: float, cfg: dict, kind) -> dict:
    key, dirty = kind_version(kind)
    now_key = _live_key(kind)
    if now_key != key:
        raise RuntimeError(f"dependency files of kind {kind.KIND!r} changed while this process "
                           f"was running ({key[:10]} -> {now_key[:10]}); cell not written")
    meta = {"code_key": key, "dirty": bool(dirty), "dirty_files": dirty, "kind": kind.KIND,
            "git_sha": git_sha(), "input_hashes": input_hashes(cfg),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(seconds, 3), "host": socket.gethostname(), "pid": os.getpid(),
            "python": sys.version.split()[0]}
    meta["packages"] = package_versions()
    return meta


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _run_one(spec: dict, cfg: dict) -> tuple[Path, str, float, str]:
    """(path, 'done'|'skipped', seconds, note); note is set when a stale cell was recomputed."""
    spec = canonical_spec(spec)
    kind = kind_of(spec)
    path = cell_path(spec, cfg)
    stored = load_cell(spec, cfg)
    why = staleness(stored, spec, cfg)
    if why is None:
        return path, "skipped", 0.0, ""
    note = f"stale -> recompute {cell_id(spec)} ({why})" if stored is not None else ""
    t0 = time.perf_counter()
    result = kind.run(spec, cfg)
    seconds = time.perf_counter() - t0
    _atomic_write(path, {"spec": spec, "result": result,
                         "meta": _provenance(seconds, cfg, kind)})
    return path, "done", seconds, note


def run_cell(spec: dict, cfg: dict) -> Path:
    """Run one cell unless a valid one is already stored; return its path."""
    return _run_one(spec, cfg)[0]


def order_longest_first(specs: list[dict]) -> list[dict]:
    def key(s):
        rounds = (max(s["budgets"]) - s["seed_size"]) / max(s["batch"], 1)
        return (_RANK.get(s["strategy"], 1), -rounds, cell_id(s))
    return sorted(specs, key=key)


def _error_file(spec: dict, cfg: dict) -> Path:
    d = cells_dir(cfg) / "_errors"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cell_id(spec)}.txt"


# --------------------------------------------------------------------------- workers

_GUARD = None


def _init_worker(n_threads: int, use_guard: bool) -> None:
    global _GUARD
    try:
        import torch
        torch.set_num_threads(max(1, n_threads))
    except ImportError:
        pass
    if use_guard:
        try:
            sys.path.insert(0, str(Path.home() / ".claude/skills/preflighting-heavy-compute"))
            from memguard import MemGuard
            _GUARD = MemGuard(min_available_gb=8.0, min_commit_gb=4.0, wait_s=300.0,
                              log=lambda m: None)
        except Exception:  # guard is optional; its absence must not stop the run
            _GUARD = None


def _work(spec: dict, cfg: dict) -> tuple[str, str, float, str, str]:
    """Pool task: never raises. Returns (id, status, seconds, error_text, note)."""
    cid = cell_id(spec)
    try:
        if _GUARD is not None and staleness(load_cell(spec, cfg), canonical_spec(spec), cfg):
            _GUARD.checkpoint(cid)
        _, status, secs, note = _run_one(spec, cfg)
        return cid, status, secs, "", note
    except Exception as e:  # noqa: BLE001 - report, stop nothing (Ctrl+C is not caught)
        if type(e).__name__ == "MemoryPressure":
            return cid, "deferred", 0.0, str(e), ""
        return cid, "failed", 0.0, traceback.format_exc(), ""


def run_cells(specs: list[dict], workers: int, cfg: dict, *, log=print) -> dict:
    """Run every unique spec (longest first) on ``workers`` spawned processes.

    workers <= 0 runs in-process (tests).  Returns counts; failures are written to
    ``cells/_errors/<id>.txt`` and never stop the other cells.
    """
    uniq: dict[str, dict] = {}
    for s in specs:
        c = canonical_spec(s)
        uniq.setdefault(cell_id(c), c)
    todo = order_longest_first(list(uniq.values()))
    n = len(todo)
    counts = {"done": 0, "skipped": 0, "failed": 0, "deferred": 0, "total": n, "stale": 0}
    by_id = {cell_id(s): s for s in todo}
    t_start = time.perf_counter()
    finished = 0

    def record(cid, status, secs, err, note=""):
        nonlocal finished
        finished += 1
        counts[status] += 1
        s = by_id[cid]
        if note:
            counts["stale"] += 1
            log(note)
        if status in ("failed", "deferred"):
            _error_file(s, cfg).write_text(err, encoding="utf-8")
        log(f"{status} {finished}/{n} {cid} {s['strategy']} {s['pool']} fold{s['fold']} "
            f"seed{s['seed']} eta{s['eta']:g} {secs:.1f}s "
            f"(elapsed {(time.perf_counter() - t_start) / 60:.1f} min)")

    if workers <= 0:
        for s in todo:
            record(*_work(s, cfg))
    else:
        ctx = mp.get_context("spawn")
        threads = max(1, (os.cpu_count() or workers) // workers)
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                    initializer=_init_worker, initargs=(threads, True)) as ex:
            futs = {ex.submit(_work, s, cfg): cell_id(s) for s in todo}
            for fut in cf.as_completed(futs):
                cid = futs[fut]
                try:
                    res = fut.result()
                except Exception:  # noqa: BLE001 - e.g. BrokenProcessPool
                    res = (cid, "failed", 0.0, traceback.format_exc(), "")
                record(*res)
    log(f"summary: done {counts['done']}, skipped {counts['skipped']}, "
        f"failed {counts['failed']}, deferred {counts['deferred']}, total {n} "
        f"(of the done, {counts['stale']} were stale and recomputed)")
    return counts


def prepare(cfg: dict | None = None) -> dict:
    """Parent-side checks before launching: frozen inputs verified once."""
    cfg = cfg or load_config()
    return verify_inputs(cfg)


try:
    _freeze_versions()
except Exception:  # noqa: BLE001 - e.g. git unavailable; fall back to lazy evaluation
    _VERSION.clear()
