"""Protocol config, frozen-input verification and label pools for the ESWA study."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PKG_ROOT / "config" / "protocol.json"

COLUMNS = ("signal_id", "panel", "actuator", "receiver", "cycle", "rep_idx", "label", "labeled_at")


class InputMismatch(RuntimeError):
    """A frozen input differs from the protocol (hash, count or missing signal)."""


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def repo_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(cfg: dict) -> dict:
    """Recorded (config) hashes of the frozen label databases."""
    return {k: v["sha256"] for k, v in cfg["pools"].items()}


def _read_pool(path: Path, freq_khz: int) -> dict[str, dict]:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT signal_id, panel, actuator, receiver, cycle, rep_idx, label, labeled_at "
            "FROM labels WHERE freq_khz = ? ORDER BY signal_id", (freq_khz,)).fetchall()
    finally:
        con.close()
    pool: dict[str, dict] = {}
    for r in rows:
        d = dict(zip(COLUMNS, r))
        d["label_int"] = 1 if d["label"] == "GOOD" else 0
        pool[d["signal_id"]] = d
    return pool


def load_pools(cfg: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    al = _read_pool(repo_path(cfg["pools"]["al"]["path"]), cfg["freq_khz"])
    rand = _read_pool(repo_path(cfg["pools"]["rand"]["path"]), cfg["freq_khz"])
    return al, rand


def reference_labels(al: dict, rand: dict) -> dict[str, int]:
    """Shared IDs -> L_AL label_int (the active-learning pool is the reference)."""
    return {s: al[s]["label_int"] for s in al if s in rand}


def h5_key(row: dict) -> str:
    return (f"{row['panel']}/50kHz/act{row['actuator']}/recv{row['receiver']}"
            f"/cycle{row['cycle']}/data")


def verify_inputs(cfg: dict) -> dict:
    """Check hashes (before reading anything), then counts and HDF5 resolvability."""
    for name, spec in cfg["pools"].items():
        db = repo_path(spec["path"])
        wal = db.with_name(db.name + "-wal")
        if wal.exists() and wal.stat().st_size > 0:
            raise InputMismatch(f"{name}: non-empty WAL file {wal} (un-checkpointed frames "
                                f"are not covered by the hash)")
        actual = sha256_file(db)
        if actual != spec["sha256"]:
            raise InputMismatch(f"{name}: sha256 {actual} != frozen {spec['sha256']}")
    al, rand = load_pools(cfg)
    shared = sorted(set(al) & set(rand))
    union = {**rand, **al}
    disagree = [s for s in shared if al[s]["label_int"] != rand[s]["label_int"]]
    missing, short = [], []
    with h5py.File(repo_path(cfg["signals_h5"]), "r") as h5:
        for sid, row in sorted(union.items()):
            key = h5_key(row)
            if key not in h5 or row["rep_idx"] >= h5[key].shape[0]:
                missing.append(sid)
            elif h5[key].shape[1] < cfg["window"]:
                short.append(sid)
    if short:
        raise InputMismatch(f"window {cfg['window']} exceeds the signal length of "
                            f"{len(short)} signals (first: {short[0]})")
    out = {
        "n_al": len(al), "n_rand": len(rand), "n_shared": len(shared),
        "n_union": len(union), "n_shared_disagree": len(disagree),
        "n_missing_in_h5": len(missing), "missing_ids": missing,
        "shared_good_al": sum(al[s]["label_int"] for s in shared),
        "shared_bad_al": sum(1 - al[s]["label_int"] for s in shared),
        "disagree": [{"signal_id": s, "label_al": al[s]["label"], "label_rand": rand[s]["label"],
                      "labeled_at_al": al[s]["labeled_at"], "labeled_at_rand": rand[s]["labeled_at"]}
                     for s in disagree],
        "input_hashes": input_hashes(cfg),
    }
    expected = {"n_al": 1011, "n_rand": 1011, "n_shared": 746, "n_union": 1276,
                "n_shared_disagree": 13, "n_missing_in_h5": 0}
    bad = {k: (out[k], v) for k, v in expected.items() if out[k] != v}
    if bad:
        raise InputMismatch(f"counts differ from the probed facts (got, expected): {bad}")
    return out
