"""Determinism check of the noise-free (eta = 0) loop cells across code versions.

The loop code was changed once (ruling R37: label noise is injected into training-pool rows only).
That change touches the eta > 0 path only, so every eta = 0 loop cell must reproduce.  This script
fingerprints each eta-0 loop cell (sha256 of its result block with timing fields removed) and
compares a stored snapshot, taken before the change, with the cells now on disk.

Usage (from anywhere; paths are relative to this file):
  python determinism_check.py snapshot <out.json>   write a fingerprint snapshot
  python determinism_check.py diff <old.json>       compare with the cells now on disk and write
                                                    results/determinism/eta0_diff_key84a0267.json
"""
import hashlib
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG / "results" / "cells"
DIFF_OUT = PKG / "results" / "determinism" / "eta0_diff_key84a0267.json"
TIMING = ("seconds", "time", "elapsed", "created", "duration", "wall")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not any(t in k.lower() for t in TIMING)}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def collect():
    out, kinds = {}, {}
    for p in sorted(ROOT.rglob("*.json")):
        if "_logs" in p.parts or "_errors" in p.parts:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        spec = d.get("spec", {})
        kind = spec.get("strategy") if spec.get("strategy") in ("tune", "ablate") else "loop"
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind != "loop" or float(spec.get("eta", -1)) != 0.0:
            continue
        blob = json.dumps(strip(d.get("result", {})), sort_keys=True).encode()
        out[str(p.relative_to(ROOT)).replace("\\", "/")] = {
            "sha": hashlib.sha256(blob).hexdigest(), "key": d.get("meta", {}).get("code_key")}
    return out, kinds


def main():
    mode, path = sys.argv[1], Path(sys.argv[2])
    cur, kinds = collect()
    print("cells by kind:", kinds, "| eta-0 loop cells:", len(cur))
    if mode == "snapshot":
        path.write_text(json.dumps(cur, indent=0))
        print("wrote", path)
        return
    old = json.loads(path.read_text())
    cur_n = {k.replace("\\", "/"): v for k, v in cur.items()}
    old_n = {k.replace("\\", "/"): v for k, v in old.items()}
    same = [k for k in old_n if k in cur_n and cur_n[k]["sha"] == old_n[k]["sha"]]
    diff = [k for k in old_n if k in cur_n and cur_n[k]["sha"] != old_n[k]["sha"]]
    missing = [k for k in old_n if k not in cur_n]
    print(f"old {len(old_n)} | identical {len(same)} | different {len(diff)} | missing now {len(missing)}")
    for k in diff[:10]:
        print("  DIFF", k)
    DIFF_OUT.parent.mkdir(parents=True, exist_ok=True)
    DIFF_OUT.write_text(json.dumps({
        "snapshot": path.name,
        "n_old": len(old_n), "n_identical": len(same), "n_different": len(diff),
        "n_missing": len(missing), "different": diff, "missing": missing,
        "old_code_keys": sorted({v["key"] for v in old_n.values()}),
        "current_code_keys": sorted({cur_n[k]["key"] for k in same}),
        "fingerprint": "sha256 of the cell result block with timing fields removed",
    }, indent=1), encoding="utf-8")
    print("wrote", DIFF_OUT)


if __name__ == "__main__":
    main()
