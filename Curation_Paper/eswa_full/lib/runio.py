"""JSON results with provenance (git SHA, input hashes, seeds, timestamp)."""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

from .protocol import PKG_ROOT, REPO_ROOT, input_hashes


def results_dir(cfg: dict | None = None) -> Path:
    d = Path(cfg["results_dir"]) if cfg and cfg.get("results_dir") else PKG_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def save_result(name: str, payload: dict, cfg: dict) -> Path:
    out = dict(payload)
    out["meta"] = {
        "git_sha": git_sha(),
        "input_hashes": input_hashes(cfg),
        "seeds": {"stage1": cfg["seeds_stage1"], "stage2": cfg["seeds_stage2"]},
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    path = results_dir(cfg) / f"{name}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_result(name: str, cfg: dict | None = None) -> dict:
    return json.loads((results_dir(cfg) / f"{name}.json").read_text(encoding="utf-8"))
