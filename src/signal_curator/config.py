"""Centralised configuration via environment variables with sensible defaults.

Override any setting via env vars before launching the app:

    SIGNAL_CURATOR_DB=/path/to/labels.db
    SIGNAL_CURATOR_DATA=/path/to/dataset.h5
    SIGNAL_CURATOR_WEIGHTS_DIR=/path/to/weights
    SIGNAL_CURATOR_WINDOW="0,500"
    SIGNAL_CURATOR_ADAPTER=morpho   # or "generic"

    streamlit run -m signal_curator.app
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration for signal-curator."""
    db_path: Path
    data_path: Path
    weights_dir: Path
    window: tuple[int, int]
    adapter: str  # "morpho" | "generic"

    @classmethod
    def from_env(cls, base: Path | None = None) -> "Config":
        base = Path(base or os.getcwd())
        db_path = Path(os.environ.get("SIGNAL_CURATOR_DB", base / "labels.db"))
        data_path = Path(os.environ.get("SIGNAL_CURATOR_DATA", base / "dataset.h5"))
        weights_dir = Path(os.environ.get("SIGNAL_CURATOR_WEIGHTS_DIR", base / "weights"))
        win_str = os.environ.get("SIGNAL_CURATOR_WINDOW", "0,500")
        try:
            w0, w1 = (int(x) for x in win_str.split(","))
        except ValueError as e:
            raise ValueError(f"SIGNAL_CURATOR_WINDOW must be 'start,end' integers, got {win_str!r}") from e
        adapter = os.environ.get("SIGNAL_CURATOR_ADAPTER", "morpho").lower()
        if adapter not in {"morpho", "generic"}:
            raise ValueError(f"SIGNAL_CURATOR_ADAPTER must be 'morpho' or 'generic', got {adapter!r}")
        weights_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            db_path=db_path,
            data_path=data_path,
            weights_dir=weights_dir,
            window=(w0, w1),
            adapter=adapter,
        )

    @property
    def input_len(self) -> int:
        return self.window[1] - self.window[0]
