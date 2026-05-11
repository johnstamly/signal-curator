"""Labelling-session data structures: SignalKey + LabelingSession.

SignalKey is the addressable coordinate for a single signal in the dataset.
LabelingSession is a thin in-memory view over the SQLite label store.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import NamedTuple


class SignalKey(NamedTuple):
    """Addressable coordinate for one signal.

    For non-MORPHO datasets you can set actuator/receiver/cycle to 0 (or any
    sentinel) and use only the panel + freq_khz + rep_idx fields.
    """
    panel: str
    freq_khz: int
    actuator: int
    receiver: int
    cycle: int
    rep_idx: int


@dataclass
class LabelingSession:
    """In-memory view of a labelling session for one frequency slice."""
    freq_khz: int
    labeled: list[dict] = field(default_factory=list)
    auto_results: list[dict] = field(default_factory=list)
    model_weights_path: str | None = None

    def add_label(self, key: SignalKey, label: str) -> None:
        """Add or update a label. User label always wins over auto."""
        for r in self.auto_results:
            if self._matches(r, key):
                r["label"] = label
                r["source"] = "user_override"
                return
        for r in self.labeled:
            if self._matches(r, key):
                r["label"] = label
                return
        self.labeled.append({
            "panel": key.panel,
            "freq_khz": key.freq_khz,
            "actuator": key.actuator,
            "receiver": key.receiver,
            "cycle": key.cycle,
            "rep_idx": key.rep_idx,
            "label": label,
            "source": "user",
        })

    def all_labeled_keys(self) -> set[SignalKey]:
        keys: set[SignalKey] = set()
        for r in self.labeled + self.auto_results:
            keys.add(SignalKey(
                r["panel"], r["freq_khz"], r["actuator"],
                r["receiver"], r["cycle"], r["rep_idx"],
            ))
        return keys

    def save(self, path: str) -> None:
        data = {
            "freq_khz": self.freq_khz,
            "labeled": self.labeled,
            "auto_results": self.auto_results,
            "model_weights_path": self.model_weights_path,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "LabelingSession":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        session = cls(
            freq_khz=data["freq_khz"],
            labeled=data.get("labeled", []),
            auto_results=data.get("auto_results", []),
            model_weights_path=data.get("model_weights_path"),
        )
        if session.model_weights_path and not os.path.exists(session.model_weights_path):
            raise FileNotFoundError(
                f"Model weights file not found: {session.model_weights_path}"
            )
        return session

    @classmethod
    def from_shared(cls, db_path: str, freq_khz: int) -> "LabelingSession":
        """Rebuild a session view from the shared SQLite label store."""
        from signal_curator.labels_db import get_all_labels

        session = cls(freq_khz=freq_khz)
        for row in get_all_labels(db_path, freq_khz):
            session.labeled.append({
                "panel": row["panel"],
                "freq_khz": row["freq_khz"],
                "actuator": row["actuator"],
                "receiver": row["receiver"],
                "cycle": row["cycle"],
                "rep_idx": row["rep_idx"],
                "label": row["label"],
                "source": row["source"],
            })
        return session

    @staticmethod
    def _matches(record: dict, key: SignalKey) -> bool:
        return (
            record["panel"] == key.panel
            and record["freq_khz"] == key.freq_khz
            and record["actuator"] == key.actuator
            and record["receiver"] == key.receiver
            and record["cycle"] == key.cycle
            and record["rep_idx"] == key.rep_idx
        )
