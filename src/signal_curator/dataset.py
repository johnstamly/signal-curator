"""Dataset adapter abstraction: pluggable HDF5 schemas for 1D signal datasets.

Two built-in adapters:

  * `MorphoAdapter` -- hierarchical schema used in the accompanying paper:
        /<panel>/<freq>kHz/act<actuator>/recv<receiver>/cycle<cycle>/data
        dataset shape (n_reps, L_full); handles 4-PZT actuator/receiver
        indexing and drive-channel exclusion.

  * `GenericAdapter` -- flat schema for non-MORPHO datasets:
        /<signal_id>/data           # 1D array, shape (L_full,)
        /<signal_id>.attrs          # freq_khz, optional other metadata

To plug in your own dataset, either pre-convert it to one of these schemas or
implement the `DatasetAdapter` protocol against your own storage format.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
import h5py

from signal_curator.session import SignalKey


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal interface every adapter must provide."""

    @property
    def frequencies(self) -> list[int]:
        """Sorted list of distinct frequencies (kHz) available in the dataset."""
        ...

    def get_signal_keys(self, freq_khz: int) -> list[SignalKey]:
        """All `SignalKey`s available at the given frequency."""
        ...

    def get_signal(self, key: SignalKey, window: tuple[int, int]) -> np.ndarray:
        """Load and window a single signal. Returns (window[1] - window[0],) float32."""
        ...

    def valid_receivers_for_actuator(self, actuator: int) -> set[int]:
        """For domain-specific receiver filtering. Returns all-receivers for generic data."""
        ...


# -----------------------------------------------------------------------------
# Generic flat-schema adapter
# -----------------------------------------------------------------------------

class GenericAdapter:
    """Adapter for the generic flat HDF5 schema.

    Schema:
        /<signal_id>/data           -- 1D array, shape (L_full,), dtype float32
        /<signal_id>.attrs.freq_khz -- int (required)
        /<signal_id>.attrs.panel    -- str (optional, defaults to 'panel0')

    All signals are treated as belonging to actuator=0, receiver=0, cycle=0.
    The `signal_id` is used directly as `rep_idx` (counter), with the original
    string preserved in a side-table for round-tripping if needed.
    """

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self._keys_by_freq: dict[int, list[SignalKey]] = {}
        self._sid_lookup: dict[SignalKey, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            counter = 0
            for sid in f.keys():
                grp = f[sid]
                if "data" not in grp:
                    continue
                freq = int(grp.attrs.get("freq_khz", 0))
                panel = str(grp.attrs.get("panel", "panel0"))
                key = SignalKey(panel=panel, freq_khz=freq, actuator=0,
                                receiver=0, cycle=0, rep_idx=counter)
                self._keys_by_freq.setdefault(freq, []).append(key)
                self._sid_lookup[key] = sid
                counter += 1

    @property
    def frequencies(self) -> list[int]:
        return sorted(self._keys_by_freq.keys())

    def get_signal_keys(self, freq_khz: int) -> list[SignalKey]:
        return list(self._keys_by_freq.get(freq_khz, []))

    def get_signal(self, key: SignalKey, window: tuple[int, int]) -> np.ndarray:
        sid = self._sid_lookup[key]
        with h5py.File(self.h5_path, "r") as f:
            data = f[sid]["data"][window[0]:window[1]]
        return np.asarray(data, dtype=np.float32)

    def valid_receivers_for_actuator(self, actuator: int) -> set[int]:
        return {0}  # generic data: no actuator/receiver structure


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def make_adapter(adapter_name: str, h5_path: str) -> DatasetAdapter:
    """Instantiate the requested adapter by name."""
    if adapter_name == "morpho":
        # Lazy import to keep generic users from depending on MORPHO code
        from signal_curator.morpho_adapter import MorphoAdapter
        return MorphoAdapter(h5_path)
    if adapter_name == "generic":
        return GenericAdapter(h5_path)
    raise ValueError(f"Unknown adapter {adapter_name!r}; expected 'morpho' or 'generic'.")
