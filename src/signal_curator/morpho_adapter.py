"""MORPHO-specific dataset adapter.

Handles the hierarchical HDF5 schema used in the H2020 MORPHO guided-wave
fatigue dataset:

    /<panel>/<freq>kHz/act<actuator>/recv<receiver>/cycle<cycle>/data
    dataset shape (n_reps, L_full), dtype float32.

Also encodes MORPHO-specific assumptions:

  * 4 piezoelectric actuators per panel; each acts as one drive channel
    and the remaining channels record. For actuator A, the drive channel is
    `recv{A-1}` -- this channel is excluded by default to avoid drive-pulse
    saturation in the wave-recording analysis.
  * 3 "wave-recording" receivers per actuator (`N_WAVE_RECEIVERS = 3`),
    selected as the first 3 channels excluding the drive.

If your dataset uses a similar hierarchical schema but different actuator
or receiver wiring, subclass `MorphoAdapter` and override `valid_receivers_for_actuator`.
"""
from __future__ import annotations
import numpy as np
import h5py

from signal_curator.session import SignalKey

N_WAVE_RECEIVERS = 3
N_TOTAL_CHANNELS = 25  # MORPHO acquisition card channel count


class MorphoAdapter:
    """Adapter for the MORPHO hierarchical HDF5 schema."""

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self._index: dict[tuple, tuple[str, int]] = {}  # (panel, freq, act, recv, cycle) -> (h5_path, n_reps)
        self._frequencies: set[int] = set()
        self._build_index()

    def _build_index(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            def _visit(name, obj):
                if not (isinstance(obj, h5py.Dataset) and name.endswith("/data")):
                    return
                parts = name.split("/")
                if len(parts) < 6:
                    return
                try:
                    panel = parts[0]
                    freq = int(parts[1].replace("kHz", ""))
                    act = int(parts[2].replace("act", ""))
                    recv = int(parts[3].replace("recv", ""))
                    cycle = int(parts[4].replace("cycle", ""))
                except (ValueError, IndexError):
                    return
                self._index[(panel, freq, act, recv, cycle)] = (name, obj.shape[0])
                self._frequencies.add(freq)
            f.visititems(_visit)

    @property
    def frequencies(self) -> list[int]:
        return sorted(self._frequencies)

    def valid_receivers_for_actuator(self, actuator: int) -> set[int]:
        """Return the 3 wave-recording receivers for the given actuator.

        The drive channel for actuator A is `recv(A-1)` and is excluded. The
        first 3 remaining channels are the wave receivers.
        """
        drive = actuator - 1
        result: set[int] = set()
        for r in range(N_TOTAL_CHANNELS):
            if r == drive:
                continue
            result.add(r)
            if len(result) == N_WAVE_RECEIVERS:
                break
        return result

    def get_signal_keys(self, freq_khz: int) -> list[SignalKey]:
        """All `SignalKey`s at the given frequency, restricted to wave receivers."""
        keys: list[SignalKey] = []
        for (panel, f, act, recv, cycle), (_, n_reps) in self._index.items():
            if f != freq_khz:
                continue
            if recv not in self.valid_receivers_for_actuator(act):
                continue
            for rep_i in range(n_reps):
                keys.append(SignalKey(panel, f, act, recv, cycle, rep_i))
        return keys

    def get_signal(self, key: SignalKey, window: tuple[int, int]) -> np.ndarray:
        idx_key = (key.panel, key.freq_khz, key.actuator, key.receiver, key.cycle)
        entry = self._index.get(idx_key)
        if entry is None:
            raise KeyError(f"No data for {idx_key}")
        h5_path, _ = entry
        with h5py.File(self.h5_path, "r") as f:
            data = f[h5_path][key.rep_idx, window[0]:window[1]]
        return np.asarray(data, dtype=np.float32)
