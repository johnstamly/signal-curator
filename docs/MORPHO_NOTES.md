# MORPHO dataset notes

This document covers the guided-wave-specific aspects of the MORPHO H2020
dataset that motivated `signal-curator` and live inside `MorphoAdapter`.
Read this if you want to:

- Reproduce the paper exactly
- Adapt the MORPHO-specific drive-channel handling to a similar guided-wave
  campaign with different PZT topology
- Understand why `valid_receivers_for_actuator` exists at all

## Dataset overview

The MORPHO project ([Zenodo](https://zenodo.org/records/paunikar2025zenodo))
provides guided-wave signal acquisitions from 5 carbon-fibre composite panels
(FOD3, FOD4, FOD5, FOD6, FOD7) subjected to 4-point bending fatigue until
failure. Each panel was instrumented with **4 standard ceramic PZTs**
(piezoelectric transducers) plus 25 screen-printed piezoelectric elements
(the latter are excluded by `MorphoAdapter`).

For each acquisition cycle, the 4 PZTs are driven sequentially in
pitch-catch mode:

- **Actuator i** is excited with a tone burst at one of 5 frequencies
  (50, 100, 150, 200, 250 kHz)
- The **other 24 channels** record the resulting waveforms
- Each tone burst is repeated **10 times** for noise averaging

This gives `4 PZTs x 5 freqs x 10 reps = 200` signal traces per acquisition,
across `N` channels.

## The drive-channel exclusion

When actuator `i` (1-indexed) is firing, channel `i-1` (0-indexed) is acting
as the drive electrode -- not a wave receiver. The raw signal on that
channel is dominated by the drive pulse and is **not a useful guided-wave
trace**. `MorphoAdapter` excludes this channel automatically:

```python
# In morpho_adapter.py:
def valid_receivers_for_actuator(self, actuator: int) -> set[int]:
    drive = actuator - 1
    result: set[int] = set()
    for r in range(N_TOTAL_CHANNELS):
        if r == drive:
            continue
        result.add(r)
        if len(result) == N_WAVE_RECEIVERS:
            break
    return result
```

The constant `N_WAVE_RECEIVERS = 3` reflects the paper's filtering choice:
the 3 closest-but-non-drive PZTs are used as wave receivers, yielding the
**3 wave-receivers per actuator** factor in the paper's `12,504` total.

## How the `12,504` number is derived

Per panel: `4 actuators x 3 wave-receivers x N_cycles x 10 reps`. The
cycle count varies by panel:

| Panel | Cycles | Signals at 50 kHz |
|---|---|---|
| FOD3 | 1,104 / (4 x 3 x 10) ≈ 9 cycles  | 1,104 |
| FOD4 | 26 cycles | 3,120 |
| FOD5 | 24 cycles | 2,880 |
| FOD6 | 23 cycles | 2,760 |
| FOD7 | 22 cycles | 2,640 |
| **Total** | | **12,504** |

The HDF5 schema preserves all 25 receivers per actuator, but `MorphoAdapter`
filters down to the 3 wave-receivers at query time.

## HDF5 schema

```
morpho_50khz_full.h5
├── FOD3/
│   ├── 50kHz/
│   │   ├── act1/
│   │   │   ├── recv0/
│   │   │   │   ├── cycle200/
│   │   │   │   │   └── data    (n_reps=10, T_full=2001) float32
│   │   │   │   ├── cycle400/
│   │   │   │   │   └── data
│   │   │   │   └── ...
│   │   │   ├── recv2/
│   │   │   ├── recv3/
│   │   │   └── ...           (recv0 is the drive for act1 and would be excluded)
│   │   ├── act2/
│   │   └── ...
│   └── ...
└── ...
```

Signal length `T_full = 2001` samples at 1 MHz acquisition; the paper uses
the leading 500 samples (corresponding to 500 μs) via the
`SIGNAL_CURATOR_WINDOW=0,500` default. This window captures the first wave
packet for the longest panel paths.

## Cycle indexing

Cycles in the HDF5 are tagged with their **cumulative fatigue cycle count**
relative to the start of the campaign. Cycle 0 entries are pre-fatigue
healthy baseline measurements (no load applied yet). Subsequent cycles
correspond to the cumulative number of load cycles applied. The mapping
from raw folder names (e.g., `4kN_200`, `12kN_400`, `Healthy_4`) to cycle
counts is encoded in `LOAD_CYCLES` in the original ingestion code.

## What's a GOOD vs BAD signal?

The paper distinguishes 4 visual fault classes (Fig. 2):

| Class | Visual signature | Likely cause |
|---|---|---|
| **GOOD** | Clean wave-packet with expected envelope and arrival times | Healthy sensor and path |
| **BAD: low-energy** | Flat, near-zero amplitude | Debonded PZT, missed trigger |
| **BAD: saturated** | Clipped or extreme amplitude | Amplifier saturation, drive overdrive |
| **BAD: noisy / glitchy** | High-frequency contamination, spikes | Loose connector, EM interference |
| **BAD: mean-drifting** | Starts low, ramps up, stabilises at non-zero mean | Sensor settling, low-freq drift, DC bias |

The synthetic demo data (`scripts/.../synthetic.py`) generates one example of
each class for newcomers to see them side-by-side.

## Adapting `MorphoAdapter` to other PZT topologies

If your campaign uses a different number of PZTs (say 6 instead of 4) or a
different drive/receive convention:

1. Subclass `MorphoAdapter`
2. Override `valid_receivers_for_actuator(actuator)` to return your valid
   set of receiver IDs
3. Pass your subclass instance directly to the app instead of going through
   `make_adapter("morpho", ...)`

Example for a 6-PZT round-robin where every channel records (no drive
exclusion):

```python
from signal_curator.morpho_adapter import MorphoAdapter, N_TOTAL_CHANNELS

class SixPZTAdapter(MorphoAdapter):
    def valid_receivers_for_actuator(self, actuator: int) -> set[int]:
        return set(range(6))  # all 6 PZTs record, no drive exclusion
```
