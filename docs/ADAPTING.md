# Adapting `signal-curator` to your own dataset

`signal-curator` is built around a generic 1D-signal data abstraction. The
labelling workflow, the model (`MicroConv1D`), the SQLite label store, and the
FP/FN re-inspection logic are all dataset-agnostic. The only piece that
"knows about MORPHO" is the `MorphoAdapter` class, which handles the
hierarchical actuator/receiver schema described in the paper.

For any other 1D-signal dataset -- vibration, accelerometer, ECG, audio
clips, or another guided-wave campaign with different sensor topology -- you
have two options.

## Option A: convert to the generic HDF5 schema (recommended)

The `GenericAdapter` reads HDF5 files with this layout:

```
your_data.h5
├── signal_001/
│   ├── data            (L,) float32 -- the signal itself
│   ├── @freq_khz       int          -- required
│   ├── @panel          str          -- optional (default: "panel0")
│   └── @<anything>     ...          -- preserved but unused
├── signal_002/
│   └── ...
└── ...
```

Each top-level group is one signal, identified by its group name (used as
the `signal_id`). The `freq_khz` attribute is required because the app
groups signals by frequency. All other attributes (panel, source, capture
time, ...) are preserved in the HDF5 but not used by the labelling workflow.

### Conversion example: from a folder of `.npy` files

```python
import h5py
import numpy as np
from pathlib import Path

src_dir = Path("my_signals/")  # contains 001.npy, 002.npy, ...
out_path = "my_data.h5"
FREQ_KHZ = 50  # whatever your dataset is sampled at, in kHz

with h5py.File(out_path, "w") as f:
    for npy_path in sorted(src_dir.glob("*.npy")):
        sig = np.load(npy_path).astype("float32")
        sid = npy_path.stem  # use filename as signal_id
        grp = f.create_group(sid)
        grp.create_dataset("data", data=sig)
        grp.attrs["freq_khz"] = FREQ_KHZ
        grp.attrs["panel"] = "my_project"
```

### Conversion example: from a single (N, L) NumPy array

```python
import h5py
import numpy as np

signals = np.load("all_signals.npy")  # shape (N, L)
out_path = "my_data.h5"
FREQ_KHZ = 50

with h5py.File(out_path, "w") as f:
    for i, sig in enumerate(signals):
        sid = f"sig_{i:06d}"
        grp = f.create_group(sid)
        grp.create_dataset("data", data=sig.astype("float32"))
        grp.attrs["freq_khz"] = FREQ_KHZ
```

### Launch

```bash
signal-curator run \
    --data my_data.h5 \
    --db ~/my_labels.db \
    --adapter generic \
    --window 0,500
```

Adjust `--window` if your signals aren't 500 samples; the model auto-resizes
to fit (use a window length divisible by 32 for clean conv-pool dimensions).

## Option B: write your own adapter class

If you can't pre-convert (e.g., signals are gigantic and you want lazy
loading from your existing storage format), implement the `DatasetAdapter`
protocol in `signal_curator.dataset`:

```python
from signal_curator.dataset import DatasetAdapter
from signal_curator.session import SignalKey
import numpy as np


class MyCustomAdapter:
    def __init__(self, source_path: str):
        self.source_path = source_path
        # build whatever index you need

    @property
    def frequencies(self) -> list[int]:
        return [50]  # or whatever your dataset has

    def get_signal_keys(self, freq_khz: int) -> list[SignalKey]:
        # Return one SignalKey per signal at that frequency.
        # For non-MORPHO data, set actuator=receiver=cycle=0 and use rep_idx
        # as a unique counter.
        ...

    def get_signal(self, key: SignalKey, window: tuple[int, int]) -> np.ndarray:
        # Load and return the (windowed) signal as a (L,) float32 array
        ...

    def valid_receivers_for_actuator(self, actuator: int) -> set[int]:
        return {0}  # generic data: no actuator/receiver structure
```

Register it by editing `signal_curator/dataset.py::make_adapter` to add a
new branch, or pass your instance directly:

```python
from signal_curator.config import Config
# launch via Python instead of CLI, set st.session_state.adapter
```

## Adjusting the model

The default `MicroConv1D` was tuned for 500-sample input windows. If your
signals are very different:

| Signal length | Recommended action |
|---|---|
| 500 (matches paper) | Use defaults |
| 100-1500 | Use defaults; `MicroConv1D` adapts the dense layer automatically |
| < 100 | Reduce conv kernel sizes (edit `model.py`) |
| > 5000 | Consider downsampling before labelling, or increase strides |

The model file (`signal_curator/model.py`) is intentionally tiny (~50 lines).
Modifying the architecture for your domain is encouraged.

## Labels and the FP/FN loop

The SQLite label schema includes MORPHO-flavoured columns (`panel`, `actuator`,
`receiver`, `cycle`, `rep_idx`). For non-MORPHO data, just stuff sensible
defaults (`panel="my_project"`, the rest = 0). The labelling workflow only
uses `signal_id` and `label`. The FP/FN re-inspection works identically.

## I have a question / want a new feature

Open an issue at https://github.com/johnstamly/signal-curator/issues.
Datasets we'd love to add support for out-of-the-box:

- WAV / audio-clip directories
- Time-series CSV files
- PSV / Wireshark capture files
- ROS bags
