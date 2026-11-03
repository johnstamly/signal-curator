# signal-curator

> Human-in-the-loop active learning labeling app for 1D signal datasets — built
> for guided-wave SHM with operator self-correction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20119847.svg)](https://doi.org/10.5281/zenodo.20119847)

`signal-curator` is a Streamlit-based labelling tool with a tight active
learning loop: a lightweight 1D CNN scores all signals, surfaces ambiguous
queries to the operator, and pairs them with **FP/FN re-inspection** of past
decisions — so the operator can correct their own labelling mistakes as the
model learns. The pipeline was developed for guided-wave structural health
monitoring (SHM) campaigns.

## Why use this

- **Faster, cleaner labelling** — small CNN ranks signals by uncertainty so the
  operator focuses on what's informative, not what's easy
- **Catches operator mistakes** — built-in FP/FN re-inspection step surfaces
  past labels the model now disagrees with, letting the operator audit and
  correct themselves
- **Tested on real data** — used to curate 12,504 signals of the MORPHO H2020
  guided-wave dataset with 1,011 operator labels (≈8%). In a retrospective
  replay, guided selection beat random selection by 5.4 percentage points of
  balanced accuracy at 200 labels and 2.9 at 500, and operator re-inspection
  recovered most injected label errors that reached the labelled set
  (see [`Curation_Paper/README.md`](Curation_Paper/README.md))
- **Frequency- and domain-agnostic core** — generic 1D-signal abstraction; the
  guided-wave actuator/receiver/drive-channel handling is isolated in a
  dedicated adapter module

## Quick start

```bash
pip install signal-curator
signal-curator demo                   # opens the app with synthetic demo data
```

Or from source:

```bash
git clone https://github.com/johnstamly/signal-curator
cd signal-curator
pip install -e .
streamlit run src/signal_curator/app.py
```

## Reproducing the paper

Every retrospective result of the paper (the replay of the labelling session,
re-inspection under injected label noise, the classifier comparison, the
ablations, thresholds, coverage and classical filters) comes from the code in
[`Curation_Paper/eswa_full/`](Curation_Paper/eswa_full/). It is published
unchanged from the code that produced the results, together with both frozen
label pools and the study results. Follow
[`Curation_Paper/README.md`](Curation_Paper/README.md) to verify the inputs,
rebuild every reported number from the stored results, or rerun the studies.

The repo also ships with the **1,011-signal labelled subset** from the paper
(`data/morpho_50khz_curated/`, ~8 MB), which opens in the labelling app:

```bash
signal-curator demo --dataset morpho_50khz_curated
```

The **full 12,504-signal MORPHO 50 kHz universe** (~93 MB) is archived at
Zenodo, DOI [10.5281/zenodo.20120069](https://doi.org/10.5281/zenodo.20120069).
Download it with:

```bash
python scripts/download_morpho_full.py
```

## Adapting to your own dataset

`signal-curator` is built around a generic 1D-signal dataset abstraction.
To use it with non-MORPHO data:

1. Convert your signals to the HDF5 schema documented in
   [`docs/ADAPTING.md`](docs/ADAPTING.md)
2. Skip the MORPHO-specific actuator/receiver/drive-channel logic by passing
   `--adapter=generic`
3. Adjust `MicroConv1D`'s `input_len` if your signals aren't 500 samples

The architecture and training recipe in `src/signal_curator/{model,trainer}.py`
are general-purpose — they don't assume guided waves or SHM. Plug in your own
HDF5 of binary-labelable 1D signals (vibration, accelerometer, ECG, audio
clips, anything) and the labelling workflow works the same.

For guided-wave specifics (4-PZT actuator/receiver indexing, drive-channel
exclusion, frequency-stratified labels), see
[`docs/MORPHO_NOTES.md`](docs/MORPHO_NOTES.md).

## Citation

```bibtex
@software{stamatelatos2026signal_curator,
  title   = {signal-curator: human-in-the-loop active learning for 1D signal datasets},
  author  = {Stamatelatos, Giannis},
  year    = {2026},
  doi     = {10.5281/zenodo.20119847},
  url     = {https://github.com/johnstamly/signal-curator},
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome. Please open an issue first for substantial
changes so we can discuss scope.

## Acknowledgments

The MORPHO dataset is provided by the H2020 MORPHO project
([Zenodo](https://doi.org/10.5281/zenodo.14627730)). The signal-curator
pipeline was developed at the
[Applied Mechanics & Vibrations Laboratory](https://www.upatras.gr/),
University of Patras, Greece.
