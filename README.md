# signal-curator

> Human-in-the-loop active learning labeling app for 1D signal datasets — built
> for guided-wave SHM with operator self-correction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`signal-curator` is a Streamlit-based labelling tool with a tight active
learning loop: a lightweight 1D CNN scores all signals, surfaces ambiguous
queries to the operator, and pairs them with **FP/FN re-inspection** of past
decisions — so the operator can correct their own labelling mistakes as the
model learns. The pipeline was developed for guided-wave structural health
monitoring (SHM) campaigns and accompanies the paper:

> *Active deep learning with operator self-correction for guided-wave SHM
> dataset curation* — Stamatelatos et al., *Measurement* (2026).

## Why use this

- **Faster, cleaner labelling** — small CNN ranks signals by uncertainty so the
  operator focuses on what's informative, not what's easy
- **Catches operator mistakes** — built-in FP/FN re-inspection step surfaces
  past labels the model now disagrees with, letting the operator audit and
  correct themselves
- **Battle-tested on real data** — achieves ≈96% validation accuracy on the
  MORPHO H2020 guided-wave dataset after labelling ≈8% of 12,504 signals
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

The repo ships with the **1,011-signal labelled subset** from the paper
(`data/morpho_50khz_curated/`, ~8 MB). To reproduce the convergence curve
(Fig 3) end-to-end:

```bash
signal-curator demo --dataset morpho_50khz_curated
# inside the app: switch frequency to 50 kHz, click "Train & Filter Dataset"
```

The **full 12,504-signal MORPHO 50 kHz universe** (~100 MB) is archived at
Zenodo (DOI: TBD upon release). Download it with:

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

If `signal-curator` helps your research, please cite the accompanying paper
**and** the code:

```bibtex
@article{stamatelatos2026curation,
  title   = {Active deep learning with operator self-correction for guided-wave SHM dataset curation},
  author  = {Stamatelatos, Giannis and Dimitriou, Dimitris K. and Loutas, Theodoros},
  journal = {Measurement},
  year    = {2026},
}

@software{stamatelatos2026signal_curator,
  title   = {signal-curator: human-in-the-loop active learning for 1D signal datasets},
  author  = {Stamatelatos, Giannis},
  year    = {2026},
  doi     = {TBD},
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
([Zenodo](https://zenodo.org/records/paunikar2025zenodo)). The signal-curator
pipeline was developed at the
[Applied Mechanics & Vibrations Laboratory](https://www.upatras.gr/),
University of Patras, Greece.
