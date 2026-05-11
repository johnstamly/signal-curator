# Tutorial: labelling a dataset end-to-end

This walks through using `signal-curator` to curate a 1D-signal dataset from
scratch, using the bundled synthetic demo data. The same workflow applies to
your own data once you have it in one of the supported HDF5 schemas (see
[ADAPTING.md](ADAPTING.md) for non-MORPHO datasets).

## 1. Install

```bash
git clone https://github.com/johnstamly/signal-curator
cd signal-curator
pip install -e .
```

This installs the `signal-curator` CLI and the `signal_curator` Python
package.

## 2. Launch the app on the demo dataset

```bash
signal-curator demo
```

Your browser opens at `http://localhost:8501`. You'll see:

- **Sidebar (left)**: frequency selector, training-config sliders, labelling
  progress counters, model status, queue stats, "Train & Filter Dataset"
  button
- **Main panel**: the current signal to label, plus GOOD / BAD / Skip buttons

## 3. Phase 1 -- seed labelling (~20 signals)

The model has not been trained yet, so the queue is a random shuffle of all
unlabelled signals. **Label ~20 signals as a seed batch.**

- Click **GOOD (valid wave)** for clean wave-packets.
- Click **BAD (sensor error)** for low-energy / saturated / noisy / drifting
  signals.
- Click **Skip** if a signal is truly uninterpretable (should be < 5% of
  labels).

Each click updates the sidebar Total / GOOD / BAD counters.

## 4. Train the first model

Once you have at least 2 GOOD and 2 BAD labels (the model needs both classes),
click **Train & Filter Dataset**. The app:

1. Trains MicroConv1D from scratch on your labels (a few seconds for the demo)
2. Scores every unlabelled signal in the dataset
3. Reshuffles the queue to surface the most ambiguous signals (those with
   probability `0.4 < p < 0.6`)
4. Shows validation results: probability histogram, true/false positives, and
   any FP/FN entries from your existing labels

## 5. Phase 2 -- active learning + self-correction

Now two things happen at every iteration:

### (a) Label the new uncertainty queue
The queue is now sorted by `|p - 0.5|`, ascending. The most ambiguous signals
appear first -- these are the ones where labelling provides the most
information per unit operator effort. Label another ~20.

### (b) Review FP/FN re-inspections
Scroll down to the "Review misclassified signals" section. The model now
disagrees with some of your earlier labels. For each disagreement, decide:

- **The model is right, my old label was wrong** -> click "Relabel GOOD" /
  "Relabel BAD". This is *operator self-correction*: the model has surfaced a
  past mistake so the operator can fix it.
- **The model is wrong, my label was right** -> ignore. The next training
  round will train on the (preserved) label and likely learn the boundary
  better.

This FP/FN re-inspection step is the key methodological contribution of the
accompanying paper -- it is what makes the pipeline robust to a labeller
who occasionally makes mistakes.

## 6. Repeat 4-5 until the queue empties

Each Train -> Label -> Relabel cycle produces a better model. You can stop
when:

- The "Uncertain (needs label)" count in the sidebar reaches zero or
  stabilises
- Validation accuracy stops improving across iterations
- You hit your operator-effort budget

For the synthetic demo data you can expect to reach ~95-100% accuracy with
~50-80 labels.

## 7. Where are my labels?

All labels are stored in the SQLite DB at
`data/demo/demo_labels.db` (or wherever `SIGNAL_CURATOR_DB` points). You can
query it directly:

```python
import sqlite3
conn = sqlite3.connect("data/demo/demo_labels.db")
rows = conn.execute("SELECT signal_id, label FROM labels").fetchall()
print(f"{len(rows)} labels")
```

Trained model weights are saved to `weights/model_<freq>kHz_weights.pt` and
auto-restore when you reopen the app at the same frequency.

## 8. Working with your own data

See [ADAPTING.md](ADAPTING.md) -- in short, convert your data to the generic
HDF5 schema (one group per signal) and launch with:

```bash
signal-curator run --data /path/to/your.h5 --db /path/to/labels.db --adapter generic
```

## 9. Reproducing the paper

The repo ships with the 1,011-signal MORPHO 50 kHz labelled subset:

```bash
signal-curator demo --dataset morpho
```

To reproduce Fig. 3 (convergence curve) end-to-end requires the full
12,504-signal pool, which is hosted on Zenodo:

```bash
python scripts/download_morpho_full.py
signal-curator run --data data/morpho_50khz_full/signals.h5 \
                   --db data/morpho_50khz_full/labels.db \
                   --adapter morpho
```
