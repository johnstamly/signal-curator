# Replay study of the paper

This folder holds the retrospective study behind the paper's results: the replay of the
labelling session, re-inspection under injected label noise, the comparison of eight
classifiers, the design ablations, the acceptance thresholds, the per-panel coverage and the
classical filters. It is published unchanged from the research repository in which the
results were computed. The folder names (`Curation_Paper/`, `eswa_full/`) are kept from that
repository because the code addresses its files by those paths.

## Provenance

Every stored result records a *code key*, the SHA-256 over the source files that can change
it (`lib/versioning.py`). The keys recomputed from the files in this repository match the keys
stored in every study result under `eswa_full/results/`, so this is the code that produced
them. The two frozen label pools are pinned by SHA-256 in `eswa_full/config/protocol.json`,
and every run stops if a hash differs.

## Layout

| Path | Contents |
|---|---|
| `shared_labels.db` (repository root) | $L_\mathrm{AL}$, the 1,011 labels of the guided session |
| `Curation_Paper/results/shared_labels_random.snapshot.db` | $L_\mathrm{rand}$, the same seed extended by 268 randomly drawn signals |
| `Curation_Paper/results/famb_trajectory.json` | ambiguous fraction recomputed for the live session |
| `Curation_Paper/eswa_full/config/protocol.json` | the frozen protocol (pools, folds, seeds, budgets) |
| `Curation_Paper/eswa_full/lib/` | the replay loop, training, metrics, aggregation |
| `Curation_Paper/eswa_full/scripts/` | one script per study (`e00` to `e14`), the cell runner, the number and figure generators |
| `Curation_Paper/eswa_full/results/` | the study results (`eNN_*.json`), the job lists and the determinism record |
| `Curation_Paper/lib/strategies.py`, `app/model.py` | the query strategies and the MicroConv1D model used by the replay |

Both label stores are frozen. Do not open them in the labelling app; it uses its own
`labels.db`.

Of the 2,866 per-cell results (about 300 MB), only the 180 fold-0 cells of E4 and E5 are
committed in `eswa_full/results/cells/`, because the clean-fold numbers are read from them
directly. The study results aggregate all the others, and `run_cells.py` rebuilds them.

## Reproducing

Run every command from the repository root.

1. Install the package with the replay extras.

   ```bash
   pip install -e ".[replay]"
   ```

2. Fetch the 12,504-signal HDF5 and the label file from the Zenodo dataset record
   ([10.5281/zenodo.20120069](https://doi.org/10.5281/zenodo.20120069)) into the folder the
   protocol expects. The HDF5 has SHA-256
   `436e22d836e1b4f11322c45995a64ac1de71718a817ce9dfb75c88c27ea7e3e0`.

   ```bash
   mkdir -p Curation_Paper/zenodo_tier3
   curl -L -o Curation_Paper/zenodo_tier3/morpho_50khz_full.h5 https://zenodo.org/records/20120070/files/morpho_50khz_full.h5
   curl -L -o Curation_Paper/zenodo_tier3/paper_labels.json https://zenodo.org/records/20120070/files/paper_labels.json
   ```

3. Verify the frozen inputs. This checks both pool hashes and that every labelled signal
   resolves in the HDF5 (746 shared signals, 13 cross-pool disagreements, none missing).

   ```bash
   python -m Curation_Paper.eswa_full.scripts.e00_verify_inputs
   ```

4. Rebuild every reported number from the stored study results. The output holds one LaTeX
   macro per number used in the article and its supplement.

   ```bash
   python Curation_Paper/eswa_full/scripts/make_numbers.py --final --require-stage2 --out numbers.tex
   ```

5. Rerun the studies (optional, many GPU hours). Rebuild the cells from a job list, then
   aggregate them with the study script, for example for E1:

   ```bash
   python Curation_Paper/eswa_full/scripts/run_cells.py --jobs Curation_Paper/eswa_full/results/jobs/stage2_loop.json --workers 6
   python Curation_Paper/eswa_full/scripts/e01_selection.py --seeds stage2
   ```

   The job lists in `results/jobs/` define every study's cells. Training is deterministic for a
   given seed on a GPU with deterministic CUDA algorithms. Runs on a CPU are not bit-identical.

6. Run the tests.

   ```bash
   python -m pytest Curation_Paper/eswa_full/tests -q
   ```

## Differences from the live tool

The replay is stricter than the labelling app in several places. Its re-inspection check
judges each labelled signal with a three-fold out-of-fold model and flags only confident
disagreements ($|p-\tfrac{1}{2}|>0.3$), whereas the app lists every labelled signal the
retrained model disagrees with at $p=0.5$. The replay also uses a 15% early-stopping split
with at most 300 epochs and patience 60, against the app's 20% split with 150 epochs and
patience 30. Supplementary Section S4 of the paper lists every difference.
