"""Export the MORPHO 1,011-signal labelled subset for distribution with the repo.

Reads the in-paper label store at <repo>/shared_labels.db and the full
dataset.h5, then writes the labelled subset in the generic flat schema:

  data/morpho_50khz_curated/signals.h5   -- 1,011 signals, generic schema
  data/morpho_50khz_curated/labels.json  -- per-signal GOOD/BAD verdicts

Schema (generic):
  /<signal_id>/data           (L,) float32   -- windowed (default 0..500)
  /<signal_id>.attrs.freq_khz int
  /<signal_id>.attrs.panel    str
  /<signal_id>.attrs.actuator int (preserved from MORPHO)
  /<signal_id>.attrs.receiver int (preserved from MORPHO)
  /<signal_id>.attrs.cycle    int (preserved from MORPHO)
  /<signal_id>.attrs.rep_idx  int (preserved from MORPHO)

This is a one-off script run by the paper author when preparing the repo
for release. End users do not need to run it.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paper-db", default="../shared_labels.db",
                   help="Path to the in-paper shared_labels.db")
    p.add_argument("--paper-h5", default="../dataset.h5",
                   help="Path to the in-paper dataset.h5")
    p.add_argument("--freq-khz", type=int, default=50)
    p.add_argument("--window", default="0,500",
                   help="Sample window 'start,end' to crop signals to.")
    p.add_argument("--out-dir", default="data/morpho_50khz_curated")
    args = p.parse_args()

    paper_db = Path(args.paper_db).resolve()
    paper_h5 = Path(args.paper_h5).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    w_start, w_end = (int(x) for x in args.window.split(","))

    if not paper_db.exists():
        print(f"Paper DB not found: {paper_db}", file=sys.stderr); sys.exit(2)
    if not paper_h5.exists():
        print(f"Paper HDF5 not found: {paper_h5}", file=sys.stderr); sys.exit(2)

    conn = sqlite3.connect(str(paper_db))
    rows = conn.execute(
        "SELECT signal_id, panel, freq_khz, actuator, receiver, cycle, rep_idx, label, labeled_at "
        "FROM labels WHERE freq_khz = ? ORDER BY signal_id",
        (args.freq_khz,),
    ).fetchall()
    conn.close()
    print(f"Read {len(rows)} labels at {args.freq_khz} kHz from {paper_db.name}")

    signals_path = out_dir / "signals.h5"
    n_written = 0
    with h5py.File(paper_h5, "r") as src, h5py.File(signals_path, "w") as dst:
        for sid, panel, freq, act, recv, cycle, rep_idx, label, labeled_at in rows:
            src_path = f"{panel}/{freq}kHz/act{act}/recv{recv}/cycle{cycle}/data"
            if src_path not in src:
                print(f"  WARN: missing {src_path}", file=sys.stderr)
                continue
            src_data = src[src_path]
            if rep_idx >= src_data.shape[0]:
                print(f"  WARN: rep_idx {rep_idx} out of bounds for {src_path}", file=sys.stderr)
                continue
            sig = src_data[rep_idx, w_start:w_end]
            grp = dst.create_group(sid)
            grp.create_dataset("data", data=sig, compression="gzip")
            grp.attrs["freq_khz"] = freq
            grp.attrs["panel"] = panel
            grp.attrs["actuator"] = act
            grp.attrs["receiver"] = recv
            grp.attrs["cycle"] = cycle
            grp.attrs["rep_idx"] = rep_idx
            grp.attrs["label"] = label
            n_written += 1

    print(f"Wrote {n_written} signals -> {signals_path}")
    print(f"  HDF5 size: {signals_path.stat().st_size / 1024 / 1024:.2f} MB")

    labels_path = out_dir / "labels.json"
    flat_labels = [
        {
            "signal_id": r[0], "panel": r[1], "freq_khz": r[2],
            "actuator": r[3], "receiver": r[4], "cycle": r[5],
            "rep_idx": r[6], "label": r[7], "labeled_at": r[8],
        }
        for r in rows
    ]
    labels_path.write_text(json.dumps({
        "freq_khz": args.freq_khz,
        "n_total": len(flat_labels),
        "n_good": sum(1 for r in flat_labels if r["label"] == "GOOD"),
        "n_bad": sum(1 for r in flat_labels if r["label"] == "BAD"),
        "window": [w_start, w_end],
        "schema": "generic",
        "adapter": "generic",
        "labels": flat_labels,
    }, indent=2))
    print(f"Wrote {labels_path}")


if __name__ == "__main__":
    main()
