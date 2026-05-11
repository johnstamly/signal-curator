"""Smoke tests -- can the core pieces load and chain together cleanly?

These tests run on the bundled demo data only (no external dependencies).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_imports():
    """Every public symbol should be importable."""
    from signal_curator import (
        Config, DatasetAdapter, GenericAdapter, LabelingSession, MicroConv1D,
        SignalKey, augment, augment_and_train, labels_db, make_adapter,
        run_inference, __version__,
    )
    assert __version__ == "0.1.0"
    assert callable(make_adapter)


def test_model_param_count():
    """MicroConv1D at 500-sample input should have ~2.5k parameters."""
    from signal_curator import MicroConv1D
    m = MicroConv1D(input_len=500)
    n_params = sum(p.numel() for p in m.parameters())
    assert 2000 < n_params < 3000, f"unexpected param count: {n_params}"


def test_demo_data_loads():
    """The bundled synthetic demo data should load via GenericAdapter."""
    from signal_curator import make_adapter
    demo_path = REPO_ROOT / "data" / "demo" / "synthetic_demo.h5"
    if not demo_path.exists():
        pytest.skip(f"demo data not present at {demo_path} (regenerate via synthetic.py)")
    ad = make_adapter("generic", str(demo_path))
    assert 50 in ad.frequencies
    keys = ad.get_signal_keys(50)
    assert len(keys) == 100
    sig = ad.get_signal(keys[0], (0, 500))
    assert sig.shape == (500,)
    assert sig.dtype == np.float32


def test_morpho_curated_subset_loads():
    """The bundled 1,011-signal MORPHO subset should load."""
    from signal_curator import make_adapter
    h5 = REPO_ROOT / "data" / "morpho_50khz_curated" / "signals.h5"
    if not h5.exists():
        pytest.skip(f"curated subset not present at {h5} (regenerate via export_morpho_curated.py)")
    ad = make_adapter("generic", str(h5))
    assert 50 in ad.frequencies
    assert len(ad.get_signal_keys(50)) == 1011


def test_train_short_run():
    """End-to-end: synthetic demo -> short 5-epoch training -> infer."""
    from signal_curator import (
        MicroConv1D, augment_and_train, make_adapter, run_inference,
    )
    demo_path = REPO_ROOT / "data" / "demo" / "synthetic_demo.h5"
    if not demo_path.exists():
        pytest.skip("demo data not present")
    ad = make_adapter("generic", str(demo_path))
    keys = ad.get_signal_keys(50)

    # Pull the underlying class label from attrs (only used inside the demo for testing)
    import h5py
    labels = []
    signals = []
    with h5py.File(str(demo_path), "r") as f:
        for k in keys:
            from signal_curator.labels_db import signal_id
            sid_lookup = list(f.keys())
        # easier: read from disk in same order as iteration
        for sid in sid_lookup:
            grp = f[sid]
            class_truth = str(grp.attrs.get("class_truth", "good"))
            labels.append(1 if class_truth == "good" else 0)
            signals.append(np.asarray(grp["data"][:], dtype=np.float32))
    labeled_buffer = [{"signal": s, "label": l} for s, l in zip(signals, labels)]

    model, metrics = augment_and_train(labeled_buffer, epochs=5, patience=3)
    assert "val_acc" in metrics
    probs = run_inference(model, np.stack(signals))
    assert probs.shape == (len(signals),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_labels_db_roundtrip(tmp_path):
    """Insert, query, update, stats should work end-to-end."""
    from signal_curator import labels_db
    db = tmp_path / "labels.db"
    labels_db.init_db(str(db))
    sid = labels_db.signal_id("FOD3", 50, 1, 2, 100, 0)
    assert labels_db.insert_label(str(db), sid, "FOD3", 50, 1, 2, 100, 0, "GOOD")
    assert not labels_db.insert_label(str(db), sid, "FOD3", 50, 1, 2, 100, 0, "GOOD")  # duplicate
    stats = labels_db.get_stats(str(db), freq_khz=50)
    assert stats == {"total": 1, "good": 1, "bad": 0}
    assert labels_db.update_label(str(db), sid, "BAD")
    stats = labels_db.get_stats(str(db), freq_khz=50)
    assert stats == {"total": 1, "good": 0, "bad": 1}
