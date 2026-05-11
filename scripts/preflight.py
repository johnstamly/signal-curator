"""One-shot pre-flight check before publishing the repo.

Simulates the full end-to-end labelling -> training -> FP/FN re-inspection
workflow without spinning up the Streamlit browser. Run from the repo root:

    python scripts/preflight.py

Exits 0 on success, non-zero on failure. Useful as the last verification
step before `git push` / Zenodo deposit / paper submission.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS = "[ OK ]"
FAIL = "[FAIL]"
SKIP = "[skip]"


class Check:
    """Tiny check runner with consistent reporting."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.start = time.time()

    def run(self, name: str, fn) -> bool:
        t0 = time.time()
        try:
            result = fn()
        except Exception as e:
            self.failed += 1
            elapsed = time.time() - t0
            print(f"  {FAIL} {name}  ({elapsed:.2f}s)")
            print(f"         {type(e).__name__}: {e}")
            return False
        elapsed = time.time() - t0
        if result is False:
            self.failed += 1
            print(f"  {FAIL} {name}  ({elapsed:.2f}s)")
            return False
        if result == "skip":
            self.skipped += 1
            print(f"  {SKIP} {name}  ({elapsed:.2f}s)")
            return True
        self.passed += 1
        print(f"  {PASS} {name}  ({elapsed:.2f}s)")
        return True

    def summary(self) -> int:
        total = self.passed + self.failed + self.skipped
        elapsed = time.time() - self.start
        print()
        print("-" * 60)
        print(f"  Passed: {self.passed:>3d}    Failed: {self.failed:>3d}    "
              f"Skipped: {self.skipped:>3d}    Total: {total}    ({elapsed:.1f}s)")
        if self.failed == 0:
            print(f"  {PASS} pre-flight green -- repo is ready to publish")
            return 0
        print(f"  {FAIL} fix the failures above before publishing")
        return 1


# -- Individual checks -------------------------------------------------------

def check_python_version() -> bool:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ required, got {sys.version}")
    return True


def check_required_files() -> bool:
    required = [
        "LICENSE", "README.md", "pyproject.toml", "CONTRIBUTING.md",
        "src/signal_curator/__init__.py",
        "src/signal_curator/app.py",
        "src/signal_curator/cli.py",
        "src/signal_curator/model.py",
        "src/signal_curator/trainer.py",
        "src/signal_curator/labels_db.py",
        "src/signal_curator/dataset.py",
        "src/signal_curator/morpho_adapter.py",
        "src/signal_curator/session.py",
        "src/signal_curator/config.py",
        "src/signal_curator/synthetic.py",
        "data/demo/synthetic_demo.h5",
        "data/morpho_50khz_curated/signals.h5",
        "data/morpho_50khz_curated/labels.json",
        "tests/test_smoke.py",
        ".github/workflows/ci.yml",
        "docs/TUTORIAL.md",
        "docs/ADAPTING.md",
        "docs/MORPHO_NOTES.md",
    ]
    missing = [p for p in required if not (REPO_ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")
    return True


def check_imports() -> bool:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import signal_curator  # noqa: F401
    from signal_curator import (
        Config, DatasetAdapter, GenericAdapter, LabelingSession, MicroConv1D,
        SignalKey, augment, augment_and_train, labels_db, make_adapter,
        run_inference,
    )
    assert signal_curator.__version__ == "0.1.0"
    return True


def check_pytest_suite() -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"pytest exit code {result.returncode}")
    return True


def check_cli_version() -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "signal_curator.cli", "--version"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=20,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"CLI exit code {result.returncode}")
    if "signal-curator 0.1.0" not in result.stdout:
        raise RuntimeError(f"unexpected --version output: {result.stdout!r}")
    return True


def check_demo_data_integrity() -> bool:
    demo = REPO_ROOT / "data" / "demo" / "synthetic_demo.h5"
    with h5py.File(demo, "r") as f:
        n_signals = len(list(f.keys()))
    if n_signals != 100:
        raise RuntimeError(f"expected 100 demo signals, got {n_signals}")
    return True


def check_morpho_subset_integrity() -> bool:
    h5 = REPO_ROOT / "data" / "morpho_50khz_curated" / "signals.h5"
    with h5py.File(h5, "r") as f:
        n_signals = len(list(f.keys()))
    if n_signals != 1011:
        raise RuntimeError(f"expected 1011 MORPHO signals, got {n_signals}")
    import json
    labels = json.loads((REPO_ROOT / "data" / "morpho_50khz_curated" / "labels.json").read_text())
    if labels["n_total"] != 1011 or labels["n_good"] != 875 or labels["n_bad"] != 136:
        raise RuntimeError(f"unexpected label counts: {labels['n_total']}, "
                           f"{labels['n_good']}, {labels['n_bad']}")
    return True


def check_end_to_end_workflow() -> bool:
    """Simulate: load demo -> label some -> train -> infer -> FP/FN flag.
    This is what would happen in the Streamlit app, minus the browser.
    """
    from signal_curator import (
        MicroConv1D, augment_and_train, make_adapter, run_inference,
    )
    from signal_curator.labels_db import (
        init_db, insert_label, get_all_labels, get_stats, signal_id, _reset,
    )
    from signal_curator.session import SignalKey

    # Clear any stale thread-local SQLite connections from prior checks
    _reset()

    demo = REPO_ROOT / "data" / "demo" / "synthetic_demo.h5"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            db_path = str(Path(tmpdir) / "labels.db")
            init_db(db_path)
            adapter = make_adapter("generic", str(demo))
            keys = adapter.get_signal_keys(50)
            if len(keys) != 100:
                raise RuntimeError(f"expected 100 demo keys, got {len(keys)}")

            with h5py.File(str(demo), "r") as f:
                sids = sorted(f.keys())
                truth = {sid: ("GOOD" if f[sid].attrs.get("class_truth", "good") == "good"
                               else "BAD") for sid in sids}

            for sid in sids[:40]:
                insert_label(db_path, sid, "demo", 50, 0, 0, 0, 0, truth[sid])

            stats = get_stats(db_path, freq_khz=50)
            if stats["total"] != 40:
                raise RuntimeError(f"expected 40 labels, got {stats['total']}")

            labelled = get_all_labels(db_path, freq_khz=50)
            labelled_buf = []
            for r in labelled:
                with h5py.File(str(demo), "r") as f:
                    sig = np.asarray(f[r["signal_id"]]["data"][0:500], dtype=np.float32)
                labelled_buf.append({"signal": sig, "label": 1 if r["label"] == "GOOD" else 0})

            model, metrics = augment_and_train(labelled_buf, epochs=10, patience=5)
            if metrics["val_acc"] < 0.5:
                raise RuntimeError(f"val_acc suspiciously low: {metrics['val_acc']}")

            labelled_sids = {r["signal_id"] for r in labelled}
            unlabelled_keys = [k for k in keys if adapter._sid_lookup[k] not in labelled_sids]
            with h5py.File(str(demo), "r") as f:
                ad_sids = [adapter._sid_lookup[k] for k in unlabelled_keys]
                sigs = np.stack([np.asarray(f[s]["data"][0:500], dtype=np.float32)
                                 for s in ad_sids])
            probs = run_inference(model, sigs)
            if probs.shape != (60,):
                raise RuntimeError(f"expected 60 probs, got {probs.shape}")
            if not ((probs >= 0).all() and (probs <= 1).all()):
                raise RuntimeError("probs out of [0,1] range")
        finally:
            # Close all sqlite connections before TemporaryDirectory cleanup;
            # otherwise Windows refuses to delete the still-open DB file.
            _reset()
    return True


def check_streamlit_app_imports() -> bool:
    """Verify the Streamlit app module compiles cleanly in a fresh subprocess.

    Run in a subprocess so streamlit's import-time st.* calls don't pollute
    state in this preflight process, and so any import error / syntax error
    surfaces clearly.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["SIGNAL_CURATOR_DATA"] = str(REPO_ROOT / "data" / "demo" / "synthetic_demo.h5")
    env["SIGNAL_CURATOR_DB"] = str(REPO_ROOT / "data" / "demo" / "_preflight_labels.db")
    env["SIGNAL_CURATOR_WEIGHTS_DIR"] = str(REPO_ROOT / "data" / "demo" / "_preflight_weights")
    env["SIGNAL_CURATOR_ADAPTER"] = "generic"
    # Suppress streamlit's bare-mode warnings entirely
    env["STREAMLIT_LOG_LEVEL"] = "error"
    result = subprocess.run(
        [sys.executable, "-c", "import signal_curator.app; print('OK')"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    # Clean up the temporary files
    for p in ("_preflight_labels.db", "_preflight_weights"):
        target = REPO_ROOT / "data" / "demo" / p
        if target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"streamlit app import failed (exit {result.returncode})")
    return True


# -- Main --------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 60)
    print("  signal-curator pre-flight check")
    print("=" * 60)
    print()

    c = Check()
    print("Environment:")
    c.run("Python >= 3.10", check_python_version)
    print()
    print("Repository structure:")
    c.run("required files present", check_required_files)
    print()
    print("Source code:")
    c.run("all modules importable", check_imports)
    c.run("CLI --version works", check_cli_version)
    c.run("Streamlit app imports cleanly", check_streamlit_app_imports)
    print()
    print("Test suite:")
    c.run("pytest suite passes", check_pytest_suite)
    print()
    print("Bundled data:")
    c.run("synthetic demo has 100 signals", check_demo_data_integrity)
    c.run("MORPHO subset has 1,011 signals (875/136)", check_morpho_subset_integrity)
    print()
    print("End-to-end workflow:")
    c.run("label -> train -> infer -> FP/FN", check_end_to_end_workflow)
    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
