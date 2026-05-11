"""Command-line entry point for signal-curator.

After `pip install signal-curator` (or `pip install -e .` from source),
invoke as:

    signal-curator demo                      # open app on bundled demo data
    signal-curator demo --dataset morpho     # open app on bundled MORPHO 1011-signal slice
    signal-curator run --data /path/to/your.h5 --db /path/to/labels.db
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from signal_curator import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # signal-curator/


def _launch_streamlit(env: dict[str, str]) -> int:
    """Spawn `streamlit run signal_curator/app.py` with the given env overrides."""
    import subprocess
    full_env = os.environ.copy()
    full_env.update(env)
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd, env=full_env)


def cmd_demo(args: argparse.Namespace) -> int:
    """Launch the app pre-configured for a bundled demo dataset."""
    if args.dataset == "synthetic":
        data_path = REPO_ROOT / "data" / "demo" / "synthetic_demo.h5"
        db_path = REPO_ROOT / "data" / "demo" / "demo_labels.db"
        adapter = "generic"
    elif args.dataset == "morpho":
        data_path = REPO_ROOT / "data" / "morpho_50khz_curated" / "signals.h5"
        db_path = REPO_ROOT / "data" / "morpho_50khz_curated" / "demo_labels.db"
        adapter = "morpho"
    else:
        print(f"Unknown demo dataset: {args.dataset}", file=sys.stderr)
        return 2
    if not data_path.exists():
        print(f"Demo data not found at {data_path}.", file=sys.stderr)
        print("Make sure you cloned the full repo (including the data/ directory),", file=sys.stderr)
        print("or run: python scripts/download_morpho_full.py", file=sys.stderr)
        return 2
    env = {
        "SIGNAL_CURATOR_DATA": str(data_path),
        "SIGNAL_CURATOR_DB": str(db_path),
        "SIGNAL_CURATOR_ADAPTER": adapter,
    }
    print(f"Launching app on demo dataset '{args.dataset}' ({data_path})", flush=True)
    return _launch_streamlit(env)


def cmd_run(args: argparse.Namespace) -> int:
    """Launch the app on user-supplied data + DB."""
    env = {
        "SIGNAL_CURATOR_DATA": str(Path(args.data).resolve()),
        "SIGNAL_CURATOR_DB": str(Path(args.db).resolve()),
        "SIGNAL_CURATOR_ADAPTER": args.adapter,
    }
    if args.window:
        env["SIGNAL_CURATOR_WINDOW"] = args.window
    return _launch_streamlit(env)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="signal-curator",
        description="Human-in-the-loop active learning labelling app for 1D signal datasets.",
    )
    p.add_argument("--version", action="version", version=f"signal-curator {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Launch the app on a bundled demo dataset.")
    p_demo.add_argument(
        "--dataset",
        choices=["synthetic", "morpho"],
        default="synthetic",
        help="Which bundled dataset to use (default: synthetic).",
    )
    p_demo.set_defaults(func=cmd_demo)

    p_run = sub.add_parser("run", help="Launch the app on your own data.")
    p_run.add_argument("--data", required=True, help="Path to your HDF5 dataset.")
    p_run.add_argument("--db", required=True, help="Path to (or for creating) the labels SQLite DB.")
    p_run.add_argument(
        "--adapter",
        choices=["morpho", "generic"],
        default="generic",
        help="Dataset adapter (default: generic).",
    )
    p_run.add_argument(
        "--window",
        default=None,
        help="Signal window 'start,end' (default uses SIGNAL_CURATOR_WINDOW or 0,500).",
    )
    p_run.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
