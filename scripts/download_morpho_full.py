"""Download the full MORPHO 50 kHz dataset (12,504 signals) from Zenodo.

This is Tier 3 in the dataset hierarchy (see README): the full universe needed
to run the complete active-learning convergence experiments. Tier 1 (synthetic
demo) and Tier 2 (1,011-signal labelled subset) are bundled in the repo.

The full file is ~100 MB. Hosted at Zenodo (DOI: TBD when released).
"""
from __future__ import annotations
import hashlib
import sys
import urllib.request
from pathlib import Path

# TODO: update with the actual Zenodo URL + SHA256 once the deposit is made.
ZENODO_URL = "https://zenodo.org/record/PLACEHOLDER/files/morpho_50khz_full.h5"
EXPECTED_SHA256 = "PLACEHOLDER_SHA256"
OUT_PATH = Path("data/morpho_50khz_full/signals.h5")


def sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        digest = sha256_file(OUT_PATH)
        if EXPECTED_SHA256 == "PLACEHOLDER_SHA256" or digest == EXPECTED_SHA256:
            print(f"{OUT_PATH} already present (sha256 {digest[:12]}...); skipping download.")
            return 0
        print(f"Existing {OUT_PATH} has wrong checksum, redownloading.")
        OUT_PATH.unlink()

    print(f"Downloading from {ZENODO_URL}...")
    print("(this is ~100 MB, may take a couple of minutes)")
    try:
        urllib.request.urlretrieve(ZENODO_URL, str(OUT_PATH))
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    digest = sha256_file(OUT_PATH)
    if EXPECTED_SHA256 != "PLACEHOLDER_SHA256" and digest != EXPECTED_SHA256:
        print(f"Checksum mismatch! Expected {EXPECTED_SHA256}, got {digest}", file=sys.stderr)
        return 2

    print(f"Downloaded to {OUT_PATH} (sha256 {digest[:12]}...)")
    print()
    print("To use it:")
    print("  signal-curator run --data data/morpho_50khz_full/signals.h5 --db ~/.signal-curator/labels.db --adapter morpho")
    return 0


if __name__ == "__main__":
    sys.exit(main())
