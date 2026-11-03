"""Mutation check for the break rule of the E13 throughput code (Task 15).  From the repo root:

    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/tests/mutation_throughput.py

Mutates the predicates marked ``# MUT:break`` and ``# MUT:seed`` in lib/throughput.py, runs
tests/test_throughput.py and requires at least one failing test.  Two mutants: the exclusion
deleted (``if True``) and the boundary flipped (``<=`` -> ``<``).  The file is restored
byte-for-byte in a finally block and its hash re-checked.  Prints a markdown table.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TARGET = REPO / "Curation_Paper" / "eswa_full" / "lib" / "throughput.py"
TEST = "Curation_Paper/eswa_full/tests/test_throughput.py"
B = "if g <= break_s]  # MUT:break"
MUTANTS = [("break exclusion deleted (if True)", B, "if True]  # MUT:break"),
           ("break boundary flipped (<= -> <)", B, "if g < break_s]  # MUT:break"),
           ("L_rand seed exclusion deleted (if True)", "if sid not in excl]  # MUT:seed",
            "if True]  # MUT:seed")]


def run_tests() -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q", "-p", "no:cacheprovider"],
                       cwd=REPO, capture_output=True, text=True)
    failed = [ln for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
    return p.returncode, "; ".join(f.split(" - ")[0].replace("FAILED ", "") for f in failed)


def main() -> int:
    raw = TARGET.read_bytes()
    h0 = hashlib.sha256(raw).hexdigest()
    src = raw.decode("utf-8")
    for _, o, _ in MUTANTS:
        assert src.count(o) == 1, f"mutation marker not found exactly once: {o}"
    rc0, _ = run_tests()
    print(f"baseline rc={rc0}")
    ok = rc0 == 0
    rows = []
    try:
        for name, o, repl in MUTANTS:
            TARGET.write_bytes(src.replace(o, repl).encode("utf-8"))
            rc, failed = run_tests()
            rows.append((name, rc, failed))
            ok &= rc != 0
    finally:
        TARGET.write_bytes(raw)
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == h0, "restore failed"
    print("| mutant | pytest rc | failing tests |\n|---|---|---|")
    for name, rc, failed in rows:
        print(f"| {name} | {rc} | {failed} |")
    print("MUTATION CHECK", "PASSED (every mutant killed)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
