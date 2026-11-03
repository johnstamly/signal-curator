"""Mutation check for the numbers generator's refusals (Task 14).  From the repo root:

    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/tests/mutation_numbers.py

For each predicate marked ``# MUT:<name>`` in lib/numbers_core.py: delete ONLY that predicate
(``if <cond>:`` -> ``if False:``; the key-path ``except (...)`` -> ``except ZeroDivisionError``),
run tests/test_make_numbers.py and require at least one failing test.  The file is restored
byte-for-byte in a finally block and its hash re-checked.  Prints a markdown table.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TARGET = REPO / "Curation_Paper" / "eswa_full" / "lib" / "numbers_core.py"
TEST = "Curation_Paper/eswa_full/tests/test_make_numbers.py"

ROWS = [  # (marker, refusal it removes)
    ("hash", "input hashes differ from the frozen ones"),
    ("nometa", "meta.input_hashes absent"),
    ("noprov_cells", "cells_code_key / cells_dirty absent"),
    ("noprov_static", "code (dirty_files) absent on a static study"),
    ("mixed_list", "cells_code_key lists several keys (also per-kind dict value)"),
    ("mixed_counts", "cells_key_counts holds several keys (also per kind)"),
    ("mixed_cross", "a cell kind has different keys across studies"),
    ("dirty_cells", "cells_dirty > 0"),
    ("dirty_files", "code.dirty_files non-empty"),
    ("nan", "NaN/inf in a macro not marked not-defined"),
    ("seeds", "comparison sides differ in seed count"),
    ("keypath", "key path does not resolve"),
    ("name", "macro name with digits"),
    ("outpath", "output under the short paper's latex/"),
    ("final", "--final with an absent study"),
    ("s20_stage", "a *_s20.json file whose stage is not 'stage2' (Task 16)"),
    ("s20_seeds", "a *_s20.json file whose seeds differ from seeds_stage2 (Task 16)"),
    ("same_stage", "a paired difference spanning a Stage-1 and a Stage-2 file (Task 16)"),
    ("require_s2", "--require-stage2 with an absent *_s20.json (Task 16)"),
]


def mutate(src: str, name: str) -> str:
    out, n = [], 0
    for ln in src.splitlines(keepends=True):
        if f"# MUT:{name}" in ln:
            new = re.sub(r"^(\s*)if .*?:(\s*# MUT:)", r"\1if False:\2", ln)
            if new == ln:
                new = re.sub(r"^(\s*)except .*?:(\s*# MUT:)", r"\1except ZeroDivisionError:\2", ln)
            if new == ln:
                new = re.sub(r"^(\s*)if (.*?):(\s*# MUT:)", r"\1if False:\3", ln)
            assert new != ln, f"cannot mutate {name}: {ln!r}"
            ln = new
            n += 1
        out.append(ln)
    assert n == 1, f"marker {name} found {n} times"
    return "".join(out)


def run_tests() -> tuple[int, list[str], str]:
    p = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q", "-p", "no:cacheprovider"],
                       cwd=REPO, capture_output=True, text=True)
    failed = re.findall(r"^FAILED \S+::(\S+)", p.stdout, flags=re.M)
    tail = [ln for ln in p.stdout.splitlines() if " passed" in ln or " failed" in ln][-1:]
    return p.returncode, failed, (tail[0] if tail else p.stdout[-200:])


def main() -> int:
    original = TARGET.read_bytes()
    if re.search(r"if False:\s*# MUT:|except ZeroDivisionError:\s*# MUT:", original.decode("utf-8")):
        print("REFUSING: numbers_core.py already carries a mutation (an earlier run was killed); "
              "restore the predicate first")
        return 3
    digest = hashlib.sha256(original).hexdigest()
    rc, failed, tail = run_tests()
    print(f"baseline: rc={rc} {tail}")
    if rc != 0:
        return 2
    rows, bad = [], 0
    try:
        for name, what in ROWS:
            TARGET.write_text(mutate(original.decode("utf-8"), name), encoding="utf-8", newline="")
            rc, failed, tail = run_tests()
            killed = rc != 0 and bool(failed)
            bad += not killed
            rows.append((name, what, "KILLED" if killed else "SURVIVED", len(failed),
                         ", ".join(failed[:3])))
            print(f"{name:14s} {'KILLED' if killed else 'SURVIVED'}  {len(failed)} failing: "
                  f"{', '.join(failed[:3])}", flush=True)
    finally:
        TARGET.write_bytes(original)
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == digest, "file not restored"
    print("\n| mutation (predicate deleted) | refusal removed | result | failing tests | e.g. |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
