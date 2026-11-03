"""Core of the numbers generator (Task 14): refusals, key-path resolution, number spelling and
formatting, macro rendering.  No study knowledge lives here (see numbers_registry.py).

Every refusal is a predicate in ``CHECKS`` with its own exception class and exit code, so the
mutation driver (tests/mutation_numbers.py) can neutralise exactly one and run the tests.

Exit codes:
  2 absent study in --final        3 input hashes differ from the frozen ones
  4 forbidden output path          5 mixed code keys
  6 dirty code / cells             7 NaN / inf value in a macro that is not "not defined"
  8 comparison sides differ in seed count (or seed list)
  9 key path does not resolve     10 invalid or duplicate macro name
 11 provenance (meta / code key) missing
 12 --require-stage2 and an E1-E4 *_s20.json file is absent (Task 16)
 13 a paired comparison spans a Stage-1 file and a Stage-2 file
 14 a *_s20.json file is not a Stage-2 file (its stage / seed list disagree with the config)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

NA_TEXT = "n/a"


class Refusal(Exception):
    code = 1


class AbsentStudy(Refusal):
    code = 2


class InputHashMismatch(Refusal):
    code = 3


class ForbiddenOutput(Refusal):
    code = 4


class MixedCodeKeys(Refusal):
    code = 5


class DirtyCode(Refusal):
    code = 6


class NanValue(Refusal):
    code = 7


class SeedMismatch(Refusal):
    code = 8


class KeyPathMissing(Refusal):
    code = 9


class BadMacroName(Refusal):
    code = 10


class NoProvenance(Refusal):
    code = 11


class Stage2Missing(Refusal):
    code = 12


class StageMismatch(Refusal):
    code = 13


class BadStage2File(Refusal):
    code = 14


# ------------------------------------------------------------------ key paths
# A path is a tuple of tokens: str = dict key, int = list index, (key, value) = the list element
# whose [key] == value (e.g. ("budget", 600)).

def render_path(path) -> str:
    out = ""
    for t in path:
        if isinstance(t, tuple):
            out += f"[{t[0]}={t[1]:g}]" if isinstance(t[1], float) else f"[{t[0]}={t[1]}]"
        elif isinstance(t, int):
            out += f"[{t}]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
            out += ("." if out else "") + t
        else:
            out += f'["{t}"]'
    return out


_TOK = re.compile(r'\.?([A-Za-z_][A-Za-z0-9_]*)|\["((?:[^"\\]|\\.)*)"\]|\[(\d+)\]|\[([A-Za-z_]\w*)=([^\]]+)\]')


def parse_path(text: str):
    """Inverse of render_path (the test parses the comments of numbers.tex with it)."""
    pos, out = 0, []
    while pos < len(text):
        m = _TOK.match(text, pos)
        if not m:
            raise ValueError(f"cannot parse key path at {text[pos:]!r}")
        pos = m.end()
        if m.group(1):
            out.append(m.group(1))
        elif m.group(2) is not None:
            out.append(m.group(2))
        elif m.group(3):
            out.append(int(m.group(3)))
        else:
            v = m.group(5)
            try:
                v = int(v)
            except ValueError:
                v = float(v) if re.fullmatch(r"-?\d+\.\d+", v) else v
            out.append((m.group(4), v))
    return tuple(out)


def resolve(data, path, *, where: str = ""):
    cur = data
    for i, t in enumerate(path):
        try:
            if isinstance(t, tuple):
                k, v = t
                hits = [e for e in cur if isinstance(e, dict) and k in e
                        and (e[k] == v or (isinstance(v, float) and abs(e[k] - v) < 1e-9))]
                if len(hits) != 1:
                    raise KeyError(f"{len(hits)} elements with {k}={v}")
                cur = hits[0]
            else:
                cur = cur[t]
        except (KeyError, IndexError, TypeError) as e:          # MUT:keypath
            raise KeyPathMissing(f"{where}: key path {render_path(path)} does not resolve "
                                 f"(failed at {render_path(path[:i + 1])}: {e})") from None
    return cur


# ------------------------------------------------------------------ number words (macro names)
_ONES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
         "Eighteen", "Nineteen"]
_TENS = {2: "Twenty", 3: "Thirty", 4: "Forty", 5: "Fifty", 6: "Sixty", 7: "Seventy",
         8: "Eighty", 9: "Ninety"}


def spell(n: int) -> str:
    """CamelCase English words of a non-negative integer < 10**9 (macro names have no digits)."""
    if isinstance(n, bool) or int(n) != n or n < 0 or n >= 10 ** 9:
        raise ValueError(f"cannot spell {n!r}")
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + "Hundred" + ("" if n % 100 == 0 else spell(n % 100))
    for unit, name in ((10 ** 6, "Million"), (1000, "Thousand")):
        if n >= unit:
            return spell(n // unit) + name + ("" if n % unit == 0 else spell(n % unit))
    raise AssertionError


def tag(value) -> str:
    """Spell a number or pass a word through: 600 -> SixHundred, 0.02 -> ZeroPointZeroTwo."""
    if isinstance(value, str):
        if re.search(r"\d", value):
            raise ValueError(f"tag {value!r} contains digits")
        return value
    if float(value).is_integer():
        return spell(int(value))
    whole, frac = f"{float(value):.6f}".rstrip("0").split(".")
    return spell(int(whole)) + "Point" + "".join(_ONES[int(c)] for c in frac)


_NAME = re.compile(r"^[A-Za-z]+$")


def check_name(name: str) -> str:
    if not _NAME.match(name):                                   # MUT:name
        raise BadMacroName(f"invalid macro name {name!r}: letters only (no digits)")
    return name


# ------------------------------------------------------------------ formatting
def _num_ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _neg(s: str) -> str:
    return s.replace("-", "\\ensuremath{-}")


def _fixed(v: float, nd: int) -> str:
    """Fixed-point text with the sign marked for TeX; a value that rounds to zero is written
    without a sign (no \\ensuremath{-}0.000 and no -0.0)."""
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return _neg(s)


def fmt_value(x, kind: str) -> str:
    """kind: pct | pp | ppnum | int | f1 | f2 | f3 | sec | bool | text.
    ppnum is a percentage-point difference written as a bare number (no plus sign, no unit),
    for prose that spells out "percentage points", e.g. the abstract."""
    if kind == "text":
        return str(x).replace("|", "\\textbar{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    if kind == "bool":
        return "yes" if x else "no"
    if kind == "int":
        if float(x) != int(x):
            raise ValueError(f"non-integer value {x!r} for kind int")
        return f"{int(x):,}".replace(",", "{,}")
    if kind == "pct":
        return _fixed(100.0 * x, 1) + "\\%"
    if kind == "pp":
        v = 100.0 * x
        return ("+" if round(v, 1) > 0 else "") + _fixed(v, 1) + "\\,pp"
    if kind == "ppnum":
        return _fixed(100.0 * x, 1)
    if kind in ("f1", "f2", "f3"):
        return _fixed(x, int(kind[1]))
    if kind == "sec":
        return f"{x:.1f}"
    raise ValueError(f"unknown kind {kind!r}")


def fmt_ci(lo, hi, kind: str) -> str:
    """'a--b' (en dash), the unit suffix once at the end."""
    a, b = fmt_value(lo, kind), fmt_value(hi, kind)
    for suf in ("\\%", "\\,pp"):
        if a.endswith(suf) and b.endswith(suf):
            a = a[: -len(suf)]
    return f"{a}--{b}"


# ------------------------------------------------------------------ provenance checks
def check_hashes(study: str, data: dict, frozen: dict) -> None:
    meta = data.get("meta")
    got = meta.get("input_hashes") if isinstance(meta, dict) else None
    if got is None:                                             # MUT:nometa
        raise NoProvenance(f"{study}: no meta.input_hashes")
    if got != frozen:                                           # MUT:hash
        raise InputHashMismatch(f"{study}: meta.input_hashes {got} differ from the frozen "
                                f"hashes {frozen}")


def code_keys(data: dict, default_kind: str = "loop") -> dict:
    """{kind: key} read from cells_code_key (a str of the study's one cell kind, or a dict per
    kind as e06 writes).  Raises on mixed keys."""
    ck = data.get("cells_code_key")
    if ck is None:
        return {}
    kinds = ck if isinstance(ck, dict) else {default_kind: ck}
    out = {}
    for kind, key in kinds.items():
        if isinstance(key, list):                               # MUT:mixed_list
            raise MixedCodeKeys(f"cells_code_key[{kind}] lists {len(key)} keys: {key}")
        if not isinstance(key, str) or not key:
            raise NoProvenance(f"cells_code_key[{kind}] is not a key string: {key!r}")
        out[kind] = key
    counts = data.get("cells_key_counts") or {}
    per_kind = counts if isinstance(ck, dict) else {default_kind: counts}
    for kind, kc in per_kind.items():
        if isinstance(kc, dict) and len(kc) > 1:                # MUT:mixed_counts
            raise MixedCodeKeys(f"cells_key_counts[{kind}] has {len(kc)} distinct keys: "
                                f"{sorted(kc)}")
    return out


def check_dirty(study: str, data: dict) -> None:
    cd = data.get("cells_dirty")
    if cd:                                                      # MUT:dirty_cells
        raise DirtyCode(f"{study}: cells_dirty = {cd!r} (cells computed from a dirty tree)")
    df = (data.get("code") or {}).get("dirty_files")
    if df:                                                      # MUT:dirty_files
        raise DirtyCode(f"{study}: code.dirty_files = {df!r}")


def check_provenance_present(study: str, data: dict, mode: str) -> None:
    """mode: cells (needs cells_code_key + cells_dirty), static (needs code), none."""
    if mode == "cells" and ("cells_code_key" not in data or "cells_dirty" not in data):  # MUT:noprov_cells
        raise NoProvenance(f"{study}: cells_code_key / cells_dirty missing")
    if mode == "static" and "code" not in data:                 # MUT:noprov_static
        raise NoProvenance(f"{study}: code (git_sha, dirty_files) missing")


def check_cross_study_keys(per_study: dict[str, dict]) -> None:
    """{study: {kind: key}} -> one key per cell kind across all studies."""
    seen: dict[str, dict[str, list]] = {}
    for study, kk in per_study.items():
        for kind, key in kk.items():
            seen.setdefault(kind, {}).setdefault(key, []).append(study)
    for kind, keys in seen.items():
        if len(keys) > 1:                                       # MUT:mixed_cross
            raise MixedCodeKeys(f"cell kind {kind!r}: {len(keys)} different code keys across "
                                f"studies: " + "; ".join(f"{k[:12]} in {v}" for k, v in keys.items()))


def check_value(macro: str, where: str, x, *, not_defined: bool) -> bool:
    """True when the value is usable; False when it is NaN but the spec allows 'n/a'."""
    if isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isfinite(x):
        if not not_defined:                                     # MUT:nan
            raise NanValue(f"{macro}: value at {where} is {x!r} and the macro is not marked "
                           f"'not defined'")
        return False
    return True


def check_seed_match(macro: str, a, b) -> None:
    if a != b:                                                  # MUT:seeds
        raise SeedMismatch(f"{macro}: comparison sides differ in seeds: {a!r} vs {b!r}")


def check_seed_identity(macro: str, a, b) -> None:
    """The two sides of a paired difference must be the same seeds in the same order (the
    per-seed ``values`` arrays are paired by position), not only the same number of seeds.  Not in
    tests/mutation_numbers.py; covered by test_refuses_comparison_of_different_seeds_with_equal_counts."""
    if list(a) != list(b):
        raise SeedMismatch(f"{macro}: comparison sides use different seeds: {list(a)} vs {list(b)}")


# ------------------------------------------------------------------ stage selection (Task 16)
def s20_name(file: str) -> str:
    """e01_selection.json -> e01_selection_s20.json (the name the aggregators write for --seeds stage2)."""
    stem, dot, ext = file.rpartition(".")
    return f"{stem}_s20.{ext}" if dot else file + "_s20"


def stage_of(data: dict) -> str:
    """'stage1' | 'stage2': the stage of the file a study was loaded from."""
    src = data.get("_source")
    return (src or {}).get("stage") or data.get("stage") or "stage1"


def check_stage2_file(study: str, data: dict, expected_seeds) -> None:
    """A *_s20.json file must say it is a Stage-2 file and carry exactly the config's Stage-2 seeds."""
    if data.get("stage") != "stage2":                           # MUT:s20_stage
        raise BadStage2File(f"{study}: says stage={data.get('stage')!r}, not 'stage2'")
    if list(data.get("seeds") or []) != list(expected_seeds):    # MUT:s20_seeds
        raise BadStage2File(f"{study}: stage-2 file carries seeds {data.get('seeds')!r}, the config's "
                            f"seeds_stage2 are {list(expected_seeds)}")


def check_same_stage(macro: str, a: str, b: str) -> None:
    """A paired difference across two studies may use only files of one stage (20 seeds vs 5 seeds)."""
    if a != b:                                                  # MUT:same_stage
        raise StageMismatch(f"{macro}: comparison spans a {a} file and a {b} file")


def check_require_stage2(stages: dict) -> None:
    """stages = {study key: 'stage1' | 'stage2' | None (absent)} for the Stage-2 capable studies."""
    bad = sorted(k for k, v in stages.items() if v != "stage2")
    if bad:                                                     # MUT:require_s2
        raise Stage2Missing("--require-stage2 refuses: no Stage-2 (*_s20.json) file for " + ", ".join(bad))


def check_output_path(path: Path, forbidden: list[Path]) -> None:
    p = Path(path).resolve()
    for f in forbidden:
        f = Path(f).resolve()
        if p == f or f in p.parents:                            # MUT:outpath
            raise ForbiddenOutput(f"refusing to write under {f}")


def check_final(missing: list[str]) -> None:
    if missing:                                                 # MUT:final
        raise AbsentStudy("--final refuses: study JSON absent for " + ", ".join(missing))


# ------------------------------------------------------------------ macro rendering
def seed_label(n) -> str:
    return "n/a (deterministic)" if n is None else str(n)
