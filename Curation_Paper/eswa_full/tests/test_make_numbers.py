"""Task 14: numbers.tex generator.  Study JSONs come from tests/study_fixtures.py (synthetic cells
-> the real aggregators) plus the six real committed JSONs.  CPU only."""
import copy
import json
import math
import re
import shutil
from pathlib import Path

import pytest

from Curation_Paper.eswa_full.lib import fold0 as F0
from Curation_Paper.eswa_full.lib import numbers_core as C
from Curation_Paper.eswa_full.lib import numbers_registry as R
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT, REPO_ROOT, input_hashes, load_config
from Curation_Paper.eswa_full.scripts import make_numbers as MN
from Curation_Paper.eswa_full.tests import study_fixtures as SF

REAL_REFERENCE = F0.reference           # captured at import, before the autouse fixture patches it

FROZEN_AL = "aaa3547f8c60969405ebc6437e9583e2189f14b1fbd2cb39a95ae796ecfd36b1"
FROZEN_RAND = "90ca86bcb09d9a296cfbde628938f3daaa2d5355f33018c47775c5899d76da3f"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = SF.build_all(tmp_path_factory.mktemp("numbers_world"))
    w["cfg"]["seeds_stage1"] = [0, 1]          # the fixture has two seeds
    return w


@pytest.fixture(autouse=True)
def fake_reference(world, monkeypatch):
    monkeypatch.setattr(F0, "reference", lambda cfg: world["ref"])


@pytest.fixture
def data(world):
    d, absent = MN.load_studies(Path(world["cfg"]["results_dir"]), world["cfg"])
    assert absent == []
    return copy.deepcopy(d)


FROZEN = {"al": FROZEN_AL, "rand": FROZEN_RAND}


def render_final(d):
    MN.check_all(d, FROZEN)
    return MN.render(d, [], "final")


# ---------------------------------------------------------------- naming
def test_frozen_hashes_equal_global_constraints():
    assert input_hashes(load_config()) == FROZEN


@pytest.mark.parametrize("n,words", [
    (0, "Zero"), (6, "Six"), (13, "Thirteen"), (20, "Twenty"), (25, "TwentyFive"), (95, "NinetyFive"),
    (100, "OneHundred"), (101, "OneHundredOne"), (250, "TwoHundredFifty"), (600, "SixHundred"),
    (1000, "OneThousand"), (12504, "TwelveThousandFiveHundredFour")])
def test_spell(n, words):
    assert C.spell(n) == words


def test_tag_and_name_rules():
    assert C.tag(0.02) == "ZeroPointZeroTwo" and C.tag("Al") == "Al" and C.tag(6.0) == "Six"
    with pytest.raises(ValueError):
        C.tag("B6")
    with pytest.raises(C.BadMacroName):
        C.check_name("EoneAcc6")
    assert C.check_name("EoneAccAlBsix") == "EoneAccAlBsix"
    with pytest.raises(ValueError):
        C.spell(-1)


def test_formatting():
    assert C.fmt_value(0.9321, "pct") == "93.2\\%"
    assert C.fmt_value(12504, "int") == "12{,}504"
    assert C.fmt_ci(0.91, 0.941, "pct") == "91.0--94.1\\%"
    assert C.fmt_value(0.023, "pp") == "+2.3\\,pp"
    assert C.fmt_value(0.023, "ppnum") == "2.3"                     # bare number, no sign or unit
    assert C.fmt_value(-0.023, "ppnum") == "\\ensuremath{-}2.3"     # a real negative stays visible
    assert C.fmt_value(-0.0214, "f3") == "\\ensuremath{-}0.021"
    assert C.fmt_value(True, "bool") == "yes"


def test_negative_zero_is_written_without_a_sign():
    assert C.fmt_value(-0.0001, "f3") == "0.000" and C.fmt_value(-0.0, "f3") == "0.000"
    assert C.fmt_value(-0.0004, "pct") == "0.0\\%"
    assert C.fmt_value(-0.0004, "pp") == "0.0\\,pp"          # no plus, no minus
    assert C.fmt_ci(-0.0004, 0.0123, "pp") == "0.0--+1.2\\,pp"
    assert C.fmt_value(-0.0006, "pct") == "\\ensuremath{-}0.1\\%"   # a real negative is kept
    assert "ensuremath{-}0.0" not in C.fmt_value(-0.00049, "f3") and C.fmt_value(-0.0006, "f3") == \
        "\\ensuremath{-}0.001"


def test_numbers_file_has_lf_line_endings(world, tmp_path):
    out = tmp_path / "n.tex"
    assert MN.main(["--final", "--results-dir", str(world["cfg"]["results_dir"]), "--seed-list", "0,1",
                    "--out", str(out)]) == 0
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")


def test_shift_pool_and_batch_macros_exist_and_agree_with_their_files(data):
    text, _ = render_final(data)
    got = {n: v for n, v, _ in _parse_macros(text)}
    e14, sh, ps = data["e14"], data["e01_shift"], data["pool_sizes"]
    inc = e14["cross_check_famb"]["increments"]
    assert [got[f"EfourteenBatch{w}"] for w in ("One", "Two", "Three")] == [str(x) for x in inc]
    assert got["EzeroNSharedOutsideSeed"] == str(sh["n_shared_outside_seed"])
    assert sh["n_shared_outside_seed"] == data["e00"]["n_shared"] - e14["seed_and_added"]["n_seed"]
    for pool, t in (("L_AL", "Al"), ("L_rand", "Rand")):
        assert got[f"EzeroNGood{t}"] == f"{sh['pools'][pool]['n_good']:,}".replace(",", "{,}")
        assert got[f"EzeroNBad{t}"] == str(sh["pools"][pool]["n_bad"])
    assert got["EzeroTrainPoolMin"] == str(ps["len_train_pool"]["min"])
    assert got["EzeroTrainPoolMax"] == str(ps["len_train_pool"]["max"])
    assert got["EoneShiftSeedAlN"] == str(sh["groups"]["seed_L_AL"]["n"]) == str(e14["seed_and_added"]["n_seed"])


# ---------------------------------------------------------------- comments are machine-resolvable
_CMT = re.compile(r"^% (?P<file>\S+) :: (?P<paths>.+) \| seeds: (?P<seeds>.+)$")


def _parse_macros(text):
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = re.match(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", ln)
        if m:
            out.append((m.group(1), m.group(2), lines[i - 1]))
    return out


def test_every_macro_comment_names_an_existing_key_path_and_seed_count(data):
    text, counts = render_final(data)
    labels = {R.BY_KEY[k].file or f"computed:{k}": data[k] for k in data}
    macros = _parse_macros(text)
    assert len(macros) == sum(counts.values()) > 300
    assert set(counts) == set(R.BY_KEY)                       # every registered study contributes
    for name, val, cmt in macros:
        assert re.fullmatch(r"[A-Za-z]+", name)
        m = _CMT.match(cmt)
        assert m, f"{name}: comment {cmt!r}"
        d = labels[m["file"]]
        seeds = m["seeds"]
        assert seeds == "n/a (deterministic)" or re.fullmatch(r"\d+", seeds), (name, seeds)
        other = re.search(r" \(([^)]*\.json)\)$", m["paths"])
        body = re.sub(r" \([^)]*\)$", "", m["paths"])
        for i, chunk in enumerate(re.split(r" & | MINUS ", body)):
            chunk = re.sub(r"\.values$", "", chunk) if " MINUS " in body else chunk
            dd = labels[other[1]] if (i == 1 and other) else d
            node = C.resolve(dd, C.parse_path(chunk), where=name)      # raises if absent
            assert node is not None


def test_macro_values_match_the_json(data):
    text, _ = render_final(data)
    got = {n: v for n, v, _ in _parse_macros(text)}
    e1 = data["e01"]
    r = next(x for x in e1["configs"]["unc_oracle_al"]["rows"] if x["budget"] == 600)
    assert got["EoneAccAlBsix"] == f"{100 * r['metrics']['acc']['mean']:.1f}\\%"
    assert got["EoneAccAlBsixCi"] == (f"{100 * r['metrics']['acc']['ci_lo']:.1f}--"
                                      f"{100 * r['metrics']['acc']['ci_hi']:.1f}\\%")
    assert got["EnineFilterTightBadRecall"] == "11.3\\%"                 # real e09: 7/62
    assert got["EzeroNUnion"] == "1{,}276" and got["EtenNSignals"] == "12{,}504"
    assert got["EelevenFullFodThreeMonoDiff"] == "0.196"                 # real e11 FOD3


def test_seed_counts_in_comments_match_the_rows(data):
    text, _ = render_final(data)
    cm = dict((n, c) for n, _, c in _parse_macros(text))
    assert cm["EoneAccAlBsix"].endswith("| seeds: 2")
    assert cm["EtenNSignals"].endswith("| seeds: 5")
    assert cm["EnineTreeBadRecall"].endswith("| seeds: n/a (deterministic)")


def test_ci_comment_names_both_paths_and_difference_names_both_sides(data):
    text, _ = render_final(data)
    cm = dict((n, c) for n, _, c in _parse_macros(text))
    assert " & " in cm["EoneAccAlBsixCi"]
    assert " MINUS " in cm["EoneSelBalAccDiffBsix"]


# ---------------------------------------------------------------- refusals
def _expect(code, exc, fn):
    with pytest.raises(exc) as ei:
        fn()
    assert ei.value.code == code


def test_refuses_input_hash_mismatch(data):
    data["e03"]["meta"]["input_hashes"]["al"] = "0" * 64
    _expect(3, C.InputHashMismatch, lambda: render_final(data))


def test_refuses_missing_meta(data):
    del data["e03"]["meta"]
    _expect(11, C.NoProvenance, lambda: render_final(data))


def _row(d, cfgname, b):
    return next(x for x in d["configs"][cfgname]["rows"] if x["budget"] == b)


def test_refuses_nan_in_a_defined_macro_and_reports_file_and_key(data):
    _row(data["e01"], "unc_oracle_al", 600)["metrics"]["acc"]["mean"] = float("nan")
    with pytest.raises(C.NanValue) as ei:
        render_final(data)
    assert ei.value.code == 7
    msg = str(ei.value)
    assert "e01_selection.json" in msg and "configs.unc_oracle_al.rows[budget=600].metrics.acc.mean" in msg


def test_not_defined_macro_emits_na_only_where_marked(data):
    text, _ = render_final(data)
    got = {n: v for n, v, _ in _parse_macros(text)}
    assert got["EthreeRecoveryOracleEtaZeroPctBsix"] == "n/a"
    assert got["EthreeRecoveryOracleEtaZeroPctBsixCi"] == "n/a"
    # the same NaN in a macro that is NOT marked refuses
    cell = data["e03"]["cells"]["unc_oracle|eta=0.1"]
    next(x for x in cell["rows"] if x["budget"] == 600)["counters"]["recovery_rate"]["mean"] = float("nan")
    with pytest.raises(C.NanValue):
        render_final(data)


def test_refuses_nan_ci_bound(data):
    _row(data["e01"], "unc_oracle_al", 600)["metrics"]["acc"]["ci_hi"] = float("nan")
    _expect(7, C.NanValue, lambda: render_final(data))


def test_refuses_comparison_with_different_seed_counts(data):
    _row(data["e01"], "random_al", 600)["n_seeds"] = 7
    _expect(8, C.SeedMismatch, lambda: render_final(data))


def test_refuses_comparison_of_different_seeds_with_equal_counts(data):
    _row(data["e01"], "random_al", 600)["seeds"] = [0, 1, 2, 3, 9]     # same count, other seed
    _expect(8, C.SeedMismatch, lambda: render_final(data))


def test_refuses_cross_study_comparison_of_different_seeds_with_equal_counts(data):
    r = next(x for x in data["e04_compare"]["models"]["microconv1d"]["rows"] if x["budget"] == 600)
    r["seeds"] = list(reversed(r["seeds"]))                              # same seeds, other order
    _expect(8, C.SeedMismatch, lambda: render_final(data))


def test_refuses_cross_study_comparison_with_different_seed_counts(data):
    r = next(x for x in data["e04_compare"]["models"]["microconv1d"]["rows"] if x["budget"] == 600)
    r["n_seeds"] = 9
    _expect(8, C.SeedMismatch, lambda: render_final(data))


def test_refuses_value_arrays_of_different_length(data):
    _row(data["e01"], "random_al", 600)["metrics"]["bal_acc"]["values"].append(0.5)
    _expect(8, C.SeedMismatch, lambda: render_final(data))


def test_refuses_mixed_code_keys_list(data):
    data["e02"]["cells_code_key"] = ["a" * 64, "c" * 64]
    _expect(5, C.MixedCodeKeys, lambda: render_final(data))


def test_refuses_mixed_key_counts(data):
    data["e02"]["cells_key_counts"] = {"a" * 64: 5, "c" * 64: 5}
    _expect(5, C.MixedCodeKeys, lambda: render_final(data))


def test_refuses_mixed_per_kind_dict(data):
    assert isinstance(data["e06"]["cells_code_key"], dict)        # the per-kind dict is accepted as is
    data["e06"]["cells_code_key"]["ablate"] = ["b" * 64, "d" * 64]
    _expect(5, C.MixedCodeKeys, lambda: render_final(data))


def test_refuses_mixed_per_kind_counts(data):
    data["e06"]["cells_key_counts"]["loop"] = {"a" * 64: 5, "e" * 64: 5}
    _expect(5, C.MixedCodeKeys, lambda: render_final(data))


def test_refuses_loop_key_that_differs_across_studies(data):
    data["e02"]["cells_code_key"] = "c" * 64
    data["e02"]["cells_key_counts"] = {"c" * 64: 10}
    _expect(5, C.MixedCodeKeys, lambda: render_final(data))
    data["e02"]["cells_code_key"] = "a" * 64
    data["e06"]["cells_code_key"]["loop"] = "9" * 64
    data["e06"]["cells_key_counts"]["loop"] = {"9" * 64: 10}
    _expect(5, C.MixedCodeKeys, lambda: render_final(data))


def test_refuses_dirty_cells(data):
    data["e07"]["cells_dirty"] = 3
    _expect(6, C.DirtyCode, lambda: render_final(data))


def test_refuses_dirty_code_files_of_static_study(data):
    data["e09"]["code"]["dirty_files"] = ["Curation_Paper/eswa_full/lib/rules_study.py"]
    _expect(6, C.DirtyCode, lambda: render_final(data))


def test_refuses_missing_cell_provenance(data):
    del data["e07"]["cells_dirty"]
    _expect(11, C.NoProvenance, lambda: render_final(data))
    data["e07"]["cells_dirty"] = 0
    del data["e09"]["code"]
    _expect(11, C.NoProvenance, lambda: render_final(data))


def test_refuses_unresolvable_key_path(data):
    macros = [R.Macro("EoneBogus", "e01", ("configs", "nope", "rows"), "pct", None)]
    labels = {"e01": "e01_selection.json"}
    _expect(9, C.KeyPathMissing, lambda: MN.evaluate(macros[0], data, labels))
    m = R.Macro("EoneBogus", "e01", ("configs", "unc_oracle_al", "rows", ("budget", 12345)), "pct", None)
    _expect(9, C.KeyPathMissing, lambda: MN.evaluate(m, data, labels))


def test_refuses_digit_or_duplicate_names():
    _expect(10, C.BadMacroName, lambda: MN.check_names([R.Macro("Bad6", "e01", ("a",), "int", None)]))
    ok = R.Macro("Same", "e01", ("a",), "int", None)
    _expect(10, C.BadMacroName, lambda: MN.check_names([ok, ok]))


def test_registry_names_are_unique_and_digit_free(data):
    macros = MN.expand(data)
    MN.check_names(macros)


def test_refuses_forbidden_output_dirs():
    for p in (REPO_ROOT / "Curation_Paper" / "latex" / "numbers.tex",
              REPO_ROOT / "Curation_Paper" / "latex" / "dce" / "x.tex",
              REPO_ROOT / "latex" / "DCE_submission" / "x.tex",
              REPO_ROOT / "latex" / "dce" / "x.tex"):
        _expect(4, C.ForbiddenOutput, lambda p=p: C.check_output_path(p, MN.FORBIDDEN_OUT))
    C.check_output_path(PKG_ROOT / "latex" / "numbers.tex", MN.FORBIDDEN_OUT)


def test_cli_final_refuses_absent_study_and_partial_lists_it(world, tmp_path, capsys):
    src = Path(world["cfg"]["results_dir"])
    dst = tmp_path / "res"
    shutil.copytree(src, dst)
    (dst / "e01_selection.json").unlink()
    out = tmp_path / "numbers.tex"
    assert MN.main(["--final", "--results-dir", str(dst), "--seed-list", "0,1", "--out", str(out)]) == 2
    assert not out.exists()
    assert "e01" in capsys.readouterr().err
    assert MN.main(["--partial", "--results-dir", str(dst), "--seed-list", "0,1", "--out", str(out)]) == 0
    head = out.read_text(encoding="utf-8")
    assert re.search(r"SKIPPED[^\n]*\be01\b", head) and "newcommand{\\EoneAccAlBsix}" not in head
    assert ("macros SKIPPED (their comparison partner study is absent): "
            "EfourTunedMinusDefaultBalAccBsix") in head


def test_cli_final_succeeds_on_complete_fixture_and_hash_refusal_writes_nothing(world, tmp_path):
    src = Path(world["cfg"]["results_dir"])
    dst = tmp_path / "res"
    shutil.copytree(src, dst)
    out = tmp_path / "numbers.tex"
    assert MN.main(["--final", "--results-dir", str(dst), "--seed-list", "0,1", "--out", str(out)]) == 0
    assert out.exists()
    out.unlink()
    d = json.loads((dst / "e08_thresholds.json").read_text())
    d["meta"]["input_hashes"]["rand"] = "f" * 64
    (dst / "e08_thresholds.json").write_text(json.dumps(d))
    assert MN.main(["--final", "--results-dir", str(dst), "--seed-list", "0,1", "--out", str(out)]) == 3
    assert not out.exists()


def test_cli_refuses_forbidden_out(tmp_path, monkeypatch):
    # a stand-in for the short paper's latex/ (a mutated predicate must not touch the real one)
    forbidden = tmp_path / "short_paper_latex"
    monkeypatch.setattr(MN, "FORBIDDEN_OUT", [forbidden])
    bad = forbidden / "sub" / "numbers.tex"
    assert MN.main(["--partial", "--out", str(bad)]) == 4
    assert not bad.exists()


def test_forbidden_out_constant_covers_the_short_paper_dirs():
    assert MN.FORBIDDEN_OUT == [REPO_ROOT / "Curation_Paper" / "latex", REPO_ROOT / "latex"]


def test_bare_nan_in_file_is_read(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": NaN}')
    assert math.isnan(MN.load_json(p)["a"])


def test_partial_on_real_results_only_uses_real_studies(tmp_path, monkeypatch, capsys):
    # The autouse fixture swaps F0.reference for the synthetic world's labels.  The real tree's
    # cells hold real signal ids, so this test must score against the real reference (this
    # mismatch was the KeyError on 'FOD3_50kHz_...' once the real e04/e05 cells existed).
    monkeypatch.setattr(F0, "reference", REAL_REFERENCE)
    assert F0.reference is REAL_REFERENCE
    out = tmp_path / "n.tex"
    rc = MN.main(["--partial", "--out", str(out)])
    # Stage 1 is complete under one loop key: the real tree must now generate cleanly.
    assert rc == 0, capsys.readouterr().err
    t = out.read_text(encoding="utf-8")
    assert "\\EtenNSignals}{12{,}504}" in t and "\\EnineTreeBadRecall}{71.0\\%}" in t
    assert "studies SKIPPED (JSON absent, --partial): none" in t
    # E1-E4 come from the Stage-2 *_s20 files when they exist (Stage 1 is the fallback) and the header says so
    results = Path(MN.__file__).resolve().parents[1] / "results"
    s2 = {"e01": "e01_selection_s20.json", "e02": "e02_reinspection_s20.json",
          "e03": "e03_noise_s20.json", "e04_compare": "e04_compare_s20.json"}
    present = {k: f for k, f in s2.items() if (results / f).is_file()}
    assert present, "expected the real tree to hold Stage-2 *_s20 files for E1-E4"
    for k, f in present.items():
        assert f"{k}=stage2 ({f}, 20 seeds)" in t, (k, f)
        assert f"{k}=stage1" not in t
    if "e01" in present:
        assert re.search(r"% e01_selection_s20\.json :: .* \| seeds: 20\n\\newcommand\{\\EoneAccAlBsix\}", t)


# ---------------------------------------------------------------- Task 16: Stage-2 (20-seed) files for E1-E4
S2_KEYS = ("e01", "e02", "e03", "e04_compare")
S2_FILES = {"e01": "e01_selection_s20.json", "e02": "e02_reinspection_s20.json",
            "e03": "e03_noise_s20.json", "e04_compare": "e04_compare_s20.json"}
CROSS = "EfourTunedMinusDefaultBalAccBsix"      # the only paired difference that spans two studies (E4 minus E1)


@pytest.fixture(scope="module")
def world2(tmp_path_factory):
    """build_all (Stage-1 files, 2 seeds) plus the Stage-2 E1-E4 files built from 20-seed cells through
    the real aggregation functions (study_fixtures.build_stage2)."""
    w = SF.build_all(tmp_path_factory.mktemp("numbers_world_s20"))
    w["cfg"]["seeds_stage1"] = [0, 1]
    w["stage2"] = SF.build_stage2(w)
    return w


@pytest.fixture(scope="module")
def loaded2(world2):
    mp = pytest.MonkeyPatch()
    mp.setattr(F0, "reference", lambda cfg: world2["ref"])
    try:
        d, absent = MN.load_studies(Path(world2["cfg"]["results_dir"]), world2["cfg"])
    finally:
        mp.undo()
    assert absent == []
    return d


@pytest.fixture
def data2(loaded2):
    return copy.deepcopy(loaded2)


def _comment_of(text, name):
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(f"\\newcommand{{\\{name}}}"))
    assert lines[i - 1].startswith("% ")
    return lines[i - 1]


def _json_dir(world, tmp_path, drop=()):
    """The world's top-level study JSONs without cells (the fold-0 studies are then absent: use --partial)."""
    src = Path(world["cfg"]["results_dir"])
    dst = tmp_path / "res"
    dst.mkdir()
    for p in src.glob("*.json"):
        if p.name not in drop:
            shutil.copy(p, dst / p.name)
    return dst


def test_stage2_files_are_preferred_for_e1_to_e4(data2):
    for k in S2_KEYS:
        assert data2[k]["_source"] == {"file": S2_FILES[k], "stage": "stage2"}, k
        assert data2[k]["n_seeds"] == 20 and C.stage_of(data2[k]) == "stage2"
    for k in ("e05", "e06", "e07", "e08", "e10_famb"):         # everything else stays Stage 1
        assert "_source" not in data2[k] and C.stage_of(data2[k]) == "stage1"
    assert data2["e05"]["n_seeds"] == 2
    assert data2["fold0_e04"]["stage"] == "stage2" and data2["fold0_e04"]["n_seeds"] == 20
    assert data2["fold0_e05"]["stage"] == "stage1" and data2["fold0_e05"]["n_seeds"] == 2


def test_stage2_macro_comments_name_the_s20_file_and_20_seeds(data2):
    text, counts = render_final(data2)
    for name, f in (("EoneAccAlBsix", "e01_selection_s20.json"), ("EtwoBalAccOracleEtaTenPctBsix", "e02_reinspection_s20.json"),
                    ("EthreeBalAccOracleEtaTenPctBsix", "e03_noise_s20.json"), ("EfourBalAccMicroBsix", "e04_compare_s20.json")):
        c = _comment_of(text, name)
        assert c.startswith(f"% {f} :: ") and c.endswith("| seeds: 20"), c
    c = _comment_of(text, CROSS)
    assert c.startswith("% e04_compare_s20.json :: ") and "(e01_selection_s20.json) | seeds: 20" in c, c
    c = _comment_of(text, "EoneNSeeds")
    assert c.startswith("% e01_selection_s20.json ::") and "\\newcommand{\\EoneNSeeds}{20}" in text
    assert "\\newcommand{\\EsevenNSeeds}{2}" in text                 # Stage-1 studies keep their own count
    assert _comment_of(text, "EfiveBalAccRankOneBsix").startswith("% e05_arch_loop.json :: ")
    assert _comment_of(text, "EfiveBalAccRankOneBsix").endswith("| seeds: 2")
    assert counts["fold0_e04"] == 8 * 6
    assert "e01=stage2 (e01_selection_s20.json, 20 seeds)" in text and "e04_compare=stage2" in text


def test_stage2_changes_no_macro_name(data, data2):
    n1 = [m.name for m in MN.expand(data)]
    n2 = [m.name for m in MN.expand(data2)]
    assert n1 == n2 and len(n1) > 500


def test_without_s20_files_stage1_is_used_and_the_comment_says_so(data):
    text, _ = render_final(data)
    c = _comment_of(text, "EoneAccAlBsix")
    assert c.startswith("% e01_selection.json :: ") and c.endswith("| seeds: 2"), c
    assert "e01=stage1 (e01_selection.json, 2 seeds)" in text and "e04_compare=stage1" in text
    assert data["fold0_e04"]["stage"] == "stage1" and data["fold0_e04"]["n_seeds"] == 2


def test_each_study_falls_back_independently(world2, tmp_path):
    dst = _json_dir(world2, tmp_path, drop=(S2_FILES["e02"],))
    cfg = dict(world2["cfg"])
    cfg["results_dir"] = str(dst)
    data, absent = MN.load_studies(dst, cfg)
    assert [C.stage_of(data[k]) for k in S2_KEYS] == ["stage2", "stage1", "stage2", "stage2"]
    MN.check_all(data, FROZEN)
    text, _ = MN.render(data, absent, "partial")
    assert _comment_of(text, "EtwoBalAccOracleEtaTenPctBsix").startswith("% e02_reinspection.json :: ")
    assert _comment_of(text, "EtwoBalAccOracleEtaTenPctBsix").endswith("| seeds: 2")
    assert _comment_of(text, "EthreeBalAccOracleEtaTenPctBsix").endswith("| seeds: 20")


def test_require_stage2_refuses_when_a_file_is_absent(world, world2, tmp_path):
    _expect(12, C.Stage2Missing, lambda: C.check_require_stage2({"e01": "stage2", "e02": None}))
    _expect(12, C.Stage2Missing, lambda: C.check_require_stage2({"e01": "stage2", "e02": "stage1"}))
    C.check_require_stage2({k: "stage2" for k in S2_KEYS})
    out = tmp_path / "numbers.tex"
    r1 = Path(world["cfg"]["results_dir"])                             # Stage 1 only: refuses, writes nothing
    assert MN.main(["--final", "--require-stage2", "--results-dir", str(r1), "--seed-list", "0,1",
                    "--out", str(out)]) == 12
    assert not out.exists()
    dst = _json_dir(world2, tmp_path, drop=(S2_FILES["e03"],))        # three of four: still refuses
    assert MN.main(["--final", "--require-stage2", "--results-dir", str(dst), "--seed-list", "0,1",
                    "--out", str(out)]) == 12
    assert not out.exists()
    r2 = Path(world2["cfg"]["results_dir"])                           # all four: passes, 20 seeds in the text
    assert MN.main(["--final", "--require-stage2", "--results-dir", str(r2), "--seed-list", "0,1",
                    "--out", str(out)]) == 0
    assert _comment_of(out.read_text(encoding="utf-8"), "EoneAccAlBsix").endswith("| seeds: 20")


def test_cli_prints_the_stage_each_study_came_from(world2, tmp_path, capsys):
    r2 = Path(world2["cfg"]["results_dir"])
    assert MN.main(["--final", "--results-dir", str(r2), "--seed-list", "0,1", "--out", str(tmp_path / "n.tex")]) == 0
    out = capsys.readouterr().out
    assert "stage sources:" in out and "e01=stage2 (e01_selection_s20.json, 20 seeds)" in out


def test_mixed_stage_cross_study_comparison_refuses(data, data2, world2, tmp_path):
    mixed = copy.deepcopy(data2)
    mixed["e01"] = copy.deepcopy(data["e01"])                          # E4 on Stage 2, E1 on Stage 1
    _expect(13, C.StageMismatch, lambda: render_final(mixed))
    mixed = copy.deepcopy(data2)
    mixed["e04_compare"] = copy.deepcopy(data["e04_compare"])          # E4 on Stage 1, E1 on Stage 2
    _expect(13, C.StageMismatch, lambda: render_final(mixed))
    # the same through the CLI: the E1 Stage-2 file is missing, the E4 one is there
    dst = _json_dir(world2, tmp_path, drop=(S2_FILES["e01"],))
    out = tmp_path / "numbers.tex"
    assert MN.main(["--partial", "--results-dir", str(dst), "--out", str(out)]) == 13
    assert not out.exists()


def test_same_stage_check_names_the_macro_and_both_stages():
    with pytest.raises(C.StageMismatch, match="EfourX.*stage2.*stage1"):
        C.check_same_stage("EfourX", "stage2", "stage1")
    C.check_same_stage("EfourX", "stage1", "stage1")


def test_cross_study_comparison_macros_are_exactly_one(data2):
    cross = [m.name for m in MN.expand(data2) if m.minus_study and m.minus_study != m.study]
    assert sorted(set(n[:-2] if n.endswith("Ci") else n for n in cross)) == [CROSS]


def test_stage2_file_must_be_a_stage2_file(world2, tmp_path):
    cfg = dict(world2["cfg"])
    for i, (field, value) in enumerate((("stage", "stage1"), ("seeds", [0, 1, 2, 3, 4]))):
        sub = tmp_path / f"v{i}"
        sub.mkdir()
        dst = _json_dir(world2, sub)
        p = dst / S2_FILES["e01"]
        d = json.loads(p.read_text())
        d[field] = value
        p.write_text(json.dumps(d))
        cfg["results_dir"] = str(dst)
        _expect(14, C.BadStage2File, lambda: MN.load_studies(dst, cfg))
    _expect(14, C.BadStage2File, lambda: C.check_stage2_file("x", {"stage": "stage2"}, cfg["seeds_stage2"]))


def test_fold0_e04_uses_the_stage2_seeds_only_when_the_s20_file_exists(world, world2, data2, data):
    assert data["fold0_e04"]["n_seeds"] == 2 and data2["fold0_e04"]["n_seeds"] == 20
    # the s20 file exists but its fold-0 cells do not: the fold-0 study is absent, never a 5-seed stand-in
    cfg = dict(world["cfg"])
    cfg["seeds_stage2"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    rdir = Path(world["cfg"]["results_dir"])
    s20 = rdir / S2_FILES["e04_compare"]
    assert not s20.exists()
    s20.write_text("{}")
    try:
        assert F0.load_e04(cfg, rdir) is None
    finally:
        s20.unlink()


def test_partial_treats_mixed_key_cell_study_as_absent_but_final_refuses(world, tmp_path):
    cfg = dict(world["cfg"])
    from Curation_Paper.eswa_full.lib import arch_study as A
    from Curation_Paper.eswa_full.lib.cells import cell_path
    groups = A.compare_specs(cfg, world["frozen"], seeds=[0, 1], models=("hgb",), budgets=cfg["budgets"])
    s0 = next(s for s in groups["hgb"] if s["fold"] == 0 and s["seed"] == 0)
    p = cell_path(s0, cfg)
    orig = p.read_text()
    try:
        d = json.loads(orig)
        d["meta"]["code_key"] = "z" * 64               # one fold-0 cell already on the new key
        p.write_text(json.dumps(d))
        rdir = Path(cfg["results_dir"])
        notes = []
        data, absent = MN.load_studies(rdir, cfg, partial=True, notes=notes)
        assert "fold0_e04" in absent and "fold0_e04" not in data
        assert any(n.startswith("fold0_e04: cells not yet consistent") for n in notes)
        assert "e04_frozen" in data                      # file studies are unaffected
        with pytest.raises(C.MixedCodeKeys):
            MN.load_studies(rdir, cfg)                   # not partial: refuse
    finally:
        p.write_text(orig)


# ---------------------------------------------------------------- fold-0 clean slice
def test_fold0_rescore_equals_direct_computation_on_fold_zero(world, data):
    cfg, ref, folds = world["cfg"], world["ref"], world["folds"]
    from Curation_Paper.eswa_full.lib import arch_study as A
    from Curation_Paper.eswa_full.lib.cells import load_cell
    import numpy as np
    from Curation_Paper.eswa_full.lib.metrics import metrics
    d = data["fold0_e04"]
    assert d["of"] == "e04" and d["n_seeds"] == 2 and d["fold"] == 0
    specs = A.compare_specs(cfg, world["frozen"], seeds=[0, 1], budgets=cfg["budgets"])["hgb"]
    accs = []
    for s in (x for x in specs if x["fold"] == 0):
        res = load_cell(s, cfg)["result"]
        rec = next(r for r in res["budgets"] if r["budget"] == 600)
        ids = folds[0]
        assert set(res["test_ids"]) == set(ids)             # fold-0 test set only
        accs.append(metrics(np.array([ref[i] for i in ids]),
                            np.array([rec["test_pred"][i] for i in ids]))["acc"])
    row = next(r for r in d["models"]["hgb"]["rows"] if r["budget"] == 600)
    assert row["metrics"]["acc"]["values"] == pytest.approx(accs)
    assert row["n_scored"] == len(folds[0]) < len(ref)
    text, counts = render_final(data)
    assert counts["fold0_e04"] == 8 * 6 and "EfourFoldZeroAccHgbBsix" in text


def test_fold0_refuses_mixed_keys_and_hash_mismatch(world, tmp_path):
    cfg = dict(world["cfg"])
    from Curation_Paper.eswa_full.lib import arch_study as A
    groups = A.compare_specs(cfg, world["frozen"], seeds=[0, 1], models=("hgb",), budgets=cfg["budgets"])
    s0 = next(s for s in groups["hgb"] if s["fold"] == 0 and s["seed"] == 0)
    from Curation_Paper.eswa_full.lib.cells import cell_path
    p = cell_path(s0, cfg)
    orig = p.read_text()
    try:
        d = json.loads(orig)
        d["meta"]["code_key"] = "z" * 64
        p.write_text(json.dumps(d))
        _expect(5, C.MixedCodeKeys, lambda: F0.fold0_study(cfg, groups, world["ref"], [600], prefix="X", of="e04"))
        d["meta"]["code_key"] = "a" * 64
        d["meta"]["input_hashes"] = {"al": "0" * 64, "rand": "1" * 64}
        p.write_text(json.dumps(d))
        _expect(3, C.InputHashMismatch, lambda: F0.fold0_study(cfg, groups, world["ref"], [600], prefix="X", of="e04"))
        d["meta"]["input_hashes"] = input_hashes(cfg)
        d["meta"]["dirty"] = True
        p.write_text(json.dumps(d))
        st = F0.fold0_study(cfg, groups, world["ref"], [600], prefix="X", of="e04")
        assert st["cells_dirty"] == 1                       # reported; refused by check_dirty
        _expect(6, C.DirtyCode, lambda: C.check_dirty("fold0_e04", st))
    finally:
        p.write_text(orig)


def test_fold0_loader_returns_none_while_cells_are_absent(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["results_dir"] = str(tmp_path)
    assert F0.load_e04(cfg, PKG_ROOT / "results") is None
    assert F0.load_e05(cfg, tmp_path) is None
