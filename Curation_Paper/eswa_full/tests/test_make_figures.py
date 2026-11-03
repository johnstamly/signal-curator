"""Task 14: figure generator on fixtures (all figures) and on the real JSONs that exist."""
import re
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Curation_Paper.eswa_full.lib import protocol, signals
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT
from Curation_Paper.eswa_full.scripts import make_figures as MF
from Curation_Paper.eswa_full.scripts import make_numbers as MN
from Curation_Paper.eswa_full.tests import study_fixtures as SF


def _assert_eswa_workflow(pdf):
    """R40a: the workflow figure carries the ESWA loop label, not the short paper's 'converges'."""
    from pypdf import PdfReader
    text = " ".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
    flat = " ".join(text.split())
    assert "proposed stopping signal" in flat, flat
    assert "converges" not in flat.lower(), flat


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Stage-1 fixture studies (2 seeds) plus the Stage-2 E1-E4 files (20 seeds, Task 16): every
    fixture figure below is drawn from the *_s20 files; the fallback tests copy the Stage-1 files only."""
    w = SF.build_all(tmp_path_factory.mktemp("figures_world"))
    w["stage2"] = SF.build_stage2(w)
    return w


def _fake_pools(cfg):
    al = {}
    for panel in ("FOD3", "FOD5", "FOD7"):
        for i, lab in enumerate((1, 0)):
            al[f"{panel}_50kHz_act1_recv{i + 1}_c200_r1"] = {"label_int": lab}
    return al, {}


def _fake_signals(ids, cfg, window=None):
    t = np.arange(500)
    return np.stack([np.sin(t / (5 + i)) * np.exp(-t / 200) for i, _ in enumerate(ids)])


def _studies():
    import Curation_Paper.eswa_full.lib.numbers_registry as R
    return [s for s in R.STUDIES if s.loader is None] + MF.AUX


def test_all_figures_drawn_from_fixtures_with_size_and_font_rules(world, tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "load_pools", _fake_pools)
    monkeypatch.setattr(signals, "signals_for", _fake_signals)
    verified = []
    monkeypatch.setattr(protocol, "verify_inputs", lambda cfg: verified.append(cfg) or {})
    data, absent = MN.load_studies(Path(world["cfg"]["results_dir"]), world["cfg"], studies=_studies())
    assert absent == []
    seen = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(self, *a, **k):
        w, h = self.get_size_inches()
        small = [t.get_text() for t in self.findobj(matplotlib.text.Text)
                 if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 7]
        seen[str(a[0])] = (round(w, 2), small)
        return real_savefig(self, *a, **k)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    status = MF.make_all(data, tmp_path)
    assert all(v == "drawn" for v in status.values()), status
    assert len(status) == len(MF.FIGS) == 13
    assert len(verified) == 1                  # the GOOD/BAD examples read the label DBs through verify_inputs
    for f in tmp_path.glob("*.pdf"):
        assert f.stat().st_size > 1000
    _assert_eswa_workflow(tmp_path / "fig_workflow.pdf")
    for name, (w, small) in seen.items():
        assert w in (3.5, 7.0), (name, w)
        assert small == [], (name, small)
    assert len([k for k in seen if k.endswith(".png")]) == 12      # every figure but the workflow (drawn in a subprocess)
    mb = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", (tmp_path / "fig_e1_curves.pdf").read_bytes())
    assert abs(float(mb[1]) - 504.0) < 1.0


def test_partial_lists_skipped_figures_and_final_refuses(world, tmp_path):
    src = Path(world["cfg"]["results_dir"])
    dst = tmp_path / "res"
    dst.mkdir()
    for name in ("e14_live_record", "e00_inputs"):
        shutil.copy(src / f"{name}.json", dst / f"{name}.json")
    out = tmp_path / "figs"
    assert MF.main(["--partial", "--results-dir", str(dst), "--out-dir", str(out), "--only",
                    "fig_campaign,fig_e1_curves"]) == 0
    assert (out / "fig_campaign.pdf").exists() and not (out / "fig_e1_curves.pdf").exists()
    assert MF.main(["--final", "--results-dir", str(dst), "--out-dir", str(out)]) == 2


def test_figures_refuse_forbidden_dir_and_hash_mismatch(world, tmp_path, monkeypatch):
    forbidden = tmp_path / "short_latex"
    monkeypatch.setattr(MN, "FORBIDDEN_OUT", [forbidden])
    assert MF.main(["--partial", "--out-dir", str(forbidden / "f")]) == 4
    assert not (forbidden / "f").exists()


def test_real_json_figures_campaign_e9_e10_e11(tmp_path):
    out = tmp_path / "figs"
    assert MF.main(["--partial", "--out-dir", str(out), "--only",
                    "fig_campaign,fig_e9_rules,fig_e10_coverage,fig_e11_downstream"]) == 0
    for n in ("fig_campaign", "fig_e9_rules", "fig_e10_coverage", "fig_e11_downstream"):
        assert (out / f"{n}.pdf").exists() and (out / f"{n}.png").exists()


def test_workflow_is_drawn_by_the_eswa_script_not_copied():
    assert MF.WORKFLOW_SCRIPT.is_file() and MF.WORKFLOW_SCRIPT.name == "fig_workflow_eswa.py"
    assert not hasattr(MF, "WORKFLOW_SRC")      # R40a: the short paper's file is never a source


def _fixture_data(world):
    data, absent = MN.load_studies(Path(world["cfg"]["results_dir"]), world["cfg"], studies=_studies())
    assert absent == []
    return data


def test_e1_main_panels_are_within_pool_and_the_boundary_is_marked(world):
    MF.style()
    data = _fixture_data(world)
    fig = MF.fig_e1(data, None)
    axs = fig.axes[:3]
    labels = [[ln.get_label() for ln in ax.lines] for ax in axs]
    assert not any("rand}" in lab for lab in labels[0] + labels[1])        # no L_rand curve in (a), (b)
    assert any("rand}" in lab for lab in labels[2])                         # it lives in (c) ...
    assert "labelling-standard" in axs[2].get_title(loc="left")             # ... titled as a standard difference
    assert "random selection" not in axs[2].get_title(loc="left")
    limit = MF.SELECTION_LIMIT * min(r["min_train_pool"] for c in data["e01"]["configs"].values()
                                     for r in c["rows"])
    for ax in axs[:2]:
        assert any(len(set(ln.get_xdata())) == 1 and ln.get_xdata()[0] == limit for ln in ax.lines), \
            "claim-boundary line at 0.75 x training pool"
    plt.close(fig)


def test_e1_hollow_markers_follow_selection_claim_ok_for_every_within_pool_series(world):
    MF.style()
    data = _fixture_data(world)
    fig = MF.fig_e1(data, None)
    n_bad = {n: sum(not r["selection_claim_ok"] for r in c["rows"]) for n, c in data["e01"]["configs"].items()}
    assert n_bad["unc_oracle_al"] > 0 and n_bad["random_al"] > 0
    hollow = [ln for ln in fig.axes[0].lines if ln.get_markerfacecolor() == "white"]
    assert len(hollow) == 2 and all(len(ln.get_xdata()) == n_bad["unc_oracle_al"] for ln in hollow)
    plt.close(fig)


def test_place_labels_never_overlap_each_other_or_markers():
    MF.style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    pts = [(17.7, 87.9), (23.2, 88.6), (23.8, 86.0), (45.0, 88.3), (52.0, 85.1), (1.4, 91.0)]
    names = ["BiGRU", "CNN-wide", "Transformer", "ResNet1D", "HGB", "MiniRocket"]
    ax.set_xscale("log")
    ax.plot(*zip(*pts), ls="none", marker="o", markersize=6)
    ax.set_xlim(0.5, 150)
    ax.set_ylim(80, 95)
    fig.tight_layout()
    boxes = MF.place_labels(ax, pts, names)
    assert len(boxes) == len(pts)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert MF._overlap(boxes[i], boxes[j], pad=0.0) == 0.0, (names[i], names[j])
    for p in pts:
        x, y = ax.transData.transform(p)
        marker = matplotlib.transforms.Bbox([[x - 3, y - 3], [x + 3, y + 3]])
        assert all(MF._overlap(b, marker, pad=0.0) == 0.0 for b in boxes), p
    plt.close(fig)


def test_end_labels_are_separated_and_inside_the_axes():
    MF.style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    items = [(9.0, 9.9, "a", "k"), (9.0, 9.95, "b", "k"), (9.0, 5.0, "c", "k")]
    MF.end_labels(ax, items)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    anns = [t for t in ax.texts]
    boxes = [matplotlib.text.Text.get_window_extent(t, r) for t in anns]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert MF._overlap(boxes[i], boxes[j], pad=0.0) == 0.0
    top = ax.get_window_extent().y1
    assert all(b.y1 <= top + 1 for b in boxes)
    plt.close(fig)


def test_famb_figure_names_its_population(world):
    MF.style()
    fig = MF.fig_e10_famb(_fixture_data(world), None)
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "unlabelled rows" in text and "12,504" in text
    plt.close(fig)


def test_real_results_draw_every_figure_final(tmp_path):
    """Stage 1 is complete: --final on the real results/ draws all 13 figures (real label DBs and
    real signals for the GOOD/BAD examples)."""
    out = tmp_path / "figs"
    assert MF.main(["--final", "--out-dir", str(out)]) == 0
    assert len(list(out.glob("*.pdf"))) == len(MF.FIGS) == 13
    assert len(list(out.glob("*.png"))) == 13      # the ESWA workflow script writes its own png too
    _assert_eswa_workflow(out / "fig_workflow.pdf")


# ---------------------------------------------------------------- Task 16: Stage-2 preference
S2_FILES = ("e01_selection_s20.json", "e02_reinspection_s20.json", "e03_noise_s20.json", "e04_compare_s20.json")


def _stage1_only_dir(world, tmp_path):
    """Top-level JSONs of the fixture world without the *_s20 files (no cells)."""
    src = Path(world["cfg"]["results_dir"])
    dst = tmp_path / "res_s1"
    dst.mkdir()
    for p in src.glob("*.json"):
        if p.name not in S2_FILES:
            shutil.copy(p, dst / p.name)
    return dst


def _fig_text(fig):
    return " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))


def test_figures_prefer_stage2_files_and_footnote_reads_the_seed_count(world, tmp_path):
    MF.style()
    data = _fixture_data(world)
    for k, fn in (("e01", S2_FILES[0]), ("e02", S2_FILES[1]), ("e03", S2_FILES[2]), ("e04_compare", S2_FILES[3])):
        assert data[k]["_source"] == {"file": fn, "stage": "stage2"} and data[k]["n_seeds"] == 20
    fig = MF.fig_e1(data, None)
    assert "mean and 95% CI over 20 seeds" in _fig_text(fig)
    plt.close(fig)
    for fn in (MF.fig_e23, MF.fig_e4):                       # drawn from 20-seed data without error
        plt.close(fn(data, None))


def test_figures_fall_back_to_stage1_and_footnote_follows(world, tmp_path):
    MF.style()
    dst = _stage1_only_dir(world, tmp_path)
    data, absent = MN.load_studies(dst, world["cfg"], studies=_studies())
    assert absent == []
    assert all(data[k]["_source"]["stage"] == "stage1" for k in ("e01", "e02", "e03", "e04_compare"))
    fig = MF.fig_e1(data, None)
    assert "mean and 95% CI over 2 seeds" in _fig_text(fig)
    plt.close(fig)


def test_figures_require_stage2_refuses_when_a_file_is_absent(world, tmp_path):
    dst = _stage1_only_dir(world, tmp_path)
    out = tmp_path / "figs"
    assert MF.main(["--final", "--require-stage2", "--results-dir", str(dst), "--out-dir", str(out)]) == 12
    assert not out.exists()
    full = Path(world["cfg"]["results_dir"])          # all four present: the flag lets the run through
    assert MF.main(["--partial", "--require-stage2", "--results-dir", str(full), "--out-dir", str(out),
                    "--only", "fig_e1_curves"]) == 0
    assert (out / "fig_e1_curves.pdf").exists()


# ---------------------------------------------------------------- fig_e23 draws E2 and E3 from one stage
def _mixed_stage_dir(world, tmp_path):
    """Stage-1 files everywhere, except E2, which keeps its 20-seed file: E2 (stage 2) next to E3 (stage 1)."""
    dst = _stage1_only_dir(world, tmp_path)
    shutil.copy(Path(world["cfg"]["results_dir"]) / "e02_reinspection_s20.json", dst / "e02_reinspection_s20.json")
    return dst


def test_fig_e23_refuses_when_e2_and_e3_come_from_different_stages(world, tmp_path):
    MF.style()
    dst = _mixed_stage_dir(world, tmp_path)
    data, absent = MN.load_studies(dst, world["cfg"], studies=_studies())
    assert absent == []
    assert data["e02"]["_source"]["stage"] == "stage2" and data["e03"]["_source"]["stage"] == "stage1"
    with pytest.raises(MF.C.StageMismatch):
        MF.fig_e23(data, None)
    # the other way round is refused as well
    data2 = _fixture_data(world)
    data2["e03"] = data["e03"]
    with pytest.raises(MF.C.StageMismatch):
        MF.fig_e23(data2, None)
    # same stage on both sides draws
    plt.close(MF.fig_e23(_fixture_data(world), None))


def test_figures_main_returns_the_refusal_code_for_a_mixed_stage_e23(world, tmp_path):
    dst = _mixed_stage_dir(world, tmp_path)
    out = tmp_path / "figs_mixed"
    rc = MF.main(["--partial", "--results-dir", str(dst), "--out-dir", str(out), "--only", "fig_e23_ablations"])
    assert rc == MF.C.StageMismatch.code
    assert not (out / "fig_e23_ablations.pdf").exists()
