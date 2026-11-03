"""Figure generator (Task 14, ruling R28; final pass on the real Stage-1 results).  Figures are drawn
only from study JSONs (and, for the GOOD/BAD examples, the read-only signal store).  From the repo root:

    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/make_figures.py --partial
    conda run --no-capture-output -n AI python Curation_Paper/eswa_full/scripts/make_figures.py --final

--partial skips a figure whose study JSON is absent (and lists it); --final refuses (exit 2).
Output: Curation_Paper/eswa_full/figures/<name>.pdf plus <name>.png preview (never the short
paper's latex/).  Style follows the dataviz skill: validated categorical palette (slots 1-3 validate
all-pairs, slots 1-4 validate on the adjacent pairlist used for lines; scripts/validate_palette.js,
light mode), Times, widths 3.5 / 7.0 in, text >= 7 pt, thin marks, a legend for every figure with
two or more series, secondary encoding (marker / line style) so identity is never colour alone,
recessive grid, no dual axes, CIs wherever the study JSON stores them.

E1 (R36): the main panels are the WITHIN-POOL comparison on L_AL (uncertainty + oracle vs random,
same labelling session).  Random order on L_rand is a separate panel titled as a labelling-standard
difference, never as "random selection".
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from Curation_Paper.eswa_full.lib import numbers_core as C  # noqa: E402
from Curation_Paper.eswa_full.lib import numbers_registry as R  # noqa: E402
from Curation_Paper.eswa_full.lib.protocol import PKG_ROOT, input_hashes, load_config  # noqa: E402
from Curation_Paper.eswa_full.scripts import make_numbers as MN  # noqa: E402

FIG_DIR = PKG_ROOT / "figures"
W1, W2 = 3.5, 7.0
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # palette slots 1-4 (light)
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e5e4e0", "#8d8c86"
WORKFLOW_SCRIPT = REPO / "Curation_Paper" / "eswa_full" / "scripts" / "fig_workflow_eswa.py"   # R40a
SELECTION_LIMIT = 0.75        # protocol: a selection effect is claimed only while budget <= 0.75 x training pool
ARCH_NAME = {"microconv1d": "MicroConv1D", "microconv1d_baseline": "MicroConv1D", "cnn_wide": "CNN-wide",
             "resnet1d": "ResNet1D", "bigru": "BiGRU", "transformer": "Transformer", "hgb": "HGB",
             "logreg": "LogReg", "minirocket": "MiniRocket"}

AUX = [R.Study("e10_scores", "e10_scores.json", "", "static")]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


# ------------------------------------------------------------------ helpers
def _row(rows, budget):
    return next(r for r in rows if r["budget"] == budget)


def _panel_letter(ax, s):
    """Panel letter; on an axes that already has a left title it becomes the title's first word."""
    t = ax.get_title(loc="left")
    if t:
        ax.set_title(f"{s} {t}", loc="left", color=INK2, fontsize=7)
    else:
        ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _x_to_data(ax, ticks):
    """Ticks and the bottom spine only where there is data (no axis beyond the last budget/threshold)."""
    ax.set_xticks(list(ticks))
    ax.spines["bottom"].set_bounds(min(ticks), max(ticks))


def _band(ax, rows, key, color, label, marker, ls="-", scale=100.0, hollow=None, lw=1.2):
    """Mean line with its 95 % CI band; ``hollow`` = predicate on a row -> white-filled marker."""
    x = [r["budget"] for r in rows]
    m = np.array([r["metrics"][key]["mean"] for r in rows]) * scale
    lo = np.array([r["metrics"][key]["ci_lo"] for r in rows]) * scale
    hi = np.array([r["metrics"][key]["ci_hi"] for r in rows]) * scale
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=color, marker=marker, ls=ls, label=label, markeredgecolor="white",
            markeredgewidth=0.6, linewidth=lw)
    if hollow is not None:
        sel = [i for i, r in enumerate(rows) if hollow(r)]
        if sel:
            ax.plot([x[i] for i in sel], [m[i] for i in sel], ls="none", marker=marker, markersize=5.5,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0)
    return x, m, lo, hi


def _fit_ylim(ax, lows, highs, pad=0.07, top=None, bottom=None):
    lo, hi = float(np.nanmin(lows)), float(np.nanmax(highs))
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - pad * span if bottom is None else bottom, hi + pad * span if top is None else top)


def legend_below(ax, ncol=1):
    """Legend under the x label (never over the data).  Call after the layout is final."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.xaxis.label.get_window_extent()
    y = ax.transAxes.inverted().transform((0, bb.y0))[1] - 0.04
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
              handlelength=2.2, columnspacing=1.4, borderaxespad=0.0)


def finish(fig, bottom_note=None, legends=()):
    """tight_layout, then outside legends, then tight_layout again (legends join the layout)."""
    rect = (0, 0.045 if bottom_note else 0, 1, 1)
    fig.tight_layout(rect=rect)
    for ax, ncol in legends:
        legend_below(ax, ncol)
    fig.tight_layout(rect=rect)
    if bottom_note:
        fig.text(0.5, 0.008, bottom_note, ha="center", va="bottom", fontsize=7, color=INK2)


def _overlap(a: Bbox, b: Bbox, pad=1.5) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0) + pad
    h = min(a.y1, b.y1) - max(a.y0, b.y0) + pad
    return w * h if (w > 0 and h > 0) else 0.0


def place_labels(ax, pts, labels, color=INK2, fontsize=7, avoid=(), marker_pts=6.0, arrow_from=9.0):
    """Label points so that no label overlaps another label, a marker, an ``avoid`` box (display
    coordinates, e.g. error bars) or leaves the axes.  Greedy over the points in x order, each trying
    eight directions at growing radii; the first free spot wins, else the least-overlapping one.
    Deterministic.  Returns the display-space boxes of the placed labels."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72.0
    axbox = ax.get_window_extent()
    xy = [ax.transData.transform(p) for p in pts]
    mk = marker_pts * scale / 2.0
    taken = [Bbox([[x - mk, y - mk], [x + mk, y + mk]]) for x, y in xy] + list(avoid)
    pref = (45, 315, 135, 225, 90, 270, 0, 180, 22.5, 337.5, 157.5, 202.5, 67.5, 292.5, 112.5, 247.5)
    dirs = []
    for a in pref:
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        dirs.append((c, s, "left" if c > 0.25 else ("right" if c < -0.25 else "center"),
                     "bottom" if s > 0.25 else ("top" if s < -0.25 else "center")))
    placed = []
    for i in sorted(range(len(pts)), key=lambda k: (xy[k][0], xy[k][1])):
        best = None
        for r in (4, 9, 16, 24, 34, 46):
            for dx, dy, ha, va in dirs:
                ann = ax.annotate(labels[i], pts[i], xytext=(dx * r, dy * r), textcoords="offset points",
                                  ha=ha, va=va, fontsize=fontsize, color=color,
                                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2)
                                  if r > arrow_from else None, annotation_clip=False)
                ann.update_positions(renderer)
                box = mtext.Text.get_window_extent(ann, renderer)
                cost = sum(_overlap(box, t) for t in taken)
                if box.x0 < axbox.x0 or box.x1 > axbox.x1 or box.y0 < axbox.y0 or box.y1 > axbox.y1:
                    cost += 1e4
                cen = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
                own = np.linalg.norm(cen - xy[i])
                if r > arrow_from:                     # a leader line must not run over another marker or bar
                    hits = 0
                    for tt in np.linspace(0.15, 1.0, 12):
                        px = xy[i] + tt * (cen - xy[i])
                        hits += any(b.x0 <= px[0] <= b.x1 and b.y0 <= px[1] <= b.y1
                                    for b in taken[:len(pts)] + list(avoid)
                                    if b is not taken[i])
                    cost += 60.0 * hits
                if r <= arrow_from and any(np.linalg.norm(cen - xy[j]) < own
                                           for j in range(len(pts)) if j != i):
                    cost += 300.0          # adjacent to another point's marker and no leader line: ambiguous
                if best is None or cost < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (cost, ann, box)
                else:
                    ann.remove()
                if best[0] == 0:
                    break
            if best[0] == 0:
                break
        taken.append(best[2])
        placed.append(best[2])
    return placed


def end_labels(ax, items, fontsize=7, pad_pts=4.0):
    """Direct labels at the right end of each line: items = [(x, y, text, color)], pushed apart
    vertically so that they never overlap each other."""
    fig = ax.figure
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    gap = fontsize * 1.35 * scale
    ys = [ax.transData.transform((x, y))[1] for x, y, _, _ in items]
    order = sorted(range(len(items)), key=lambda k: ys[k])
    adj = list(ys)
    for a, b in zip(order, order[1:]):
        if adj[b] - adj[a] < gap:
            adj[b] = adj[a] + gap
    mid_shift = (np.mean(ys) - np.mean(adj))
    ab = ax.get_window_extent()                          # keep every label inside the axes
    lo_adj, hi_adj = min(adj) + mid_shift, max(adj) + mid_shift
    if hi_adj > ab.y1 - gap / 2:
        mid_shift -= hi_adj - (ab.y1 - gap / 2)
    elif lo_adj < ab.y0 + gap / 2:
        mid_shift += (ab.y0 + gap / 2) - lo_adj
    for k in range(len(items)):
        x, y, text, col = items[k]
        ax.annotate(text, (x, y), xytext=(pad_pts, (adj[k] + mid_shift - ys[k]) / scale),
                    textcoords="offset points", ha="left", va="center", fontsize=fontsize, color=col,
                    annotation_clip=False)


# ------------------------------------------------------------------ figure functions
def fig_campaign(d, out):
    e = d["e14"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    days = sorted(set(e["labels_per_day"]["L_AL"]) | set(e["labels_per_day"]["L_rand"]))
    x = np.arange(len(days))
    for off, pool, col, lab in ((-0.19, "L_AL", BLUE, "$L_{AL}$"), (0.19, "L_rand", ORANGE, "$L_{rand}$")):
        v = [e["labels_per_day"][pool].get(dy, 0) for dy in days]
        bars = ax.bar(x + off, v, 0.36, color=col, label=lab, linewidth=0)
        if pool == "L_rand":  # the first-day bar is the seed copied from the L_AL session
            k = days.index(e["seed_and_added"]["first_day"])
            bars[k].set_hatch("////")
            bars[k].set_edgecolor("white")
            bars[k].set_linewidth(0)
            ax.annotate("copied seed\n(labelled in the $L_{AL}$ session)",
                        xy=(x[k] + off + 0.18, v[k] * 0.55), xytext=(x[k] + 0.75, v[k] * 0.55),
                        fontsize=7, color=INK2, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6))
        for xi, vi in zip(x + off, v):
            if vi:
                ax.text(xi, vi + 12, str(vi), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([dy[5:] for dy in days])
    ax.set_xlabel("labelling day (2026, MM-DD)")
    ax.set_ylabel("labels stamped")
    ax.set_ylim(0, 860)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _panel_letter(ax, "(a)")
    ax = axs[1]
    it = e["famb_trajectory"]["iterations"]
    n = [i["n_labels"] for i in it]
    f = [100 * i["f_amb_pool_12504"] for i in it]
    ax.plot(n, f, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("labels available at the iteration")
    ax.set_ylabel("live $f_{amb}$ on the 12,504-signal universe (%)")
    ax.set_ylim(0, max(f) * 1.35)
    ax.set_xlim(min(n) - 40, max(n) + 60)
    _panel_letter(ax, "(b)")
    ax.text(0.98, 0.04, "reconstructed afterwards,\nnot logged live", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=INK2)
    fig.tight_layout()
    # iteration labels go on the side away from the line (vector from the point to its neighbours)
    fig.canvas.draw()
    disp = np.array([ax.transData.transform(p) for p in zip(n, f)])
    for k, (ni, fi, i) in enumerate(zip(n, f, it)):
        nb = [disp[j] for j in (k - 1, k + 1) if 0 <= j < len(disp)]
        v = np.mean(nb, axis=0) - disp[k]
        u = -v / (np.linalg.norm(v) or 1.0)
        ax.annotate(f"iter {i['iteration']}", (ni, fi), xytext=(10 * u[0], 10 * u[1]),
                    textcoords="offset points", ha="center", va="center", fontsize=7, color=INK2)
    return fig


def fig_good_bad_examples(d, out, pools=None, signals=None):
    """GOOD vs BAD examples (the label stores hold no fault-class field: label in {GOOD, BAD}).
    Written to fig_fault_gallery.pdf because the manuscript includes that file name."""
    from Curation_Paper.eswa_full.lib import protocol as P
    from Curation_Paper.eswa_full.lib import signals as S
    cfg = load_config()
    if pools is None:
        P.verify_inputs(cfg)          # frozen hashes, counts and HDF5 resolvability before any read
        al = P.load_pools(cfg)[0]
    else:
        al = pools
    get = signals if signals is not None else (lambda ids: S.signals_for(ids, cfg))
    pick = {}
    for cls, lab in (("GOOD", 1), ("BAD", 0)):
        ids = sorted(s for s, r in al.items() if r["label_int"] == lab)
        chosen, seen = [], set()
        for s in ids:
            p = S.parse_id(s)["panel"]
            if p not in seen and S.parse_id(s)["cycle"] >= 200:
                seen.add(p)
                chosen.append(s)
            if len(chosen) == 3:
                break
        pick[cls] = chosen
    # The last BAD example is the BAD signal with the highest raw peak (co-author review: show a
    # signal that "went really high"). Each panel is divided by its own peak, so the peak is
    # printed as a multiple of the median GOOD peak to keep the height visible.
    all_ids = sorted(al)
    peaks = {s: float(np.abs(np.asarray(x, dtype=float)).max()) for s, x in zip(all_ids, get(all_ids))}
    good_med = float(np.median([peaks[s] for s in all_ids if al[s]["label_int"] == 1]))
    top_bad = max((s for s in all_ids if al[s]["label_int"] == 0 and S.parse_id(s)["cycle"] >= 200),
                  key=peaks.get)
    if top_bad not in pick["BAD"]:
        pick["BAD"][-1] = top_bad
    fig, axs = plt.subplots(2, 3, figsize=(W2, 3.4), sharex=True, sharey=True)
    for r, (cls, col) in enumerate((("GOOD", BLUE), ("BAD", ORANGE))):
        X = get(pick[cls])
        for c, (sid, x) in enumerate(zip(pick[cls], X)):
            ax = axs[r, c]
            x = np.asarray(x, dtype=float)
            ax.plot(np.arange(len(x)), x / (np.abs(x).max() or 1.0), color=col, linewidth=0.8)
            p = S.parse_id(sid)
            ax.set_title(f"{cls} example: {p['panel']} act{p['actuator']}/recv{p['receiver']}, cycle {p['cycle']}"
                         f"\npeak = {peaks[sid] / good_med:.1f}× median GOOD peak",
                         fontsize=7, loc="left")
            ax.set_ylim(-1.1, 1.1)
            if r == 1:
                ax.set_xlabel("sample index (500-sample window)")
            if c == 0:
                ax.set_ylabel("amplitude / peak")
    fig.tight_layout()
    return fig


def fig_e1(d, out):
    e = d["e01"]
    cf = e["configs"]
    limit = SELECTION_LIMIT * min(r["min_train_pool"] for c in cf.values() for r in c["rows"])
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.9))
    within = (("unc_oracle_al", BLUE, "o", "-", "uncertainty + oracle"),
              ("random_al", ORANGE, "s", "--", "random order"))
    not_ok = lambda r: not r["selection_claim_ok"]            # noqa: E731
    xmax = max(r["budget"] for c in cf.values() for r in c["rows"])
    for ax, key, yl, letter in ((axs[0], "bal_acc", "balanced accuracy (%)", "(a)"),
                                (axs[1], "bad_recall", "BAD recall (%)", "(b)")):
        lows, highs = [], []
        for name, col, mk, ls, lab in within:
            if name in cf:
                _, _, lo, hi = _band(ax, cf[name]["rows"], key, col, lab, mk, ls, hollow=not_ok)
                lows.append(lo)
                highs.append(hi)
        _fit_ylim(ax, np.concatenate(lows), np.concatenate(highs), top=None)
        yl0, yl1 = ax.get_ylim()
        ax.axvspan(limit, xmax + 15, color=MUTED, alpha=0.18, linewidth=0, zorder=0)
        ax.axvline(limit, color=INK2, linewidth=0.6, ls=":")
        ax.text((limit + xmax + 15) / 2, yl0 + 0.04 * (yl1 - yl0), "no selection claim", fontsize=7,
                color=INK2, ha="center", va="bottom", rotation=90)
        ax.set_xlim(85, xmax + 15)
        _x_to_data(ax, [r["budget"] for r in cf["unc_oracle_al"]["rows"]])
        ax.set_xlabel("labelling budget")
        ax.set_ylabel(yl)
        ax.set_title("within $L_{AL}$ (one labelling session)" if letter == "(a)" else "within $L_{AL}$",
                     loc="left", color=INK2)
        _panel_letter(ax, letter)
    ax = axs[2]
    lows, highs = [], []
    for name, col, mk, ls, lab in (("random_al", ORANGE, "s", "--", "random order, $L_{AL}$ labels"),
                                   ("random_rand", AQUA, "^", ":", "random order, $L_{rand}$ labels")):
        if name in cf:
            _, _, lo, hi = _band(ax, cf[name]["rows"], "acc", col, lab, mk, ls)
            lows.append(lo)
            highs.append(hi)
    _fit_ylim(ax, np.concatenate(lows), np.concatenate(highs))
    ax.set_xlim(85, xmax + 15)
    _x_to_data(ax, [r["budget"] for r in cf["random_al"]["rows"]])
    ax.set_xlabel("labelling budget")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("labelling-standard difference\n(same order, two sessions' labels)", loc="left", color=INK2)
    _panel_letter(ax, "(c)")
    finish(fig, f"mean and 95% CI over {e['n_seeds']} seeds. Hollow markers and grey band: budget > "
                f"{SELECTION_LIMIT:g} x training pool, no selection effect claimed.",
           legends=[(axs[0], 1), (axs[1], 1), (axs[2], 1)])
    return fig


def fig_e23(d, out):
    e2, e3 = d["e02"], d["e03"]
    # Panel (a) is E3 and panel (b) is E2 in one figure: a 20-seed file next to a 5-seed file would put two
    # different seed counts under one footnote and one set of interval bars.
    C.check_same_stage("fig_e23_ablations", C.stage_of(e2), C.stage_of(e3))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
    ax = axs[0]
    etas = sorted({c["eta"] for c in e3["cells"].values()})
    lows, highs = [], []
    for strat, col, mk, ls, lab, off in (("uncertainty", ORANGE, "s", "--", "uncertainty, no re-inspection", -0.35),
                                         ("unc_oracle", BLUE, "o", "-", "uncertainty + oracle re-inspection", 0.35)):
        xs, ys, lo, hi = [], [], [], []
        for eta in etas:
            k = f"{strat}|eta={eta:g}"
            if k not in e3["cells"]:
                continue
            r = _row(e3["cells"][k]["rows"], 600)["metrics"]["bal_acc"]
            xs.append(100 * eta + off)
            ys.append(100 * r["mean"])
            lo.append(100 * r["ci_lo"])
            hi.append(100 * r["ci_hi"])
        ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], color=col, marker=mk, ls=ls, capsize=2,
                    label=lab, markeredgecolor="white", markeredgewidth=0.6, linewidth=1.0, elinewidth=0.8)
        lows += lo
        highs += hi
    _fit_ylim(ax, lows, highs)
    ax.set_xticks([100 * x for x in etas])
    ax.set_xticklabels([f"{100 * x:g}" for x in etas])
    ax.set_xlabel("label noise $\\eta$ (% of the labels flipped)")
    ax.set_ylabel("balanced accuracy at budget 600 (%)")
    ax.set_title("E3: noise levels", loc="left", color=INK2)
    _panel_letter(ax, "(a)")
    ax = axs[1]
    eta = 0.1
    lows, highs = [], []
    ends = []
    for strat, col, mk, ls, lab in (("uncertainty", ORANGE, "s", "--", "no re-inspection"),
                                    ("unc_oracle", BLUE, "o", "-", "oracle re-inspection"),
                                    ("unc_selftrain", AQUA, "^", ":", "self-trained flags")):
        k = f"{strat}|eta={eta:g}"
        if k in e2["cells"]:
            x, m, lo, hi = _band(ax, e2["cells"][k]["rows"], "bal_acc", col, lab, mk, ls)
            lows.append(lo)
            highs.append(hi)
            ends.append((x[-1], m[-1], lab, col))
    _fit_ylim(ax, np.concatenate(lows), np.concatenate(highs))
    ax.set_xlim(85, 600 + 270)
    _x_to_data(ax, x)
    ax.set_xlabel("labelling budget")
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_title(f"E2: trajectories at $\\eta$ = {100 * eta:g}%", loc="left", color=INK2)
    _panel_letter(ax, "(b)")
    finish(fig, legends=[(axs[0], 1), (axs[1], 1)])
    end_labels(axs[1], ends)
    return fig


def fig_e4(d, out):
    e = d["e04_compare"]
    pts = []
    for m, mv in e["models"].items():
        r = _row(mv["rows"], 600)["metrics"]["bal_acc"]
        pts.append(dict(m=m, t=mv["train_seconds_per_fit_mean"], size=mv["size"], unit=mv["size_unit"],
                        y=100 * r["mean"], lo=100 * r["ci_lo"], hi=100 * r["ci_hi"], cal=mv["calibrated"]))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0))
    for ax, xk, panel in ((axs[0], "t", "(a)"), (axs[1], "size", "(b)")):
        use = pts if xk == "t" else [p for p in pts if p["unit"] == "parameters"]
        for p in use:
            col = BLUE if p["m"] == "microconv1d" else MUTED
            ax.errorbar([p[xk]], [p["y"]], yerr=[[p["y"] - p["lo"]], [p["hi"] - p["y"]]], color=col, capsize=2,
                        linewidth=0.8, ls="none", zorder=2)
            ax.plot([p[xk]], [p["y"]], marker="o" if p["cal"] else "D", color=col, markeredgecolor="white",
                    markeredgewidth=0.6, markersize=6, ls="none", zorder=3)
        ax.set_xscale("log")
        lows, highs = [p["lo"] for p in use], [p["hi"] for p in use]
        _fit_ylim(ax, lows, highs, pad=0.12)
        if xk == "t":
            ax.set_xlim(min(p["t"] for p in use) / 2.2, max(p["t"] for p in use) * 3.2)
            ax.set_xlabel("mean training time per fit (s, log scale)")
            ax.set_title("all eight models", loc="left", color=INK2)
        else:
            ax.set_xlim(min(p["size"] for p in use) / 2.2, max(p["size"] for p in use) * 3.2)
            ax.set_xlabel("trainable parameters (log scale)")
            ax.set_title("neural networks only (parameters)", loc="left", color=INK2)
        ax.set_ylabel("balanced accuracy at budget 600 (%)")
        _panel_letter(ax, panel)
    axs[0].plot([], [], marker="o", color=MUTED, ls="none", label="calibrated probabilities")
    axs[0].plot([], [], marker="D", color=MUTED, ls="none", label="uncalibrated score (MiniRocket)")
    axs[0].plot([], [], marker="o", color=BLUE, ls="none", label="MicroConv1D (deployed)")
    finish(fig, legends=[(axs[0], 1)])
    # labels last: the layout is final, so the display positions used for the overlap test are final
    for ax, xk in ((axs[0], "t"), (axs[1], "size")):
        use = pts if xk == "t" else [p for p in pts if p["unit"] == "parameters"]
        avoid = []           # error bars are obstacles for the labels
        for p in use:
            (x0, y0), (x1, y1) = ax.transData.transform((p[xk], p["lo"])), ax.transData.transform((p[xk], p["hi"]))
            avoid.append(Bbox([[x0 - 3, y0], [x1 + 3, y1]]))
        place_labels(ax, [(p[xk], p["y"]) for p in use], [ARCH_NAME.get(p["m"], p["m"]) for p in use],
                     avoid=avoid)
    return fig


def fig_e5(d, out):
    e = d["e05"]
    series = []
    if "microconv1d_baseline" in e:
        series.append(("microconv1d_baseline", e["microconv1d_baseline"]["rows"], BLUE, "o", "-", 1.6))
    for sel, (col, mk, ls) in zip(e["selected"], ((ORANGE, "s", "--"), (AQUA, "^", ":"), (YELLOW, "D", "-."))):
        series.append((sel["arch"], e["models"][sel["arch"]]["rows"], col, mk, ls, 1.2))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
    for ax, key, yl, letter in ((axs[0], "bal_acc", "balanced accuracy (%)", "(a)"),
                                (axs[1], "auroc", "AUROC (%)", "(b)")):
        lows, highs, ends = [], [], []
        for name, rows, col, mk, ls, lw in series:
            x, m, lo, hi = _band(ax, rows, key, col, ARCH_NAME.get(name, name), mk, ls, lw=lw)
            lows.append(lo)
            highs.append(hi)
            ends.append((x[-1], m[-1], ARCH_NAME.get(name, name), INK2))
        _fit_ylim(ax, np.concatenate(lows), np.concatenate(highs))
        ax.set_xlim(85, max(r["budget"] for r in series[0][1]) + 230)
        _x_to_data(ax, [r["budget"] for r in series[0][1]])
        ax.set_xlabel("labelling budget")
        ax.set_ylabel(yl)
        ax.set_title("uncertainty + oracle on $L_{AL}$, $\\eta$ = 0" if letter == "(a)" else "same loops", loc="left",
                     color=INK2)
        _panel_letter(ax, letter)
        axs_ends = ends
        ax._ends = axs_ends
    finish(fig, f"mean and 95% CI over {e['n_seeds']} seeds; each architecture runs its own tuned recipe inside the loop.",
           legends=[(axs[0], 2)])
    for ax in axs:
        end_labels(ax, ax._ends)
    return fig


def fig_e6(d, out):
    e = d["e06"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    names = [v for v, ve in e["ablate"]["variants"].items() if ve.get("paired_vs_default")]
    y = np.arange(len(names))
    for off, key, col, mk, lab in ((-0.14, "bal_acc", BLUE, "o", "balanced accuracy"),
                                   (0.14, "auroc", ORANGE, "s", "AUROC")):
        m = [100 * e["ablate"]["variants"][v]["paired_vs_default"][key]["mean"] for v in names]
        lo = [100 * e["ablate"]["variants"][v]["paired_vs_default"][key]["ci_lo"] for v in names]
        hi = [100 * e["ablate"]["variants"][v]["paired_vs_default"][key]["ci_hi"] for v in names]
        ax.errorbar(m, y + off, xerr=[np.array(m) - lo, np.array(hi) - m], color=col, marker=mk, ls="none",
                    capsize=2, label=lab, markeredgecolor="white", markeredgewidth=0.6)
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("difference to the default recipe (percentage points, 95% CI)")
    _panel_letter(ax, "(a)")
    ax = axs[1]
    pr = e["warm_start"]["paired"]
    x = [r["budget"] for r in pr]
    m = np.array([100 * r["warm_minus_scratch"]["bal_acc"]["mean"] for r in pr])
    lo = np.array([100 * r["warm_minus_scratch"]["bal_acc"]["ci_lo"] for r in pr])
    hi = np.array([100 * r["warm_minus_scratch"]["bal_acc"]["ci_hi"] for r in pr])
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xlabel("labelling budget")
    ax.set_ylabel("warm start minus scratch, bal. acc. (pp)")
    _panel_letter(ax, "(b)")
    finish(fig, legends=[(axs[0], 2)])
    return fig


def fig_e8(d, out):
    e = d["e08"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
    ax = axs[0]
    bins = [b for b in e["reliability"] if b["pooled_over_seeds"]["n"] > 0]
    mp = np.array([b["mean_p"]["mean"] for b in bins])
    og = np.array([b["obs_good_rate"]["mean"] for b in bins])
    lo = np.array([b["obs_good_rate"]["ci_lo"] for b in bins])
    hi = np.array([b["obs_good_rate"]["ci_hi"] for b in bins])
    n = np.array([b["pooled_over_seeds"]["n"] for b in bins], dtype=float)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.6, ls="--", label="perfect calibration")
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.errorbar(mp[ok], og[ok], yerr=[(og - lo)[ok], (hi - og)[ok]], color=BLUE, ls="none", elinewidth=0.7,
                capsize=1.5, zorder=2)
    ax.plot(mp, og, color=BLUE, linewidth=0.8, zorder=2)
    ax.scatter(mp, og, s=10 + 70 * np.sqrt(n / n.max()), color=BLUE, edgecolor="white", linewidth=0.6,
               zorder=3, label="bins (area grows with share of signals)", clip_on=False)
    ax.set_xlabel("predicted p(GOOD), mean over seeds")
    ax.set_ylabel("observed GOOD rate (mean and 95% CI)")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("reliability at budget 600", loc="left", color=INK2)
    _panel_letter(ax, "(a)")
    ax = axs[1]
    t = [100 * r["p_accept"] for r in e["thresholds"]]
    lows, highs, ends = [], [], []
    for key, col, mk, ls, lab in (("precision", BLUE, "o", "-", "precision of the kept set"),
                                  ("bad_recall", ORANGE, "s", "--", "BAD recall"),
                                  ("fraction_kept", AQUA, "^", ":", "fraction kept")):
        m = np.array([100 * r[key]["mean"] for r in e["thresholds"]])
        lo_ = np.array([100 * r[key]["ci_lo"] for r in e["thresholds"]])
        hi_ = np.array([100 * r[key]["ci_hi"] for r in e["thresholds"]])
        ax.fill_between(t, lo_, hi_, color=col, alpha=0.15, linewidth=0)
        ax.plot(t, m, color=col, marker=mk, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.6)
        lows.append(lo_)
        highs.append(hi_)
        ends.append((t[-1], m[-1], lab, INK2))
    _fit_ylim(ax, np.concatenate(lows), np.concatenate(highs), top=100.8)
    ax.set_xlim(t[0] - 3, t[-1] + 34)
    _x_to_data(ax, [v for v in (30, 40, 50, 60, 70, 80, 90) if t[0] <= v <= t[-1]] + [t[-1]])
    ax.set_xlabel("acceptance threshold $p_{accept}$ (%)")
    ax.set_ylabel("%")
    ax.set_title("kept = p(GOOD) $\\geq p_{accept}$", loc="left", color=INK2)
    _panel_letter(ax, "(b)")
    finish(fig, legends=[(axs[0], 1), (axs[1], 1)])
    end_labels(axs[1], ends)
    return fig


def fig_e9(d, out):
    e = d["e09"]
    items = [(f"{k} filter", e["filters"][k]["bad_recall"], e["filters"][k]["good_kept"], None, None)
             for k in e["filters"]]
    items.append(("depth-3 tree", e["tree"]["pooled"]["bad_recall"], e["tree"]["pooled"]["good_kept"],
                  None, None))
    pending = None
    if "e03" in d:
        r = _row(d["e03"]["cells"]["unc_oracle|eta=0"]["rows"], 600)["metrics"]
        items.append(("learned (MicroConv1D, E3)", r["bad_recall"]["mean"], r["good_retention"]["mean"],
                      (r["bad_recall"]["ci_lo"], r["bad_recall"]["ci_hi"]),
                      (r["good_retention"]["ci_lo"], r["good_retention"]["ci_hi"])))
    else:
        pending = "learned model: pending E3 (cells not yet aggregated)"
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    x = np.arange(len(items))
    ax.bar(x - 0.19, [100 * i[1] for i in items], 0.36, color=BLUE, label=f"BAD recall (% of the {e['n_shared_bad']} BAD flagged)",
           linewidth=0)
    ax.bar(x + 0.19, [100 * i[2] for i in items], 0.36, color=ORANGE,
           label=f"GOOD kept (% of the {e['n_shared'] - e['n_shared_bad']} GOOD)", linewidth=0)
    for xi, i in zip(x, items):
        for off, v, ci in ((-0.19, i[1], i[3]), (0.19, i[2], i[4])):
            top = 100 * v
            if ci is not None:
                ax.errorbar([xi + off], [100 * v], yerr=[[100 * (v - ci[0])], [100 * (ci[1] - v)]], color=INK2,
                            capsize=2, linewidth=0.8)
                top = 100 * ci[1]
            ax.text(xi + off, top + 1.5, f"{100 * v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([i[0] for i in items])
    ax.set_ylabel(f"% of the {e['n_shared']} shared signals' class")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    if pending:
        ax.text(0.99, 0.96, pending, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2,
                style="italic")
    finish(fig, "learned model: mean and 95% CI over seeds (E3, unc_oracle, eta 0, budget 600).",
           legends=[(ax, 2)])
    return fig


def fig_e10(d, out):
    from Curation_Paper.eswa_full.lib.signals import parse_id
    s = d["e10_scores"]
    p = np.asarray(s["mean_p"], dtype=float)
    by = {}
    for sid, pv in zip(s["ids"], p):
        q = parse_id(sid)
        by.setdefault(q["panel"], {}).setdefault(q["cycle"], []).append(pv >= 0.5)
    panels = sorted(by)
    fig, axs = plt.subplots(1, len(panels), figsize=(W2, 2.2), sharey=True)
    for ax, pn in zip(np.atleast_1d(axs), panels):
        cyc = sorted(by[pn])
        fr = [100 * np.mean(by[pn][c]) for c in cyc]
        ax.axhline(50, color=INK2, linewidth=0.6, ls=":")
        ax.plot(cyc, fr, color=BLUE, marker="o", markersize=2.5, linewidth=0.9)
        ax.set_title(pn, fontsize=8)
        ax.set_xlabel("cycle")
        ax.set_ylim(-3, 103)
        ax.tick_params(axis="x", labelsize=7)
        ax.locator_params(axis="x", nbins=3)
    np.atleast_1d(axs)[0].set_ylabel("signals predicted GOOD (%)")
    fig.tight_layout()
    return fig


def fig_e10_famb(d, out):
    """Simulated f_amb of the replay (E10): over the POOL's unlabelled rows, not the 12,504 universe."""
    e = d["e10_famb"]
    rows = e["rows"]
    x = [r["budget"] for r in rows]
    m = np.array([100 * r["over_seeds"]["mean"] for r in rows])
    lo = np.array([100 * r["over_seeds"]["ci_lo"] for r in rows])
    hi = np.array([100 * r["over_seeds"]["ci_hi"] for r in rows])
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6)
    ax.set_ylim(0, max(hi) * 1.12)
    ax.set_xlim(x[0] - 20, x[-1] + 20)
    ax.set_xlabel("labelling budget")
    ax.set_ylabel(f"simulated $f_{{amb}}$ (%), $|p-0.5|<{e['ambiguous_band']:g}$")
    ax.text(0.97, 0.96, "replay, uncertainty + oracle, $L_{AL}$, $\\eta$ = 0:\nshare of the pool's unlabelled rows,\n"
            "not of the 12,504-signal universe", transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=INK2)
    finish(fig, f"mean and 95% CI over {len(e['seeds'])} seeds.")
    return fig


def fig_e11(d, out):
    e = d["e11"]
    groups = (("Full", ("panels",), BLUE, "o", "curated vs uncurated"),
              ("Common", ("common_cycle", "panels"), ORANGE, "s", "same path-cycles only"),
              ("LabelsCommon", ("labels_only", "common_cycle"), AQUA, "^", "labels-only curation, same path-cycles"))
    panels = sorted(k for k in e["panels"] if k.startswith("FOD"))
    fig, ax = plt.subplots(figsize=(W2, 2.8))
    y = np.arange(len(panels))
    for off, (gname, root, col, mk, lab) in zip((-0.22, 0.0, 0.22), groups):
        node = e
        for t in root:
            node = node[t]
        m, lo, hi = [], [], []
        for pn in panels:
            pp = node[pn]["path_paired_monotonicity"]
            m.append(pp["mean_diff"]); lo.append(pp["ci95"][0]); hi.append(pp["ci95"][1])
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.errorbar(m, y + off, xerr=[m - lo, hi - m], color=col, marker=mk, ls="none", capsize=2, label=lab,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(panels); ax.invert_yaxis()
    ax.set_xlabel("change in monotonicity (mean $|$Spearman$|$ of path DI vs cycle), 95% CI over paths")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_workflow(d, out):
    """Drawn by fig_workflow_eswa.py (ruling R40a), never copied from the short paper's figure.
    The script is run from the repo root and writes fig_workflow.pdf/.png into `out`."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ESWA_WORKFLOW_OUT=str(out))
    r = subprocess.run([sys.executable, str(WORKFLOW_SCRIPT)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0 or not (out / "fig_workflow.pdf").is_file():
        raise RuntimeError(f"fig_workflow_eswa.py failed (rc {r.returncode}):\n{r.stdout}\n{r.stderr}")
    return None


# name, studies needed, function
FIGS = [
    ("fig_campaign", ["e14"], fig_campaign),
    ("fig_fault_gallery", [], fig_good_bad_examples),
    ("fig_workflow", [], fig_workflow),
    ("fig_e1_curves", ["e01"], fig_e1),
    ("fig_e23_ablations", ["e02", "e03"], fig_e23),
    ("fig_e4_scatter", ["e04_compare"], fig_e4),
    ("fig_e5_architectures", ["e05"], fig_e5),
    ("fig_e6_ablation", ["e06"], fig_e6),
    ("fig_e8_threshold", ["e08"], fig_e8),
    ("fig_e9_rules", ["e09"], fig_e9),
    ("fig_e10_coverage", ["e10_scores"], fig_e10),
    ("fig_e10_famb", ["e10_famb"], fig_e10_famb),
    ("fig_e11_downstream", ["e11"], fig_e11),
]


def make_all(data: dict, out_dir: Path, *, only=None, fault_gallery=True) -> dict:
    """{name: 'drawn' | 'skipped: <missing>'}.  Pure function of the loaded JSON."""
    style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, needs, fn in FIGS:
        if only and name not in only:
            continue
        miss = [n for n in needs if n not in data]
        if miss:
            status[name] = "skipped: " + ", ".join(miss)
            continue
        if name == "fig_fault_gallery" and not fault_gallery:
            status[name] = "skipped: fault gallery disabled"
            continue
        fig = fn(data, out_dir)
        if fig is not None:
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"{name}.{ext}")
            plt.close(fig)
        status[name] = "drawn"
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="generate figures from the study JSONs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--partial", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--require-stage2", action="store_true",
                    help="refuse (exit 12) unless e01/e02/e03/e04_compare are all read from *_s20.json")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None, help="comma list of figure names")
    args = ap.parse_args(argv)
    cfg = load_config()
    rdir = Path(args.results_dir) if args.results_dir else PKG_ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else FIG_DIR
    studies = [s for s in R.STUDIES if s.loader is None] + AUX
    try:
        C.check_output_path(out_dir, MN.FORBIDDEN_OUT)
        data, absent = MN.load_studies(rdir, cfg, studies=studies)
        if args.require_stage2:
            C.check_require_stage2({st.key: C.stage_of(data[st.key]) if st.key in data else None
                                    for st in studies if st.stage2})
        if args.final:
            C.check_final(absent)
        MN.check_all(data, input_hashes(cfg), studies=studies)
    except C.Refusal as e:
        print(f"REFUSING ({type(e).__name__}, exit {e.code}): {e}", file=sys.stderr)
        return e.code
    print("stage sources:", "; ".join(f"{k}={stg} ({f}, {C.seed_label(n)} seeds)"
                                      for k, stg, f, n in MN.stage_report(data, studies)) or "none")
    try:
        status = make_all(data, out_dir, only=args.only.split(",") if args.only else None)
    except C.Refusal as e:                       # e.g. fig_e23 refuses a mixed-stage E2/E3 pair
        print(f"REFUSING ({type(e).__name__}, exit {e.code}): {e}", file=sys.stderr)
        return e.code
    for k, v in status.items():
        print(f"{k:22s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
