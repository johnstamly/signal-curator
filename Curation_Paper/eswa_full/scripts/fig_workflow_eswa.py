"""ESWA variant of Curation_Paper/scripts/23_workflow_schematic.py (ruling R40a).

A copy of that script with two changes: the loop label reads "repeat; f_amb is shown as a
proposed stopping signal" (the original says "until f_amb converges", which this paper does not
claim), and the outputs go to eswa_full/figures.  Everything else is identical.

Run from the repository root:
    python Curation_Paper/eswa_full/scripts/fig_workflow_eswa.py

Outputs:
- Curation_Paper/eswa_full/figures/fig_workflow.pdf  (vector)
- Curation_Paper/eswa_full/figures/fig_workflow.png  (300 dpi preview)

Replaces the generated raster fig2a_pipeline.png. The figure is drawn at its final
printed size (7.0 in, the elsarticle 5p text width), so font sizes are true points.
The build fails if any label is below 7 pt, leaves the canvas, or overlaps another
label. The waveform in the signal box is real: the GOOD-labelled signal of median
peak-to-peak amplitude (cycle >= 200) in the curated Zenodo deposit, first 500
samples (the classifier's input window).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

DATA = Path("Curation_Paper/zenodo_tier3")
# ESWA variant: only the loop label differs from script 23.  make_figures sets ESWA_WORKFLOW_OUT so
# the figure lands in its --out-dir; run standalone it writes to eswa_full/figures.
OUT = Path(os.environ.get("ESWA_WORKFLOW_OUT", "Curation_Paper/eswa_full/figures"))
STEM = "fig_workflow"

# Times to match elsarticle; orange = operator step, blue = model step
INK, INK2, RIM = "#1d1d1b", "#52514e", "#3a3a37"
MODEL_FC, MODEL_EC = "#e6eef9", "#2a5f9e"
HUMAN_FC, HUMAN_EC = "#fbeadf", "#b5541c"
DATA_FC, DATA_EC = "#f1f0ec", "#6f6e69"
WAVE = "#2a78d6"
MIN_PT = 7.0
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42})

# ---- real waveform for the signal box ----
labels = json.loads((DATA / "paper_labels.json").read_text())["labels"]
h5 = h5py.File(DATA / "morpho_50khz_full.h5", "r")
good = [L for L in labels if L["label"] == "GOOD" and L["cycle"] >= 200]


def signal(L):
    d = h5[f"{L['panel']}/50kHz/act{L['actuator']}/recv{L['receiver']}/cycle{L['cycle']}/data"]
    return np.asarray(d[L["rep_idx"], :500], dtype=float)


p2p = np.array([np.ptp(signal(L)) for L in good])
pick = good[int(np.argsort(p2p)[len(p2p) // 2])]
WAVEFORM = signal(pick)
print("waveform:", pick["signal_id"])


class Canvas:
    """Axes in inches over the whole figure, so coordinates and font sizes are physical."""

    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.texts = []

    def text(self, x, y, s, size=7.5, **kw):
        assert size >= MIN_PT, (s, size)
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        t = self.ax.text(x, y, s, fontsize=size, **kw)
        self.texts.append(t)
        return t

    def box(self, x, y, w, h, fc, ec, lw=0.8):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                         fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(self, p, q, color=RIM, lw=0.9):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                          color=color, lw=lw, zorder=1, shrinkA=0, shrinkB=0,
                                          mutation_scale=9, joinstyle="miter"))

    def route(self, pts, color=RIM, lw=0.9, ls="-"):
        """Orthogonal connector through pts; arrowhead on the last segment (drawn solid)."""
        xs, ys = zip(*pts)
        self.ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1] - np.sign(ys[-1] - ys[-2]) * 0.06,),
                     color=color, lw=lw, ls=ls, zorder=1, solid_joinstyle="miter", solid_capstyle="butt")
        self.arrow((xs[-1], ys[-1] - np.sign(ys[-1] - ys[-2]) * 0.07), pts[-1], color=color, lw=lw)

    def lane_label(self, x, y, s, size=7.5, color=HUMAN_EC):
        return self.text(x, y, s, size=size, color=color, zorder=5,
                         bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    def person(self, x, y, s=0.075, color=HUMAN_EC):
        """Operator glyph (head and shoulders) centred at x, y."""
        self.ax.add_patch(Circle((x, y + 0.55 * s), 0.38 * s, fc=color, ec="none", zorder=4))
        self.ax.add_patch(Wedge((x, y - 0.55 * s), 0.75 * s, 0, 180, fc=color, ec="none", zorder=4))

    def wave(self, x, y, w, h, color=WAVE):
        s = WAVEFORM / np.abs(WAVEFORM).max()
        self.ax.plot(np.linspace(x, x + w, len(s)), y + s * h / 2, color=color, lw=0.6, zorder=4)

    def p_axis(self, x, y, w, band=0.1):
        """Score axis p in [0, 1] with the ambiguous zone |p - 1/2| < 0.1 shaded (glyph, not data)."""
        self.ax.plot([x, x + w], [y, y], color=INK2, lw=0.6, zorder=4)
        for v in (0, 0.5, 1):
            self.ax.plot([x + v * w] * 2, [y - 0.03, y + 0.03], color=INK2, lw=0.6, zorder=4)
        self.ax.add_patch(FancyBboxPatch((x + (0.5 - band) * w, y - 0.045), 2 * band * w, 0.09,
                                         boxstyle="square,pad=0", fc=HUMAN_EC, ec="none", alpha=0.55, zorder=3))

    def check_and_save(self, stem):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        bbs = [(t.get_text(), t.get_window_extent(r)) for t in self.texts]
        W, H = self.fig.bbox.width, self.fig.bbox.height
        for s, b in bbs:
            assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= W and b.y1 <= H, f"text off canvas: {s!r}"
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                assert not bbs[i][1].overlaps(bbs[j][1]), f"text collision: {bbs[i][0]!r} / {bbs[j][0]!r}"
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUT / f"{stem}.pdf")
        self.fig.savefig(OUT / f"{stem}.png", dpi=300)
        plt.close(self.fig)
        print("wrote", OUT / f"{stem}.pdf")


c = Canvas(7.00, 1.72)
yc, bh = 0.80, 0.62
B = dict(sig=(0.05, 0.94), L=(1.44, 0.94), tr=(2.68, 1.05), sc=(4.05, 0.95))
colours = dict(sig=(DATA_FC, DATA_EC), L=(HUMAN_FC, HUMAN_EC), tr=(MODEL_FC, MODEL_EC), sc=(MODEL_FC, MODEL_EC))
for k, (x, w) in B.items():
    c.box(x, yc - bh / 2, w, bh, *colours[k])
for a, b in (("sig", "L"), ("L", "tr"), ("tr", "sc")):
    c.arrow((B[a][0] + B[a][1], yc), (B[b][0], yc))


def mid(k, dx=0.0):
    return B[k][0] + B[k][1] / 2 + dx


# signal universe -> seed -> label store
c.wave(B["sig"][0] + 0.08, yc + 0.16, B["sig"][1] - 0.16, 0.18)
c.text(mid("sig"), yc - 0.10, "12,504 signals", size=8)
c.text(mid("sig"), yc - 0.23, "50 kHz, FOD3–FOD7", size=7, color=INK2)
gx = (B["sig"][0] + B["sig"][1] + B["L"][0]) / 2
c.text(gx, yc + 0.09, "seed", size=7, color=INK2)
c.text(gx, yc - 0.09, "743 labels", size=7, color=INK2)

c.person(B["L"][0] + 0.15, yc + 0.15)
c.text(mid("L", 0.10), yc + 0.15, "Label store $L$", size=8)
c.text(mid("L"), yc - 0.06, "operator verdicts", size=7, color=INK2)
c.text(mid("L"), yc - 0.19, "GOOD / BAD", size=7, color=INK2)

c.text(mid("tr"), yc + 0.15, "Train MicroConv1D", size=8)
c.text(mid("tr"), yc - 0.06, "from scratch on $L$", size=7, color=INK2)
c.text(mid("tr"), yc - 0.19, "new 20 % validation split", size=7, color=INK2)

c.text(mid("sc"), yc + 0.19, "Score all signals", size=8)
c.p_axis(B["sc"][0] + 0.145, yc - 0.06, B["sc"][1] - 0.29)
c.text(mid("sc"), yc + 0.045, r"$f_\mathrm{amb}$", size=7, color=HUMAN_EC)
c.text(mid("sc", -0.33), yc - 0.20, "0", size=7, color=INK2)
c.text(mid("sc"), yc - 0.20, "$p$", size=7.5, color=INK2)
c.text(mid("sc", 0.33), yc - 0.20, "1", size=7, color=INK2)

# two operator query streams
qx, qw, qh, qU, qF = 5.36, 1.58, 0.40, 1.05, 0.55
for yq, title, sub in ((qU, "(i) Uncertainty query", r"unlabelled, min $|p-\frac{1}{2}|$"),
                       (qF, "(ii) FP/FN re-inspection", "validation disagreements")):
    c.box(qx, yq - qh / 2, qw, qh, HUMAN_FC, HUMAN_EC)
    c.person(qx + 0.15, yq)
    c.text(qx + qw / 2 + 0.10, yq + 0.08, title, size=8)
    c.text(qx + qw / 2 + 0.10, yq - 0.09, sub, size=7, color=INK2)
sx = B["sc"][0] + B["sc"][1]
c.arrow((sx, yc + 0.08), (qx, qU))
c.arrow((sx, yc - 0.08), (qx, qF))

# return paths: (i) adds verdicts to L, (ii) confirms or corrects existing ones
Lx, xq = mid("L"), qx + qw / 2
c.route([(xq, qU + qh / 2), (xq, 1.42), (Lx, 1.42), (Lx, yc + bh / 2)], color=HUMAN_EC)
c.route([(xq, qF - qh / 2), (xq, 0.17), (Lx, 0.17), (Lx, yc - bh / 2)], color=HUMAN_EC, ls=(0, (3, 1.6)))
c.lane_label(3.80, 1.42, "operator labels new signals: added to $L$ (≈80–100 per iteration)", size=7)
c.lane_label(3.80, 0.17, r"operator confirms or corrects earlier verdicts ($|L|$ unchanged)", size=7)
# stopping rule applies to the whole cycle, so it sits above both loops
c.text((Lx + xq) / 2, 1.62, r"repeat; $f_\mathrm{amb}^{(k)}$ is shown as a proposed stopping signal", size=7.5, color=INK)

c.check_and_save(STEM)
