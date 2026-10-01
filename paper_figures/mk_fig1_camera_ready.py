"""Figure 1 of the camera-ready: the controlled 2x2 minimal pair, measured.

Same content as mk_fig1_minimal_pair.py (one AB-swap image pair crossed with one word-swap caption
pair, four measured cosine similarities, Delta(S) < 0), redrawn after the conventions of the
reference papers in conference_paper/: English text inside the figure, plain table-like cells, and
coloured boxes that mark the objects on the images. The boxes make the AB-swap construction visible:
I+ is the edit with the cup removed (the pizza, O, remains), I- is the edit with the pizza removed
(the cups, O^C, remain). Box coordinates come from differencing each edit against the original
COCO image (COCO_val2014_000000294475.jpg), in original pixel units (640 x 482).

Run from the repo root:  python paper_figures/mk_fig1_camera_ready.py [example.json] [out.png]
(the two optional paths default to the camera-ready files below; the re-run passes its own)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.unicode_minus"] = False

EXAMPLE = sys.argv[1] if len(sys.argv) > 1 else "paper_figures/fig1_example_pair.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_figures/fig1_camera_ready.png"

INK, MUTE, EDGE = "#1a1a1a", "#7a7a7a", "#9a9a9a"
RED = "#c0392b"          # the violated comparison
C_O, C_OC = "#e67e22", "#1f77b4"   # pizza (O) and cup (O^C)
FS, FS_L = 6.2, 7.2

PIZZA = [(173, 181, 399, 321)]
CUPS = [(358, 92, 403, 149), (477, 181, 538, 254)]


def draw_segments(ax, xc, y, segs, fs):
    """Centre a line made of differently coloured pieces at (xc, y) in axes coordinates."""
    r = ax.figure.canvas.get_renderer()
    items = []
    for s, c in segs:
        t = ax.text(0, y, s, fontsize=fs, color=c, ha="left", va="center")
        items.append((t, t.get_window_extent(renderer=r).width))
    x_px = ax.transAxes.transform((xc, y))[0] - sum(w for _, w in items) / 2
    inv = ax.transAxes.inverted()
    for t, w in items:
        t.set_x(inv.transform((x_px, 0))[0])
        x_px += w


def boxed_image(ax, path, rect, boxes, color):
    box = ax.inset_axes(rect)
    img = mpimg.imread(path)
    box.imshow(img)
    for x0, y0, x1, y1 in boxes:
        box.add_patch(Rectangle((x0 - 5, y0 - 5), x1 - x0 + 10, y1 - y0 + 10, fill=False,
                                edgecolor=color, lw=0.9))
    box.set_xticks([]); box.set_yticks([])
    for s in box.spines.values():
        s.set_edgecolor(EDGE); s.set_linewidth(0.4)


def main():
    ex = json.load(open(EXAMPLE))
    for p in (ex["IP"], ex["IN"]):
        assert os.path.exists(p), "missing example image: " + p

    fig = plt.figure(figsize=(3.30, 1.42), dpi=400)
    ax = fig.add_axes([0.0, 0.085, 1.0, 0.915])
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    COLX = (0.555, 0.845)
    ROWY = (0.425, 0.035)
    CW, CH = 0.262, 0.355
    PX, PW = 0.108, 0.235

    ax.text(COLX[0], 0.945, "$T_{+}$", ha="center", va="center", fontsize=FS_L, color=INK)
    draw_segments(ax, COLX[0], 0.872, [("“no ", MUTE), ("cup", C_OC), (",", MUTE)], FS)
    draw_segments(ax, COLX[0], 0.815, [("but a ", MUTE), ("pizza", C_O), ("”", MUTE)], FS)
    ax.text(COLX[1], 0.945, "$T_{-}$", ha="center", va="center", fontsize=FS_L, color=INK)
    draw_segments(ax, COLX[1], 0.872, [("“no ", MUTE), ("pizza", C_O), (",", MUTE)], FS)
    draw_segments(ax, COLX[1], 0.815, [("but a ", MUTE), ("cup", C_OC), ("”", MUTE)], FS)

    rows = ((ex["IP"], ROWY[0], "$I_{+}$", "Presence", PIZZA, C_O),
            (ex["IN"], ROWY[1], "$I_{-}$", "Absence", CUPS, C_OC))
    for path, y, lab, sub, boxes, color in rows:
        boxed_image(ax, path, [PX, y, PW, CH], boxes, color)
        ax.text(0.056, y + CH * 0.62, lab, ha="center", va="center", fontsize=FS_L, color=INK)
        ax.text(0.056, y + CH * 0.28, sub, ha="center", va="center", fontsize=FS - 1.2, color=MUTE)

    # u = text polarity, v = image presence; matched cells on the diagonal
    cells = (((COLX[0], ROWY[0]), "+", "+", ex["Spp"], "matched"),
             ((COLX[1], ROWY[0]), "-", "+", ex["Smp"], "violating"),
             ((COLX[0], ROWY[1]), "+", "-", ex["Spm"], "plain"),
             ((COLX[1], ROWY[1]), "-", "-", ex["Smm"], "matched"))
    style = {"matched": (INK, "-", 0.8, INK),
             "violating": (RED, (0, (2.2, 1.4)), 0.9, RED),
             "plain": (EDGE, "-", 0.5, MUTE)}
    for (cx, cy), u, v, val, kind in cells:
        ec, ls, lw, tc = style[kind]
        ax.add_patch(Rectangle((cx - CW / 2, cy), CW, CH, facecolor="white",
                               edgecolor=ec, ls=ls, lw=lw))
        ax.text(cx, cy + CH * 0.67, "$S_{(%s,%s)}$" % (u, v), ha="center", va="center",
                fontsize=FS_L, color=tc)
        ax.text(cx, cy + CH * 0.29, "%.4f" % val, ha="center", va="center", fontsize=FS_L, color=tc)

    fig.text(0.5, 0.042, "$\\Delta(S)=%.4f-%.4f=%.3f<0$" % (ex["Smm"], ex["Smp"], ex["delta"]),
             ha="center", va="center", fontsize=FS_L, color=RED)
    fig.savefig(OUT, dpi=400, facecolor="white")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
