"""Figure 1 of the 0827 draft: the controlled 2x2 minimal pair, measured.

One BEAF counterfactual image pair (same scene, one object inpainted out) crossed with
one AB-swap caption pair (same word set, binding swapped). The four measured cosine
similarities fill the cells; the mismatched cell S(-,+) beats the matched cell S(-,-),
so Delta(S) < 0 and the judgment fails on this pair.

This is the panel-(a)-only variant of mk_fig1_binding_condition.py: it carries no
alpha/beta/gamma decomposition, so it illustrates the draft's own Delta(S) definition
without importing machinery the draft does not introduce.

Run from the repo root:  python paper_figures/mk_fig1_minimal_pair.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
except Exception:
    pass
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

EXAMPLE = "paper_figures/fig1_example_pair.json"
OUT = "paper_figures/fig1_minimal_pair.png"

INK, MUTE = "#1a1a1a", "#8a8a8a"
RED = "#b3402f"
FS, FS_L = 6.4, 7.5


def main():
    ex = json.load(open(EXAMPLE))

    fig = plt.figure(figsize=(3.26, 1.52), dpi=400)
    ax = fig.add_axes([0.005, 0.075, 0.99, 0.915])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    COLX = (0.545, 0.845)
    ROWY = (0.435, 0.055)
    CW, CH = 0.275, 0.315
    PX, PW = 0.105, 0.225

    heads = ((COLX[0], "$T_{+}$", "“no cup,\nbut a pizza”"),
             (COLX[1], "$T_{-}$", "“no pizza,\nbut a cup”"))
    for x, lab, cap in heads:
        ax.text(x, 0.955, lab, ha="center", va="center", fontsize=FS_L, color=INK)
        ax.text(x, 0.845, cap, ha="center", va="center", fontsize=FS, color=MUTE,
                linespacing=1.15)

    rows = ((ex["IP"], ROWY[0], "$I_{+}$", "Presence"),
            (ex["IN"], ROWY[1], "$I_{-}$", "Absence"))
    for path, y, lab, sub in rows:
        assert os.path.exists(path), "missing example image: " + path
        box = ax.inset_axes([PX, y, PW, CH])
        box.imshow(mpimg.imread(path))
        box.set_xticks([]); box.set_yticks([])
        for s in box.spines.values():
            s.set_edgecolor(MUTE); s.set_linewidth(0.4)
        ax.text(0.038, y + CH * 0.62, lab, ha="center", va="center",
                fontsize=FS_L, color=INK)
        ax.text(0.038, y + CH * 0.20, sub, ha="center", va="center",
                fontsize=FS - 0.6, color=MUTE)

    # u = text polarity, v = image presence
    cells = (((COLX[0], ROWY[0]), "+", "+", ex["Spp"], "matched"),
             ((COLX[1], ROWY[0]), "-", "+", ex["Smp"], "winner"),
             ((COLX[0], ROWY[1]), "+", "-", ex["Spm"], "plain"),
             ((COLX[1], ROWY[1]), "-", "-", ex["Smm"], "matched"))
    vals = [c[3] for c in cells]
    lo, hi = min(vals), max(vals)
    STYLE = {"matched": (INK, "-", 0.8, INK),
             "winner": (RED, (0, (2.0, 1.4)), 0.9, RED),
             "plain": ("none", "-", 0.0, MUTE)}
    for (cx, cy), u, v, val, kind in cells:
        ec, ls, lw, tc = STYLE[kind]
        shade = 0.96 - 0.14 * (val - lo) / (hi - lo)
        ax.add_patch(Rectangle((cx - CW / 2, cy), CW, CH, facecolor=str(shade),
                               edgecolor=ec, ls=ls, lw=lw))
        ax.text(cx, cy + CH * 0.66, "$S_{(%s,%s)}$" % (u, v), ha="center", va="center",
                fontsize=FS_L, color=tc)
        ax.text(cx, cy + CH * 0.26, "%.4f" % val, ha="center", va="center",
                fontsize=FS_L, color=tc)

    fig.text(0.50, 0.038,
             "$\\Delta(S)=%.4f-%.4f=%.3f<0$" % (ex["Smm"], ex["Smp"], ex["delta"]),
             ha="center", va="center", fontsize=FS_L, color=RED)

    fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
