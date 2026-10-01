"""Figure 1 of the 2-page draft: the controlled 2x2 and the score decomposition.

(a) One BEAF counterfactual image pair crossed with one AB-swap caption pair, with the
    measured cosine similarities in the four cells. The mismatched cell S(-,+) beats the
    matched cell S(-,-), so the judgment fails.
(b) The same four scores as deviations from the shared constant C, drawn as a waterfall of
    the identity S_{u,v} = C + u*alpha + v*beta + u*v*gamma. The interaction is a sliver.

Notation follows the paper: u is the text polarity label, v is the image presence label,
both in {-1,+1}; alpha = m_I . d_T is the text main effect, beta = d_I . m_T the image
main effect, gamma = d_I . d_T the interaction.

`panel_c` (per-model distance to the success condition) is retained but not called: it is
a full-width panel and does not fit the single-column figure. Re-enable it in main().

Run from the repo root:  python paper_figures/mk_fig1_binding_condition.py
"""
import csv
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
B = "logs/evaluation/01_paper/"
P2_CONCEPT = B + "2026-09-02_peakpatch_2x2_audit/peakpatch_per_concept.csv"
P2_MODELS = B + "2026-09-03_target_and_aim/target_and_aim.json"
P2_PEAKPATCH = B + "2026-09-02_peakpatch_2x2_audit/peakpatch_2x2_summary.json"
OUT = "paper_figures/fig1_binding_condition.png"

INK, MUTE, FAINT = "#1a1a1a", "#8a8a8a", "#c9c9c9"
GREEN, RED = "#2e7d4f", "#b3402f"
FS, FS_L = 7.0, 7.5          # the 7 pt floor


def load_concept_ratios():
    rows = list(csv.DictReader(open(P2_CONCEPT)))
    return ({pre: [float(r[f"{pre}_gamma"])
                   / max(float(r[f"{pre}_abs_alpha"]), float(r[f"{pre}_abs_beta"]))
                   for r in rows] for pre in ("cosine", "ecn", "ecn+scn")}, len(rows))


def load_model_points():
    rows = json.load(open(P2_MODELS))["rows"]
    pts = [(r["name"], r["gamma"] / max(r["abs_alpha"], r["abs_beta"]), r["family"])
           for r in rows]
    s = json.load(open(P2_PEAKPATCH))["summary"]["ecn+scn"]
    pts.append(("PeakPatch ECN+SCN",
                s["gamma"] / max(s["abs_alpha"], s["abs_beta"]), "intervention"))
    return pts


# ------------------------------------------------------------------- panel (a)
def panel_a(fig, rect, ex):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    COLX = (0.520, 0.830)
    ROWY = (0.490, 0.100)
    CW, CH = 0.285, 0.325
    PX, PW = 0.100, 0.230

    for x, lab in ((COLX[0], "$T_{+}$"), (COLX[1], "$T_{-}$")):
        ax.text(x, 0.925, lab, ha="center", va="center", fontsize=FS_L, color=INK)

    for path, y, lab in ((ex["IP"], ROWY[0], "$I_{+}$"), (ex["IN"], ROWY[1], "$I_{-}$")):
        assert os.path.exists(path), "missing example image: " + path
        box = ax.inset_axes([PX, y, PW, CH])
        box.imshow(mpimg.imread(path))
        box.set_xticks([]); box.set_yticks([])
        for s in box.spines.values():
            s.set_edgecolor(MUTE); s.set_linewidth(0.4)
        ax.text(0.030, y + CH / 2, lab, ha="center", va="center", fontsize=FS_L, color=INK)

    # u = text label, v = image label
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
        ax.text(cx, cy + CH * 0.66, "$S_{%s,%s}$" % (u, v), ha="center", va="center",
                fontsize=FS_L, color=tc)
        ax.text(cx, cy + CH * 0.26, "%.4f" % val, ha="center", va="center",
                fontsize=FS_L, color=tc)


# ------------------------------------------------------------------- panel (b)
def panel_b(fig, rect, ex):
    al, be, ga = ex["alpha"], ex["beta"], ex["gamma"]
    order = ((+1, +1, ex["Spp"], True), (-1, +1, ex["Smp"], False),
             (+1, -1, ex["Spm"], False), (-1, -1, ex["Smm"], True))
    for u, v, val, _ in order:              # verify the identity before drawing it
        assert abs((ex["C"] + u * al + v * be + u * v * ga) - val) < 1e-6

    ax = fig.add_axes(rect)
    ax.set_ylim(-0.6, 3.6)
    ax.invert_yaxis()
    ax.set_yticks(range(4))
    ax.set_yticklabels(["$S_{%s,%s}$" % ("+" if u > 0 else "-", "+" if v > 0 else "-")
                        for u, v, _, _ in order], fontsize=FS)
    for tick, o in zip(ax.get_yticklabels(), order):
        tick.set_color(INK if o[3] else MUTE)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=FS, length=1.5, pad=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)

    lim = 0.0225
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-0.02, 0, 0.02])
    ax.set_xticklabels(["−.02", "0", "+.02"], fontsize=FS)
    ax.axvline(0, color=INK, lw=0.5, zorder=1)

    for i, (u, v, _, _) in enumerate(order):
        run = 0.0
        for val, col in ((u * al, FAINT), (v * be, MUTE), (u * v * ga, GREEN)):
            ax.barh(i, val, left=run, height=0.52, color=col, lw=0, zorder=2)
            run += val

    handles = [Rectangle((0, 0), 1, 1, color=FAINT), Rectangle((0, 0), 1, 1, color=MUTE),
               Rectangle((0, 0), 1, 1, color=GREEN)]
    ax.legend(handles, ["$u\\alpha$", "$v\\beta$", "$uv\\gamma$"],
              loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=3, frameon=False,
              fontsize=FS, handlelength=0.8, handleheight=0.7, handletextpad=0.35,
              columnspacing=0.8, borderpad=0.0)


# ------------------------------------------------------------------- panel (c)
# Retained for a full-width version or for slides; not called by main().
def panel_c(fig, rect, ratios, n_concepts, models):
    ax = fig.add_axes(rect)
    back = [m for m in models if m[2] == "backbone"]
    inter = [m for m in models if m[2] == "intervention"]
    ordered = back + inter
    n = len(ordered)
    SHORT = {"ViT-B/32 (OpenAI)": "ViT-B/32", "NegCLIP-NegFull": "NegCLIP-NF",
             "CLIP-NegFull": "CLIP-NF"}
    CLOUD = {"ViT-B/32": "cosine", "PeakPatch ECN": "ecn",
             "PeakPatch ECN+SCN": "ecn+scn"}
    rmap = {SHORT.get(m[0], m[0]): m[1] for m in ordered}
    ax.set_ylim(n + 1.40, -1.20)
    ax.set_xlim(-0.16, 1.22)
    ax.set_yticks(range(n))
    ax.set_yticklabels([SHORT.get(m[0], m[0]) for m in ordered], fontsize=FS)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=FS, length=1.5, pad=1.5)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_bounds(-0.16, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.plot([1.0, 1.0], [-0.95, n - 0.55], color=RED, lw=0.9, zorder=4)
    ax.axvline(0.0, color=FAINT, lw=0.5, zorder=1)
    ax.text(0.985, -1.16, "성공 조건  $r>1$", ha="right", va="top",
            fontsize=FS_L, color=RED)
    r0 = rmap["ViT-B/32"]
    ax.plot([r0, r0], [0, n - 1], color=GREEN, lw=0.5, ls=(0, (1.6, 1.6)), zorder=1)
    for i, (name, r, fam) in enumerate(ordered):
        short = SHORT.get(name, name)
        ax.plot([0, r], [i, i], color=FAINT, lw=0.5, zorder=1)
        cloud = CLOUD.get(short)
        if cloud:
            ax.scatter(ratios[cloud], [i] * n_concepts, s=4.5, color=MUTE,
                       alpha=0.42, lw=0, zorder=2)
        ax.plot([r], [i], marker="D" if fam == "backbone" else "o", ms=3.4,
                color=INK if fam == "backbone" else GREEN, zorder=3)
    r_pp = rmap["PeakPatch ECN+SCN"]
    ax.annotate("", xy=(r_pp, n - 1), xytext=(r0, n - 1),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8,
                                shrinkA=1.5, shrinkB=1.5), zorder=5)
    ax.axhline(len(back) - 0.5, color=FAINT, lw=0.5, ls=(0, (2.0, 1.8)), zorder=1)


def main():
    ex = json.load(open(EXAMPLE))

    fig = plt.figure(figsize=(3.26, 1.44), dpi=400)
    panel_a(fig, [0.005, 0.085, 0.560, 0.900], ex)
    panel_b(fig, [0.720, 0.335, 0.250, 0.580], ex)
    # panel_c(fig, [0.0, 0.0, 1.0, 1.0], *load_concept_ratios(), load_model_points())

    fig.text(0.283, 0.038,
             "$\\Delta(S)=%.4f-%.4f=%.3f<0$" % (ex["Smm"], ex["Smp"], ex["delta"]),
             ha="center", va="center", fontsize=FS_L, color=RED)

    fig.savefig(OUT, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    r = ex["gamma"] / max(abs(ex["alpha"]), abs(ex["beta"]))
    print("saved:", OUT)
    print("  caption T+: " + ex["TP"])
    print("  caption T-: " + ex["TN"])
    print("  example pair ratio gamma/max(|alpha|,|beta|) = %.3f" % r)
if __name__ == "__main__":
    main()
