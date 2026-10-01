"""Figure 1 of the 2-page draft: the 2x2 construction and why cosine cannot satisfy it.

Left  -- the controlled 2x2. One BEAF counterfactual image pair (a pizza present / inpainted
         away, background held fixed) crossed with one NegBench caption pair that swaps only
         which object is negated. The judgment is correct only when both matched cells beat
         both mismatched ones.
Right -- what those scores decompose into over the 42 concepts: the interaction gamma that
         carries negation is a fifth of the dominant main effect, so the success condition
         gamma > max(|alpha|,|beta|) fails almost everywhere.

Coefficients: logs/evaluation/01_paper/2026-08-31_negfam_e2_abswap/vitb32_openai/
(42 concepts, 2,480 pairs, ViT-B/32 OpenAI), concept-mean of per-concept means, x1e3.

Run from the repo root:  python paper_figures/mk_paper_fig1_overview.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

try:
    fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
except Exception:
    pass
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

IMG_POS = "data/coco/images/val2014/COCO_val2014_000000008749_01.png"
IMG_NEG = "data/coco/images/val2014/COCO_val2014_000000008749_03.png"
ABS_A, ABS_B, GAM = 3.804, 6.284, 1.211        # |alpha|, |beta|, gamma  (x1e-3)

GREEN, RED, INK, MUTE = "#2e7d4f", "#b3402f", "#1a1a1a", "#8a8a8a"


def main():
    fig = plt.figure(figsize=(8.27 / 2.54, 3.05 / 2.54), dpi=400)

    # ---------------- left: the controlled 2x2 -------------------------------
    ax = fig.add_axes([0.0, 0.0, 0.55, 1.0]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    COLX = (0.575, 0.855)
    for x, lab, cap in ((COLX[0], "T\u208a", "“a pizza,\nbut no cup”"),
                        (COLX[1], "T\u208b", "“a cup,\nbut no pizza”")):
        ax.text(x, 0.995, lab, ha="center", va="top", fontsize=6.4, color=INK,
                style="italic")
        ax.text(x, 0.900, cap, ha="center", va="top", fontsize=4.5, color=MUTE,
                linespacing=1.2)

    ROWY = (0.400, 0.105)
    for img, y, lab in ((IMG_POS, ROWY[0], "I\u208a"), (IMG_NEG, ROWY[1], "I\u208b")):
        box = ax.inset_axes([0.135, y, 0.245, 0.265])
        box.imshow(mpimg.imread(img)); box.set_xticks([]); box.set_yticks([])
        for s in box.spines.values():
            s.set_edgecolor(MUTE); s.set_linewidth(0.4)
        ax.text(0.075, y + 0.137, lab, ha="center", va="center", fontsize=6.4, color=INK,
                style="italic")
    ax.text(0.075, ROWY[0] + 0.045, "객체\n있음", ha="center", va="top",
            fontsize=4.0, color=MUTE, linespacing=1.2)
    ax.text(0.075, ROWY[1] + 0.045, "객체\n제거", ha="center", va="top",
            fontsize=4.0, color=MUTE, linespacing=1.2)
    ax.text(0.2625, 0.995, "배경 고정,\n객체 유무만 변인", ha="center", va="top",
            fontsize=3.9, color=MUTE, linespacing=1.2)

    for (cx, cy), lab, col, mark in (((COLX[0], ROWY[0]), "$S_{+,+}$", GREEN, "✓"),
                                     ((COLX[1], ROWY[0]), "$S_{+,-}$", RED, "✗"),
                                     ((COLX[0], ROWY[1]), "$S_{-,+}$", RED, "✗"),
                                     ((COLX[1], ROWY[1]), "$S_{-,-}$", GREEN, "✓")):
        ax.add_patch(Rectangle((cx - 0.125, cy), 0.25, 0.265, facecolor=col, alpha=0.11,
                               edgecolor=col, lw=0.5))
        ax.text(cx, cy + 0.160, lab, ha="center", va="center", fontsize=6.4, color=col)
        ax.text(cx, cy + 0.058, mark, ha="center", va="center", fontsize=4.8, color=col)

    ax.text(0.50, 0.030,
            r"$\Delta(S)=\min(S_{+,+},S_{-,-})-\max(S_{+,-},S_{-,+})>0$",
            ha="center", va="center", fontsize=4.8, color=INK)

    # ---------------- right: what the score decomposes into ------------------
    ax2 = fig.add_axes([0.740, 0.320, 0.230, 0.440])
    ax2.set_xlim(0, 7.0); ax2.set_ylim(-0.6, 2.6)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels([r"$\gamma$ 교차항", r"$|\beta|$ 이미지", r"$|\alpha|$ 텍스트"],
                        fontsize=4.6)
    ax2.tick_params(axis="y", length=0, pad=1)
    ax2.tick_params(axis="x", labelsize=4.2, length=1.5, pad=1)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_linewidth(0.4)
    ax2.set_xticks([0, 2, 4, 6])

    ax2.barh([2, 1, 0], [ABS_A, ABS_B, GAM], height=0.52,
             color=[MUTE, MUTE, GREEN])
    for y, v in ((2, ABS_A), (1, ABS_B), (0, GAM)):
        ax2.text(v + 0.18, y, f"{v:.2f}", va="center", fontsize=4.3, color=INK)
    ax2.axvline(max(ABS_A, ABS_B), color=RED, lw=0.6, ls=(0, (2.2, 1.4)), ymax=0.93)

    fig.text(0.858, 0.985, r"$S_{ab}=C+a\beta+b\alpha+ab\gamma$", ha="center", va="top",
             fontsize=5.2, color=INK)
    fig.text(0.858, 0.865, r"$\alpha=m_I\!\cdot\!d_T,\ \beta=d_I\!\cdot\!m_T,\ "
                           r"\gamma=d_I\!\cdot\!d_T$", ha="center", va="top",
             fontsize=4.3, color=MUTE)
    fig.text(0.858, 0.085, r"성공 조건  $\gamma>\max(|\alpha|,|\beta|)$" "\n"
                           "42개 중 41개 개념에서 위배  ($\\times 10^{-3}$)",
             ha="center", va="center", fontsize=4.3, color=INK, linespacing=1.55)

    fig.savefig("paper_figures/fig_paper_overview.png", dpi=400,
                facecolor="white", pad_inches=0.015, bbox_inches="tight")
    print("saved: paper_figures/fig_paper_overview.png")


if __name__ == "__main__":
    main()
