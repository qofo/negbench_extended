"""
The audit figure: what does a high NegBench score buy on the controlled 2x2?

NegBench MCQ fixes the image and asks the model to pick a caption, which in the
factor coordinates is the single condition gamma > |alpha|. The controlled
counterfactual pairs also let us score the other direction, gamma > |beta|
(fix the caption, pick the image), and their conjunction, the Winoground group
score gamma > max(|alpha|,|beta|).

Panel (a) puts those two evaluations on one plane for nine models. Panel (b)
splits the group score into the two conditions it conjoins, so the failure can
be located rather than just reported.

Coefficient source is the single-object (6col) runs, matching mk_external_data.py
-- see its header for why the AB-swap sweep is not used for external comparisons.

Usage:
    python paper_figures/mk_audit.py
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
# Same font the other Korean-labelled figures use (mk_external.py, mk_design_comparison.py);
# the CJK JP face carries the Hangul glyphs and is the only CJK family installed here.
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

OUT = "logs/evaluation/01_paper/2026-08-31_negfam_mcq"
P = "logs/evaluation/01_paper/"

# (display name, mcq run dir, e2 dir, family)
M = [("ViT-B/32 (OpenAI)", "vitb32_openai",   P + "2026-08-28_r6_main_effect_ablation_33concepts/", "pre"),
     ("ViT-B/16",          "vitb16_openai",   P + "2026-08-30_r8_vitb16/e2_hadamard_decomposition/", "pre"),
     ("ViT-L/14",          "vitl14_openai",   P + "2026-08-30_r8_vitl14/e2_hadamard_decomposition/", "pre"),
     ("LAION-2B",          "vitb32_laion2b",  P + "2026-08-30_negfam_laion2b/e2/", "pre"),
     ("SigLIP B/16",       "vitb16_siglip",   P + "2026-08-30_negfam_siglip_b16/e2/", "pre"),
     ("CoN-CLIP",          "conclip",         P + "2026-08-30_negfam_conclip/e2/", "ft"),
     ("NegCLIP",           "negclip",         P + "2026-08-30_negfam_negclip/e2/", "ft"),
     ("NegCLIP-NegFull",   "negclip_negfull", P + "2026-08-30_negfam_negclip_negfull/e2/", "ft"),
     ("CLIP-NegFull",      "clip_negfull",    P + "2026-08-30_negfam_clip_negfull/e2/", "ft")]

CHANCE_MCQ = 25.0      # four-choice
CHANCE_PAIR = 25.0     # one of four orderings puts the right score first
CHANCE_GROUP = 100 / 6  # 16.67%, one of C(4,2) orderings

# Reference-palette categorical slots 1, 2 and 7. Validated --pairs all on the light
# surface: worst CVD dE 13.0, worst normal-vision dE 16.3, all three >= 3:1 contrast.
# Slot 3 (aqua) was the first choice and fails the contrast check against this surface.
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"


def parse_log(run):
    """Pull the 'Eval Epoch' metric lines out of out.log."""
    txt = open(os.path.join(OUT, run, "out.log"), encoding="utf-8", errors="replace").read()
    return {f"{m.group(1)}-{m.group(2)}": float(m.group(3))
            for m in re.finditer(r"(coco-mcq|voc2007-mcq)-(\w+_accuracy):\s*([0-9.]+)", txt)}


def collect():
    rows = []
    for name, run, e2, fam in M:
        g = parse_log(run)
        p = pd.read_csv(e2 + "e2_per_pair_decomposition.csv")
        larger = np.maximum(p.abs_alpha, p.abs_beta)
        rows.append(dict(
            model=name, family=fam,
            # The three Winoground conditions, each a direct inequality on the coefficients.
            caption=(p.gamma > p.abs_alpha).mean() * 100,
            image=(p.gamma > p.abs_beta).mean() * 100,
            group=(p.gamma > larger).mean() * 100,
            coco=g.get("coco-mcq-total_accuracy", np.nan) * 100,
            coco_neg=g.get("coco-mcq-negative_accuracy", np.nan) * 100,
            voc=g.get("voc2007-mcq-total_accuracy", np.nan) * 100,
        ))
    return pd.DataFrame(rows)


def render(d, out_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK_3)

    # ── (a) the dissociation ────────────────────────────────────────────
    band = ax1.axhspan(0, CHANCE_GROUP, color=INK_3, alpha=0.10, zorder=0)
    ax1.axhline(CHANCE_GROUP, color=INK_3, ls="--", lw=1.0, zorder=1)
    ax1.axvline(CHANCE_MCQ, color=INK_3, ls=":", lw=1.0, zorder=1)

    for fam, color, marker, label in (("pre", BLUE, "o", "사전학습 백본"),
                                      ("ft", ORANGE, "^", "부정 특화 미세조정")):
        s = d[d.family == fam]
        ax1.scatter(s.coco, s.group, s=110, c=color, marker=marker,
                    edgecolors=SURFACE, linewidths=2.0, zorder=3, label=label)

    # Selective direct labels: the two ends of the horizontal spread plus the
    # fine-tuned models, which are the point of the panel. The rightmost two are
    # anchored right so they stay inside the axes.
    right_anchored = ("NegCLIP-NegFull", "CLIP-NegFull")
    for _, r in d.iterrows():
        if r.model in right_anchored + ("CoN-CLIP", "ViT-B/32 (OpenAI)", "NegCLIP"):
            right = r.model in right_anchored
            ax1.annotate(r.model, (r.coco, r.group), textcoords="offset points",
                         xytext=(-8, 7) if right else (8, 5),
                         ha="right" if right else "left",
                         fontsize=8.5, color=INK_2)

    ax1.text(22.4, CHANCE_GROUP - 0.5, "우연 16.67%",
             fontsize=8.5, color=INK_3, va="top")
    ax1.text(CHANCE_MCQ + 0.7, 18.6, "우연 25%", fontsize=8.5, color=INK_3, va="top")
    ax1.set_xlabel("NegBench COCO MCQ 정확도 (%)", fontsize=11, color=INK)
    ax1.set_ylabel("통제된 2×2 group score (%)", fontsize=11, color=INK)
    ax1.set_title("(a) 벤치마크 성능은 31pp 움직이는데 group score는 5pp 안에 갇혀 있다",
                  fontsize=11, fontweight="bold", color=INK, loc="left")
    ax1.set_xlim(22, 59)
    ax1.set_ylim(0, 19)
    ax1.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK_2)
    ax1.grid(axis="y", ls="-", lw=0.6, color=INK_3, alpha=0.25, zorder=0)
    ax1.set_axisbelow(True)

    # ── (b) the group score split into its two conditions ───────────────
    order = d.sort_values("coco", ascending=True).reset_index(drop=True)
    y = np.arange(len(order))
    specs = [("caption", VIOLET, "s", r"캡션 선택  $\gamma>|\alpha|$"),
             ("image", ORANGE, "D", r"이미지 선택  $\gamma>|\beta|$"),
             ("group", BLUE, "o", r"group  $\gamma>\max(|\alpha|,|\beta|)$")]

    ax2.axvline(CHANCE_GROUP, color=INK_3, ls="--", lw=1.0, zorder=1)
    ax2.axvline(CHANCE_PAIR, color=INK_3, ls=":", lw=1.0, zorder=1)
    # A hairline per model ties its three markers together, so a row reads as one
    # model rather than three unrelated points at the same height.
    for i, r in order.iterrows():
        lo, hi = min(r.group, r.caption, r.image), max(r.group, r.caption, r.image)
        ax2.plot([lo, hi], [i, i], color=INK_3, lw=0.8, alpha=0.45, zorder=2)
    for col, color, marker, label in specs:
        ax2.scatter(order[col], y, s=78, c=color, marker=marker,
                    edgecolors=SURFACE, linewidths=1.6, zorder=3, label=label)

    ax2.set_yticks(y)
    ax2.set_yticklabels(order.model, fontsize=9, color=INK_2)
    ax2.set_xlabel("2×2 판정 정확도 (%)", fontsize=11, color=INK)
    ax2.set_title("(b) 세 판정 어느 것도 우연을 넘지 못한다 (9/9)",
                  fontsize=11, fontweight="bold", color=INK, loc="left")
    top = len(order) - 0.12
    ax2.text(CHANCE_GROUP - 0.4, top, "group 우연 16.67%", fontsize=8, color=INK_3,
             ha="right", va="top")
    ax2.text(CHANCE_PAIR + 0.4, top, "캡션·이미지 우연 25%", fontsize=8, color=INK_3,
             va="top", ha="right")
    ax2.set_xlim(0, 30)
    ax2.set_ylim(-1.5, len(order) + 0.15)
    ax2.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    ax2.grid(axis="x", ls="-", lw=0.6, color=INK_3, alpha=0.25, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    d = collect()
    pd.set_option("display.width", 200)
    print(d.round(2).to_string(index=False))

    csv_path = "paper_figures/fig_audit_data.csv"
    d.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    render(d, "paper_figures/fig_audit.png")

    print("\n  --- invariants ---")
    print(f"  models above group chance ({CHANCE_GROUP:.2f}%): "
          f"{(d.group > CHANCE_GROUP).sum()} / {len(d)}")
    print(f"  models above caption chance (25%): {(d.caption > 25).sum()} / {len(d)}")
    print(f"  models above image chance (25%):   {(d.image > 25).sum()} / {len(d)}")
    print(f"  COCO MCQ range: {d.coco.min():.2f} – {d.coco.max():.2f} "
          f"({d.coco.max() - d.coco.min():.2f} pp spread)")
    print(f"  group range:    {d.group.min():.2f} – {d.group.max():.2f} "
          f"({d.group.max() - d.group.min():.2f} pp spread)")
    r = d[["coco", "group"]].corr().iloc[0, 1]
    print(f"  Pearson r(COCO MCQ, group) = {r:+.3f}  (n = {len(d)})")
