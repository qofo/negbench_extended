"""Figure 3 — the cross-modal double dissociation.

alpha (text main effect) predicts the caption-selection failure axis;
beta (image main effect) predicts the T2I-retrieval failure axis; neither crosses over.
Run from the repo root:  python paper_figures/mk_external.py
"""
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

r = pd.read_csv("paper_figures/fig_retrieval_data.csv").set_index("model")
m = pd.read_csv("paper_figures/fig_external_data.csv").set_index("model")
d = r.join(m[["coco_pos", "coco_neg"]]).reset_index()
d["mcq_gap"] = d.coco_pos - d.coco_neg
d["ret_gap"] = d.drop1

FAM = {"ViT-B/32 (OpenAI)":"A","ViT-B/16":"A","ViT-L/14":"A","LAION-2B":"B",
       "SigLIP B/16":"B","CoN-CLIP":"C","NegCLIP":"C","NegCLIP-NegFull":"C","CLIP-NegFull":"C"}
COL = {"A":"#4c72b0","B":"#55a868","C":"#c44e52"}
LAB = {"A":"아키텍처 3종","B":"학습 데이터 · 목적함수","C":"부정 특화 미세조정 4종"}

# (predictor, target, matched?, title, xlabel, ylabel)
CELLS = [("alpha_signed","mcq_gap",True,
          "(a) α → 캡션 선택 축   ○ 정합",
          "α  텍스트 주효과 (부호, ×10³)", "MCQ 긍정 − 부정 정확도 (%p)"),
         ("alpha_signed","ret_gap",False,
          "(b) α → 이미지 검색 축   × 교차",
          "α  텍스트 주효과 (부호, ×10³)", "T2I 표준 − 부정 R@1 (%p)"),
         ("beta_signed","mcq_gap",False,
          "(c) β → 캡션 선택 축   × 교차",
          "β  이미지 주효과 (부호, ×10³)", "MCQ 긍정 − 부정 정확도 (%p)"),
         ("beta_signed","ret_gap",True,
          "(d) β → 이미지 검색 축   ○ 정합",
          "β  이미지 주효과 (부호, ×10³)", "T2I 표준 − 부정 R@1 (%p)")]

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6))
for ax, (xc, yc, matched, title, xlab, ylab) in zip(axes.ravel(), CELLS):
    seen = set()
    for _, row in d.iterrows():
        f = FAM[row.model]
        ax.scatter(row[xc], row[yc], s=95,
                   color=COL[f] if matched else "#b9b9b9",
                   edgecolor="black", lw=.65, zorder=3,
                   label=LAB[f] if (matched and f not in seen) else None)
        seen.add(f)
    x, y = d[xc].values, d[yc].values
    rr, pv = pearsonr(x, y); rs, ps = spearmanr(x, y)
    b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color="#8b1a1a" if matched else "#9a9a9a",
            ls="--", lw=1.6 if matched else 1.1, alpha=.85 if matched else .6, zorder=2)
    if matched:                                   # name the extreme points only
        for name in (["CoN-CLIP"] if yc == "mcq_gap" else ["SigLIP B/16", "NegCLIP-NegFull"]):
            row = d[d.model == name].iloc[0]
            ax.annotate(name, (row[xc], row[yc]), textcoords="offset points",
                        xytext=(9, -4), fontsize=8.6, color="#333")
    ax.set_xlabel(xlab, fontsize=10.5); ax.set_ylabel(ylab, fontsize=10.5)
    ax.set_title(f"{title}\nPearson r = {rr:+.3f} (p = {pv:.4f}) · ρ = {rs:+.3f}",
                 fontsize=11.2, fontweight="bold",
                 color="#111" if matched else "#777")
    ax.axhline(0, color="#666", ls=":", lw=1.0, zorder=1)
    ax.axvline(0, color="#666", ls=":", lw=1.0, zorder=1)
    ax.grid(True, ls="--", alpha=.3); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(2.0 if matched else 0.8)
        sp.set_color("#8b1a1a" if matched else "#cccccc")
axes[0, 0].legend(fontsize=8.8, loc="lower right", framealpha=.95,
                  title="모델 계열", title_fontsize=8.8)
plt.tight_layout()
plt.savefig("paper_figures/fig_external.png", dpi=300, bbox_inches="tight")
print("saved fig_external (2x2 dissociation)")
