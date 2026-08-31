import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

d = pd.read_csv("paper_figures/fig5_data.csv")
d["gap"] = d.coco_pos - d.coco_neg
FAM = {"ViT-B/32 (OpenAI)":"A","ViT-B/16":"A","ViT-L/14":"A","LAION-2B":"B",
       "SigLIP B/16":"B","CoN-CLIP":"C","NegCLIP":"C","NegCLIP-NegFull":"C","CLIP-NegFull":"C"}
COL = {"A":"#4c72b0","B":"#55a868","C":"#c44e52"}
LAB = {"A":"아키텍처 3종","B":"학습 데이터 · 목적함수","C":"부정 특화 미세조정 4종"}
OFF_A = {"NegCLIP-NegFull":((7,-14),"left"), "CLIP-NegFull":((-8,6),"right"),
         "ViT-B/16":((7,-4),"left"), "LAION-2B":((8,-13),"left"),
         "SigLIP B/16":((7,-13),"left"), "CoN-CLIP":((-9,5),"right"),
         "ViT-B/32 (OpenAI)":((8,4),"left"), "ViT-L/14":((8,-12),"left")}
OFF_B = {"ViT-B/32 (OpenAI)":((-9,7),"right"), "ViT-B/16":((-9,-14),"right"),
         "ViT-L/14":((8,-4),"left"), "CLIP-NegFull":((-9,4),"right"),
         "NegCLIP-NegFull":((-9,-14),"right"), "NegCLIP":((8,-4),"left"),
         "LAION-2B":((8,4),"left"), "SigLIP B/16":((8,-12),"left"),
         "CoN-CLIP":((11,-2),"left")}

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
PANEL = [("ratio", "coco",
          "제안 지표  γ / max(|α|, |β|)", "NegBench COCO MCQ 총 정확도 (%)",
          "(a) 총 정확도는 예측되지 않는다", 25.0, "MCQ 우연 25%"),
         ("alpha_signed", "gap",
          "텍스트 주효과 α  (부호 있는 매크로 평균 × 10³)",
          "MCQ 긍정 문항 − 부정 문항 정확도 (%p)",
          "(b) 실패 축은 예측된다", 0.0, "격차 0")]

for ax, off, (xc, yc, xlab, ylab, title, hl, hll) in zip(axes, (OFF_A, OFF_B), PANEL):
    seen = set()
    for _, r in d.iterrows():
        f = FAM[r.model]
        ax.scatter(r[xc], r[yc], s=115, color=COL[f], edgecolor="black", lw=.7,
                   zorder=3, label=LAB[f] if f not in seen else None); seen.add(f)
        xy, ha = off.get(r.model, ((7, 6), "left"))
        ax.annotate(r.model, (r[xc], r[yc]), textcoords="offset points",
                    xytext=xy, ha=ha, fontsize=8.4, color="#333")
    x, y = d[xc].values, d[yc].values
    rr, pv = pearsonr(x, y); rs, ps = spearmanr(x, y)
    b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color="#8b1a1a", ls="--", lw=1.5, alpha=.8, zorder=2)
    ax.axhline(hl, color="#555", ls=":", lw=1.2, zorder=1)
    ax.set_xlabel(xlab, fontsize=11); ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(f"{title}\nPearson r = {rr:+.3f} (p = {pv:.4f}) · Spearman ρ = {rs:+.3f} (p = {ps:.4f})",
                 fontsize=11.5, fontweight="bold")
    ax.grid(True, ls="--", alpha=.35); ax.set_axisbelow(True)
    ax.text(ax.get_xlim()[1] - .02*(ax.get_xlim()[1]-ax.get_xlim()[0]), hl + 1.2,
            hll, fontsize=8.5, color="#555", ha="right")

axes[1].axvline(0, color="#555", ls=":", lw=1.2, zorder=1)
axes[1].text(0.9, -24, "α < 0  ←  |  →  α > 0", fontsize=8.6, color="#555", ha="left")
axes[0].legend(fontsize=9, loc="upper left", framealpha=.95,
               title="모델 계열", title_fontsize=9)
axes[1].set_ylim(top=axes[1].get_ylim()[1] + 8)
plt.tight_layout()
plt.savefig("paper_figures/fig5_external.png", dpi=300, bbox_inches="tight")
print("saved fig5")
