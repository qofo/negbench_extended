import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv("paper_figures/fig4_data.csv")
GROUP = {"OpenAI CLIP":"아키텍처", "ViT-B/16":"아키텍처", "ViT-L/14":"아키텍처",
         "LAION-2B":"데이터·목적함수", "SigLIP B/16":"데이터·목적함수",
         "NegCLIP":"부정 미세조정", "CoN-CLIP":"부정 미세조정",
         "CLIP-NegFull":"부정 미세조정", "NegCLIP-NegFull":"부정 미세조정"}
COL = {"아키텍처":"#4c72b0", "데이터·목적함수":"#55a868", "부정 미세조정":"#c44e52"}
MRK = {"아키텍처":"o", "데이터·목적함수":"^", "부정 미세조정":"s"}
df["grp"] = df["name"].map(GROUP)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
OFF = {
 "ratio": {"OpenAI CLIP":(-58,-4), "CoN-CLIP":(6,-14), "ViT-L/14":(-14,8),
           "LAION-2B":(-20,9), "SigLIP B/16":(4,-15), "ViT-B/16":(5,7),
           "NegCLIP-NegFull":(-30,-17), "NegCLIP":(6,6), "CLIP-NegFull":(-76,7)},
 "auc":   {"OpenAI CLIP":(7,-4), "CoN-CLIP":(-52,4), "ViT-L/14":(9,-14),
           "LAION-2B":(-22,9), "SigLIP B/16":(6,-4), "ViT-B/16":(6,4),
           "NegCLIP-NegFull":(6,6), "NegCLIP":(-20,10), "CLIP-NegFull":(4,-14)},
}
SPECS = [("ratio", "γ / max(|α|, |β|)      — 본 연구가 제안하는 지표",
          "(a) 제안 지표는 예측한다", axes[0]),
         ("auc", "E1 최소쌍 존재 탐지 AUC      — \"정보가 있는가\"를 재는 통계량",
          "(b) 정보 존재 지표는 예측하지 못한다", axes[1])]
for col, xlab, title, ax in SPECS:
    x, y = df[col].values, df["acc"].values
    r, pr = stats.pearsonr(x, y); s, ps = stats.spearmanr(x, y)
    xs = np.linspace(x.min()*0.92, x.max()*1.08, 50)
    k, b = np.polyfit(x, y, 1)
    ax.plot(xs, k*xs+b, "-", color="#555", lw=1.6, alpha=.55, zorder=1)
    for g in COL:
        m = df["grp"] == g
        ax.scatter(x[m.values], y[m.values], s=95, marker=MRK[g], color=COL[g],
                   edgecolor="black", lw=.8, zorder=3, label=g if col == "ratio" else None)
    for xi, yi, nm in zip(x, y, df["name"]):
        ax.annotate(nm, (xi, yi), textcoords="offset points",
                    xytext=OFF[col].get(nm, (7, 5)), fontsize=8.2)
    ax.axhline(100/6, color="#8b1a1a", ls="--", lw=1.8, zorder=2)
    ax.set_xlabel(xlab, fontsize=10.5)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.grid(True, ls="--", alpha=.35); ax.set_axisbelow(True)
    box = "Pearson r = %+.3f  (p = %.3f)\nSpearman ρ = %+.3f  (p = %.3f)" % (r, pr, s, ps)
    ax.text(.03, .97, box, transform=ax.transAxes, fontsize=10, va="top",
            fontweight="bold" if col == "ratio" else "normal",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff7d6" if col=="ratio" else "#eeeeee",
                      ec="#999", alpha=.95))
axes[0].set_ylabel("2×2 동시 정답률 (%)", fontsize=11)
axes[0].set_ylim(-1.2, 19)
axes[0].text(0.208, 100/6 - 1.2, "우연 16.67%", color="#8b1a1a", fontsize=9.5,
             ha="right", fontweight="bold")
axes[0].legend(fontsize=9.5, loc="lower right", title="모델 계열", title_fontsize=9.5)
fig.suptitle("9개 모델 — 아키텍처 3종 · 학습 데이터 3종 · 목적함수 2종 · 부정 미세조정 4종",
             fontsize=11, y=1.005, color="#444")
plt.tight_layout()
plt.savefig("paper_figures/fig4_metric_predicts.png", dpi=300, bbox_inches="tight")
print("saved")
