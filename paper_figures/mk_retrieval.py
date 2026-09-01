"""Figure 6 -- the T2I retrieval external check, and its coordinate dependence.

(a) In the single-object coordinate the signed image main effect predicts how much R@1 falls
    when the query is negated.
(b) The same regression on AB-swap coefficients is flat. The reason is not that the retrieval
    captions match one design and not the other -- they are affirm-and-negate sentences, the
    AB-swap shape -- but that AB-swap removes the signed main effects altogether: the between-
    model range of signed beta collapses from 7.73 units (all positive) to 1.44 (mixed sign).

Run from the repo root, after both passes of mk_retrieval_data.py:
    python paper_figures/mk_retrieval_data.py
    python paper_figures/mk_retrieval_data.py abswap
    python paper_figures/mk_retrieval.py
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import pearsonr

GROUP = {"ViT-B/32 (OpenAI)": "아키텍처", "ViT-B/16": "아키텍처", "ViT-L/14": "아키텍처",
         "LAION-2B": "데이터·목적함수", "SigLIP B/16": "데이터·목적함수",
         "NegCLIP": "부정 미세조정", "CoN-CLIP": "부정 미세조정",
         "CLIP-NegFull": "부정 미세조정", "NegCLIP-NegFull": "부정 미세조정"}
COL = {"아키텍처": "#4c72b0", "데이터·목적함수": "#55a868", "부정 미세조정": "#c44e52"}
MRK = {"아키텍처": "o", "데이터·목적함수": "^", "부정 미세조정": "s"}

six = pd.read_csv("paper_figures/fig_retrieval_data.csv")
ab = pd.read_csv("paper_figures/fig_retrieval_data_abswap.csv")
ab = ab.set_index("model").loc[six.model].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
OFF = {"CoN-CLIP": (8, -13), "NegCLIP-NegFull": (-30, -15), "CLIP-NegFull": (7, 5),
       "SigLIP B/16": (-58, 2), "LAION-2B": (7, -4), "ViT-B/32 (OpenAI)": (-30, 9),
       "ViT-B/16": (7, -11), "ViT-L/14": (7, 4), "NegCLIP": (7, -3)}

for ax, d, xlab, title in [
        (axes[0], six, r"부호 있는 $\beta$ ×10³  (단일 객체 좌표)",
         "(a) 구성이 다른 좌표 — 예측한다"),
        (axes[1], ab, r"부호 있는 $\beta$ ×10³  (AB-swap 좌표)",
         "(b) 구성이 같은 좌표 — 예측하지 못한다")]:
    x, y = d.beta_signed.values, six.drop1.values
    for _, r in d.iterrows():
        g = GROUP[r.model]
        ax.scatter(r.beta_signed, six.set_index("model").loc[r.model, "drop1"],
                   s=78, color=COL[g], marker=MRK[g], zorder=3, edgecolor="white", lw=0.8)
        dx, dy = OFF.get(r.model, (7, 4))
        ax.annotate(r.model, (r.beta_signed, six.set_index("model").loc[r.model, "drop1"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.6, color="#333")
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b * xs + a, color="#666", ls="--", lw=1.3, zorder=1)
    r, p = pearsonr(x, y)
    ax.text(0.03, 0.95, f"r = {r:+.3f}  (p = {p:.3f})", transform=ax.transAxes,
            fontsize=11, va="top", fontweight="bold",
            color="#1d3557" if p < 0.05 else "#9a031e")
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("부정 질의에서의 R@1 하락폭 (%p)", fontsize=10)
handles = [plt.Line2D([], [], marker=MRK[g], color=COL[g], ls="", ms=8, label=g) for g in COL]
axes[1].legend(handles=handles, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("paper_figures/fig_retrieval.png", dpi=170)
print("saved paper_figures/fig_retrieval.png")
