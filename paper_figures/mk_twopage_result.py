"""Figure 2 of the two-page version -- the whole empirical claim in one panel.

(a) the three coefficients, showing the interaction is the smallest by a factor of five;
(b) the ladder from cosine to two decoded bits to oracle, showing what the same embeddings
    support once the scoring rule changes.

Run from the repo root:
    python paper_figures/mk_twopage_result.py
"""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E2 = "logs/evaluation/01_paper/2026-08-31_negfam_e2_abswap/vitb32_openai/"
CEIL = "logs/evaluation/01_paper/2026-08-31_binary_ceiling/vitb32_openai/"
CEIL_NB = "logs/evaluation/01_paper/2026-08-31_binary_ceiling/vitb32_openai_no_bias/"
CTRL = "logs/evaluation/01_paper/2026-09-01_binary_ceiling_controls/"
SINGLE_W = "logs/evaluation/01_paper/2026-09-01_single_w_abswap/"
CHANCE = 100 / 6

c = pd.read_csv(E2 + "e2_per_concept_decomposition.csv")
s_bias = json.load(open(CEIL + "binary_ceiling_summary.json"))["pooled"]
s_nb = json.load(open(CEIL_NB + "binary_ceiling_summary.json"))["pooled"]
s_ct = json.load(open(CTRL + "binary_ceiling_summary.json"))["pooled"]
fam = {f["family"]: f for f in json.load(open(SINGLE_W + "single_w_summary.json"))["families"]}

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))

# ---- (a) the three coefficients
ax = axes[0]
vals = [c.abs_alpha_mean.values, c.abs_beta_mean.values, c.gamma_mean.values]
labels = [r"$|\alpha|$" "\n텍스트 주효과", r"$|\beta|$" "\n이미지 주효과", r"$\gamma$" "\n교차항"]
cols = ["#c44e52", "#4c72b0", "#2a9d8f"]
bp = ax.boxplot(vals, labels=labels, widths=0.55, patch_artist=True, showfliers=False)
for patch, col in zip(bp["boxes"], cols):
    patch.set_facecolor(col); patch.set_alpha(0.45); patch.set_edgecolor(col)
for med in bp["medians"]:
    med.set_color("#222"); med.set_linewidth(1.6)
for i, (v, col) in enumerate(zip(vals, cols)):
    ax.scatter(np.random.default_rng(0).normal(i + 1, 0.055, len(v)), v,
               s=11, color=col, alpha=0.55, zorder=3, linewidths=0)
    ax.text(i + 1, v.max() * 1.06, f"{v.mean():.5f}", fontsize=9.5, color=col,
            va="bottom", ha="center", fontweight="bold")
ax.set_ylabel("계수 크기 (개념 단위)", fontsize=10)
ax.set_title("(a) 교차항이 지배 주효과의 1/5.2\n42개 개념 2,480쌍 · ViT-B/32", fontsize=11)
ax.set_ylim(0, max(v.max() for v in vals) * 1.22)
ax.grid(axis="y", alpha=0.25)

# ---- (b) the ladder
ax = axes[1]
rungs = [("무작위 외적\n(대조)", 100 * s_ct["random_outer"], "#c9ccd1"),
         ("코사인\n$v\\cdot t$", 100 * s_bias["cosine"], "#8d99ae"),
         ("두 비트 합성\n$f(\\hat a,\\hat b)$\n(개념별)", 100 * s_bias["binary"], "#2a9d8f"),
         ("공유 $W$, rank 32\n(미지 개념)", fam["lowrank_32"]["oof_pooled_acc_pct"], "#e9c46a"),
         ("oracle 비트\n$f(a,b)$", 100 * s_bias["oracle"], "#264653")]
x = np.arange(len(rungs))
bars = ax.bar(x, [r[1] for r in rungs], color=[r[2] for r in rungs], width=0.62)
ax.axhline(CHANCE, color="crimson", ls="--", lw=1.5)
ax.text(-0.45, CHANCE + 2.8, f"교환 가능 기준 {CHANCE:.2f}%", color="crimson", fontsize=8.5, ha="left")
null_v = 100 * s_ct["crossconcept_null"]
ax.axhline(null_v, color="#1d3557", ls=":", lw=1.5)
ax.text(1.55, null_v + 2.8, f"결합 없는 귀무 {null_v:.2f}%", color="#1d3557", fontsize=8.5, ha="left")
for b, r in zip(bars, rungs):
    ax.text(b.get_x() + b.get_width() / 2, r[1] + 2.2, f"{r[1]:.2f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rungs], fontsize=7.8)
ax.set_ylabel("2×2 정답률 (%)", fontsize=10); ax.set_ylim(0, 112)
ax.set_title("(b) 같은 임베딩, 채점 규칙만 교체", fontsize=11)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig("paper_figures/fig_twopage_result.png", dpi=170)
print(f"coefficients: |a|={c.abs_alpha_mean.mean():.5f} |b|={c.abs_beta_mean.mean():.5f} "
      f"g={c.gamma_mean.mean():.5f}")
print(f"ladder: rand {100*s_ct['random_outer']:.2f} | cos {100*s_bias['cosine']:.2f} | "
      f"bits {100*s_bias['binary']:.2f} | sharedW32 {fam['lowrank_32']['oof_pooled_acc_pct']:.2f} | "
      f"oracle {100*s_bias['oracle']:.2f} | null {100*s_ct['crossconcept_null']:.2f}")
print("saved paper_figures/fig_twopage_result.png")
