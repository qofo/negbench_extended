import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

P = "logs/evaluation/01_paper/"
SRC = [
 ("OpenAI CLIP","A", P+"2026-08-28_r6_main_effect_ablation_33concepts/e2_per_concept_decomposition.csv"),
 ("ViT-B/16","A",     P+"2026-08-30_r8_vitb16/e2_hadamard_decomposition/e2_per_concept_decomposition.csv"),
 ("ViT-L/14","A",     P+"2026-08-30_r8_vitl14/e2_hadamard_decomposition/e2_per_concept_decomposition.csv"),
 ("LAION-2B","B",     P+"2026-08-30_negfam_laion2b/e2/e2_per_concept_decomposition.csv"),
 ("SigLIP B/16","B",  P+"2026-08-30_negfam_siglip_b16/e2/e2_per_concept_decomposition.csv"),
 ("CoN-CLIP","C",     P+"2026-08-30_negfam_conclip/e2/e2_per_concept_decomposition.csv"),
 ("NegCLIP","C",      P+"2026-08-30_negfam_negclip/e2/e2_per_concept_decomposition.csv"),
 ("NegCLIP-NegFull","C", P+"2026-08-30_negfam_negclip_negfull/e2/e2_per_concept_decomposition.csv"),
 ("CLIP-NegFull","C", P+"2026-08-30_negfam_clip_negfull/e2/e2_per_concept_decomposition.csv"),
]
FAM = {"A": ("아키텍처 3종", "#4c72b0"),
       "B": ("학습 데이터 · 목적함수", "#55a868"),
       "C": ("부정 특화 미세조정 4종", "#c44e52")}
FLOOR = 3e-4
rows = [(nm, fam, pd.read_csv(f)) for nm, fam, f in SRC]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.1),
                              gridspec_kw={"width_ratios": [1.28, 1.0]})

# ---- (a) ratio strip plot, 9 models --------------------------------------
rng = np.random.default_rng(0)
n = len(rows); n_neg_tot = 0
seen = set()
for i, (nm, fam, d) in enumerate(rows):
    y0 = n - 1 - i                                   # first model on top
    thr = np.maximum(d.abs_alpha_mean, d.abs_beta_mean)
    ratio = (d.gamma_mean / thr).values
    n_neg_tot += int((ratio <= 0).sum())
    plotted = np.clip(ratio, FLOOR, None)
    lab = FAM[fam][0] if fam not in seen else None; seen.add(fam)
    ax.scatter(plotted, y0 + rng.uniform(-0.17, 0.17, len(plotted)), s=34,
               facecolor=FAM[fam][1], edgecolor="black", lw=.45, alpha=.72,
               zorder=3, label=lab)
    med = float(np.median(ratio))
    ax.plot([med, med], [y0 - 0.32, y0 + 0.32], color="black", lw=2.6, zorder=4)
    ax.text(med * 1.13, y0 + 0.30, f"{med:.3f}", ha="left", va="center",
            fontsize=8.8, fontweight="bold")

for b in (2.5, 4.5):                                  # family separators
    ax.axhline(b, color="#aaa", lw=.9, ls=":", zorder=1)
ax.axvline(1.0, color="#8b1a1a", ls="--", lw=2, zorder=2)
ax.axvspan(1.0, 4, color="#2ca02c", alpha=.07, zorder=0)
ax.text(1.09, n - 0.35, "성공 문턱\nγ = max(|α|,|β|)", color="#8b1a1a", fontsize=10,
        va="top", fontweight="bold", linespacing=1.5)
ax.set_xscale("log"); ax.set_xlim(2e-4, 4); ax.set_ylim(-0.85, n - 0.05)
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10)
ax.set_xlabel("γ / max(|α|, |β|)      — 1.0을 넘어야 부정 검색이 성공한다", fontsize=11)
ax.set_title("(a) 개념-모델 조합 297개(33개 × 9개 모델) 중 문턱을 넘은 것은 0개",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", ls="--", alpha=.35); ax.set_axisbelow(True)
ax.text(FLOOR, -0.70, f"← 음수 γ {n_neg_tot}개는 좌단에 고정", fontsize=8.5, color="#555")
ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.075, 0.03),
          framealpha=.95, title="모델 계열", title_fontsize=9, markerscale=1.3)

# ---- (b) three coefficients, log scale -----------------------------------
x = np.arange(n); w = 0.27
series = [("|α|  텍스트 주효과", "abs_alpha_mean", "#4c72b0"),
          ("|β|  이미지 주효과", "abs_beta_mean", "#dd8452"),
          ("γ   교차항",         "gamma_mean",    "#55a868")]
neg_alpha = [bool(r[2]["alpha_mean"].mean() < 0) for r in rows]
for i, (lab, col, c) in enumerate(series):
    vals = [r[2][col].mean() * 1000 for r in rows]
    hatch = ["///" if (i == 0 and na) else "" for na in neg_alpha]
    for j, v in enumerate(vals):
        ax2.bar(x[j] + (i - 1) * w, v, w, color=c, edgecolor="black", lw=.5,
                hatch=hatch[j], zorder=3, label=lab if j == 0 else None)
ax2.set_yscale("log"); ax2.set_ylim(0.35, 130)
for j, (nm, fam, d) in enumerate(rows):
    ax2.text(j, 46, f"{d.abs_alpha_mean.mean()/d.gamma_mean.mean():.1f}배",
             ha="center", fontsize=8.6, fontweight="bold", color="#333")
ax2.text(n - 0.5, 80, "|α| / γ", ha="right", fontsize=9, color="#333",
         fontweight="bold")
for b in (2.5, 4.5):
    ax2.axvline(b, color="#aaa", lw=.9, ls=":", zorder=1)
ax2.set_xticks(x)
ax2.set_xticklabels([r[0] for r in rows], fontsize=9, rotation=32, ha="right")
ax2.set_ylabel("계수 × 1000  (로그 축)", fontsize=11)
ax2.set_title("(b) 값은 모델마다 움직이고, 교차항은 언제나 꼴찌다\n"
              "빗금친 |α| = 부호가 음수 (부정 캡션을 더 높게 채점)",
              fontsize=12, fontweight="bold")
ax2.grid(True, axis="y", ls="--", alpha=.35); ax2.set_axisbelow(True)
ax2.legend(fontsize=9.2, loc="upper center", ncol=3, framealpha=.95,
           columnspacing=1.0, handlelength=1.3)
plt.tight_layout()
plt.savefig("paper_figures/fig2_coefficients.png", dpi=300, bbox_inches="tight")
print("saved", n_neg_tot)
