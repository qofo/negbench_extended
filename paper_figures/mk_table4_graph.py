"""표 4를 그래프로: 항등 보존 잔차 W = I + UV^T 를 검색 대조(InfoNCE) 목적함수로 rank별 학습.
항등 유지(c=0.1, ||W-I||_F≈1-2) 대 표류 허용(c=0). 각 조건에서 2x2 판정 정확도와 COCO T2I R@1.

Data: logs/evaluation/01_paper/2026-09-10_infonce_idinit/final_sweep.csv
Run from repo root:  python paper_figures/mk_table4_graph.py
"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

df = pd.read_csv("logs/evaluation/01_paper/2026-09-10_infonce_idinit/final_sweep.csv")
rep = json.load(open("logs/evaluation/01_paper/2026-09-10_infonce_idinit/final_report.json"))
cos = rep["cosine"]
CHANCE = 100 / 6
OUT = "paper_figures/fig_table4_graph.png"

keep = df[df.c == 0.1].sort_values("rank")      # 항등 유지
drift = df[df.c == 0.0].sort_values("rank")     # 표류 허용
BLUE, RED = "#4c72b0", "#c44e52"

fig, ax = plt.subplots(figsize=(7.3, 4.7))
fig.subplots_adjust(left=0.09, right=0.99, top=0.885, bottom=0.125)

ax.axhline(CHANCE, color="#888", ls=":", lw=1.1)
ax.text(1.0, CHANCE + 1.0, f"우연 {CHANCE:.2f}%", fontsize=8.3, color="#666")
ax.axhline(cos["std_r1"], color="#999", ls="-", lw=1.0)
ax.text(600, cos["std_r1"] - 3.2, f"코사인 R@1 {cos['std_r1']:.1f}%", fontsize=8.3, color="#777", ha="right")
ax.axhline(cos["acc_2x2"], color="#999", ls="-", lw=1.0)
ax.text(600, cos["acc_2x2"] + 0.7, f"코사인 2×2 {cos['acc_2x2']:.2f}%", fontsize=8.3, color="#777", ha="right")

ax.plot(keep["rank"], keep.acc_2x2, "o-", color=BLUE, lw=2.0, ms=6,
        label="항등 유지 · 2×2 판정")
ax.plot(keep["rank"], keep.std_r1, "s--", color=BLUE, lw=1.7, ms=5, mfc="white",
        label="항등 유지 · 검색 R@1")
ax.plot(drift["rank"], drift.acc_2x2, "o-", color=RED, lw=2.0, ms=6,
        label="표류 허용 · 2×2 판정")
ax.plot(drift["rank"], drift.std_r1, "s--", color=RED, lw=1.7, ms=5, mfc="white",
        label="표류 허용 · 검색 R@1")

ax.annotate("항등을 지키면 검색은 코사인 이상,\n2×2는 rank와 무관하게 정체", xy=(256, keep.acc_2x2.iloc[-2] + 0.5),
            xytext=(2.6, 25.5), fontsize=8.3, color=BLUE, ha="left",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0, connectionstyle="arc3,rad=-0.15"))
ax.annotate("표류를 허용하면 2×2는 오르나\n검색 R@1이 무너짐", xy=(340, 3.0),
            xytext=(9, 43), fontsize=8.3, color=RED, ha="left",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0, connectionstyle="arc3,rad=0.22"))

ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.minorticks_off()
ax.set_xlabel("저계수 보정의 rank $r$", fontsize=11)
ax.set_ylabel("2×2 판정 정확도 · 검색 R@1 (%)", fontsize=11)
ax.set_xlim(0.85, 620)
ax.set_ylim(-2, 52)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=8.3, loc="upper left", framealpha=0.95, ncol=2,
          bbox_to_anchor=(0.0, 1.135), borderaxespad=0)

fig.savefig(OUT, dpi=180)
print("keep  ", list(zip(keep["rank"], keep.acc_2x2.round(1), keep.std_r1.round(1))))
print("drift ", list(zip(drift["rank"], drift.acc_2x2.round(1), drift.std_r1.round(1))))
print("saved", OUT)
