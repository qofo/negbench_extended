"""표류 허용(c=0) 조건의 2×2 판정 정확도만 rank축에 그린다. LABCLIP 원본 recipe(공식 구현
그대로: identity-init 완전 Linear, InfoNCE)의 2×2를 별표로 병기하고, 우연 기준 16.67%를 표시.
검색 R@1은 포함하지 않는다.

drift 곡선: logs/.../2026-09-10_infonce_idinit/final_sweep.csv  (c == 0)
LABCLIP 원본: logs/.../2026-09-10_labclip_rank_strategy_ext/rank_strategy_report.json  (A, rank 512)

Run from repo root:  python paper_figures/mk_drift_2x2_labclip.py
"""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

df = pd.read_csv("logs/evaluation/01_paper/2026-09-10_infonce_idinit/final_sweep.csv")
drift = df[df.c == 0.0].sort_values("rank")
plain = pd.read_csv("logs/evaluation/01_paper/2026-09-10_infonce_idinit_plain/final_sweep.csv").sort_values("rank")
rr = json.load(open("logs/evaluation/01_paper/2026-09-10_labclip_rank_strategy_ext/rank_strategy_report.json"))
labclip = [x for x in rr["conditions"]["A_posthoc_svd"] if x["rank"] == 512][0]["delta_s"]
CHANCE = 100 / 6
OUT = "paper_figures/fig_drift_2x2_labclip.png"

fig, ax = plt.subplots(figsize=(7.0, 4.5))
fig.subplots_adjust(left=0.10, right=0.98, top=0.965, bottom=0.135)

ax.axhline(CHANCE, color="#888", ls="--", lw=1.2)
ax.text(1.0, CHANCE + 1.2, f"random {CHANCE:.2f}%", fontsize=9, color="#666")

ax.plot(drift["rank"], drift.acc_2x2, "o-", color="#c44e52", lw=2.1, ms=6.5,
        label="Low rank bilinear transform")

ax.plot(plain["rank"], plain.acc_2x2, "^--", color="#dd8452", lw=2.0, ms=6.5,
        label="Low rank bilinear transform (random init)")

ax.scatter([512], [labclip], marker="*", s=280, color="#8c2d2d", edgecolor="white",
           lw=0.8, zorder=6, label=f"LABCLIP(full bilinear transform)")

ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.minorticks_off()
ax.set_xlabel("rank $r$", fontsize=11)
ax.set_ylabel("2×2 판정 정확도 (%)", fontsize=11)
ax.set_xlim(0.85, 640)
ax.set_ylim(0, 62)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

fig.savefig(OUT, dpi=180)
print("drift 2x2:", list(zip(drift["rank"], drift.acc_2x2.round(1))))
print("plain 2x2:", list(zip(plain["rank"], plain.acc_2x2.round(1))))
print("LABCLIP:", round(labclip, 2))
print("saved", OUT)
