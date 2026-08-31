import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np, json

S = json.load(open("logs/evaluation/01_paper/2026-08-30_i2_single_w_vitb32/single_w_summary.json"))
F = {f["family"]: f for f in S["families"]}
ORDER = ["identity", "diagonal", "lowrank_1", "lowrank_2", "lowrank_4",
         "lowrank_8", "lowrank_16", "lowrank_32", "full"]
LABEL = {"identity": "항등\n(코사인)", "diagonal": "대각\ndiag(w)", "full": "완전\nW"}
for r in (1, 2, 4, 8, 16, 32): LABEL[f"lowrank_{r}"] = f"저계수\nr={r}"

x = np.arange(len(ORDER))
ins = [F[k]["in_sample_acc_pct"] for k in ORDER]
oof = [F[k]["oof_pooled_acc_pct"] for k in ORDER]
lo  = [F[k]["oof_concept_ci_lower"] for k in ORDER]
hi  = [F[k]["oof_concept_ci_upper"] for k in ORDER]
par = [F[k]["n_trainable_params"] for k in ORDER]
CH = S["chance_pct"]

fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.axhspan(0, CH, color="#c44e52", alpha=.06, zorder=0)
ax.axhline(CH, color="#8b1a1a", ls="--", lw=2, zorder=2)
ax.axhline(67.43, color="#2ca02c", ls="-.", lw=1.8, zorder=2)

ax.fill_between(x, lo, hi, color="#c44e52", alpha=.18, zorder=2, label="홀드아웃 개념 단위 95% CI")
ax.plot(x, ins, "o-", color="#4c72b0", lw=2.2, ms=7, zorder=4,
        label="학습에 쓰인 개념 (in-sample)")
ax.plot(x, oof, "s-", color="#c44e52", lw=2.6, ms=8, zorder=5,
        label="학습에 없던 개념 (홀드아웃)")

ax.text(8.35, CH + 1.4, f"우연 {CH:.2f}%", color="#8b1a1a", fontsize=10,
        fontweight="bold", ha="right")
ax.text(8.35, 62.5, "쌍별 사영 상한 67.43%", color="#1a6b2a", fontsize=10,
        fontweight="bold", ha="right")

best = int(np.argmax(oof))
ax.annotate(f"최고 {oof[best]:.1f}%\n(우연의 1.4배, 코사인의 26배)\n그러나 CI 하한 {lo[best]:.1f}% < 우연",
            xy=(x[best], oof[best]), xytext=(x[best] + .5, 40),
            fontsize=9.5, ha="left", linespacing=1.5,
            arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.6))
ax.annotate("자유도가 커질수록\n홀드아웃은 단조 감소",
            xy=(8, oof[8]), xytext=(6.15, 3.5), fontsize=9.5, ha="left", linespacing=1.5,
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))

ax.set_xticks(x)
ax.set_xticklabels([f"{LABEL[k]}\n{p:,}p" for k, p in zip(ORDER, par)], fontsize=8.8)
ax.set_ylabel("2×2 동시 정답률 (%)", fontsize=11)
ax.set_xlabel("공유 W 하나의 자유도  (학습 파라미터 수)", fontsize=11)
ax.set_title("단일 W는 보지 못한 개념에서 우연을 인증 가능하게 넘지 못한다",
             fontsize=12.5, fontweight="bold")
ax.set_ylim(0, 84); ax.set_xlim(-.5, 8.5)
ax.grid(True, axis="y", ls="--", alpha=.35); ax.set_axisbelow(True)
ax.legend(fontsize=9.5, loc="upper left", framealpha=.95)
plt.tight_layout()
plt.savefig("paper_figures/fig3_single_w.png", dpi=300, bbox_inches="tight")
print("saved; best =", ORDER[best], oof[best])
