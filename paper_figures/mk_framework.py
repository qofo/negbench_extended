"""Figure 1 -- the 2x2 minimal pair and its factor decomposition.

Draws one real BEAF counterfactual pair against its AB-swap caption pair, scores the four
cells with the baseline model, and lays the four similarities beside the coordinate change
they are rewritten into. The bottom line states the success condition the rest of the paper
tests.

Run from the repo root:
    python paper_figures/mk_framework.py
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import torch
from PIL import Image

import sys, os
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "benchmarks"),
                os.path.join(os.getcwd(), "benchmarks", "src")]
from benchmarks.src.analysis.model_loader import load_clip_for_eval

PAIR = "benchmarks/data/coco/images/val2014/COCO_val2014_000000008749_{}.png"
POS = "The scene features a pizza, but lacks a cup."
NEG = "The scene features a cup, but lacks a pizza."
GREEN, RED = "#2a9d8f", "#e76f51"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, tokenizer = load_clip_for_eval("ViT-B-32", "openai", device)

imgs = [Image.open(PAIR.format(k)).convert("RGB") for k in ("01", "03")]
with torch.no_grad():
    v = model.encode_image(torch.stack([preprocess(i) for i in imgs]).to(device)).float()
    t = model.encode_text(tokenizer([POS, NEG]).to(device)).float()
v = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()
t = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()

S = v @ t.T                       # rows: present / absent   cols: positive / negative
S_pp, S_pm, S_mp, S_mm = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
C = (S_pp + S_mp + S_pm + S_mm) / 4
beta = (S_pp - S_mp + S_pm - S_mm) / 4
alpha = (S_pp + S_mp - S_pm - S_mm) / 4
gamma = (S_pp - S_mp - S_pm + S_mm) / 4

fig = plt.figure(figsize=(13.6, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.10,
                      left=0.055, right=0.985, top=0.90, bottom=0.14)

# ---------------------------------------------------------------- left: the 2x2 block
axL = fig.add_subplot(gs[0, 0]); axL.axis("off")
axL.set_xlim(0, 10); axL.set_ylim(0, 10)
axL.text(5.0, 9.6, "2×2 최소쌍", ha="center", fontsize=13, fontweight="bold")

for col, (lab, sub) in enumerate([("긍정 캡션", '"…a pizza, but lacks a cup."'),
                                  ("부정 캡션", '"…a cup, but lacks a pizza."')]):
    x = 4.2 + col * 2.9
    axL.text(x, 8.75, lab, ha="center", fontsize=10.5, fontweight="bold")
    axL.text(x, 8.30, sub, ha="center", fontsize=7.2, color="#444")

cells = [(S_pp, 0, 0, True), (S_pm, 0, 1, False), (S_mp, 1, 0, False), (S_mm, 1, 1, True)]
names = {(0, 0): "$S_{++}$", (0, 1): "$S_{+-}$", (1, 0): "$S_{-+}$", (1, 1): "$S_{--}$"}
for val, r, c, diag in cells:
    x, y = 3.0 + c * 2.9, 5.15 - r * 2.55
    axL.add_patch(Rectangle((x, y), 2.5, 2.2, facecolor=GREEN if diag else RED,
                            alpha=0.16, edgecolor=GREEN if diag else RED, lw=1.8))
    axL.text(x + 1.25, y + 1.42, names[(r, c)], ha="center", fontsize=13,
             color=GREEN if diag else RED, fontweight="bold")
    axL.text(x + 1.25, y + 0.52, f"{val:.4f}", ha="center", fontsize=13, fontweight="bold")

for r, (im, lab) in enumerate(zip(imgs, ["pizza 있음\n(cup 제거)", "pizza 제거\n(cup 있음)"])):
    ax = fig.add_axes([0.040, 0.495 - r * 0.290, 0.120, 0.240])
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#333"); sp.set_linewidth(1.4)
    ax.set_ylabel(lab, fontsize=9.6, labelpad=6)

axL.text(5.0, 0.42, "같은 장면에서 객체 하나만 제거 · 같은 단어 집합으로 결합만 교체", 
         ha="center", fontsize=9.2, color="#333")

# ---------------------------------------------------------------- right: decomposition
axR = fig.add_subplot(gs[0, 1]); axR.axis("off")
axR.set_xlim(0, 10); axR.set_ylim(0, 10)
axR.text(5.0, 9.6, "요인 분해 — 가정 없는 좌표 변환", ha="center", fontsize=13, fontweight="bold")
axR.text(5.0, 8.85, r"$S_{ab} = C + a\beta + b\alpha + ab\,\gamma$",
         ha="center", fontsize=15)

rows = [("C", "전체 평균", C, [[1, 1], [1, 1]], "#8d99ae"),
        ("β", "이미지 주효과 (행 차이)", beta, [[1, 1], [-1, -1]], "#4c72b0"),
        ("α", "텍스트 주효과 (열 차이)", alpha, [[1, -1], [1, -1]], "#c44e52"),
        ("γ", "교차항 (대각 − 반대각)", gamma, [[1, -1], [-1, 1]], "#2a9d8f")]
for i, (sym, desc, val, pat, col) in enumerate(rows):
    y = 7.45 - i * 1.62
    for r in range(2):
        for c in range(2):
            axR.add_patch(Rectangle((0.35 + c * 0.46, y + 0.52 - r * 0.46), 0.44, 0.44,
                                    facecolor=col, alpha=0.85 if pat[r][c] > 0 else 0.16,
                                    edgecolor="white", lw=1.0))
    axR.text(1.72, y + 0.34, sym, fontsize=15, color=col, fontweight="bold")
    axR.text(2.45, y + 0.40, desc, fontsize=9.4, va="center")
    axR.text(9.5, y + 0.40, f"{val:+.5f}", fontsize=12, ha="right", va="center",
             fontweight="bold", color=col)

axR.add_patch(FancyArrowPatch((5.0, 1.62), (5.0, 1.15), arrowstyle="-|>",
                              mutation_scale=16, color="#333", lw=1.4))
axR.add_patch(Rectangle((0.2, 0.05), 9.6, 1.05, facecolor="#f4f4f6",
                        edgecolor="#333", lw=1.4))
axR.text(5.0, 0.76, r"성공 $\Leftrightarrow \min(S_{++},S_{--}) > \max(S_{+-},S_{-+})"
                    r"\Leftrightarrow \gamma > \max(|\alpha|,|\beta|)$",
         ha="center", fontsize=10.8)
axR.text(5.0, 0.28, f"이 쌍: γ = {gamma:.5f} vs max(|α|,|β|) = "
                    f"{max(abs(alpha), abs(beta)):.5f}   →   "
                    f"{'성공' if gamma > max(abs(alpha), abs(beta)) else '실패'}"
                    "     (무작위 기준 1/C(4,2) = 16.67%)",
         ha="center", fontsize=9.2, color="#333")

fig.savefig("paper_figures/fig_framework.png", dpi=170)
print(f"S++={S_pp:.4f} S+-={S_pm:.4f} S-+={S_mp:.4f} S--={S_mm:.4f}")
print(f"C={C:.5f} beta={beta:+.5f} alpha={alpha:+.5f} gamma={gamma:+.5f}")
print("saved paper_figures/fig_framework.png")
