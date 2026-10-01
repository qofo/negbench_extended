"""Figure 1 for PAPER_3STAGE.md -- the 2x2 minimal pair.

Draws one real BEAF counterfactual image pair against its AB-swap caption pair and
scores the four cells with the baseline model. Unlike `mk_framework.py`, this figure
carries no factor decomposition panel: PAPER_3STAGE.md states the success condition
directly on the four similarities and never introduces alpha/beta/gamma.

Run from the repo root:
    python paper_figures/mk_3stage_fig1.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from PIL import Image

sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "benchmarks"),
                os.path.join(os.getcwd(), "benchmarks", "src")]
from benchmarks.src.analysis.model_loader import load_clip_for_eval

PAIR = "benchmarks/data/coco/images/val2014/COCO_val2014_000000008749_{}.png"
POS = "The scene features a pizza, but lacks a cup."
NEG = "The scene features a cup, but lacks a pizza."
OUT = "paper_figures/fig_3stage_1.png"
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
delta = min(S_pp, S_mm) - max(S_pm, S_mp)

fig = plt.figure(figsize=(7.6, 4.9))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

for col, (lab, sub) in enumerate([("긍정 캡션", '"…a pizza, but lacks a cup."'),
                                  ("부정 캡션", '"…a cup, but lacks a pizza."')]):
    x = 4.4 + col * 2.9
    ax.text(x, 9.05, lab, ha="center", fontsize=11, fontweight="bold")
    ax.text(x, 8.55, sub, ha="center", fontsize=7.4, color="#444")

cells = [(S_pp, 0, 0, True), (S_pm, 0, 1, False), (S_mp, 1, 0, False), (S_mm, 1, 1, True)]
names = {(0, 0): "$S_{++}$", (0, 1): "$S_{+-}$", (1, 0): "$S_{-+}$", (1, 1): "$S_{--}$"}
for val, r, c, diag in cells:
    x, y = 3.2 + c * 2.9, 5.30 - r * 2.55
    ax.add_patch(Rectangle((x, y), 2.5, 2.2, facecolor=GREEN if diag else RED,
                           alpha=0.16, edgecolor=GREEN if diag else RED, lw=1.8))
    ax.text(x + 1.25, y + 1.42, names[(r, c)], ha="center", fontsize=13,
            color=GREEN if diag else RED, fontweight="bold")
    ax.text(x + 1.25, y + 0.50, f"{val:.4f}", ha="center", fontsize=13, fontweight="bold")

for r, (im, lab) in enumerate(zip(imgs, ["pizza 있음\n(cup 제거)", "pizza 제거\n(cup 있음)"])):
    axi = fig.add_axes([0.055, 0.515 - r * 0.263, 0.195, 0.222])
    axi.imshow(im)
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_edgecolor("#333")
        sp.set_linewidth(1.4)
    axi.set_ylabel(lab, fontsize=9.4, labelpad=6)

ax.text(5.4, 1.30, "같은 장면에서 객체 하나만 제거 · 같은 단어 집합으로 결합만 교체",
        ha="center", fontsize=9.2, color="#333")
ax.add_patch(Rectangle((1.05, 0.18), 8.8, 0.82, facecolor="#f4f4f6",
                       edgecolor="#333", lw=1.3))
ax.text(5.45, 0.53,
        r"$\Delta(S) = \min(S_{++},S_{--}) - \max(S_{+-},S_{-+}) = $"
        f"{delta:+.4f}"
        f"   →   {'성공' if delta > 0 else '실패'}",
        ha="center", va="center", fontsize=10.4)

fig.savefig(OUT, dpi=170)
print(f"S++={S_pp:.4f} S+-={S_pm:.4f} S-+={S_mp:.4f} S--={S_mm:.4f}  delta={delta:+.5f}")
print(f"saved {OUT}")
