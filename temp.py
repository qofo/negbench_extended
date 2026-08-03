import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# 1. 데이터 로드
csv_path = "logs/evaluation/coco_val_mcq_top100_paired/image_text_similarity.csv"
df = pd.read_csv(csv_path)

# 2. 통계 지표 계산 (Pearson r & Spearman rho)
pos_sim = df["sim_image_pos"]
neg_sim = df["sim_image_neg"]

p_r, p_val_r = pearsonr(pos_sim, neg_sim)
s_r, s_val_r = spearmanr(pos_sim, neg_sim)

# 3. 그래프 스타일링 및 렌더링
plt.figure(figsize=(8, 7), dpi=130)

in_img = df[df["object_in_image"] == True]
not_in_img = df[df["object_in_image"] == False]

# 산점도 플롯: 객체 존재 (파란 동그라미) vs 객체 부재 (분홍 세모)
plt.scatter(
    in_img["sim_image_pos"], in_img["sim_image_neg"],
    c="#52a0fd", label="Object IN image", alpha=0.75,
    edgecolors="#0055c4", linewidth=0.5, s=35
)
plt.scatter(
    not_in_img["sim_image_pos"], not_in_img["sim_image_neg"],
    c="#e0667e", label="Object NOT in image", marker="^", alpha=0.75,
    edgecolors="#942337", linewidth=0.5, s=35
)

# y=x 대각선 (완전 상관관계 기준선)
lims = [0.09, 0.38]
plt.plot(lims, lims, "--", color="gray", alpha=0.7, label="y=x (perfect correlation)")

# 축 라벨 및 타이틀
plt.xlabel('cos_sim(image, "There is A")', fontsize=12)
plt.ylabel('cos_sim(image, "There is no A")', fontsize=12)
plt.title(
    f"Image-Text Similarity: Positive vs Negative\n"
    f"Pearson r={p_r:.3f} (p={p_val_r:.1e}), Spearman ρ={s_r:.3f} (p={s_val_r:.1e})",
    fontsize=13, fontweight="bold"
)

plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=10, loc="upper left")
plt.tight_layout()

# 4. 이미지 저장
output_path = "image_text_correlation.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"그래프 저장이 완료되었습니다: {output_path}")
