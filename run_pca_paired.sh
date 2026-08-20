#!/bin/bash

# ==============================================================================
# CLIP 부정어 이해 분석: PCA + Cosine Similarity (Experiment 1, 2)
# 이미지-텍스트 상관관계 분석 (Experiment 3, 이미지 존재 시)
# ==============================================================================

# ── 입력 / 출력 설정 ──
PAIRED_CSV="COCO_val_mcq_top100_paired.csv"
OUTPUT_DIR="logs/pca/coco_val_mcq_top100_paired"

# ── 모델 설정 ──
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"    # 옵션: eot (기본), mean, all
MAX_SAMPLES=500
BATCH_SIZE=256

# ── 이미지 경로 (Experiment 3용, 비어있으면 Exp3 스킵) ──
# 이미지가 있는 서버에서 실행 시 아래 경로를 설정해주세요
# 예: IMAGE_ROOT="."  (CSV의 image_path가 "data/coco/images/..." 이므로)
IMAGE_ROOT=""

echo "=========================================================="
echo "  CLIP Negation Analysis"
echo "  Input    : ${PAIRED_CSV}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Model    : ${MODEL_NAME} (${PRETRAINED})"
echo "  Token    : ${TARGET_TOKEN}"
echo "  Exp 3    : $([ -z \"${IMAGE_ROOT}\" ] && echo 'SKIP (no image_root)' || echo ${IMAGE_ROOT})"
echo "=========================================================="

CMD="python -m benchmarks.src.analysis.run_analysis \
    --csv_path ${PAIRED_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --target_token ${TARGET_TOKEN} \
    --max_samples ${MAX_SAMPLES} \
    --batch_size ${BATCH_SIZE}"

# Experiment 3: 이미지 경로가 설정되어 있으면 추가
if [ -n "${IMAGE_ROOT}" ]; then
    CMD="${CMD} --image_root ${IMAGE_ROOT}"
fi

eval ${CMD}

echo "=========================================================="
echo "  Complete! Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "  Output files:"
echo "    - pca_grid_${TARGET_TOKEN}.png                        (Exp 1: PCA)"
echo "    - pca_report_${TARGET_TOKEN}.txt                      (Exp 1: PCA)"
echo "    - cosine_similarity_by_layer_${TARGET_TOKEN}.csv      (Exp 2: layer stats)"
echo "    - cosine_similarity_pairs_${TARGET_TOKEN}.csv         (Exp 2: per-pair)"
echo "    - cosine_similarity_distribution_${TARGET_TOKEN}.png  (Exp 2: box plot)"
echo "    - cosine_similarity_final_histogram_${TARGET_TOKEN}.png (Exp 2: histogram)"
echo "    - cosine_similarity_by_group_${TARGET_TOKEN}.png      (Exp 2: in/not in image)"
if [ -n "${IMAGE_ROOT}" ]; then
echo "    - image_text_similarity.csv                           (Exp 3: per-pair)"
echo "    - image_text_correlation.png                          (Exp 3: scatter)"
fi
echo "=========================================================="
