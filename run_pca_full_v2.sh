#!/bin/bash

# ==============================================================================
# Script: run_pca_full_v2.sh
# Purpose: CLIP Negation Analysis using improved COCO_val_full_paired_v2.csv
# Dataset: COCO_val_full_paired_v2.csv (29,631 paired captions, diverse templates)
# Output Directory: logs/evaluation/coco_val_full_paired_v2
# ==============================================================================

PAIRED_CSV="COCO_val_full_paired_v2.csv"
OUTPUT_DIR="logs/evaluation/coco_val_full_paired_v2"

MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"
MAX_SAMPLES=60000   # Process all 29,631 pairs
BATCH_SIZE=256

# 이미지 경로 (이미지가 저장되어 있는 서버에서 실행 시 설정)
IMAGE_ROOT=""

echo "=========================================================="
echo "  Starting Full COCO Dataset CLIP Negation Analysis (v2)"
echo "  Input CSV   : ${PAIRED_CSV}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "  Target Token: ${TARGET_TOKEN}"
echo "  Max Samples : ${MAX_SAMPLES}"
echo "  Image Root  : $([ -z \"${IMAGE_ROOT}\" ] && echo 'SKIP (no image_root)' || echo ${IMAGE_ROOT})"
echo "=========================================================="

CMD="python -m benchmarks.src.analysis.run_analysis \
    --csv_path ${PAIRED_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --target_token ${TARGET_TOKEN} \
    --max_samples ${MAX_SAMPLES} \
    --batch_size ${BATCH_SIZE}"

if [ -n "${IMAGE_ROOT}" ]; then
    CMD="${CMD} --image_root ${IMAGE_ROOT}"
fi

eval ${CMD}

echo "=========================================================="
echo "Full Dataset Analysis (v2) Complete!"
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "  Output files:"
echo "    - pca_grid_${TARGET_TOKEN}.png"
echo "    - pca_report_${TARGET_TOKEN}.txt"
echo "    - cosine_similarity_by_layer_${TARGET_TOKEN}.csv"
echo "    - cosine_similarity_pairs_${TARGET_TOKEN}.csv"
echo "    - cosine_similarity_distribution_${TARGET_TOKEN}.png"
echo "    - cosine_similarity_final_histogram_${TARGET_TOKEN}.png"
echo "    - cosine_similarity_by_group_${TARGET_TOKEN}.png"
echo "=========================================================="
