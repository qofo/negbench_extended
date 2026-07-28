#!/bin/bash

# ==============================================================================
# CLIP 부정어 이해 분석: PCA + Cosine Similarity (Experiment 1, 2)
# 이미지-텍스트 상관관계 분석 (Experiment 3, 이미지 존재 시)
# ==============================================================================

# ── 입력 / 출력 설정 ──
PAIRED_CSV="benchmarks/data/images/COCO_val_full_paired.csv"
OUTPUT_DIR="logs/pca/coco_val_full_paired"

# ── 모델 설정 ──
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"    # 옵션: eot (기본), mean, all
MAX_SAMPLES=60000
BATCH_SIZE=256

# ── 이미지 경로 (Experiment 3용, 비어있으면 Exp3 스킵) ──
# 이미지가 있는 서버에서 실행 시 아래 경로를 설정해주세요
# 예: IMAGE_ROOT="."  (CSV의 image_path가 "data/coco/images/..." 이므로)
IMAGE_ROOT=benchmarks
#!/bin/bash

# ==============================================================================
# Script: run_pca_full.sh
# Purpose: CLIP Negation Analysis across ALL 5,000 COCO validation images
# Dataset: COCO_val_full_paired.csv (54,645 paired captions)
# Output Directory: logs/evaluation/coco_val_full_paired
# ==============================================================================


echo "=========================================================="
echo "  Starting Full COCO Dataset CLIP Negation Analysis"
echo "  Input CSV   : ${PAIRED_CSV}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "  Target Token: ${TARGET_TOKEN}"
echo "  Max Samples : ${MAX_SAMPLES}"
echo "  Image Root  : $([ -z \"${IMAGE_ROOT}\" ] && echo 'SKIP (no image_root)' || echo ${IMAGE_ROOT})"
echo "=========================================================="

CMD="python benchmarks/src/evaluation/pca_text_encoder.py \
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
