#!/bin/bash
set -e

# ==============================================================================
# Script: run_analysis.sh
# Academic Pipeline Runner for Modular Representation Analysis in CLIP
# ==============================================================================

PAIRED_CSV="benchmarks/data/images/COCO_val_full_paired.csv"
if [ ! -f "${PAIRED_CSV}" ]; then
    PAIRED_CSV="COCO_val_full_paired.csv"
fi

OUTPUT_DIR="logs/analysis_modular/openai_vit_b32"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"
MAX_SAMPLES=60000
BATCH_SIZE=256

IMAGE_ROOT="benchmarks"

echo "=========================================================="
echo "  Executing Modular CLIP Negation Analysis Pipeline"
echo "  Input CSV   : ${PAIRED_CSV}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "  Batch Size  : ${BATCH_SIZE}"
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
echo "Analysis Complete! Generated experimental outputs in ${OUTPUT_DIR}/"
echo "=========================================================="
