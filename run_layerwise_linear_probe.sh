#!/bin/bash

# ==============================================================================
# Script: run_layerwise_linear_probe.sh
# Purpose: Execute Stratified 5-Fold Linear Probe Analysis across ALL layers
# Dataset: COCO_val_full_paired.csv (or COCO_val_mcq_top100_paired.csv)
# Output Directory: logs/evaluation/linear_probe_layerwise
# ==============================================================================

PAIRED_CSV="COCO_val_full_paired.csv"
if [ ! -f "${PAIRED_CSV}" ]; then
    PAIRED_CSV="COCO_val_mcq_top100_paired.csv"
fi

OUTPUT_DIR="logs/evaluation/linear_probe_layerwise"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
TARGET_TOKEN="eot"
MAX_SAMPLES=60000
BATCH_SIZE=256
N_SPLITS=5

echo "=========================================================="
echo "  Executing CLIP Layer-wise Linear Probe Analysis"
echo "  Input CSV   : ${PAIRED_CSV}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "  Target Token: ${TARGET_TOKEN}"
echo "  5-Fold CV   : ${N_SPLITS} splits"
echo "=========================================================="

python -m benchmarks.src.evaluation.eval_layerwise_linear_probe \
    --csv_path ${PAIRED_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --target_token ${TARGET_TOKEN} \
    --max_samples ${MAX_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --n_splits ${N_SPLITS}

echo "=========================================================="
echo "Layer-wise Linear Probe Analysis Complete!"
echo "Generated output files:"
echo "  1. ${OUTPUT_DIR}/layerwise_linear_probe.csv"
echo "  2. ${OUTPUT_DIR}/layerwise_linear_probe.json"
echo "  3. ${OUTPUT_DIR}/layerwise_linear_probe.png"
echo "=========================================================="
