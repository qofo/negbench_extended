#!/bin/bash

# ==============================================================================
# Script: run_probe_failure_inspector.sh
# Purpose: Execute Vision & Text Linear Probing and Inspect Out-of-Fold Failures
# References:
#   - Vision: beaf_per_obj_test4 (~70.5% Val Acc via GroupKFold)
#   - Text: negation_existence_probe2 (75.35% Val Acc via StratifiedKFold)
# ==============================================================================

VISION_CSV="benchmarks/data/images/beaf_counterfactual_6col.csv"
TEXT_CSV="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv"
IMAGE_ROOT="benchmarks/data/images"
OUTPUT_DIR="logs/evaluation/probe_failure_inspection"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
BATCH_SIZE=128

echo "=========================================================="
echo "  Executing Vision & Text Failure Inspector"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "=========================================================="

python -m benchmarks.src.evaluation.eval_probe_failure_inspector \
    --vision_csv ${VISION_CSV} \
    --text_csv ${TEXT_CSV} \
    --image_root ${IMAGE_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --batch_size ${BATCH_SIZE}

echo "=========================================================="
echo "Failure Inspection Complete!"
echo "Inspect Failure Records:"
echo "  1. ${OUTPUT_DIR}/vision_probing_failures.csv"
echo "  2. ${OUTPUT_DIR}/text_probing_failures.csv"
echo "  3. ${OUTPUT_DIR}/top_failed_objects_breakdown.csv"
echo "  4. ${OUTPUT_DIR}/1_vision_train_val_summary.png"
echo "  5. ${OUTPUT_DIR}/1_text_train_val_summary.png"
echo "  6. ${OUTPUT_DIR}/fig_probe_failures_by_object.png"
echo "  7. ${OUTPUT_DIR}/fig_text_failure_patterns_by_cue.png"
echo "=========================================================="
