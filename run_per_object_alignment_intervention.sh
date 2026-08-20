#!/bin/bash

# ==============================================================================
# Script: run_per_object_alignment_intervention.sh
# Purpose: Execute Per-Object Probe Alignment Causal Intervention (d_I vs d_T)
# Compares:
#   1. Baseline Cosine (A = I)
#   2. Closed-Form 2D Rotation (R in SO(d), cos = 1.0)
#   3. Rank-1 Polar Adapter (A_rank1)
#   4. Learned Bilinear (W)
#   5. Random Rotation (R_rand)
# ==============================================================================

OUTPUT_DIR="logs/evaluation/per_object_alignment_intervention"
VISION_CSV="benchmarks/data/images/beaf_counterfactual_6col.csv"
TEXT_CSV="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv"
IMAGE_ROOT="benchmarks/data/images"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"

echo "=========================================================="
echo "  Executing Per-Object Probe Alignment Intervention"
echo "  Output Dir : ${OUTPUT_DIR}"
echo "  Model      : ${MODEL_NAME} (${PRETRAINED})"
echo "=========================================================="

python -m benchmarks.src.evaluation.eval_per_object_alignment_intervention \
    --vision_csv ${VISION_CSV} \
    --text_csv ${TEXT_CSV} \
    --image_root ${IMAGE_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED}

echo "=========================================================="
echo "Intervention Complete!"
echo "Inspect Results:"
echo "  1. ${OUTPUT_DIR}/per_object_intervention_results.csv"
echo "  2. ${OUTPUT_DIR}/fig_intervention_5conditions_bar.png"
echo "  3. ${OUTPUT_DIR}/fig_alignment_vs_gain_scatter.png"
echo "  4. ${OUTPUT_DIR}/fig_per_object_gain_waterfall.png"
echo "  5. ${OUTPUT_DIR}/per_object_intervention_summary.json"
echo "=========================================================="
