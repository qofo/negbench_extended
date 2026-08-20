#!/bin/bash

# ==============================================================================
# Script: run_ab_swap_diagnostic.sh
# Purpose: BEAF A/B Swap Compositional Negation Diagnostic
#          (Text Sanity Probe + Unary vs Compound + Margin Analysis)
# Dataset: beaf_counterfactual_ab_swap.csv (2,771 counterfactual pairs)
# Output:  logs/evaluation/ab_swap_diagnostic/
# ==============================================================================

CSV_PATH="benchmarks/data/images/beaf_counterfactual_ab_swap.csv"
OUTPUT_DIR="logs/evaluation/ab_swap_diagnostic"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
BATCH_SIZE=256

echo "=========================================================="
echo "  BEAF A/B Swap Compositional Negation Diagnostic"
echo "  Input CSV   : ${CSV_PATH}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "=========================================================="

python -m benchmarks.src.evaluation.eval_ab_swap_negation_diagnostic \
    --csv_path ${CSV_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --batch_size ${BATCH_SIZE}

echo "=========================================================="
echo "Diagnostic Complete!"
echo "Generated output files:"
echo "  1. ${OUTPUT_DIR}/exp1_layerwise_sanity_probe.png"
echo "  2. ${OUTPUT_DIR}/exp1_layerwise_sanity_probe.csv"
echo "  3. ${OUTPUT_DIR}/exp2_unary_vs_compound.png (or exp2_text_separability.png)"
echo "  4. ${OUTPUT_DIR}/exp2_6score_matrix.csv (if images available)"
echo "  5. ${OUTPUT_DIR}/exp2_summary.json"
echo "  6. ${OUTPUT_DIR}/exp3_margin_histogram.png (if images available)"
echo "  7. ${OUTPUT_DIR}/exp3_margin_summary.json (if images available)"
echo "  8. ${OUTPUT_DIR}/full_diagnostic_report.json"
echo "=========================================================="
