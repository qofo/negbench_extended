#!/bin/bash

# ==============================================================================
# Script: run_unary_mechanistic_pipeline.sh
# Purpose: Execute 4-Stage Unary Mechanistic Analysis (E1 ~ E4)
# Stages:
#   - E1: Representation Presence/Polarity Probing
#   - E2: Cross-Modal Alignment (Probe Normal & Centroid Shift vs Random Null)
#   - E3: Alignment vs Cosine Correctness Margin (M) Correlation
#   - E4: Why Cosine Fails: Diagonal vs Off-Diagonal Bilinear Ablation (W = D + O)
# ==============================================================================

CSV_PATH="benchmarks/data/images/beaf_counterfactual_6col.csv"
OUTPUT_DIR="logs/evaluation/unary_mechanistic_analysis"
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"
MIN_PAIRS=6
BATCH_SIZE=256

echo "=========================================================="
echo "  Executing 4-Stage Unary Mechanistic Pipeline (E1 ~ E4)"
echo "  Input CSV   : ${CSV_PATH}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Model       : ${MODEL_NAME} (${PRETRAINED})"
echo "=========================================================="

python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis \
    --csv_path ${CSV_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} \
    --pretrained ${PRETRAINED} \
    --min_pairs ${MIN_PAIRS} \
    --batch_size ${BATCH_SIZE}

echo "=========================================================="
echo "Pipeline Execution Complete!"
echo "Generated Figures and Reports:"
echo "  1. ${OUTPUT_DIR}/fig1_representation_probing.png"
echo "  2. ${OUTPUT_DIR}/fig2_alignment_distribution.png"
echo "  3. ${OUTPUT_DIR}/fig3_alignment_vs_margin_scatter.png"
echo "  4. ${OUTPUT_DIR}/fig4_bilinear_ablation_bar.png"
echo "  5. ${OUTPUT_DIR}/e1_to_e4_summary_table.csv"
echo "  6. ${OUTPUT_DIR}/full_mechanistic_report.json"
echo "=========================================================="
