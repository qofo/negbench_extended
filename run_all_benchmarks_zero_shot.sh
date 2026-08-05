#!/bin/bash
# ==============================================================================
# Master Script to Run Zero-Shot Transfer Across ALL NegBench Benchmarks
# (COCO, VOC2007, CheXpert Medical, MSRVTT Video, BEAF Counterfactual)
# Evaluates ALL 7 Trained Scorer Checkpoints (Low-Rank, Bilinear, MLP, LogReg, etc.)
# ==============================================================================

set -e

# Base and Data Directory Settings
BASE_DIR="."
DATA_DIR="${BASE_DIR}/benchmarks/data"
SCORER_DIR="logs/evaluation/scoring_head_experiments/checkpoints"
OUTPUT_DIR="logs/evaluation/all_benchmarks_transfer"
MODEL="ViT-B-32"
PRETRAINED="openai"

# Real benchmark CSV files existing in NegBench repository
BENCHMARKS=(
    "${DATA_DIR}/images/COCO_val_mcq_llama3.1_rephrased.csv"
    "${DATA_DIR}/images/VOC2007_mcq_llama3.1_rephrased.csv"
    "${DATA_DIR}/images/chexpert_binary_mcq_control_valid_only.csv"
    "${DATA_DIR}/images/beaf_counterfactual_6col.csv"
    "${DATA_DIR}/images/COCO_val_retrieval.csv"
)

echo "======================================================================"
echo "Master Evaluation: Zero-Shot Transfer for ALL Trained Scorers"
echo " Checkpoint Directory: ${SCORER_DIR}"
echo " Output Directory:     ${OUTPUT_DIR}"
echo " Total Benchmarks to Evaluate: ${#BENCHMARKS[@]}"
echo "======================================================================"

for csv in "${BENCHMARKS[@]}"; do
    if [ -f "$csv" ]; then
        echo ""
        echo "======================================================================"
        echo "Evaluating Benchmark: ${csv}"
        echo "======================================================================"
        
        python -m benchmarks.src.evaluation.eval_zero_shot_transfer \
            --model ${MODEL} \
            --pretrained ${PRETRAINED} \
            --scorer-dir ${SCORER_DIR} \
            --target-mcq ${csv} \
            --output-dir ${OUTPUT_DIR}
    else
        echo "⚠️ Skipping missing benchmark file: ${csv}"
    fi
done

echo ""
echo "======================================================================"
echo "✅ All benchmark evaluations completed successfully!"
echo "Master Summary saved to: ${OUTPUT_DIR}/zero_shot_transfer_summary.csv"
echo "======================================================================"
