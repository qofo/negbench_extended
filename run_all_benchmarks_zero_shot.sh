#!/bin/bash
# ==============================================================================
# Master Script to Run Zero-Shot Transfer Across ALL Negation Benchmarks
# ==============================================================================

set -e

# Scorer Checkpoint and Output Directory Settings
SCORER_CKPT="logs/evaluation/scoring_head_experiments/checkpoints/deep_mlp_scorer.pt"
OUTPUT_DIR="logs/evaluation/all_benchmarks_transfer"
MODEL="ViT-B-32"
PRETRAINED="openai"

# Array of all target benchmark CSV files available in the workspace
BENCHMARKS=(
    "COCO_val_mcq_llama3.1_rephrased.csv"
    "beaf_counterfactual_6col.csv"
    "COCO_val_full_paired_v2.csv"
    "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    "COCO_val_retrieval.csv"
)

echo "======================================================================"
echo "Master Evaluation: Running Zero-Shot Transfer Across ALL Benchmarks"
echo " Scorer Checkpoint: ${SCORER_CKPT}"
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
            --scorer-ckpt ${SCORER_CKPT} \
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
