#!/bin/bash
# ==============================================================================
# Script to Run Zero-Shot Cross-Dataset Transfer Evaluation for Scoring Heads
# ==============================================================================

set -e

# Data and Checkpoint Paths
SCORER_CKPT="logs/evaluation/scoring_head_experiments/checkpoints/deep_mlp_scorer.pt"
TARGET_MCQ="beaf_counterfactual_6col.csv"
OUTPUT_DIR="logs/evaluation/zero_shot_transfer"
MODEL="ViT-B-32"
PRETRAINED="openai"

echo "======================================================================"
echo "Running Zero-Shot Cross-Dataset Transfer Evaluation"
echo " Target Benchmark: ${TARGET_MCQ}"
echo " Scorer Checkpoint: ${SCORER_CKPT}"
echo "======================================================================"

python -m benchmarks.src.evaluation.eval_zero_shot_transfer \
    --model ${MODEL} \
    --pretrained ${PRETRAINED} \
    --scorer-ckpt ${SCORER_CKPT} \
    --target-mcq ${TARGET_MCQ} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "======================================================================"
echo "✅ Zero-Shot Transfer evaluation completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "======================================================================"
