#!/bin/bash
# ==============================================================================
# Script to Run Rank Sweep & Concept Vector Subtraction Experiments
# ==============================================================================

set -e

# Data and Output Settings
CSV_PATH="COCO_val_mcq_llama3.1_rephrased.csv"
OUTPUT_DIR="logs/evaluation/scoring_head_experiments"
MODEL="ViT-B-32"
PRETRAINED="openai"
EPOCHS=15
BATCH_SIZE=64
LR=1e-3

echo "======================================================================"
echo "1. Running Rank-k Sweep Experiment (Low-Rank & Non-Linear Bi-Encoder)"
echo "======================================================================"
python -m benchmarks.src.evaluation.eval_rank_sweep \
    --model ${MODEL} \
    --pretrained ${PRETRAINED} \
    --coco-mcq ${CSV_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --ranks 1 2 4 8 16 32 64 128 256 512 \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR}

echo ""
echo "======================================================================"
echo "2. Running Concept Vector Subtraction (Negation Vector Ablation)"
echo "======================================================================"
python -m benchmarks.src.evaluation.eval_concept_ablation \
    --model ${MODEL} \
    --pretrained ${PRETRAINED} \
    --coco-mcq ${CSV_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --best-rank 32 \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR}

echo ""
echo "======================================================================"
echo "✅ All experiments completed successfully! Results saved to: ${OUTPUT_DIR}"
echo "======================================================================"
