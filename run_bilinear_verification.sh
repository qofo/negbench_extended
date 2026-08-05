#!/bin/bash
# ==============================================================================
# Script to Run Bilinear Scorer Verification Experiments (Exp 1, Exp 2, Exp 3)
# ==============================================================================

set -e

# Data and Output Settings
CSV_PATH="COCO_val_mcq_llama3.1_rephrased.csv"
OUTPUT_DIR="logs/evaluation/scoring_head_experiments"
MODEL="ViT-B-32"
PRETRAINED="openai"
BATCH_SIZE=64
LR=1e-3

echo "======================================================================"
echo "Running Bilinear Verification Experiments:"
echo " 1. Full Bilinear Identity Init vs. Random Normal(0.02) Init"
echo " 2. Epoch Convergence Sweep (15, 30, 50, 100 Epochs)"
echo " 3. Mathematical Equivalence & Weight Transfer Check (W = A^T B)"
echo "======================================================================"

python -m benchmarks.src.evaluation.eval_bilinear_verification \
    --model ${MODEL} \
    --pretrained ${PRETRAINED} \
    --coco-mcq ${CSV_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --epochs-list 15 30 50 100 \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR}

echo ""
echo "======================================================================"
echo "✅ Verification experiments completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}/bilinear_verification_results.json"
echo "Plot saved to:    ${OUTPUT_DIR}/bilinear_verification_convergence.png"
echo "======================================================================"
