#!/bin/bash
# ==============================================================================
# Script to Run Top-Priority Unexecuted Experiments
# 1. Concept Vector Subtraction (Intervention Mode A vs. Retrained Mode B)
# 2. Counterfactual Word-Swap Text Probe (Token-Presence Bias Disentanglement)
# ==============================================================================

set -e

# Data and Output Settings
CSV_PATH="COCO_val_mcq_llama3.1_rephrased.csv"
OUTPUT_DIR="logs/evaluation/top_priority_experiments"
MODEL="ViT-B-32"
PRETRAINED="openai"
EPOCHS=15
BATCH_SIZE=64
LR=1e-3

echo "======================================================================"
echo "1. Running Concept Vector Subtraction (Mode A Intervention vs Mode B Retrain)"
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
echo "2. Running Counterfactual Word-Swap Text Probe (Token-Presence Bias)"
echo "======================================================================"
python -m benchmarks.src.evaluation.eval_word_swap_probe \
    --model ${MODEL} \
    --pretrained ${PRETRAINED} \
    --coco-mcq ${CSV_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE}

echo ""
echo "======================================================================"
echo "✅ All top-priority unexecuted experiments completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "======================================================================"
