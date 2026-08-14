#!/bin/bash
# ==============================================================================
# Script: run_vision_ablation_shortcut.sh
# Critique #3 — Vision Ablation Shortcut Diagnostic
#
# Tests whether trained scoring heads (Bilinear 74.8%, Deep MLP 86.7%) are
# actually using image information or exploiting text-only shortcuts.
#
# Ablation conditions:
#   Original  : Normal image embeddings (reference)
#   Zero      : All image embeddings = 0 vector
#   Shuffle   : Randomly permuted image embeddings across samples
#   Gaussian  : Random Gaussian vectors with matched norm
#
# If ablated accuracy ≈ original → TEXT-ONLY SHORTCUT DETECTED
# ==============================================================================

set -e

# Data and Output Settings
CSV_PATH="COCO_val_mcq_llama3.1_rephrased.csv"
OUTPUT_DIR="logs/evaluation/vision_ablation_shortcut"
MODEL="ViT-B-32"
PRETRAINED="openai"
EPOCHS=15
BATCH_SIZE=64
LR=1e-3
SEED=42

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmarks:$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PYTHONPATH"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo "  Vision Ablation Shortcut Diagnostic (Critique #3)"
echo "  Model      : ${MODEL} (${PRETRAINED})"
echo "  MCQ Dataset: ${CSV_PATH}"
echo "  Output Dir : ${OUTPUT_DIR}"
echo "======================================================================"

python -m benchmarks.src.evaluation.eval_vision_ablation_shortcut \
    --model ${MODEL} \
    --pretrained ${PRETRAINED} \
    --coco-mcq ${CSV_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED}

echo ""
echo "======================================================================"
echo "  Vision Ablation Shortcut Diagnostic Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "======================================================================"
