#!/bin/bash
# ==============================================================================
# Script: run_scoring_head_evaluations.sh
# Evaluates 6 Scoring Functions (Cosine, Weighted Cosine, Bilinear, Logistic Regression,
# Shallow MLP, Deep MLP) on CLIP MCQ benchmark using 5-Fold Cross Validation.
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$BASE_DIR/.."
DATA_DIR="$ROOT_DIR"
LOGS_DIR="$BASE_DIR/../logs/evaluation/scoring_head_experiments"

export PYTHONPATH="$BASE_DIR:$ROOT_DIR:$PYTHONPATH"

MODEL="ViT-B-32"
PRETRAINED="openai"
COCO_MCQ="$DATA_DIR/COCO_val_mcq_llama3.1_rephrased.csv"

if [ ! -f "$COCO_MCQ" ]; then
    COCO_MCQ="$BASE_DIR/data/images/COCO_val_mcq_llama3.1_rephrased.csv"
fi

mkdir -p "$LOGS_DIR"

cd "$BASE_DIR"

echo "=========================================================="
echo "  Executing Expressive Scoring Head Evaluation Pipeline"
echo "  Model      : $MODEL ($PRETRAINED)"
echo "  COCO MCQ   : $COCO_MCQ"
echo "  Logs Dir   : $LOGS_DIR"
echo "=========================================================="

python -m src.evaluation.eval_scoring_heads \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --coco-mcq "$COCO_MCQ" \
    --output-dir "$LOGS_DIR" \
    --n-splits 5 \
    --epochs 15 \
    --lr 0.001 \
    --batch-size 64 \
    --seed 42

echo ""
echo "=========================================================="
echo "✅ Scoring Head Evaluation Complete!"
echo "   Artifacts saved in: $LOGS_DIR"
echo "=========================================================="
