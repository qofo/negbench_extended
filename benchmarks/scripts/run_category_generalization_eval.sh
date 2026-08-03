#!/bin/bash
# ==============================================================================
# Script: run_category_generalization_eval.sh
# Evaluates Category Cross-Generalization (100% Unseen Category Split) for 6 Scoring Heads
# via standalone appended module src.evaluation.eval_category_generalization.
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$BASE_DIR/.."
DATA_DIR="$ROOT_DIR"
LOGS_DIR="$BASE_DIR/../logs/evaluation/category_generalization_experiments"

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
echo "  Executing Category Cross-Generalization Evaluation"
echo "  Model          : $MODEL ($PRETRAINED)"
echo "  Evaluation Mode: 100% Unseen Category Split (GroupKFold)"
echo "  COCO MCQ       : $COCO_MCQ"
echo "  Logs Dir       : $LOGS_DIR"
echo "=========================================================="

python -m src.evaluation.eval_category_generalization \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --coco-mcq "$COCO_MCQ" \
    --output-dir "$LOGS_DIR" \
    --group-col object_name \
    --n-splits 5 \
    --epochs 15 \
    --lr 0.001 \
    --batch-size 64 \
    --seed 42

echo ""
echo "=========================================================="
echo "✅ Category Cross-Generalization Evaluation Complete!"
echo "   Artifacts saved in: $LOGS_DIR"
echo "=========================================================="
