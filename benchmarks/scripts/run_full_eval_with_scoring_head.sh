#!/bin/bash
# ==============================================================================
# Script: run_full_eval_with_scoring_head.sh
# 1. Trains and exports Deep MLP Scorer checkpoint (.pt)
# 2. Runs eval_negation.py using the trained Deep MLP Scorer across ALL benchmarks:
#    - VOC2007 MCQ
#    - CheXpert MCQ / Binary MCQ
#    - MSR-VTT Video MCQ
#    - COCO Image Retrieval & Negated Retrieval
#    - MSR-VTT Video Retrieval & Negated Retrieval
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$BASE_DIR/.."
DATA_DIR="$ROOT_DIR"
LOGS_DIR="$BASE_DIR/../logs/evaluation/scoring_head_full_benchmarks"
CKPT_DIR="$LOGS_DIR/checkpoints"

export PYTHONPATH="$BASE_DIR:$ROOT_DIR:$PYTHONPATH"

MODEL="ViT-B-32"
PRETRAINED="openai"
SCORER_TYPE="deep_mlp"
SCORER_CKPT="$CKPT_DIR/deep_mlp_scorer.pt"

COCO_MCQ="$DATA_DIR/COCO_val_mcq_llama3.1_rephrased.csv"
VOC_MCQ="$DATA_DIR/VOC2007_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="$DATA_DIR/COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="$DATA_DIR/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
MSRVTT_RETRIEVAL="$DATA_DIR/videos/MSRVTT/msr_vtt_retrieval.csv"
MSRVTT_MCQ="$DATA_DIR/videos/MSRVTT/negation/msr_vtt_mcq_rephrased_llama.csv"

mkdir -p "$LOGS_DIR"
mkdir -p "$CKPT_DIR"

cd "$BASE_DIR"

echo "=========================================================="
echo " Stage 1: Train & Export Deep MLP Scorer Checkpoint"
echo " Model      : $MODEL ($PRETRAINED)"
echo " Save Path  : $SCORER_CKPT"
echo "=========================================================="

python -m src.evaluation.eval_scoring_heads \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --coco-mcq "$COCO_MCQ" \
    --output-dir "$LOGS_DIR" \
    --save-scorer-path "$SCORER_CKPT" \
    --save-scorer-model "Deep MLP" \
    --epochs 15 \
    --lr 0.001 \
    --batch-size 64

echo ""
echo "=========================================================="
echo " Stage 2: Evaluate Deep MLP Scorer across ALL NegBench Datasets"
echo " (VOC2007 MCQ, CheXpert, MSRVTT Video, COCO Retrieval, etc.)"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --name "deep_mlp_scoring_head_full_eval" \
    --logs "$LOGS_DIR" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq="$COCO_MCQ" \
    --coco-retrieval="$COCO_RETRIEVAL" \
    --coco-negated-retrieval="$COCO_NEGATED_RETRIEVAL" \
    --voc2007-mcq="$VOC_MCQ" \
    --negation-method "scoring_head" \
    --scorer-checkpoint "$SCORER_CKPT" \
    --scorer-type "$SCORER_TYPE" \
    --batch-size=64

echo ""
echo "=========================================================="
echo "✅ Full Evaluation with Trained Deep MLP Scorer Complete!"
echo "   Artifacts saved in: $LOGS_DIR"
echo "=========================================================="
