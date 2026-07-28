#!/bin/bash
# ==============================================================================
# Script: run_hypothesis_evaluations.sh
# 3-Stage Hypothesis Verification & Evaluation Runner
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$BASE_DIR/.."
DATA_DIR="$BASE_DIR/data"
LOGS_DIR="$BASE_DIR/../logs/evaluation/hypothesis_experiments"
SUBSPACE_DIR="$BASE_DIR/../logs/subspace_analysis"

# Symlink root 'data' to 'benchmarks/data' if missing
if [ ! -e "$ROOT_DIR/data" ] && [ -d "$DATA_DIR" ]; then
    ln -sf "$DATA_DIR" "$ROOT_DIR/data" || true
fi

export PYTHONPATH="$BASE_DIR:$ROOT_DIR:$PYTHONPATH"

MODEL="ViT-B-32"
PRETRAINED="openai"

COCO_MCQ="$DATA_DIR/images/COCO_val_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="$DATA_DIR/images/COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="$DATA_DIR/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"

mkdir -p "$LOGS_DIR"
mkdir -p "$SUBSPACE_DIR"

cd "$BASE_DIR"

echo "=========================================================="
echo "  Executing 3-Stage Hypothesis Verification Pipeline"
echo "  Base Directory: $BASE_DIR"
echo "  PYTHONPATH    : $PYTHONPATH"
echo "  Logs Directory: $LOGS_DIR"
echo "=========================================================="

# ------------------------------------------------------------------------------
# Stage 1: Global Negation Subspace & Procrustes Causal Alignment
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 1-A] Running Global Negation Subspace Analysis (H3)..."
python -m src.analysis.subspace_analysis \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --output_dir "$SUBSPACE_DIR"

SUBSPACE_BASIS="$SUBSPACE_DIR/negation_subspace_basis_top5.npy"
PROBE_WEIGHTS="$SUBSPACE_DIR/linear_probe_weights.npz"

echo ""
echo ">>> [Stage 1-B] Evaluating H1: Procrustes Orthogonal Alignment (Isometric Test)..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --name "h1_procrustes_orthogonal" \
    --logs "$LOGS_DIR" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq="$COCO_MCQ" \
    --coco-retrieval="$COCO_RETRIEVAL" \
    --coco-negated-retrieval="$COCO_NEGATED_RETRIEVAL" \
    --negation-method procrustes_orthogonal \
    --batch-size=64

# ------------------------------------------------------------------------------
# Stage 2: Hyperplane Projection-Guided Metric (H2)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 2] Evaluating H2: Hyperplane Projection-Guided Cosine Metric..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --name "h2_hyperplane_projection" \
    --logs "$LOGS_DIR" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq="$COCO_MCQ" \
    --coco-retrieval="$COCO_RETRIEVAL" \
    --coco-negated-retrieval="$COCO_NEGATED_RETRIEVAL" \
    --negation-method hyperplane_projection \
    --hyperplane-weight-path "$PROBE_WEIGHTS" \
    --hyperplane-lambda 0.5 \
    --batch-size=64

# ------------------------------------------------------------------------------
# Stage 3: Subspace-Constrained Bilinear Metric Tensor (H4)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 3] Evaluating H4: Subspace-Constrained Bilinear Metric Tensor..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --name "h4_subspace_bilinear" \
    --logs "$LOGS_DIR" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq="$COCO_MCQ" \
    --coco-retrieval="$COCO_RETRIEVAL" \
    --coco-negated-retrieval="$COCO_NEGATED_RETRIEVAL" \
    --negation-method subspace_bilinear \
    --subspace-basis-path "$SUBSPACE_BASIS" \
    --bilinear-alpha 0.5 \
    --batch-size=64

echo ""
echo "=========================================================="
echo "✅ Hypothesis Verification Evaluation Pipeline Complete!"
echo "   Results saved in: $LOGS_DIR"
echo "=========================================================="
