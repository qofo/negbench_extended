#!/bin/bash
# ==============================================================================
# Script: run_cc12m_negfull_experiments.sh
# Pipeline Runner for CC12M_negfull Fine-Tuned CLIP Model Analysis & Experiments
# ==============================================================================

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ------------------------------------------------------------------------------
# User Configuration: Specify your CC12M_negfull checkpoint path below
# ------------------------------------------------------------------------------
# Example checkpoint paths:
#   CHECKPOINT_PATH="benchmarks/models/cc12m_negfull.pt"
#   CHECKPOINT_PATH="/path/to/your/cc12m_negfull_checkpoint.pt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-benchmarks/models/cc12m_negfull.pt}"
MODEL_ARCH="${MODEL_ARCH:-ViT-B-32}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "⚠️ Warning: Checkpoint file not found at: ${CHECKPOINT_PATH}"
    echo "Please set CHECKPOINT_PATH variable or pass it as an environment variable:"
    echo "  CHECKPOINT_PATH=/path/to/cc12m_negfull.pt bash run_cc12m_negfull_experiments.sh"
    echo ""
fi

# Add root and benchmarks directories to PYTHONPATH
export PYTHONPATH="${ROOT_DIR}/benchmarks:${ROOT_DIR}:${PYTHONPATH}"

PAIRED_CSV="benchmarks/data/images/COCO_val_full_paired.csv"
if [ ! -f "${PAIRED_CSV}" ]; then PAIRED_CSV="data/images/COCO_val_full_paired.csv"; fi
if [ ! -f "${PAIRED_CSV}" ]; then PAIRED_CSV="COCO_val_full_paired.csv"; fi

COCO_MCQ="benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv"
if [ ! -f "${COCO_MCQ}" ]; then COCO_MCQ="data/images/COCO_val_mcq_llama3.1_rephrased.csv"; fi
if [ ! -f "${COCO_MCQ}" ]; then COCO_MCQ="COCO_val_mcq_llama3.1_rephrased.csv"; fi

COCO_RETRIEVAL="benchmarks/data/images/COCO_val_retrieval.csv"
if [ ! -f "${COCO_RETRIEVAL}" ]; then COCO_RETRIEVAL="data/images/COCO_val_retrieval.csv"; fi
if [ ! -f "${COCO_RETRIEVAL}" ]; then COCO_RETRIEVAL="COCO_val_retrieval.csv"; fi

COCO_NEGATED_RETRIEVAL="benchmarks/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
if [ ! -f "${COCO_NEGATED_RETRIEVAL}" ]; then COCO_NEGATED_RETRIEVAL="data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"; fi
if [ ! -f "${COCO_NEGATED_RETRIEVAL}" ]; then COCO_NEGATED_RETRIEVAL="COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"; fi

ANALYSIS_OUTPUT_DIR="${ROOT_DIR}/logs/analysis_modular/cc12m_negfull"
SUBSPACE_DIR="${ROOT_DIR}/logs/subspace_analysis_cc12m_negfull"
LOGS_DIR="${ROOT_DIR}/logs/evaluation/cc12m_negfull_experiments"

mkdir -p "${ANALYSIS_OUTPUT_DIR}"
mkdir -p "${SUBSPACE_DIR}"
mkdir -p "${LOGS_DIR}"

echo "=========================================================="
echo "  Executing CC12M_negfull Fine-Tuned CLIP Experiments"
echo "  Root Directory : ${ROOT_DIR}"
echo "  Architecture   : ${MODEL_ARCH}"
echo "  Checkpoint Path: ${CHECKPOINT_PATH}"
echo "  Analysis Output: ${ANALYSIS_OUTPUT_DIR}"
echo "  Subspace Dir   : ${SUBSPACE_DIR}"
echo "  Eval Logs Dir  : ${LOGS_DIR}"
echo "=========================================================="

# ------------------------------------------------------------------------------
# Step 1: Modular Representation Analysis on CC12M_negfull CLIP
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 1] Running Modular Representation Geometry Analysis..."
python -m benchmarks.src.analysis.run_analysis \
    --csv_path "${PAIRED_CSV}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}" \
    --model "${MODEL_ARCH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --target_token "eot" \
    --max_samples 60000 \
    --batch_size 256

# ------------------------------------------------------------------------------
# Step 2: Global Subspace Analysis (H3) on CC12M_negfull CLIP
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 2] Running Global Negation Subspace Analysis (H3)..."
python -m benchmarks.src.analysis.subspace_analysis \
    --model "${MODEL_ARCH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --csv_path "${PAIRED_CSV}" \
    --output_dir "${SUBSPACE_DIR}"

SUBSPACE_BASIS="${SUBSPACE_DIR}/negation_subspace_basis_top5.npy"
PROBE_WEIGHTS="${SUBSPACE_DIR}/linear_probe_weights.npz"

# ------------------------------------------------------------------------------
# Change working directory to benchmarks/ for eval_negation relative image paths
# ------------------------------------------------------------------------------
cd "${ROOT_DIR}/benchmarks"

# ------------------------------------------------------------------------------
# Step 3-A: Pure Projection Removal Ablation (Layer 12 Raw)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 3-A] Evaluating Pure Projection Removal (Layer 12 Raw)..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL_ARCH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --name "cc12m_pure_layer12_raw" \
    --logs "${LOGS_DIR}" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq "${COCO_MCQ}" \
    --coco-retrieval "${COCO_RETRIEVAL}" \
    --coco-negated-retrieval "${COCO_NEGATED_RETRIEVAL}" \
    --negation-method layer12_raw \
    --batch-size 64

# ------------------------------------------------------------------------------
# Step 3-B: H1 Procrustes Orthogonal Causal Alignment (Isometric Test)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 3-B] Evaluating H1: Procrustes Orthogonal Alignment..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL_ARCH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --name "cc12m_procrustes_orthogonal" \
    --logs "${LOGS_DIR}" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq "${COCO_MCQ}" \
    --coco-retrieval "${COCO_RETRIEVAL}" \
    --coco-negated-retrieval "${COCO_NEGATED_RETRIEVAL}" \
    --negation-method procrustes_orthogonal \
    --batch-size 64

# ------------------------------------------------------------------------------
# Step 3-C: H2 Hyperplane Projection-Guided Cosine Metric
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 3-C] Evaluating H2: Hyperplane Projection-Guided Cosine Metric..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL_ARCH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --name "cc12m_hyperplane_projection" \
    --logs "${LOGS_DIR}" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq "${COCO_MCQ}" \
    --coco-retrieval "${COCO_RETRIEVAL}" \
    --coco-negated-retrieval "${COCO_NEGATED_RETRIEVAL}" \
    --negation-method hyperplane_projection \
    --hyperplane-weight-path "${PROBE_WEIGHTS}" \
    --hyperplane-lambda 0.5 \
    --batch-size 64

# ------------------------------------------------------------------------------
# Step 3-D: H4 Subspace-Constrained Bilinear Metric Tensor
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 3-D] Evaluating H4: Subspace-Constrained Bilinear Metric Tensor..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL_ARCH}" \
    --pretrained "${CHECKPOINT_PATH}" \
    --name "cc12m_subspace_bilinear" \
    --logs "${LOGS_DIR}" \
    --dataset-type csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --coco-mcq "${COCO_MCQ}" \
    --coco-retrieval "${COCO_RETRIEVAL}" \
    --coco-negated-retrieval "${COCO_NEGATED_RETRIEVAL}" \
    --negation-method subspace_bilinear \
    --subspace-basis-path "${SUBSPACE_BASIS}" \
    --bilinear-alpha 0.5 \
    --batch-size 64

echo ""
echo "=========================================================="
echo "✅ CC12M_negfull Fine-Tuned CLIP Experiments Complete!"
echo "   Analysis Results: ${ANALYSIS_OUTPUT_DIR}"
echo "   Subspace Reports: ${SUBSPACE_DIR}"
echo "   Benchmark Logs:   ${LOGS_DIR}"
echo "=========================================================="
