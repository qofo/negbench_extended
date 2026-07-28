#!/bin/bash
# ==============================================================================
# Script: run_hypothesis_experiments.sh
# Master Execution Runner for 3-Stage Hypothesis Verification Pipeline
# ==============================================================================

set -e

# Automatically detect current script directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Create root symlink for 'data' if missing, so relative CSV image paths 'data/coco/...' resolve seamlessly
if [ ! -e "${ROOT_DIR}/data" ] && [ -d "${ROOT_DIR}/benchmarks/data" ]; then
    ln -sf "${ROOT_DIR}/benchmarks/data" "${ROOT_DIR}/data" || true
fi

# Add both root and benchmarks directory to PYTHONPATH to resolve internal imports (src.evaluation...)
export PYTHONPATH="${ROOT_DIR}/benchmarks:${ROOT_DIR}:${PYTHONPATH}"

MODEL="ViT-B-32"
PRETRAINED="openai"

# Locate dataset CSVs (check both root and benchmarks/ subdirectories)
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

SUBSPACE_DIR="${ROOT_DIR}/logs/subspace_analysis"
LOGS_DIR="${ROOT_DIR}/logs/evaluation/hypothesis_experiments"

mkdir -p "${SUBSPACE_DIR}"
mkdir -p "${LOGS_DIR}"

echo "=========================================================="
echo "  Executing Master Hypothesis Verification Pipeline"
echo "  Root Directory : ${ROOT_DIR}"
echo "  PYTHONPATH     : ${PYTHONPATH}"
echo "  Model          : ${MODEL} (${PRETRAINED})"
echo "  Paired CSV     : ${PAIRED_CSV}"
echo "  Subspace Dir   : ${SUBSPACE_DIR}"
echo "  Logs Dir       : ${LOGS_DIR}"
echo "=========================================================="

# ------------------------------------------------------------------------------
# Stage 1-A: Global Negation Subspace & Cross-Category Transfer Probe (H3)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 1-A] Executing Global Negation Subspace Analysis (H3)..."
python -m benchmarks.src.analysis.subspace_analysis \
    --model "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --csv_path "${PAIRED_CSV}" \
    --output_dir "${SUBSPACE_DIR}"

SUBSPACE_BASIS="${SUBSPACE_DIR}/negation_subspace_basis_top5.npy"
PROBE_WEIGHTS="${SUBSPACE_DIR}/linear_probe_weights.npz"

# ------------------------------------------------------------------------------
# Change working directory to benchmarks/ for eval_negation relative image paths
# ------------------------------------------------------------------------------
cd "${ROOT_DIR}/benchmarks"

# ------------------------------------------------------------------------------
# Stage 1-B: H1 Procrustes Orthogonal Causal Alignment (Isometric Test)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 1-B] Evaluating H1: Procrustes Orthogonal Alignment..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --name "h1_procrustes_orthogonal" \
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
# Stage 2: H2 Hyperplane Projection-Guided Cosine Metric
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 2] Evaluating H2: Hyperplane Projection-Guided Cosine Metric..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --name "h2_hyperplane_projection" \
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
# Stage 3: H4 Subspace-Constrained Bilinear Metric Tensor
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 3] Evaluating H4: Subspace-Constrained Bilinear Metric Tensor..."
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.eval_negation \
    --model "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --name "h4_subspace_bilinear" \
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
echo "✅ Hypothesis Verification Experiments Complete!"
echo "   Subspace Reports: ${SUBSPACE_DIR}"
echo "   Benchmark Logs:   ${LOGS_DIR}"
echo "=========================================================="
