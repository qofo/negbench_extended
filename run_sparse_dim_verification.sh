#!/bin/bash
# ==============================================================================
# Pipeline: Sparse Dimension Analysis (Critique #1) for Text and Vision
# ==============================================================================

set -e

BASE_DIR="."
CSV_PATH="${BASE_DIR}/benchmarks/data/images/COCO_val_full_paired.csv"
BEAF_CSV="${BASE_DIR}/benchmarks/data/images/beaf_counterfactual_6col.csv"
OUT_DIR="${BASE_DIR}/logs/evaluation/sparse_text_dimensions"

echo "======================================================================"
echo "🚀 Running Sparse Dimension Verification (Critique #1) - Text & Vision"
echo "======================================================================"

python -m benchmarks.src.analysis.eval_sparse_text_dimensions \
    --model ViT-B-32 \
    --pretrained openai \
    --csv_path "${CSV_PATH}" \
    --beaf_csv "${BEAF_CSV}" \
    --output_dir "${OUT_DIR}"

echo "======================================================================"
echo "✅ Done! Output directory: ${OUT_DIR}"
echo "======================================================================"
